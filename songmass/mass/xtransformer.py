# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from collections import OrderedDict

from fairseq import utils
from fairseq.models import FairseqMultiModel, register_model, register_model_architecture, BaseFairseqModel

from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
)

from .masked_attention_decoder_layer import MaskedAttentionDecoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F


class XTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.mask_idx = dictionary.mask_index

    def forward(self, src_tokens, src_lengths,
                source_sent_ids=None, target_sent_ids=None):

        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx) | src_tokens.eq(self.mask_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)
            
        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'source_sent_ids': source_sent_ids # B x S
        }
    
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['source_sent_ids'] is not None:
            encoder_out['source_sent_ids'] = \
                encoder_out['source_sent_ids'].index_select(0, new_order)      
        
        return encoder_out


class XTransformerDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        
        self.layers = nn.ModuleList([])
        self.layers.extend([
            MaskedAttentionDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])
        self.cnt = 0
        
    def forward(self, prev_output_tokens, encoder_out=None,
                incremental_state=None, positions=None):
        if encoder_out is not None and type(encoder_out) == type(dict()) \
            and 'source_sent_ids' in encoder_out.keys() and encoder_out['source_sent_ids'] is not None:

            src_len = encoder_out['source_sent_ids'].size()[-1]
            tgt_len = prev_output_tokens.size()[1]
            beam_batch_size = prev_output_tokens.size()[0]

            source_sent_ids = encoder_out['source_sent_ids']
            is_sep = prev_output_tokens.eq(5).int()
            target_sent_ids = is_sep.cumsum(dim=1)
            
            # T is current time step
            s = source_sent_ids.unsqueeze(1).repeat(1, tgt_len, 1)
            t = target_sent_ids.unsqueeze(2).repeat(1, 1, src_len)
            sent_mask = torch.ne(s, t) 
            sent_mask = sent_mask[:, -1, :]
            sent_mask = sent_mask.unsqueeze(1)
            encoder_out['encoder_padding_mask'] = sent_mask

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
            positions=positions,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn, attns = None, []

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn, _ = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                need_attn=True,
            )
            inner_states.append(x)
            attns.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states, 'attns': attns}


@register_model('xtransformer')
class XTransformerModel(BaseFairseqModel):
    def __init__(self, encoders, decoders, eval_lang_pair=None):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.tgt_key = None
        if eval_lang_pair is not None:
            self.source_lang = eval_lang_pair.split('-')[0]
            self.target_lang = eval_lang_pair.split('-')[1]

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif hasattr(self, 'decoders'):
            return self.decoders[self.tgt_key].get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        return None

    def max_decoder_positions(self):
        return min(decoder.max_positions() for decoder in self.decoders.values())

    def forward(self, src_tokens, src_lengths, prev_output_tokens,
                source_sent_ids, target_sent_ids, src_key, tgt_key, positions=None):

        encoder_out = self.encoders[src_key](src_tokens, src_lengths)

        input_encoder_out = encoder_out['encoder_out']
        input_encoder_padding_mask = encoder_out['encoder_padding_mask']

        src_len = src_tokens.size()[1]
        tgt_len = prev_output_tokens.size()[1]
        # (B, S) -> (B,1,S) -> (B,T,S)
        s = source_sent_ids.unsqueeze(1).repeat(1, tgt_len, 1)
        # (B, T) -> (B,T,1) -> (B,T,S)
        t = target_sent_ids.unsqueeze(2).repeat(1, 1, src_len)

        sent_mask = torch.ne(s, t)
        encoder_out['encoder_padding_mask'] = sent_mask

        decoder_out = self.decoders[tgt_key](
            prev_output_tokens,
            encoder_out=encoder_out,
            positions=positions
        )
        self.tgt_key = tgt_key
        return decoder_out

    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')

    @classmethod
    def build_model(cls, args, task):
        langs = [lang for lang in args.langs]

        embed_tokens = {}
        for lang in langs:
            if len(embed_tokens) == 0 or args.share_all_embeddings is False:
                embed_token = build_embedding(
                    task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path,
                )
                embed_tokens[lang] = embed_token
            else:
                embed_tokens[lang] = embed_tokens[langs[0]]

        args.share_decoder_input_output_embed = True
        encoders, decoders = {}, {}

        for lang in langs:
            encoder_embed_tokens = embed_tokens[lang]
            decoder_embed_tokens = encoder_embed_tokens
            if lang in args.source_langs:
                encoder = XTransformerEncoder(args, task.dicts[lang], encoder_embed_tokens)
                encoders[lang] = encoder
            if lang in args.target_langs:
                decoder = XTransformerDecoder(args, task.dicts[lang], decoder_embed_tokens)
                decoders[lang] = decoder
        return XTransformerModel(encoders, decoders, args.eval_lang_pair)

    @property
    def decoder(self):
        return self.decoders[self.target_lang]

    @property
    def encoder(self):
        return self.encoders[self.source_lang]


@register_model_architecture('xtransformer', 'xtransformer')
def base_x_transformer(args):
    base_architecture(args)


def build_embedding(dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb
