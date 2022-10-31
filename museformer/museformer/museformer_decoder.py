import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from fairseq import utils
from fairseq.models import FairseqDecoder
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.layer_norm import LayerNorm

from .tools import instant_info_construction as iic
from .tools import arg_tools, computation_tools
from .data_structures.four_dim_pocket import FourDimPocket
from .kernels.range_fill import range_fill
from .kernels.segment_arange import segment_arange
from .museformer_decoder_layer import MuseformerDecoderLayer
from .data_structures.attention_scheme import AttentionScheme
from .embedding.tao_embedding import TaoEmbedding
from .datasets.add_beat_dataset import get_beat_ids


def construct_reg_bar_ids(chunk_ranges, num_chunks, reg_len):
    device = chunk_ranges.device
    bar_ids = []
    for sample_ranges, sample_num_chunk in zip(chunk_ranges, num_chunks):
        if sample_num_chunk == 0:
            sample_bar_ids = torch.zeros(reg_len, dtype=torch.long, device=device)
        else:
            sample_bar_ids = range_fill(sample_ranges, torch.arange(1, sample_num_chunk + 1, device=device),
                                        reg_len, pad_value=0)
        bar_ids.append(sample_bar_ids)
    bar_ids = torch.stack(bar_ids, dim=0)
    return bar_ids  # (bsz, reg_len)


def construct_reg_token_in_chunk_ids(chunk_ranges, num_chunks, reg_len):
    device = chunk_ranges.device
    ids = []
    for sample_ranges, sample_num_chunk in zip(chunk_ranges, num_chunks):
        if sample_num_chunk == 0:
            sample_ids = torch.zeros(reg_len, dtype=torch.long, device=device)
        else:
            sample_ranges = sample_ranges[:sample_num_chunk]
            sample_ids = segment_arange(sample_ranges, 1, reg_len, 0, dtype=torch.long, no_cuda_kernel=False)
        ids.append(sample_ids)
    ids = torch.stack(ids, dim=0)
    return ids


class MuseformerDecoder(FairseqDecoder):
    _submodules = (MuseformerDecoderLayer,)

    @classmethod
    def add_args(cls, parser):
        # === Implementation ===
        parser.add_argument('--attention-impl', choices=('mask', 'blocksparse', 'sparta'))
        parser.add_argument('--block-size', type=int, choices=(64, 32))
        parser.add_argument('--attention-mode', choices=('v2s1',))

        # === Transformer ===
        parser.add_argument('--attention-embed-dim', type=int)
        parser.add_argument('--num-layers', type=int)
        parser.add_argument('--num-attention-heads', type=eval)
        parser.add_argument('--normalize-before', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--ffn-embed-dim', type=int)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--attention-dropout', type=float)
        parser.add_argument('--activation-fn', choices=utils.get_available_activation_fns())
        parser.add_argument('--no-final-norm', type=arg_tools.str_bool_with_default_error)

        # === Attention Scheme ===
        parser.add_argument('--con2con', type=eval)
        parser.add_argument('--con2con-self', choices=('default', 'none', 'full'))
        parser.add_argument('--con2sum', type=eval)
        parser.add_argument('--con2sum-self', choices=('default', 'none', 'full'))
        parser.add_argument('--sum2con', type=eval)
        parser.add_argument('--sum2con-self', choices=('default', 'none', 'full'))
        parser.add_argument('--sum2sum', type=eval)
        parser.add_argument('--sum2sum-self', choices=('default', 'none', 'full'))
        parser.add_argument('--con2con-causal', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--pref2pref-mode', choices=('full', 'lt', 'none'))
        parser.add_argument('--sum2pref-mode', choices=('none',))
        parser.add_argument('--pref2sum-mode', choices=('none',))
        parser.add_argument('--con2pref-mode', choices=('full',))
        parser.add_argument('--pref2con-mode', choices=('none',))

        # === Summary Tokens ===
        parser.add_argument('--num-summary-tokens-each-chunk', type=int)
        parser.add_argument('--max-summary-tokens', type=int)

        # === Embedding ===
        parser.add_argument('--concat-sum-embeddings', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--concat-reg-embeddings', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--proj-sum-embeddings', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--proj-reg-embeddings', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--share-embedding-projection', type=arg_tools.str_bool_with_default_error)

        parser.add_argument('--share-input-output-embed', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--no-scale-embedding', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--layernorm-embedding', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--share-layernorm-embedding', type=arg_tools.str_bool_with_default_error)

        # token embedding
        parser.add_argument('--token-embed-dim', type=int)

        # absolute token in-chunk-position embedding
        parser.add_argument('--use-token-in-chunk-abs-pos', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--learned-token-in-chunk-abs-pos', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--max-token-in-chunk-abs-pos', type=int)
        parser.add_argument('--token-in-chunk-abs-pos-embed-dim', type=int)

        # absolute token position embedding
        parser.add_argument('--use-token-abs-pos', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--learned-token-abs-pos', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--max-token-abs-pos', type=int)
        parser.add_argument('--token-abs-pos-embed-dim', type=int)

        # absolute bar position embedding
        parser.add_argument('--use-bar-abs-pos', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--learned-bar-abs-pos', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--max-bar-abs-pos', type=int)
        parser.add_argument('--bar-abs-pos-embed-dim', type=int)
        parser.add_argument('--valid-parts-for-bar-abs-pos', type=arg_tools.comma_split_tuple_of_specific_type(str))

        # absolute beat position embedding
        parser.add_argument('--use-beat-abs-pos', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--learned-beat-abs-pos', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--max-beat-abs-pos', type=int)
        parser.add_argument('--beat-abs-pos-embed-dim', type=int)
        parser.add_argument('--beat-abs-pos-padding-on-sum', type=arg_tools.str_bool_with_default_error)

        # === Gradient Checkpointing ===
        parser.add_argument('--gradient-checkpointing', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--gradient-checkpointing-every-n-layer', type=int)
        parser.add_argument('--gradient-checkpointing-layers',
                            type=lambda x: tuple([int(item) for item in x.split(',')]))

        arg_tools.add_submodule_args(cls, parser)

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args

        # === Meta Preparations ===
        # pocket to store things that can be used everywhere
        self.pocket = FourDimPocket()
        self.pocket['constant'] = self.pocket_constant = {}
        self.pocket['instant'] = self.pocket_instant = {}

        # --- control parameters ---
        self.attention_impl = self.args.attention_impl
        self.block_size = getattr(self.args, "block_size", None)
        if self.attention_impl == 'blocksparse':
            if self.block_size is None:
                self.block_size = 64
        self.attention_mode = getattr(self.args, "attention_mode", "simu")
        self.attn_mask_gen_parts = None  # which parts are involved for attn_mask
        if self.attention_mode == 'v2s1':
            self.attn_mask_combination = {'sx': ('ss', 'sr'), 'rx': ('rs', 'rr')}
        else:
            raise NotImplementedError
        self.need_embedding_summary_tokens = True

        self.pocket_constant['attention_impl'] = self.attention_impl
        self.pocket_constant['block_size'] = self.block_size
        self.pocket_constant['attention_mode'] = self.attention_mode
        self.pocket_constant['attn_mask_combination'] = self.attn_mask_combination
        self.pocket_constant['attn_mask_gen_parts'] = self.attn_mask_gen_parts
        self.pocket_constant['need_embedding_summary_tokens'] = self.need_embedding_summary_tokens

        # === Basic Parameters ===
        self.embed_dim = self.args.attention_embed_dim
        self.num_layers = self.args.num_layers
        self.num_attention_heads = arg_tools.possibly_extend_tuple(
            self.args.num_attention_heads, self.num_layers
        )  # tuple of size num_layers
        self.valid_dictionary_len = getattr(self.args, 'valid_dict_len', len(dictionary))

        # === Embedding ===
        self.embed_scale = None if args.no_scale_embedding else math.sqrt(self.args.token_embed_dim)

        # --- Regular Token Embedding ---
        self.embed_regular_tokens = nn.Embedding(
            self.valid_dictionary_len, self.args.token_embed_dim, self.dictionary.pad()
        )
        # --- Summary Token Embedding ---
        self.num_summary = self.args.num_summary_tokens_each_chunk
        assert self.num_summary >= 0
        if self.num_summary == 0 or not self.need_embedding_summary_tokens:
            self.embed_summary_tokens = None
        else:
            self.embed_summary_tokens = nn.Embedding(
                getattr(self.args, 'max_summary_tokens', self.num_summary) + 1,
                self.args.token_embed_dim,
                padding_idx=0
            )

        # --- Absolute Token In-Chunk Absolute Position Embedding ---
        self.use_token_in_chunk_abs_pos = getattr(self.args, 'use_token_in_chunk_abs_pos', False)
        if self.use_token_in_chunk_abs_pos:
            token_in_chunk_abs_pos_embed_dim = getattr(self.args, "token_in_chunk_abs_pos_embed_dim",
                                                       self.args.token_embed_dim)
            self.embed_token_in_chunk_abs_pos = TaoEmbedding(
                self.args.max_token_in_chunk_abs_pos + 1, token_in_chunk_abs_pos_embed_dim, padding_idx=0,
                learned=getattr(self.args, 'learned_token_in_chunk_abs_pos', False)
            )
        else:
            self.embed_token_in_chunk_abs_pos = None
            token_in_chunk_abs_pos_embed_dim = 0

        # --- Absolute Token Position Embedding ---
        self.use_token_abs_pos = getattr(self.args, 'use_token_abs_pos', False)
        if self.use_token_abs_pos:
            token_abs_pos_embed_dim = getattr(self.args, "token_abs_pos_embed_dim", self.args.token_embed_dim)
            if getattr(self.args, 'learned_token_abs_pos', False):
                raise NotImplementedError("The support for overly long token absolute position embedding is not done.")
            self.embed_token_abs_pos = TaoEmbedding(
                self.args.max_token_abs_pos, token_abs_pos_embed_dim, padding_idx=0,
                learned=getattr(self.args, 'learned_token_abs_pos', False)
            )
        else:
            self.embed_token_abs_pos = None
            token_abs_pos_embed_dim = 0

        # --- Absolute Bar Position Embedding ---
        self.use_bar_abs_pos = getattr(self.args, 'use_bar_abs_pos', False)
        if self.use_bar_abs_pos:
            bar_abs_pos_embed_dim = getattr(self.args, "bar_abs_pos_embed_dim", self.args.token_embed_dim)
            # if getattr(self.args, 'learned_bar_abs_pos', False):
            #     raise NotImplementedError("The support for overly long bar absolute position embedding is not done.")
            self.embed_bar_abs_pos = TaoEmbedding(
                self.args.max_bar_abs_pos + 1, bar_abs_pos_embed_dim, padding_idx=0,
                learned=getattr(self.args, 'learned_bar_abs_pos', False)
            )
            self.valid_parts_for_bar_abs_pos = getattr(self.args, 'valid_parts_for_bar_abs_pos', None)
        else:
            self.embed_bar_abs_pos = None
            bar_abs_pos_embed_dim = 0
            self.valid_parts_for_bar_abs_pos = None

        # --- Absolute Beat Position Embedding ---
        self.use_beat_abs_pos = getattr(self.args, 'use_beat_abs_pos', False)
        if self.use_beat_abs_pos:
            beat_abs_pos_embed_dim = getattr(self.args, "beat_abs_pos_embed_dim", self.args.token_embed_dim)
            # if getattr(self.args, 'learned_beat_abs_pos', False):
            #     raise NotImplementedError("The support for overly long beat absolute position embedding is not done.")
            self.embed_beat_abs_pos = TaoEmbedding(
                self.args.max_beat_abs_pos + 1, beat_abs_pos_embed_dim, padding_idx=0,
                learned=getattr(self.args, 'learned_beat_abs_pos', False)
            )
            self.valid_parts_for_beat_abs_pos = getattr(self.args, 'valid_parts_for_beat_abs_pos', None)
        else:
            self.embed_beat_abs_pos = None
            beat_abs_pos_embed_dim = 0
            self.valid_parts_for_beat_abs_pos = None

        # --- Conclude Embedding ---
        self.sum_proj_embeddings = None
        self.reg_proj_embeddings = None
        self.concat_reg_embeddings = getattr(self.args, "concat_reg_embeddings", False)
        if self.concat_reg_embeddings:
            if getattr(self.args, 'proj_reg_embeddings', False):
                self.reg_proj_embeddings = nn.Linear(
                    self.args.token_embed_dim + token_in_chunk_abs_pos_embed_dim +
                    token_abs_pos_embed_dim +
                    (
                        bar_abs_pos_embed_dim
                        if (self.valid_parts_for_bar_abs_pos is None
                            or 'r' in self.valid_parts_for_bar_abs_pos)
                        else 0
                    ) + beat_abs_pos_embed_dim,
                    self.embed_dim
                )
        self.concat_sum_embeddings = False
        self.beat_abs_pos_padding_on_sum = getattr(self.args, 'beat_abs_pos_padding_on_sum', False)
        if self.num_summary > 0 and self.need_embedding_summary_tokens:
            self.concat_sum_embeddings = getattr(self.args, "concat_sum_embeddings", False)
            if getattr(self.args, 'proj_sum_embeddings', False):
                if getattr(self.args, 'share_embedding_projection', False):
                    self.sum_proj_embeddings = self.reg_proj_embeddings
                else:
                    self.sum_proj_embeddings = nn.Linear(
                        self.args.token_embed_dim + (
                            bar_abs_pos_embed_dim
                            if (self.valid_parts_for_bar_abs_pos is None
                                or 's' in self.valid_parts_for_bar_abs_pos)
                            else 0
                        ) +
                        (
                            beat_abs_pos_embed_dim if self.beat_abs_pos_padding_on_sum else 0
                        ),
                        self.embed_dim
                    )

        self.reg_layernorm_embedding = None
        self.sum_layernorm_embedding = None
        if getattr(args, "layernorm_embedding", False):
            self.reg_layernorm_embedding = LayerNorm(self.embed_dim)
            if self.num_summary > 0 and self.need_embedding_summary_tokens:
                if getattr(self.args, "share_layernorm_embedding", False):
                    self.sum_layernorm_embedding = self.reg_layernorm_embedding
                else:
                    self.sum_layernorm_embedding = LayerNorm(self.embed_dim)

        # === Attention Scheme ===
        self.attn_scheme = AttentionScheme(
            self.args.sum2sum, self.args.sum2sum_self,
            self.args.sum2con, self.args.sum2con_self,
            self.args.con2sum, self.args.con2sum_self,
            self.args.con2con, self.args.con2con_self,
            self.args.con2con_causal,
            self.args.pref2pref_mode,
            self.args.sum2pref_mode,
            self.args.pref2sum_mode,
            self.args.con2pref_mode,
            self.args.pref2con_mode,
            self.num_layers, self.num_attention_heads
        )
        self.layer_to_sv = {}
        self.sv_to_layers = {}
        attn_scheme_set = set(self.attn_scheme)
        for layer_idx, layer_scheme in enumerate(self.attn_scheme):
            for sv, unique_scheme in enumerate(attn_scheme_set):
                if layer_scheme == unique_scheme:
                    self.layer_to_sv[layer_idx] = sv
                    if sv not in self.sv_to_layers:
                        self.sv_to_layers[sv] = []
                    self.sv_to_layers[sv].append(layer_idx)
        for sv in self.sv_to_layers:
            self.sv_to_layers[sv] = set(self.sv_to_layers[sv])
        self.pocket_constant['layer_to_sv'] = self.layer_to_sv
        self.pocket_constant['sv_to_layers'] = self.sv_to_layers

        # === Transformer Blocks ===
        self.dropout_module = FairseqDropout(
            self.args.dropout, module_name=self.__class__.__name__
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(
                    args, layer_idx, self.num_attention_heads[layer_idx],
                    self.attn_scheme[layer_idx],
                    args.attention_dropout
                ) for layer_idx in range(self.num_layers)
            ]
        )

        if args.normalize_before and not getattr(args, "no_final_norm", False):
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

        # === Output ===
        if self.args.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_regular_tokens.weight.shape[1],
                self.embed_regular_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_regular_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.embed_dim, self.valid_dictionary_len, bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.embed_dim ** -0.5
            )

        self.gradient_checkpointing = getattr(self.args, 'gradient_checkpointing', False)
        if self.gradient_checkpointing:
            checkpointing_layers = getattr(self.args, 'gradient_checkpointing_layers', None)
            if checkpointing_layers is None:
                gradient_checkpointing_every_n_layer = getattr(self.args, 'gradient_checkpointing_every_n_layer', 1)
                checkpointing_layers = tuple(range(0, self.num_layers, gradient_checkpointing_every_n_layer))
            self.checkpointing_layers = checkpointing_layers

    def build_decoder_layer(
        self,
        args,
        layer_idx,
        num_attention_heads,
        layer_attention_scheme,
        attention_dropout,
        **kwargs
    ):
        return MuseformerDecoderLayer(
            args,
            layer_idx,
            num_attention_heads,
            layer_attention_scheme,
            attention_dropout,
            **kwargs
        )

    def forward(
        self,
        src_tokens,  # (batch, reg_len)
        src_lengths=None,
        chunk_points=None,  # (batch, max_chunk + 1)
        num_chunks=None,  # (batch,)
        num_complete_chunks=None,  # (batch,)
        num_pref=None,  # (batch,)
        beat_ids=None,  # (batch, reg_len)

        last_state_only=False,
        features_only=False,

        **kwargs
    ):
        # print('Beg', src_tokens.shape)
        x, extra = self.extract_features(
            src_tokens,
            src_lengths=src_lengths,
            reg_chunk_points=chunk_points,
            num_chunks=num_chunks,
            num_complete_chunks=num_complete_chunks,
            num_pref=num_pref,
            beat_ids=beat_ids,
            last_state_only=last_state_only,
            **kwargs
        )
        if not features_only:
            x = self.output_layer(x)

        return x, extra

    def extract_features(
        self,
        reg_tokens,  # (batch, reg_len)
        src_lengths=None,
        reg_chunk_points=None,  # (batch, max_chunk + 1)
        num_chunks=None,  # (batch,)
        num_complete_chunks=None,  # (batch,)
        num_pref=None,  # (batch,)
        beat_ids=None,

        last_state_only=False,

        **kwargs
    ):
        self.pocket_instant.clear()

        bsz, reg_len = reg_tokens.shape
        device = reg_tokens.device

        # ===== Instant Info Construction ===
        # Construct the missing input. Only for inference.
        if not self.training:
            if all([item is None for item in (reg_chunk_points, num_chunks, num_complete_chunks, num_pref)]):
                if src_lengths is None:
                    src_lengths = reg_tokens.ne(self.dictionary.pad()).sum(dim=-1)
                reg_chunk_points, num_chunks, num_complete_chunks, num_pref = iic.construct_bar_chunk_info(
                    reg_tokens, src_lengths,
                    self.dictionary.index(getattr(self.args, 'eob_token', 'b-1')),
                    begin_idx=1,
                    only_bos_prefix=True,
                    device=device
                )
                if getattr(self.args, 'take_bos_as_bar', False):
                    assert reg_chunk_points.shape[0] == 1
                    reg_chunk_points = torch.cat(
                        (reg_chunk_points.new_tensor([[0]]), reg_chunk_points), dim=-1
                    )
                    num_pref = num_pref.new_zeros(1,)
                    num_chunks = num_chunks + 1
                    num_complete_chunks = num_complete_chunks + 1

            if getattr(self.args, 'use_beat_abs_pos', False) and beat_ids is None:
                beat_ids = get_beat_ids(
                    reg_tokens[0], self.dictionary,
                    ts_instead_of_tempo=getattr(self.args, 'beat_mask_ts', False)
                )[None]

        # ===== Input Checking ===
        if num_complete_chunks is None:
            num_complete_chunks = num_chunks  # (bsz,)
        else:
            temp_check = num_chunks - num_complete_chunks
            assert (temp_check.eq(0) | temp_check.eq(1)).all()
            del temp_check
        max_chunk = int(num_chunks.max())
        assert reg_chunk_points.shape == (bsz, max_chunk + 1)

        reg_chunk_ranges = computation_tools.transfer_chunk_points_to_ranges(reg_chunk_points)  # (bsz, max_chunk, 2)
        assert reg_chunk_ranges.shape == (bsz, max_chunk, 2)
        del reg_chunk_points

        # ===== Summary Sequence and Embeddings =====
        max_comp_chunk = int(num_complete_chunks.max())
        sum_len = max_comp_chunk * self.num_summary

        # ===== Padding for Blocksparse Computation =====
        sum_pad_len = 0
        reg_pad_len = 0
        real_sum_len = sum_len
        real_reg_len = reg_len
        if self.block_size is not None:
            if sum_len > 0:
                sum_pad_len = self.block_size - sum_len % self.block_size
            if sum_pad_len == self.block_size:
                sum_pad_len = 0
            if sum_pad_len > 0:
                sum_len = real_sum_len + sum_pad_len
            reg_pad_len = self.block_size - reg_len % self.block_size
            if reg_pad_len == self.block_size:
                reg_pad_len = 0
            if reg_pad_len > 0:
                reg_len = real_reg_len + reg_pad_len

        # ===== Embedding Layer =====
        # --- Summary and Regular Token Embeddings ---
        sum_x = None
        sum_key_padding_mask = None
        sum_token_ids = None
        if self.num_summary > 0:
            sum_key_padding_mask = torch.arange(max_comp_chunk, device=device)[None].expand(
                bsz, -1).ge(num_complete_chunks[:, None])[:, :, None].expand(-1, -1, self.num_summary).reshape(
                bsz, real_sum_len
            )  # (bsz, real_sum_len)
            if sum_pad_len > 0:
                sum_key_padding_mask = torch.cat(
                    (sum_key_padding_mask, sum_key_padding_mask.new_ones(bsz, sum_pad_len)), dim=1
                )  # (bsz, sum_len)
            sum_token_ids = torch.arange(1, self.num_summary + 1, device=device)[None, None].repeat(
                bsz, max_comp_chunk, 1
            ).reshape(bsz, real_sum_len)
            if sum_pad_len > 0:
                sum_token_ids = torch.cat(
                    (sum_token_ids, sum_token_ids.new_zeros(bsz, sum_pad_len)), dim=1
                )
            sum_token_ids.masked_fill_(sum_key_padding_mask, 0)
            if self.need_embedding_summary_tokens:
                sum_x = self.embed_summary_tokens(sum_token_ids.transpose(0, 1))
                if self.embed_scale is not None:
                    sum_x.mul_(self.embed_scale)
        self.pocket_instant['sum_token_ids'] = sum_token_ids
        if reg_pad_len > 0:
            reg_tokens = torch.cat((reg_tokens, reg_tokens.new_full((bsz, reg_pad_len), self.dictionary.pad())), dim=1)
            beat_ids = torch.cat((beat_ids, beat_ids.new_full((bsz, reg_pad_len), 0)), dim=1)
        reg_key_padding_mask = reg_tokens.eq(self.embed_regular_tokens.padding_idx)  # (bsz, reg_len)
        reg_x = self.embed_regular_tokens(reg_tokens.transpose(0, 1))  # (reg_len, bsz, token_embed_dim)
        del reg_tokens
        if self.embed_scale is not None:
            reg_x.mul_(self.embed_scale)

        # --- Absolute Token In-Chunk Position Embedding ---
        if self.embed_token_in_chunk_abs_pos is not None:
            token_in_chunk_abs_pos = construct_reg_token_in_chunk_ids(reg_chunk_ranges, num_chunks, reg_len)
            token_in_chunk_abs_pos_embed = self.embed_token_in_chunk_abs_pos(token_in_chunk_abs_pos).transpose(0, 1) \
                # (reg_len, bsz, dim)
            del token_in_chunk_abs_pos
        else:
            token_in_chunk_abs_pos_embed = None

        # --- Absolute Beat Position Embedding ---
        reg_beat_abs_pos_embed = None
        sum_beat_abs_pos_embed = None
        if self.embed_beat_abs_pos is not None:
            reg_beat_abs_pos_embed = self.embed_beat_abs_pos(beat_ids.transpose(0, 1))  # (l, bsz, dim)
            sum_beat_abs_pos_embed = None
            if sum_x is not None and self.beat_abs_pos_padding_on_sum:
                sum_beat_abs_pos_embed = reg_x.new_zeros(sum_x.shape[0], bsz, self.embed_beat_abs_pos.embedding_dim)
        del beat_ids

        # --- Absolute Token Position Embedding ---
        token_abs_pos_ids = None
        if self.use_token_abs_pos:
            token_abs_pos_ids = torch.arange(1, reg_len + 1, device=device)[None].repeat(bsz, 1)
            token_abs_pos_ids.masked_fill_(reg_key_padding_mask, 0)
        if self.embed_token_abs_pos is not None:
            token_abs_pos_embed = self.embed_token_abs_pos(token_abs_pos_ids.transpose(0, 1))
        else:
            token_abs_pos_embed = None
        del token_abs_pos_ids

        # --- Absolute Bar Position Embedding ---
        sum_bar_pos_ids = None
        reg_bar_pos_ids = None
        if self.use_bar_abs_pos:
            reg_bar_pos_ids = construct_reg_bar_ids(reg_chunk_ranges, num_chunks, reg_len)  # (bsz, reg_len)
            if self.num_summary > 0:
                sum_bar_pos_ids = torch.arange(1, max_comp_chunk + 1, device=device)  # (max_comp_chunk,)
                sum_bar_pos_ids = sum_bar_pos_ids[None, :, None].expand(bsz, -1, self.num_summary).reshape(
                    bsz, real_sum_len
                )  # (bsz, sum_len)
                sum_bar_pos_ids = torch.cat((sum_bar_pos_ids, sum_bar_pos_ids.new_zeros(bsz, sum_pad_len)), dim=1)
                sum_bar_pos_ids.masked_fill_(sum_key_padding_mask, 0)
        sum_bar_abs_pos_embed = None
        reg_bar_abs_pos_embed = None
        if self.embed_bar_abs_pos is not None:
            if (
                self.num_summary > 0 and self.need_embedding_summary_tokens and
                (self.valid_parts_for_bar_abs_pos is None or 's' in self.valid_parts_for_bar_abs_pos)
            ):
                sum_bar_abs_pos_embed = self.embed_bar_abs_pos(sum_bar_pos_ids.transpose(0, 1))  # (l, bsz, dim)
            if self.valid_parts_for_bar_abs_pos is None or 'r' in self.valid_parts_for_bar_abs_pos:
                reg_bar_abs_pos_embed = self.embed_bar_abs_pos(reg_bar_pos_ids).transpose(0, 1)  # (l, bsz, dim)
        del sum_bar_pos_ids, reg_bar_pos_ids

        # --- Conclude Embeddings ---
        sum_x = [item for item in (sum_x, sum_bar_abs_pos_embed, sum_beat_abs_pos_embed) if item is not None]
        if len(sum_x) == 0:
            sum_x = None
        reg_x = [item for item in (
            reg_x, token_in_chunk_abs_pos_embed, token_abs_pos_embed, reg_bar_abs_pos_embed, reg_beat_abs_pos_embed
        ) if item is not None]
        del sum_bar_abs_pos_embed
        del token_in_chunk_abs_pos_embed, token_abs_pos_embed, reg_bar_abs_pos_embed

        if self.concat_reg_embeddings:
            reg_x = torch.cat(reg_x, dim=-1)
            if self.reg_proj_embeddings is not None:
                reg_x = self.reg_proj_embeddings(reg_x)
        else:
            reg_x = sum(reg_x)
        if self.num_summary > 0 and self.need_embedding_summary_tokens:
            if self.concat_sum_embeddings:
                sum_x = torch.cat(sum_x, dim=-1)
                if self.sum_proj_embeddings is not None:
                    sum_x = self.sum_proj_embeddings(sum_x)
            else:
                sum_x = sum(sum_x)

        if self.sum_layernorm_embedding is not None and sum_x is not None:
            sum_x = self.sum_layernorm_embedding(sum_x)
        if self.reg_layernorm_embedding is not None:
            reg_x = self.reg_layernorm_embedding(reg_x)

        if sum_x is not None:
            sum_x = self.dropout_module(sum_x)
        reg_x = self.dropout_module(reg_x)

        key_padding_mask = computation_tools.may_bi_cat(sum_key_padding_mask, reg_key_padding_mask, dim=1)
        if key_padding_mask is not None and not key_padding_mask.any():
            key_padding_mask = None
        del sum_key_padding_mask, reg_key_padding_mask

        # with open('meta.bin', 'wb') as f:
        #     torch.save((sum_len, reg_len), f)
        #     print('saved meta')

        # === Transformer Layers ===
        (sum_x, reg_x), inner_states = self.run_layers(
            (sum_x, reg_x),
            reg_chunk_ranges=reg_chunk_ranges,
            num_chunks=num_chunks,
            num_complete_chunks=num_complete_chunks,
            num_pref=num_pref,
            sum_len=sum_len,
            reg_len=reg_len,

            key_padding_mask=key_padding_mask,
            attn_mask=None,

            need_weights=False,
            need_head_weights=False,

            last_state_only=last_state_only,
        )

        if sum_x is not None:
            sum_x = sum_x[:real_sum_len]
            sum_len = real_sum_len
            sum_x = sum_x.transpose(0, 1)
            assert sum_x.shape == (bsz, sum_len, self.embed_dim), (sum_x.shape, (bsz, sum_len, self.embed_dim))

        reg_x = reg_x[:real_reg_len]
        reg_len = real_reg_len
        if self.layer_norm is not None:
            reg_x = self.layer_norm(reg_x)
        reg_x = reg_x.transpose(0, 1)
        assert reg_x.shape == (bsz, reg_len, self.embed_dim), (reg_x.shape, (bsz, reg_len, self.embed_dim))

        others = {
            # Uncomment if needed
            # 'summary': sum_x,
            'attn': None,
            # 'inner_states': inner_states,
        }

        return reg_x, others

    def run_layers(
        self,
        x,
        reg_chunk_ranges,
        num_chunks,
        num_complete_chunks,
        num_pref,
        sum_len,
        reg_len,

        key_padding_mask,
        attn_mask,

        need_weights,
        need_head_weights,

        last_state_only
    ):
        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            layer_idx = layer.layer_idx

            if (
                getattr(self.args, "gradient_checkpointing", False) and self.training and
                layer_idx in self.checkpointing_layers
            ):
                x, _ = checkpoint(
                    layer,
                    x,
                    reg_chunk_ranges,
                    num_chunks,
                    num_complete_chunks,
                    num_pref,
                    sum_len,
                    reg_len,

                    key_padding_mask,
                    None if attn_mask is None else attn_mask[layer_idx],

                    need_weights,
                    need_head_weights,
                )
            else:
                x, _ = layer(
                    x,
                    reg_chunk_ranges,
                    num_chunks,
                    num_complete_chunks,
                    num_pref,
                    sum_len,
                    reg_len,

                    key_padding_mask=key_padding_mask,
                    attn_mask=None if attn_mask is None else attn_mask[layer_idx],

                    need_weights=need_weights,
                    need_head_weights=need_head_weights,
                )

            if attn_mask is not None:
                attn_mask[layer_idx] = None

            if not last_state_only:
                inner_states.append(x)

        if last_state_only:
            inner_states = [x]

        return x, inner_states

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        return self.output_projection(features)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if getattr(self.args, 'use_token_abs_pos', False) and getattr(self.args, 'learned_token_abs_pos', False):
            return min(self.args.max_target_positions, self.args.max_token_abs_pos)
        return self.args.max_target_positions

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        # assert sample is not None

        logits, others = net_output  # (batch, seq_len, vocab_len)
        embed_dim = logits.shape[-1]
        if sample is not None and 'target_mask' in sample:
            target_mask = sample['target_mask']
        else:
            target_mask = None
        if target_mask is not None:
            logits = logits.masked_select(target_mask.unsqueeze(-1)).view(-1, embed_dim)
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def get_targets(self, sample, net_output):
        target = sample['target']
        _, others = net_output
        if 'target_mask' in sample:
            target_mask = sample['target_mask']
        else:
            target_mask = None
        if target_mask is not None:
            target = target.masked_select(target_mask)
        return target
