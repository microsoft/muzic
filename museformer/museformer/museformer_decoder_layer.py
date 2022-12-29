import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from fairseq import utils
from fairseq.modules import FairseqDropout, LayerNorm

from .tools import arg_tools, computation_tools
from . import attention
from .attention_mask_generation.attention_mask_generation import LayerAttentionMaskGeneration, \
    combine_part_masks, transfer_attn_mask_to_block_layout, combine_attn_mask_and_key_padding_mask_
from .data_structures.four_dim_pocket import FourDimPocket

logger = logging.getLogger(__name__)


def generate_layer_attn_mask(
    layer_gen: LayerAttentionMaskGeneration,
    reg_chunk_ranges, num_chunks, num_complete_chunks,
    num_summary, num_pref, sum_len, reg_len,
    key_padding_mask=None,
    highlight_prefix=False,
    combine_parts=None,
    block_size=None,
    avoid_empty_row_or_column=True
):
    # all_seq_len = sum_len + reg_len
    bsz = reg_chunk_ranges.shape[0]
    attn_mask = layer_gen(
        reg_chunk_ranges, num_chunks, num_complete_chunks,
        num_summary, num_pref, sum_len, reg_len
    )  # each part: (bsz, 1, part_tgt_len, part_src_len)

    # combine key_padding_mask into attn_mask
    if key_padding_mask is not None:
        sum_key_padding_mask = key_padding_mask[:, :sum_len]  # (bsz, sum_len)
        if sum_key_padding_mask.any():
            for key in ('ss', 'rs'):
                if key not in attn_mask:
                    continue
                combine_attn_mask_and_key_padding_mask_(attn_mask[key], sum_key_padding_mask)
        reg_key_padding_mask = key_padding_mask[:, sum_len:]  # (bsz, reg_len)
        if reg_key_padding_mask.any():
            for key in ('sr', 'rr'):
                if key not in attn_mask:
                    continue
                combine_attn_mask_and_key_padding_mask_(attn_mask[key], reg_key_padding_mask)
        del sum_key_padding_mask, reg_key_padding_mask
    del key_padding_mask

    if highlight_prefix:
        raise NotImplementedError
        rr_attn_mask = attn_mask['rr']
        assert rr_attn_mask.shape[1:] == (1, reg_len, reg_len)
        rr_attn_mask = rr_attn_mask.type(torch.uint8)  # (bsz, 1, reg_len, reg_len)
        num_unique_pref = len(torch.unique(num_pref))
        if num_unique_pref == 1:
            unique_prefix_number = num_pref[0].item()
            change_2_mask = rr_attn_mask[:, :, :, :unique_prefix_number].eq(0)
            rr_attn_mask[:, :, :, :unique_prefix_number][change_2_mask] = 2
        else:
            raise NotImplementedError
        attn_mask['rr'] = rr_attn_mask

    # for key in attn_mask:
    #     print(key)
    #     temp_mask = attn_mask[key]
    #     if temp_mask is not None:
    #         print(temp_mask.dtype)
    #         temp_mask = temp_mask.long()
    #         temp_mask = temp_mask[0, 0]
    #
    #         for line in temp_mask:
    #             for col in line:
    #                 print(int(col), end=' ')
    #             print()
    #     else:
    #         print('None')

    if combine_parts == 'all':
        attn_mask = combine_part_masks(attn_mask)  # (bsz, 1, all_seq_len, all_seq_len)
        attn_mask = {'all': attn_mask}
    elif isinstance(combine_parts, dict):
        new_attn_mask = {}
        processed_parts = set()
        for key in combine_parts:
            part1, part2 = combine_parts[key]
            assert part1 not in processed_parts
            assert part2 not in processed_parts
            assert part1[0] == part2[0]  # only ss + sr, or rs + rr can be combined
            new_attn_mask[key] = computation_tools.may_bi_cat(attn_mask[part1], attn_mask[part2], dim=3)
            processed_parts.add(part1)
            processed_parts.add(part2)
        for key in attn_mask:
            if key in processed_parts:
                continue
            new_attn_mask[key] = attn_mask[key]
        attn_mask = new_attn_mask
        del processed_parts, new_attn_mask
    elif combine_parts is None:
        pass
    else:
        raise ValueError('combine_parts has wrong value:', combine_parts)

    if block_size is None:
        return attn_mask

    for key in attn_mask:
        batch_key_attn_mask = []
        for sample_idx in range(bsz):
            if attn_mask[key] is None:
                r = None
            else:
                r = transfer_attn_mask_to_block_layout(attn_mask[key][sample_idx], block_size,
                                                       avoid_empty_row_or_column=avoid_empty_row_or_column)
            batch_key_attn_mask.append(r)
        attn_mask[key] = batch_key_attn_mask
        # real_part_sample_mask: layout, block_mask
        #     layout: (head, num_tgt_blocks, num_src_blocks)
        #     block_mask: (-1, block, block)

    return attn_mask


class MuseformerDecoderLayer(nn.Module):
    _submodules = (attention,)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--share-self-attention-layer-norm', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--share-ffn', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--share-final-layer-norm', type=arg_tools.str_bool_with_default_error)
        parser.add_argument('--chunk-ffn', type=int)

        arg_tools.add_submodule_args(cls, parser)

    def __init__(self,
                 args,
                 layer_idx,
                 num_attention_heads,
                 layer_attention_scheme,
                 attention_dropout,
                 **kwargs):
        super().__init__()

        self.pocket = FourDimPocket()
        self.pocket_constant = self.pocket['constant']
        self.pocket_instant = self.pocket['instant']

        # === Basic Settings ===
        self.args = args
        self.layer_idx = layer_idx

        self.attention_impl = self.pocket_constant['attention_impl']
        self.block_size = self.pocket_constant['block_size']
        self.attention_mode = self.pocket_constant['attention_mode']
        self.attn_mask_combination = self.pocket_constant['attn_mask_combination']
        self.attn_mask_gen_parts = self.pocket_constant['attn_mask_gen_parts']
        self.need_embedding_summary_tokens = self.pocket_constant['need_embedding_summary_tokens']

        self.layer_to_sv = self.pocket_constant['layer_to_sv']
        self.sv_to_layers = self.pocket_constant['sv_to_layers']
        self.layer_sv = self.layer_to_sv[self.layer_idx]

        self.num_attention_heads = num_attention_heads
        self.embed_dim = args.attention_embed_dim
        self.num_summary = args.num_summary_tokens_each_chunk
        self.normalize_before = args.normalize_before

        self.chunk_ffn = getattr(args, 'chunk_ffn', None)
        if self.chunk_ffn is not None:
            if self.chunk_ffn <= 0:
                self.chunk_ffn = None

        # === Layer Attention Mask Generator ===
        self.layer_attention_scheme = layer_attention_scheme
        layer_attention_mask_generator_label = ('layer_attention_mask_generator', self.layer_sv)
        if layer_attention_mask_generator_label in self.pocket_constant:
            self.layer_attention_mask_generator = self.pocket_constant[layer_attention_mask_generator_label]
        else:
            self.layer_attention_mask_generator = LayerAttentionMaskGeneration(self.layer_attention_scheme,
                                                                               gen_parts=self.attn_mask_gen_parts)
            self.pocket_constant[layer_attention_mask_generator_label] = self.layer_attention_mask_generator

        # === Self Attention ===
        self.self_attn = self.build_self_attention(
            self.args,
            self.embed_dim,
            self.num_attention_heads,
            attention_dropout,
        )

        # === Other Modules ===
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        # activation_dropout_p = getattr(args, "activation_dropout", None)
        # if activation_dropout_p is not None:
        #     assert activation_dropout_p > 0
        #     self.activation_dropout_module = FairseqDropout(
        #         float(activation_dropout_p), module_name=self.__class__.__name__
        #     )
        # else:
        #     self.activation_dropout_module = None

        self.reg_self_attn_layer_norm = LayerNorm(self.embed_dim, export=False)
        if self.need_embedding_summary_tokens:
            if getattr(self.args, 'share_self_attention_layer_norm', False):
                self.sum_self_attn_layer_norm = self.reg_self_attn_layer_norm
            else:
                self.sum_self_attn_layer_norm = LayerNorm(self.embed_dim, export=False)
        else:
            self.sum_self_attn_layer_norm = None

        self.reg_fc1 = nn.Linear(self.embed_dim, args.ffn_embed_dim)
        self.reg_fc2 = nn.Linear(args.ffn_embed_dim, self.embed_dim)
        share_ffn = getattr(self.args, 'share_ffn', False)
        if self.need_embedding_summary_tokens:
            if share_ffn:
                self.sum_fc1 = self.reg_fc1
                self.sum_fc2 = self.reg_fc2
            else:
                self.sum_fc1 = nn.Linear(self.embed_dim, args.ffn_embed_dim)
                self.sum_fc2 = nn.Linear(args.ffn_embed_dim, self.embed_dim)
        else:
            self.sum_fc1 = None
            self.sum_fc2 = None

        self.reg_final_layer_norm = LayerNorm(self.embed_dim, export=False)
        if self.need_embedding_summary_tokens:
            if getattr(self.args, 'share_final_layer_norm', False):
                self.sum_final_layer_norm = self.reg_final_layer_norm
            else:
                self.sum_final_layer_norm = LayerNorm(self.embed_dim, export=False)
        else:
            self.sum_final_layer_norm = None

    def build_self_attention(
        self,
        args,
        embed_dim,
        num_attention_heads,
        dropout,
        **kwargs,
    ):
        return attention.create_attention(
            implementation=self.attention_impl,
            attention_mode=self.attention_mode,
            block_size=self.block_size,

            embed_dim=embed_dim,
            num_heads=num_attention_heads,

            num_summary=self.args.num_summary_tokens_each_chunk,

            layer_idx=self.layer_idx,

            dropout=dropout,

            query_proj_bias=getattr(args, 'attn_query_proj_bias', False),
            key_proj_bias=getattr(args, 'attn_key_proj_bias', False),
            value_proj_bias=getattr(args, 'attn_value_proj_bias', False),
            out_proj_bias=getattr(args, 'attn_out_proj_bias', False),

            single_head_masks=self.layer_attention_mask_generator.single_head,

            add_different_kqv_bias_for_sum_and_reg=getattr(args, 'add_different_kqv_bias_for_sum_and_reg', False),
            add_different_out_bias_for_sum_and_reg=getattr(args, 'add_different_out_bias_for_sum_and_reg', False),

            sum_key2_proj_bias=getattr(args, 'attn_sum_key2_proj_bias', False),
            sum_value2_proj_bias=getattr(args, 'attn_sum_value2_proj_bias', False),
            share_query_proj=getattr(args, 'attn_share_query_proj', False),
            share_key_proj=getattr(args, 'attn_share_key_proj', False),
            share_value_proj=getattr(args, 'attn_share_value_proj', False),
            share_out_proj=getattr(args, 'attn_share_out_proj', False),
            share_key2_value2_proj_weight=getattr(args, 'attn_share_key2_value2_proj_weight', False),

            no_sum_out=((self.layer_idx == self.args.num_layers - 1)
                        if getattr(args, 'add_different_out_bias_for_sum_and_reg', False) else False
                        ), # to make compatible with previous checkpoints
        )

    def forward(
        self,
        x,  # tuple of (Optional[Tensor], Tensor)  # (sum_len, bsz, embed_dim)  (reg_len, bsz, embed_dim)
        reg_chunk_ranges,  # (bsz, max_chunk, 2)
        num_chunks,  # (bsz,)
        num_complete_chunks,  # (bsz,)
        num_pref,  # (batch,)
        sum_len,
        reg_len,

        key_padding_mask=None,  # (bsz, all_seq_len)
        attn_mask=None,  # (bsz, num_heads, all_seq_len, all_seq_len)

        need_weights=False,
        need_head_weights=False,
    ):
        if need_head_weights:
            need_weights = True

        if attn_mask is None:
            attn_mask_label = 'attn_mask_sv%d' % self.layer_sv
            if attn_mask_label in self.pocket_instant:
                attn_mask = self.pocket_instant[attn_mask_label]
            else:
                attn_mask = generate_layer_attn_mask(
                    self.layer_attention_mask_generator,
                    reg_chunk_ranges, num_chunks, num_complete_chunks,
                    self.args.num_summary_tokens_each_chunk,
                    num_pref, sum_len, reg_len,
                    key_padding_mask=key_padding_mask,
                    highlight_prefix=False,
                    combine_parts=self.attn_mask_combination,
                    block_size=self.block_size if self.attention_impl == 'blocksparse' else None,
                    avoid_empty_row_or_column=True
                )
                self.pocket_instant[attn_mask_label] = attn_mask
        else:
            raise NotImplementedError('Passing attn_mask into a Museformer layer is not supported yet.')
        key_padding_mask = None  # key_padding_mask is useless, after combined into attn_mask

        residual = x
        if self.normalize_before:
            x = computation_tools.may_bi_op(
                self.sum_self_attn_layer_norm, self.reg_self_attn_layer_norm,
                x, sum_len, reg_len, self.embed_dim, as_tuple=True
            )

        # st = time.time()
        # try:
        x, attn = self.run_self_attn(
            x, x, x,
            sum_len, reg_len,
            key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            incremental_state=None,
            need_weights=need_weights, need_head_weights=need_head_weights,
        )
        # except RuntimeError:
        #     print('x:', x.shape)
        #     print('num_heads:', self.num_attention_heads)
        #     print('attn_mask:', attn_mask.shape)
        #     print('key_padding_mask:', key_padding_mask.shape)
        #     raise
        # et = time.time()
        # if debug:
        #     logger.info('Run Self-Attn: %f' % (et - st))

        x = computation_tools.may_bi_op(self.dropout_module, self.dropout_module, x,
                                        sum_len, reg_len, self.embed_dim, as_tuple=True)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = computation_tools.may_bi_op(
                self.sum_self_attn_layer_norm, self.reg_self_attn_layer_norm,
                x, sum_len, reg_len, self.embed_dim, as_tuple=True
            )

        residual = x
        if self.normalize_before:
            x = computation_tools.may_bi_op(
                self.sum_final_layer_norm, self.reg_final_layer_norm,
                x, sum_len, reg_len, self.embed_dim, as_tuple=True
            )

        x = self.run_ffn(x, sum_len, reg_len)

        x = computation_tools.may_bi_op(self.dropout_module, self.dropout_module, x,
                                        sum_len, reg_len, self.embed_dim, as_tuple=True)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = computation_tools.may_bi_op(
                self.sum_final_layer_norm, self.reg_final_layer_norm,
                x, sum_len, reg_len, self.embed_dim, as_tuple=True
            )

        return x, attn

    @staticmethod
    def residual_connection(x, residual):
        return residual[0] + x[0] if (residual[0] is not None and x[0] is not None) else None, residual[1] + x[1]

    def run_self_attn(
        self,
        query,  # (all_seq_len, bsz, embed_dim)
        key,  # (all_seq_len, bsz, embed_dim)
        value,  # (all_seq_len, bsz, embed_dim)
        sum_len,
        reg_len,
        key_padding_mask=None,
        attn_mask=None,
        incremental_state=None,
        need_weights=True,
        need_head_weights=False,
        **kwargs,
    ):
        assert incremental_state is None
        r, weight = self.self_attn(
            query,
            self.pocket_instant['sum_token_ids'],
            sum_len, reg_len,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
            need_head_weights=need_head_weights,
            **kwargs,
        )
        return r, weight

    def run_ffn(self, x, sum_len, reg_len):
        sum_x, reg_x = x
        del x

        def reg_ffn(input_x):
            input_x = self.reg_fc1(input_x)
            input_x = self.activation_fn(input_x)
            input_x = self.reg_fc2(input_x)
            return input_x

        if sum_x is None:
            sum_ffn = None
        else:
            if getattr(self.args, 'share_ffn', False):
                sum_ffn = reg_ffn
            else:
                def sum_ffn(input_x):
                    input_x = self.sum_fc1(input_x)
                    input_x = self.activation_fn(input_x)
                    input_x = self.sum_fc2(input_x)
                    return input_x

        if self.chunk_ffn is None:
            if sum_x is not None:
                sum_x = sum_ffn(sum_x)
            reg_x = reg_ffn(reg_x)
        else:
            def do_chunk_ffn(input_x, ffn, split_size):
                input_x = torch.split(input_x, split_size, dim=0)
                result = []
                for chunk_x in input_x:
                    chunk_x = ffn(chunk_x)
                    result.append(chunk_x)
                del input_x
                result = torch.cat(result, dim=0)
                return result
            if reg_x.requires_grad:
                if id(reg_ffn) == id(sum_ffn):
                    def do_multi_chunk_ffn(input_x, ffn, split_size):
                        return [do_chunk_ffn(one_x, ffn, split_size) for one_x in input_x]
                    sum_x, reg_x = do_multi_chunk_ffn((sum_x, reg_x), reg_ffn, self.chunk_ffn)
                else:
                    reg_x = checkpoint(do_chunk_ffn, reg_x, reg_ffn, self.chunk_ffn)
                    if sum_x is not None:
                        sum_x = do_chunk_ffn(sum_x, sum_ffn, self.chunk_ffn)
            else:
                reg_x = do_chunk_ffn(reg_x, reg_ffn, self.chunk_ffn)
                if sum_x is not None:
                    sum_x = do_chunk_ffn(sum_x, sum_ffn, self.chunk_ffn)

        return sum_x, reg_x
