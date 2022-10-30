import torch
import torch.nn as nn

from .multihead_block_selection_generation import MultiheadBlockSelectionGeneration, get_block_ranges
from ..kernels.block_fill import block_fill_
from ..tools import computation_tools
from ..data_structures.attention_scheme import LayerAttentionScheme

PREF2PREF_CHOICES = ('full', 'lt', 'none')
SUM2PREF_CHOICES = ('none',)
PREF2SUM_CHOICES = ('none',)
CON2PREF_CHOICES = ('full',)
PREF2CON_CHOICES = ('none',)


def construct_sum_chunk_ranges(max_complete_chunk, num_summary, device):
    ranges = torch.arange(0, max_complete_chunk * num_summary + num_summary, num_summary, device=device)
    ranges = torch.stack((ranges[:-1], ranges[1:]), dim=-1)  # (max_complete_chunk, 2)
    return ranges


def select_self_(selection, max_comp_chunk, mode):
    if mode == 'default':
        return selection
    ndim = selection.ndim
    eye = torch.eye(max_comp_chunk, dtype=torch.bool, device=selection.device)
    if ndim >= 3:
        num_heads = selection.shape[-1]
        eye = eye[:, :, None].expand(-1, -1, num_heads)
    if mode == 'none':
        selection[eye] = False
    elif mode == 'full':
        selection[eye] = True
    else:
        raise ValueError(mode)
    return selection


def fill_mask_with_selection_(selection_gen: MultiheadBlockSelectionGeneration,
                              row_ranges, col_ranges, selection_mask, fill_value, out):
    """

    :param selection_gen:
    :param row_ranges: (num_row_chunks, 2)
    :param col_ranges: (num_col_ranges, 2)
    :param selection_mask: (num_row_chunks, num_col_ranges, num_heads)
    :param fill_value: bool
    :param out: (num_heads, row_len, col_len)
    :return:
    """
    assert row_ranges.ndim == 2
    assert col_ranges.ndim == 2
    assert out.is_contiguous()
    block_indices, command_masks = selection_gen.get_block_indices_and_masks_from_selection_masks(
        selection_mask, overall_mask=None
    )  # (num_selected_blocks, 2)  (num_selected_blocks, num_heads)
    block_ranges = get_block_ranges(row_ranges, col_ranges, block_indices)  # (num_selected_blocks, 4)
    block_fill_(out, block_ranges, command_masks, fill_value)
    return out


def partially_combine_part_masks(part_masks_dict):
    row_sum = computation_tools.may_bi_cat(part_masks_dict['ss'], part_masks_dict['sr'], dim=3)
    row_reg = computation_tools.may_bi_cat(part_masks_dict['rs'], part_masks_dict['rr'], dim=3)
    return row_sum, row_reg


def combine_part_masks(part_masks_dict):
    row_sum, row_reg = partially_combine_part_masks(part_masks_dict)
    return computation_tools.may_bi_cat(row_sum, row_reg, dim=2)


def combine_attn_mask_and_key_padding_mask_(attn_mask, key_padding_mask):
    """

    :param attn_mask: (bsz, num_heads, seq_len, seq_len)
    :param key_padding_mask: (bsz, seq_len)
    :return:
    """
    if attn_mask is None:
        return attn_mask
    attn_mask = attn_mask.contiguous()
    attn_mask.masked_fill_(key_padding_mask[:, None, None], True)
    return attn_mask


def layout_full_zero_check(input_layout):
    row_check = input_layout.sum(dim=2).eq(0)  # (H, L // block)
    col_check = input_layout.sum(dim=1).eq(0)
    row_answer = bool(row_check.any())
    col_answer = bool(col_check.any())
    return row_answer or col_answer, row_answer, col_answer, row_check, col_check


def transfer_attn_mask_to_block_layout(attn_mask, block_size, avoid_empty_row_or_column=True):
    """

    :param attn_mask: (bsz_heads, tgt_len, src_len)  bool  False for selected
    :param block_size: int
    :param avoid_empty_row_or_column
    :return:
    """
    bsz_heads, tgt_len, src_len = attn_mask.shape
    num_tgt_blocks = tgt_len // block_size
    num_src_blocks = src_len // block_size

    assert num_tgt_blocks * block_size == tgt_len
    assert num_src_blocks * block_size == src_len

    assert attn_mask.dtype in (torch.bool, torch.uint8)
    attn_mask = attn_mask.view(
        bsz_heads, num_tgt_blocks, block_size, num_src_blocks, block_size
    ).permute(0, 1, 3, 2, 4)  # (bsz * num_heads, num_tgt_blocks, num_src_blocks, block_size, block_size)
    layout = (~(attn_mask.bool())).reshape(
        bsz_heads, num_tgt_blocks, num_src_blocks, block_size * block_size
    ).sum(dim=-1).gt(0)  # (bsz * num_heads, num_tgt_blocks, num_src_blocks)
    if not layout.any():  # for this part of attn_mask, there's nothing to compute
        return None, None
    if avoid_empty_row_or_column:
        from ..blocksparse import layout_full_zero_check
        answer, row_answer, col_answer, row_check, col_check = layout_full_zero_check(layout)
        if answer:
            if row_answer:
                row_zero_indices = row_check.nonzero()
                # print('row:')
                # print(row_zero_indices)
                layout[row_zero_indices[:, 0], row_zero_indices[:, 1], 0] = True
            if col_answer:
                col_zero_indices = col_check.nonzero()
                # print('col:')
                # print(col_zero_indices)
                layout[col_zero_indices[:, 0], 0, col_zero_indices[:, 1]] = True
        assert not layout_full_zero_check(layout)[0]
    block_mask = attn_mask.masked_select(layout[:, :, :, None, None]).view(-1, block_size, block_size)

    return layout, block_mask


class LayerAttentionMaskGeneration(nn.Module):
    def __init__(
        self,
        layer_attention_scheme: LayerAttentionScheme,
        gen_parts=None,
        single_head=True,
        init_max_chunk=None
    ):
        super().__init__()

        self.layer_attention_scheme = layer_attention_scheme
        self.single_head = single_head
        if not self.layer_attention_scheme.same_for_all_heads:
            raise NotImplementedError("Not support for different attention schemes over different heads.")

        self.num_heads = self.layer_attention_scheme.num_layer_heads
        self.num_fake_heads = 1
        self.single_head_scheme = self.layer_attention_scheme[0]

        assert self.single_head_scheme.head_pref2pref_mode in PREF2PREF_CHOICES
        self.pref2pref_mode = self.single_head_scheme.head_pref2pref_mode
        assert self.single_head_scheme.head_sum2pref_mode in SUM2PREF_CHOICES
        self.sum2pref_mode = self.single_head_scheme.head_sum2pref_mode
        assert self.single_head_scheme.head_pref2sum_mode in PREF2SUM_CHOICES
        self.pref2sum_mode = self.single_head_scheme.head_pref2sum_mode
        assert self.single_head_scheme.head_con2pref_mode in CON2PREF_CHOICES
        self.con2pref_mode = self.single_head_scheme.head_con2pref_mode
        assert self.single_head_scheme.head_pref2con_mode in PREF2CON_CHOICES
        self.pref2con_mode = self.single_head_scheme.head_pref2con_mode

        self.gen_parts = gen_parts

        self.con2con_gen, self.con2sum_gen, self.sum2con_gen, self.sum2sum_gen = None, None, None, None

        if gen_parts is None or 'rr' in gen_parts:
            self.con2con_gen = MultiheadBlockSelectionGeneration((self.single_head_scheme.head_con2con,),
                                                                 init_max_chunk=init_max_chunk)
        if gen_parts is None or 'rs' in gen_parts:
            self.con2sum_gen = MultiheadBlockSelectionGeneration((self.single_head_scheme.head_con2sum,),
                                                                 init_max_chunk=init_max_chunk)
        if gen_parts is None or 'sr' in gen_parts:
            self.sum2con_gen = MultiheadBlockSelectionGeneration((self.single_head_scheme.head_sum2con,),
                                                                 init_max_chunk=init_max_chunk)
        if gen_parts is None or 'ss' in gen_parts:
            self.sum2sum_gen = MultiheadBlockSelectionGeneration((self.single_head_scheme.head_sum2sum,),
                                                                 init_max_chunk=init_max_chunk)

    def gen_ss_mask(
        self,
        num_complete_chunks: torch.Tensor,  # (bsz,)
        num_summary: int,
        bsz, sum_len,
        device,
    ):
        if sum_len <= 0 or num_summary <= 0:
            return None
        max_complete_chunk = int(num_complete_chunks.max())
        temp_sum_len = max_complete_chunk * num_summary
        mask = self.sum2sum_gen.get_block_selection_masks(max_complete_chunk).to(device) \
            # (max_complete_chunk, max_complete_chunk, 1)  # True for selected
        mask = select_self_(mask, max_complete_chunk, self.single_head_scheme.head_sum2sum_self)  # Add Self
        mask = ~mask  # False for selected
        mask = mask.permute(2, 0, 1)  # (num_heads, max_comp_chunk, max_comp_chunk)
        mask = mask[None].repeat(bsz, 1, 1, 1).contiguous()  # (bsz, num_heads, max_comp_chunk, max_comp_chunk)
        for idx, sample_comp_chunk in enumerate(num_complete_chunks):
            mask[idx, :, sample_comp_chunk:] = True
            mask[idx, :, :, sample_comp_chunk:] = True
        mask = mask[:, :, :, :, None, None].expand(-1, -1, -1, -1, num_summary, num_summary)
        mask = mask.permute(0, 1, 2, 4, 3, 5).reshape(bsz, self.num_fake_heads, temp_sum_len, temp_sum_len) \
            # (bsz, num_heads, temp_sum_len, temp_sum_len)
        if sum_len != temp_sum_len:
            temp_mask = torch.ones(bsz, self.num_fake_heads, sum_len, sum_len, dtype=torch.bool, device=device)
            temp_mask[:, :, :temp_sum_len, :temp_sum_len] = mask
            mask = temp_mask
        return mask

    def gen_sr_mask(
        self,
        sum_chunk_ranges, reg_chunk_ranges, num_complete_chunks,
        sum_len, reg_len,
        num_pref,
        sum2pref_mode,
        bsz,
        device
    ):
        if sum_len <= 0:
            return None
        mask = torch.ones(bsz, self.num_fake_heads, sum_len, reg_len, dtype=torch.bool, device=device)
        for sample_idx, sample_comp_chunk in enumerate(num_complete_chunks):
            if sample_comp_chunk <= 0:
                continue
            sample_selections = self.sum2con_gen.get_block_selection_masks(sample_comp_chunk)  \
                # (sample_comp_chunk, sample_comp_chunk, num_heads)
            sample_selections = select_self_(sample_selections, sample_comp_chunk,
                                             self.single_head_scheme.head_sum2con_self)
            fill_mask_with_selection_(self.sum2con_gen,
                                      sum_chunk_ranges[sample_idx], reg_chunk_ranges[sample_idx],
                                      sample_selections,
                                      False, mask[sample_idx])
        if sum2pref_mode == 'none':
            pass
        else:
            raise NotImplementedError(sum2pref_mode)
        return mask

    def gen_rs_mask(
        self,
        reg_chunk_ranges, sum_chunk_ranges, num_complete_chunks, num_chunks,
        sum_len, reg_len,
        num_pref,
        pref2sum_mode,
        bsz, device
    ):
        if sum_len <= 0:
            return None
        mask = torch.ones(bsz, self.num_fake_heads, reg_len, sum_len, dtype=torch.bool, device=device)
        for sample_idx, (sample_comp_chunk, sample_chunk) in enumerate(zip(num_complete_chunks, num_chunks)):
            if sample_chunk <= 0:
                continue
            sample_selections = self.con2sum_gen.get_block_selection_masks(sample_chunk)
            sample_selections = select_self_(sample_selections, sample_chunk, self.single_head_scheme.head_con2sum_self)
            sample_selections = sample_selections[:, :sample_comp_chunk]
            fill_mask_with_selection_(self.con2sum_gen,
                                      reg_chunk_ranges[sample_idx], sum_chunk_ranges[sample_idx],
                                      sample_selections,
                                      False, mask[sample_idx])
        if pref2sum_mode == 'none':
            pass
        else:
            raise NotImplementedError(pref2sum_mode)
        return mask

    def gen_rr_mask(
        self,
        reg_chunk_ranges, num_chunks,
        reg_len,
        num_pref,
        pref2pref_mode, pref2con_mode, con2pref_mode,
        bsz, device
    ):
        mask = torch.ones(bsz, self.num_fake_heads, reg_len, reg_len, dtype=torch.bool, device=device)
        for sample_idx, sample_chunk in enumerate(num_chunks):
            if sample_chunk <= 0:
                continue
            sample_selections = self.con2con_gen.get_block_selection_masks(sample_chunk)
            sample_selections = select_self_(
                sample_selections, sample_chunk,
                'full' if self.single_head_scheme.head_con2con_self == 'lt'
                else self.single_head_scheme.head_con2con_self
            )
            fill_mask_with_selection_(self.con2con_gen,
                                      reg_chunk_ranges[sample_idx], reg_chunk_ranges[sample_idx],
                                      sample_selections,
                                      False, mask[sample_idx])
            sample_num_pref = num_pref[sample_idx]
            if self.single_head_scheme.head_con2con_causal:
                up_triangle = torch.ones(reg_len - sample_num_pref, reg_len - sample_num_pref,
                                         dtype=torch.bool, device=device)
                up_triangle.triu_(1)  # (con_len, con_len)
                mask[sample_idx, :, sample_num_pref:, sample_num_pref:].masked_fill_(up_triangle[None], True)
        if pref2pref_mode == 'none':
            pass
        elif pref2pref_mode == 'full':
            for sample_idx, sample_num_pref in enumerate(num_pref):
                mask[sample_idx, :, :sample_num_pref, :sample_num_pref] = False
        elif pref2pref_mode == 'lt':
            for sample_idx, sample_num_pref in enumerate(num_pref):
                if sample_num_pref == 1:
                    mask[sample_idx, :, :sample_num_pref, :sample_num_pref] = False
                    continue
                mask[sample_idx, :, :sample_num_pref, :sample_num_pref].triu_(1)
        else:
            raise NotImplementedError(pref2pref_mode)

        if pref2con_mode == 'none':
            pass
        else:
            raise NotImplementedError(pref2con_mode)

        if con2pref_mode == 'none':
            pass
        elif con2pref_mode == 'full':
            for sample_idx, sample_num_pref in enumerate(num_pref):
                sample_num_chunks = num_chunks[sample_idx]
                if sample_num_chunks == 0:
                    continue
                sample_reg_len = reg_chunk_ranges[sample_idx, sample_num_chunks - 1, -1]
                mask[sample_idx, :, sample_num_pref: sample_reg_len, :sample_num_pref] = False
        else:
            raise NotImplementedError(con2pref_mode)

        return mask

    def forward(
        self,
        reg_chunk_ranges: torch.Tensor,  # (bsz, max_chunk, 2)
        num_chunks: torch.Tensor,  # (bsz,)
        num_complete_chunks: torch.Tensor,  # (bsz,)
        num_summary: int,
        num_pref: torch.Tensor,  # (bsz,)
        sum_len: int,
        reg_len: int,
    ):
        # print('===attn_mask===')
        # print(reg_chunk_ranges)
        # print(num_chunks)
        # print(num_complete_chunks)
        # print(num_summary)
        # print(num_pref)
        # print(sum_len, reg_len)
        # print('=====')
        # === Preliminaries ===
        device = reg_chunk_ranges.device
        bsz = num_chunks.shape[0]
        # all_seq_len = temp_sum_len + reg_len
        max_complete_chunk = int(num_complete_chunks.max())

        if num_summary <= 0 or sum_len <= 0:
            sum_chunk_ranges = None
        else:
            sum_chunk_ranges = construct_sum_chunk_ranges(max_complete_chunk, num_summary, device=device)  \
                # (max_comp_chunk, 2)
            sum_chunk_ranges = sum_chunk_ranges[None].repeat(bsz, 1, 1).contiguous()  # (bsz, max_comp_chunk, 2)

        # === Generate Masks for Parts (True is for the masked out)
        ss_mask, sr_mask, rs_mask, rr_mask = None, None, None, None
        if self.gen_parts is None or 'ss' in self.gen_parts:
            ss_mask = self.gen_ss_mask(num_complete_chunks, num_summary, bsz, sum_len, device) \
                # (bsz, num_heads, temp_sum_len, temp_sum_len)
            assert ss_mask is None or ss_mask.dtype == torch.bool
        if self.gen_parts is None or 'sr' in self.gen_parts:
            sr_mask = self.gen_sr_mask(sum_chunk_ranges, reg_chunk_ranges, num_complete_chunks, sum_len, reg_len,
                                       num_pref, self.sum2pref_mode, bsz, device)
            assert sr_mask is None or sr_mask.dtype == torch.bool
        if self.gen_parts is None or 'rs' in self.gen_parts:
            rs_mask = self.gen_rs_mask(reg_chunk_ranges, sum_chunk_ranges, num_complete_chunks, num_chunks,
                                       sum_len, reg_len, num_pref, self.pref2sum_mode, bsz, device)
            assert rs_mask is None or rs_mask.dtype == torch.bool
        if self.gen_parts is None or 'rr' in self.gen_parts:
            rr_mask = self.gen_rr_mask(reg_chunk_ranges, num_chunks, reg_len,
                                       num_pref, self.pref2pref_mode, self.pref2con_mode, self.con2pref_mode,
                                       bsz, device)
            assert rr_mask.dtype == torch.bool

        if not self.single_head and self.num_heads > 1:
            if ss_mask is not None:
                ss_mask = ss_mask.expand(-1, self.real_num_heads, -1, -1)
            if sr_mask is not None:
                sr_mask = sr_mask.expand(-1, self.real_num_heads, -1, -1)
            if rs_mask is not None:
                rs_mask = rs_mask.expand(-1, self.real_num_heads, -1, -1)
            if rr_mask is not None:
                rr_mask = rr_mask.expand(-1, self.real_num_heads, -1, -1)

        attn_mask = {
            'ss': ss_mask,  # (bsz, num_heads, temp_sum_len, temp_sum_len)
            'sr': sr_mask,  # (bsz, num_heads, temp_sum_len, reg_len)
            'rs': rs_mask,  # (bsz, num_heads, reg_len, temp_sum_len)
            'rr': rr_mask,  # (bsz, num_heads, reg_len, reg_len)
        }

        for key in ('ss', 'sr', 'rs', 'rr'):
            if self.gen_parts is not None and key not in self.gen_parts:
                attn_mask.pop(key)

        return attn_mask
