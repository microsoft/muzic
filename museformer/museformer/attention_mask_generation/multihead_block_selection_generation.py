from functools import lru_cache

import torch
import torch.nn as nn

from .block_selection_template_manager import BlockSelectionTemplateManager


class MultiheadBlockSelectionGeneration(nn.Module):
    """
    Generate a set of block masks according to a set of range commands, typically for heads in a layer.
    """
    def __init__(self, multihead_range_commands, init_max_chunk=None):
        """

        :param multihead_range_commands: tuple consisting of head_range_command
        :param init_max_chunk: int
        """
        super().__init__()

        self._manager = BlockSelectionTemplateManager(init_max_chunk)
        self.range_commands = multihead_range_commands

        for range_command in self.range_commands:
            self._manager.register_range_command(range_command)

    @lru_cache(maxsize=128, typed=False)
    def get_block_selection_masks(self, num_chunks: int):
        """
        Get selection masks of size (num_chunks, num_chunks, num_commands) corresponding to all the range_commands.
        :param num_chunks:
        :return:
        """
        assert num_chunks > 0
        self._manager.update(num_chunks)

        commands_masks = []
        false_chunk = torch.zeros(num_chunks, num_chunks, dtype=torch.bool, device=self._manager.device)
        considered_range_commands = set()
        for idx, range_command in enumerate(self.range_commands):
            mask = self._manager.mask(range_command)
            if mask is None:
                assert self.range_commands[idx] is None
                commands_masks.append(false_chunk)
            else:
                mask = mask[:num_chunks, :num_chunks]
                commands_masks.append(mask)
            considered_range_commands.add(range_command)
        commands_masks = torch.stack(commands_masks, dim=-1)  # (num_chunks, num_chunks, num_commands)

        return commands_masks

    @staticmethod
    def get_overall_mask_from_commands_masks(commands_masks):
        return commands_masks.sum(dim=-1).gt(0)

    def get_block_indices_and_masks_from_selection_masks(self, commands_masks, overall_mask=None):
        """

        :param commands_masks:
        :param overall_mask:
        :return:
        """
        if overall_mask is None:
            overall_mask = self.get_overall_mask_from_commands_masks(commands_masks)
        else:
            assert overall_mask.ndim == 2
        num_chunks_1, num_chunks_2 = overall_mask.shape[:2]
        index_matrix = self._manager.index_matrix[:num_chunks_1, :num_chunks_2]
        indices = index_matrix.masked_select(overall_mask.unsqueeze(-1)).view(-1, 2)
        commands_masks = commands_masks.masked_select(overall_mask.unsqueeze(-1)).view(-1, len(self.range_commands))
        return indices, commands_masks

    @lru_cache(maxsize=128, typed=False)
    def get_block_indices_and_masks(self, num_chunks: int):
        """

        :param num_chunks: int, number of chunks in one sample
        :return: block indices (query, key) to compute  (num, 2);
                 mask indicating whether to compute in each c (head)
        """
        commands_masks = self.get_block_selection_masks(num_chunks)
        overall_mask = self.get_overall_mask_from_commands_masks(commands_masks)
        return self.get_block_indices_and_masks_from_selection_masks(commands_masks, overall_mask=overall_mask)

    def get_diagonal_indices(self, num_chunks):
        return self._manager.get_diagonal_indices(num_chunks)


def get_block_ranges(row_ranges, col_ranges, block_indices):
    """

    :param row_ranges: begins and endings for chunks on row dimension. (num_tgt_chunks, 2)
    :param col_ranges: begins and endings for chunks on col dimension. (num_src_chunks, 2)
    :param block_indices: row and col indices of selected blocks. (num_blocks, 2)
    :return:
    """
    tgt_ranges = row_ranges[block_indices[:, 0]]  # (num_blocks, 2)
    src_ranges = col_ranges[block_indices[:, 1]]  # (num_blocks, 2)
    block_ranges = torch.cat((tgt_ranges, src_ranges), dim=1)  # (num_blocks, 4)
    return block_ranges
