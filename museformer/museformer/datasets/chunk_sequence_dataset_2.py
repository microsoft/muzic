# import os
# from copy import deepcopy
# import pickle
import logging

# import numpy as np
import torch
# from fairseq.data import data_utils, FairseqDataset
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
# from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder


logger = logging.getLogger(__name__)


def get_bar_chunk_points(seq: torch.Tensor, eob_index, begin_idx=0, take_bos_as_bar=False, bos_index=None):
    # seq: (seq_len,)
    # eob_index: int
    is_complete_bar = seq[-1] == eob_index
    indices = seq.eq(eob_index).nonzero(as_tuple=False).squeeze(1)  # (num_bars,)
    indices = indices + 1
    indices = torch.cat(
        (indices.new_tensor([begin_idx]), indices), dim=0
    )
    len_seq = len(seq)
    if not is_complete_bar and len_seq > begin_idx:
        indices = torch.cat(
            (indices, indices.new_tensor([len_seq])), dim=0
        )
    if take_bos_as_bar:
        assert seq[0] == bos_index
        assert begin_idx == 1
        indices = torch.cat(
            (torch.tensor([0]), indices), dim=0
        )

    return indices, is_complete_bar


class BarChunkSequenceDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset, src_dict, eob,
        eos_appended=True,
        offset=1,
        take_bos_as_bar=True,
        bos_index=None,
    ):
        super().__init__(dataset)
        self.src_dict = src_dict
        self.eos_appended = eos_appended
        self.eob = eob
        self.offset = offset
        self.take_bos_as_bar = take_bos_as_bar
        self.bos_index = bos_index
        self.cache = [None] * len(self.dataset)

    def __iter__(self):
        len_dataset = len(self)
        for idx in range(len_dataset):
            yield self[idx]

    def __getitem__(self, index):
        src, tgt = self.dataset[index]  # all include eoc
        chunk_points = self.cache[index]
        if chunk_points is None:
            chunk_points, complete = get_bar_chunk_points(
                src, self.eob, begin_idx=self.offset,
                take_bos_as_bar=self.take_bos_as_bar, bos_index=self.bos_index
            )
            assert complete
            self.cache[index] = chunk_points
        return src, tgt, chunk_points

    def collater(self, samples):
        raise NotImplementedError("Dataset class %s is not designed for collating samples." % self.__class__.__name__)


class FixedChunkingLengthDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset, fixed_chunking_length
    ):
        assert fixed_chunking_length is not None
        super().__init__(dataset)
        self.fixed_chunking_length = fixed_chunking_length

    def __iter__(self):
        len_dataset = len(self)
        for idx in range(len_dataset):
            yield self[idx]

    def __getitem__(self, index):
        src, tgt = self.dataset[index]  # all include eoc

        sample_len = len(src)
        chunk_points = torch.arange(0, sample_len, self.fixed_chunking_length)
        chunk_points = torch.cat((chunk_points, chunk_points.new_tensor([sample_len])), dim=0)

        return src, tgt, chunk_points

    def collater(self, samples):
        raise NotImplementedError("Dataset class %s is not designed for collating samples." % self.__class__.__name__)


def ChunkSequenceDataset(
    dataset, src_dict,
    eob, eoc,
    chunking_scheme='bar_aware',
    chunking_length=None,
    dataset_name=None,
    cache_data_label=None,
    cache_sequence=None,
    offset=0,
    take_bos_as_bar=False, bos_index=None
):
    if chunking_scheme == 'bar_aware':
        return BarChunkSequenceDataset(
            dataset, src_dict, eob,
            offset=offset,
            take_bos_as_bar=take_bos_as_bar,
            bos_index=bos_index
        )
    elif chunking_scheme == 'fixed':
        return FixedChunkingLengthDataset(
            dataset, chunking_length
        )

    raise NotImplementedError(chunking_scheme)
