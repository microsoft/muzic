import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from fairseq.data import data_utils
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset

logger = logging.getLogger(__name__)


class PostProcessDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index):
        src, tgt, chunk_points, beat_ids, num_chunks, num_complete_chunks, num_prefix = self.dataset[index]
        seq_len = src.shape[0]

        new_sample = {
            'id': index,
            'src_tokens': src,
            'src_length': seq_len,
            'target': tgt,
            'chunk_points': chunk_points,
            'num_chunks': num_chunks,
            'num_complete_chunks': num_complete_chunks,
            'num_pref': num_prefix,
            'beat_ids': beat_ids,
        }

        return new_sample

    @property
    def sizes(self):
        return self.dataset.sizes

    def size(self, index):
        return self.dataset.size(index)

    def num_tokens(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        bsz = len(samples)
        sample_id = torch.tensor([s['id'] for s in samples])
        src_tokens = [s['src_tokens'] for s in samples]
        src_lengths = [s['src_length'] for s in samples]
        target = [s['target'] for s in samples]
        chunk_points = [s['chunk_points'] for s in samples]
        num_chunks = [s['num_chunks'] for s in samples]
        num_complete_chunks = [s['num_complete_chunks'] for s in samples]
        num_prefix = [s['num_pref'] for s in samples]
        beat_ids = [s['beat_ids'] for s in samples]
        ntokens = sum(src_lengths)

        src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=0)
        src_lengths = torch.tensor(src_lengths, dtype=torch.long)
        target = pad_sequence(target, batch_first=True, padding_value=0)
        chunk_points = data_utils.collate_tokens(
            chunk_points, 0
        )
        num_chunks = torch.tensor(num_chunks, dtype=torch.long)
        num_complete_chunks = torch.tensor(num_complete_chunks, dtype=torch.long)
        num_prefix = torch.tensor(num_prefix, dtype=torch.long)
        beat_ids = pad_sequence(beat_ids, batch_first=True, padding_value=0)

        batch = {
            'id': sample_id,
            'nsentences': bsz,
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'chunk_points': chunk_points,
                'num_chunks': num_chunks,
                'num_complete_chunks': num_complete_chunks,
                'num_pref': num_prefix,
                'beat_ids': beat_ids,
            },
            'target': target,
        }

        return batch

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        # print(indices)
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = data_utils._filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        if len(ignored) > 0:
            print(self.sizes)
            print(ignored)
            print(max_sizes)
        return indices, ignored

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        """
        Given an ordered set of indices, return batches according to
        *max_tokens*, *max_sentences* and *required_batch_size_multiple*.
        """
        from fairseq.data import data_utils

        fixed_shapes = self.get_batch_shapes()
        if fixed_shapes is not None:

            def adjust_bsz(bsz, num_tokens):
                if bsz is None:
                    assert max_tokens is not None, "Must specify --max-tokens"
                    bsz = max_tokens // num_tokens
                if max_sentences is not None:
                    bsz = min(bsz, max_sentences)
                elif (
                    bsz >= required_batch_size_multiple
                    and bsz % required_batch_size_multiple != 0
                ):
                    bsz -= bsz % required_batch_size_multiple
                return bsz

            fixed_shapes = np.array(
                [
                    [adjust_bsz(bsz, num_tokens), num_tokens]
                    for (bsz, num_tokens) in fixed_shapes
                ]
            )

        return data_utils.batch_by_size(
            indices,
            num_tokens_fn=self.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            fixed_shapes=fixed_shapes,
        )
