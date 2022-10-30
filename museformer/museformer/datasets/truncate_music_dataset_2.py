import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset


class TruncateMusicDataset(BaseWrapperDataset):
    def __init__(self, dataset, truncated_length):
        super().__init__(dataset)
        assert truncated_length is None or truncated_length > 1
        self.truncated_length = truncated_length

        self._sizes = self.dataset.sizes.copy()
        self._sizes[self._sizes > self.truncated_length] = self.truncated_length

    def __getitem__(self, idx):
        src, tgt, chunk_points, beat_ids = self.dataset[idx]
        num_chunks = len(chunk_points) - 1
        if self.truncated_length is None or self.dataset.size(idx) <= self.truncated_length:
            return src, tgt, chunk_points, beat_ids, num_chunks, num_chunks  # num_complete_chunks

        src = src[:self.truncated_length]
        tgt = tgt[:self.truncated_length]
        beat_ids = beat_ids[:self.truncated_length]
        chunk_points = chunk_points[chunk_points.le(self.truncated_length)]
        if chunk_points[-1] == self.truncated_length:
            num_chunks = len(chunk_points) - 1
            num_complete_chunks = num_chunks
        else:
            num_chunks = len(chunk_points)
            num_complete_chunks = num_chunks - 1
            chunk_points = torch.cat((chunk_points, chunk_points.new_tensor([self.truncated_length])))
        return src, tgt, chunk_points, beat_ids, num_chunks, num_complete_chunks

    @property
    def sizes(self):
        return self._sizes

    def size(self, index):
        return self._sizes[index]

    def num_tokens(self, index):
        return self._sizes[index]

    def collater(self, samples):
        raise NotImplementedError
