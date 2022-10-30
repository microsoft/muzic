# Author: Botao Yu

import numpy as np
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset


class RemoveLongSizeSamplesDataset(BaseWrapperDataset):
    def __init__(self, dataset, max_size):
        super().__init__(dataset)
        self.max_size = max_size
        max_token_select = self.dataset.sizes <= self.max_size
        final_select = max_token_select
        self.selected_index = np.nonzero(final_select)[0]

    def __getitem__(self, index):
        origin_index = self.selected_index[index]
        return self.dataset[origin_index]

    def __len__(self):
        return len(self.selected_index)

    def __iter__(self):
        len_dataset = len(self)
        for idx in range(len_dataset):
            yield self[idx]

    @property
    def sizes(self):
        return self.dataset.sizes[self.selected_index]

    def size(self, index):
        origin_index = self.selected_index[index]
        return self.dataset.size(origin_index)

    def num_tokens(self, index):
        origin_index = self.selected_index[index]
        return self.dataset.num_tokens(origin_index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self), dtype=np.int64)


def MayRemoveLongSizeSamplesDataset(dataset, max_size):
    if max_size is None:
        return dataset

    new_dataset = RemoveLongSizeSamplesDataset(dataset, max_size)
    if len(new_dataset) == len(dataset):
        del new_dataset
        return dataset
    return new_dataset
