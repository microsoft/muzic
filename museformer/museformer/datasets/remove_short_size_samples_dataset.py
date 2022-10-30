# Author: Botao Yu

import numpy as np
from fairseq.data import FairseqDataset


class RemoveShortSizeSamplesDataset(FairseqDataset):
    def __init__(self, dataset, min_size):
        super().__init__()
        # st = time.time()
        self.dataset = dataset

        self.min_size = min_size

        min_token_select = self.dataset.sizes >= self.min_size

        final_select = min_token_select
        self.selected_index = np.nonzero(final_select)[0]

        # et = time.time()
        # print('%s dataset: %.2fs' % (self.__class__.__name__, et - st))

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


def MayRemoveShortSizeSamplesDataset(dataset, min_size):
    if min_size is None or min_size == 0:
        return dataset

    new_dataset = RemoveShortSizeSamplesDataset(dataset, min_size)
    if len(new_dataset) == len(dataset):
        del new_dataset
        return dataset
    return new_dataset
