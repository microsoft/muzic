from fairseq.data import FairseqDataset
from torch.utils.data.dataloader import default_collate


class ExtendedWrapperDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, "collater"):
            return self.dataset.collater(samples)
        else:
            return default_collate(samples)

    @property
    def sizes(self):
        return self.dataset.sizes

    def num_tokens(self, index):
        return self.dataset.sizes[index]

    def size(self, index):
        return self.dataset.sizes[index]

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def attr(self, attr: str, index: int):
        return self.dataset.attr(attr, index)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

