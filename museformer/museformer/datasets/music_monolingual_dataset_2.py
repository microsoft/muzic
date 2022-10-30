import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset


class MusicMonolingualDataset(BaseWrapperDataset):
    def __int__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]  # (len,)
        return torch.cat((sample[-1:], sample[:-1]), dim=0), sample

    def collater(self, samples):
        raise NotImplementedError
