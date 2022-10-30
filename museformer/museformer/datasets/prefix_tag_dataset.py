from fairseq.data.base_wrapper_dataset import BaseWrapperDataset


class PrefixTagDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_prefix):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_prefix = num_prefix

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return (*sample, self.num_prefix)

    def collater(self, samples):
        raise NotImplementedError("Dataset class %s is not designed for collating samples." % self.__class__.__name__)