import logging
import os
import numpy as np
from fairseq.data import data_utils
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder
from ..dictionary.compound_dictionary import CompoundDictionary


logger = logging.getLogger(__name__)


def get_beat_ids(src_tokens, dictionary, ts_instead_of_tempo=False):
    change = 's' if ts_instead_of_tempo else 't'

    beat_mask = src_tokens.ge(dictionary.type_dict['o'][0]) & src_tokens.lt(dictionary.type_dict['o'][1])
    change_mask = src_tokens.ge(dictionary.type_dict[change][0]) & src_tokens.lt(dictionary.type_dict[change][1])
    bar_mask = src_tokens.ge(dictionary.type_dict['b'][0]) & src_tokens.lt(dictionary.type_dict['b'][1])
    special_mask = src_tokens.lt(4)
    no_beat_mask = change_mask | bar_mask | special_mask
    del change_mask, bar_mask
    src_tokens = src_tokens.clone() - (dictionary.type_dict['o'][0] - 1)
    cur_beat = 0
    for idx, (token, beat_token) in enumerate(zip(src_tokens, beat_mask)):
        if beat_token:
            cur_beat = token
        else:
            src_tokens[idx] = cur_beat
    src_tokens.masked_fill_(no_beat_mask, 0)
    return src_tokens


class AddBeatDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset, dictionary: CompoundDictionary,
        cache_data_label='default', dataset_name=None,
        mask_ts_instead_of_tempo=False,
    ):
        super().__init__(dataset)
        self.dictionary = dictionary
        self.cache_data_label = cache_data_label
        self.dataset_name = dataset_name
        self.mask_ts_instead_of_tempo = mask_ts_instead_of_tempo

        self.cache_list = None

        cache_name = 'add-beat-dataset_%s_%s' % (self.cache_data_label, self.dataset_name)
        cache_dataset_dir = 'cached_datasets/'
        if cache_data_label is not None:
            cache_dataset_dir = os.path.join(cache_dataset_dir, cache_data_label)
        cache_path = os.path.join(cache_dataset_dir, cache_name)

        if all([os.path.isfile(cache_path + suffix) for suffix in ('.beat_ids.bin', '.beat_ids.idx')]):
            pass
        else:
            logger.info('Building up beat_ids dataset for %s ...' % dataset_name)
            os.makedirs(cache_dataset_dir, exist_ok=True)
            self.beat_ids_builder = MMapIndexedDatasetBuilder(
                cache_path + '.beat_ids.bin', dtype=np.int32
            )

            self.__prepare_dataset()

            self.beat_ids_builder.finalize(cache_path + '.beat_ids.idx')
            del self.beat_ids_builder

        self.beat_ids_dataset = data_utils.load_indexed_dataset(cache_path + '.beat_ids',
                                                                dictionary=None, dataset_impl='mmap')
        assert len(self.beat_ids_dataset) == len(self.dataset)
        for idx, beat_ids in enumerate(self.beat_ids_dataset):
            assert len(beat_ids) == self.dataset.size(idx), (idx, self.dataset.size(idx), len(beat_ids))
        logger.info('Checked the cached beat_ids dataset.')

    def __prepare_dataset(self):
        for sample in self.dataset:
            src_tokens = sample[0]
            beat_ids = get_beat_ids(src_tokens, self.dictionary, ts_instead_of_tempo=self.mask_ts_instead_of_tempo)
            self.beat_ids_builder.add_item(beat_ids)

    def __getitem__(self, idx):
        beat_ids = self.beat_ids_dataset[idx]
        sample = self.dataset[idx]
        return (*sample, beat_ids)

    def collater(self, samples):
        raise NotImplementedError
