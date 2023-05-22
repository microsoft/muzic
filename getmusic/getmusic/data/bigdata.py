from torch.utils.data import Dataset
import numpy as np
import torch
from getmusic.data.indexed_datasets import IndexedDataset
import random
import itertools as it

class BigDataset(Dataset):
    def __init__(self, prefix, vocab_size, path=None):
        self.data_dir = path
        self.prefix = prefix
        self.ds_name = 'train' if prefix == 'train' else 'valid'
        self.size = np.load(f'{self.data_dir}/{self.ds_name}_length.npy')
        self.empty_idx = vocab_size - 1
        self.indexed_ds = None
     
    def __len__(self):
        return self.size
    
    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.ds_name}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        item = self._get_item(index)
        return item

    def collater(self, samples):
        if len(samples) == 0:
            assert 1==0
            return {}

        batch = {}
        batch['tempo'] = torch.LongTensor(np.array([s[0,-1] for s in samples]))
        batch['data'] = []
        batch['condition_pos'] = []
        batch['not_empty_pos'] = []
        
        for sample in samples:
            sample = sample[:,:-1]
            assert sample.shape == (14, 512)
            
            figure_size = 512

            track_not_empty_bool = torch.tensor((sample == self.empty_idx).astype(float).sum(-1) != figure_size).float()
            
            # because preprocessed music has at least 2 tracks
            # we have to randomly perform a single track generation 
            if random.randint(0,9) == 0:
                not_empty_track_index = torch.nonzero(track_not_empty_bool[:-2])  # can't only keep chord
                not_empty_track_index = [i // 2 for i in not_empty_track_index.view(-1).tolist() if i % 2 == 0]
                single_track_index = random.choice(not_empty_track_index)
                sample[:2 * single_track_index] = self.empty_idx
                sample[2 * single_track_index + 2:-2] = self.empty_idx
            
            # which track is empty
            track_not_empty_bool = torch.tensor((sample == self.empty_idx).astype(float).sum(-1) != figure_size).float()

            track_not_empty_num = track_not_empty_bool.sum()

            content_num = random.randint(1, int(track_not_empty_num / 2) - 1) # chord can not be a content track

            condition_bool = track_not_empty_bool.clone()

            # if content num != not empty num, conditional generation
            if content_num != track_not_empty_num:
                not_empty_track_index = torch.nonzero(track_not_empty_bool[:-2])  
                not_empty_track_index = [i // 2 for i in not_empty_track_index.view(-1).tolist() if i % 2 == 0]
                content_combination = it.combinations(not_empty_track_index, content_num)
                content_combination = [i for i in content_combination]
                content_track_index = random.choice(content_combination)
                for c_idx in content_track_index:
                    condition_bool[c_idx * 2] = 0
                    condition_bool[c_idx * 2 + 1] = 0

            # randomly disable chord guidance
            if random.randint(0,3) < 3:
                sample[-2:] = self.empty_idx
                condition_bool[-2:] = 0

            assert (track_not_empty_bool * condition_bool != track_not_empty_bool).any()
            condition_bool = condition_bool.view(14,1).repeat(1,figure_size)

            track_not_empty_bool = (torch.tensor(sample) != self.empty_idx).float()

            batch['data'].append(sample)
            batch['condition_pos'].append(condition_bool)
            batch['not_empty_pos'].append(track_not_empty_bool)
        
        batch['data'] = torch.LongTensor(np.array(batch['data']))
        batch['condition_pos'] = torch.stack(batch['condition_pos'], dim=0)
        batch['not_empty_pos'] = torch.stack(batch['not_empty_pos'], dim=0)
        
        return batch