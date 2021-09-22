# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

import numpy as np
import torch

from fairseq import utils

from fairseq.data import data_utils, FairseqDataset


class MusicMassDataset(FairseqDataset):
    """Masked Language Pair dataset (only support for single language)
       [x1, x2, x3, x4, x5]
                 |
                 V
       src: [x1, _, _, x4, x5]
       tgt: [x1, x2] => [x2, x3]
    """

    def __init__(
        self, src, sizes, vocab,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, lang_id=None, ratio=None, training=True,
        pred_probs=None, lang="",
    ):
        self.src = src
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.lang_id = lang_id
        self.ratio = ratio
        self.training = training
        self.pred_probs = pred_probs

        self.sep_token = vocab.nspecial
        self.align_token = self.sep_token + 1
        self.lang = lang
        self.mask_len_expect_per_segment = 10
        self.pitch_start = self.align_token + 1
        self.duration_start = self.align_token + 129

    def __getitem__(self, index):
        if self.training is False:
            src_item = self.src[index]
            src_list = src_item.tolist()
            sep_positions = [i for i, x in enumerate(src_list) if x == self.sep_token]
            sep_positions.insert(0, -1)
            source = []
            source_sent_ids = []
            for i in range(len(sep_positions)-1):
                sent = src_list[sep_positions[i] + 1: sep_positions[i + 1]]
                sent = [ch for ch in sent if ch != self.align_token]
                source.extend(sent)
                source_sent_ids.extend([i] * len(sent))
            source.append(self.vocab.eos_index)
            source_sent_ids.append(-1)  # -1 non word for lyric
            source.insert(0, self.vocab.eos_index)
            source_sent_ids.insert(0, -1)

            output = source[1:]
            target_sent_ids = source_sent_ids[1:]
            target = source[:-1]
        else:
            src_item = self.src[index]
            src_list = src_item.tolist()

            sep_positions = [
                i for i, x in enumerate(src_list) if x == self.sep_token
            ]
            sep_positions.insert(0, -1)

            s = []
            source_sent_ids = []
            for i in range(len(sep_positions)-1):
                sent = src_list[sep_positions[i] + 1:sep_positions[i + 1]]
                sent = [ch for ch in sent if ch != self.align_token]
                s.extend(sent)
                source_sent_ids.extend([i] * len(sent))

            segment_num = round(len(s) / (
                self.mask_len_expect_per_segment / self.ratio
            ))
            segment_num = max(1, segment_num)
            seg_len = len(s) // segment_num

            source = []
            output = []
            target = []
            target_sent_ids = []

            for i in range(segment_num):
                seg_start = i * seg_len
                seg_end = (i+1) * seg_len
                if i == segment_num - 1:
                    seg_end = len(s)
                if self.lang == 'melody':
                    assert len(s) % 2 == 0
                    if seg_start % 2 == 1:
                        seg_start -= 1
                    if seg_end % 2 == 1:
                        seg_end -= 1
                    mask_start, mask_length = self.mask_interval_align(seg_start, seg_end)
                else:
                    mask_start, mask_length = self.mask_interval(seg_start, seg_end)

                output.extend(s[mask_start: mask_start + mask_length].copy())

                for j in range(mask_start, mask_start + mask_length):
                    target_sent_ids.append(source_sent_ids[j])
                if mask_start == 0:
                    t = [self.vocab.eos_index] + s[mask_start: mask_start + mask_length - 1].copy()
                else:
                    t = s[mask_start - 1: mask_start + mask_length - 1].copy()

                if self.lang == 'lyric':
                    for w in t:
                        target.append(self.random_word(w, self.pred_probs))
                    for i in range(seg_start, seg_end):
                        w = s[i]
                        if i >= mask_start and i < mask_start + mask_length:
                            w = self.mask_word(w)
                        if w is not None:
                            source.append(w)
                else:
                    t = t[1:] + [t[0]]
                    t2 = []
                    for i in range(0, len(t), 2):
                        pit, dur = self.random_pitch_duration(t[i], t[i + 1], self.pred_probs)
                        t2.append(pit)
                        t2.append(dur)
                    t = [t2[-1]] + t2[:-1]
                    target.extend(t)

                    assert seg_start % 2 == 0
                    assert seg_end % 2 == 0
                    for i in range(seg_start, seg_end, 2):
                        pit = s[i]
                        dur = s[i+1]
                        if i >= mask_start and i + 1 < mask_start + mask_length:
                            pit, dur = self.mask_pitch_duration(pit, dur) 
                        if pit is not None and dur is not None:
                            source.append(pit)
                            source.append(dur)

            source.append(self.vocab.eos_index)
            source_sent_ids.append(-1)
            assert len(output) == len(target)
            assert len(source_sent_ids) == len(source)
            assert len(target_sent_ids) == len(target)

        return {
            'id': index,
            'source': torch.LongTensor(source),
            'target': torch.LongTensor(target),
            'output': torch.LongTensor(output),
            'source_sent_ids': torch.LongTensor(source_sent_ids),
            'target_sent_ids': torch.LongTensor(target_sent_ids)
        }

    def __len__(self):
        return len(self.src)

    def _collate(self, samples, pad_idx, eos_idx, segment_label):

        def merge(key, left_pad):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad,
            )

        def merge_sentId(key, left_pad, pad_idx=pad_idx):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=self.left_pad_source)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        ntokens = sum(len(s['target']) for s in samples)

        prev_output_tokens = merge('target', left_pad=self.left_pad_target)
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

        target = merge('output', left_pad=self.left_pad_target)
        target = target.index_select(0, sort_order)

        source_sent_ids = merge_sentId(
            'source_sent_ids', left_pad=self.left_pad_target, pad_idx=-1
        )
        source_sent_ids = source_sent_ids.index_select(0, sort_order)
        target_sent_ids = merge_sentId(
            'target_sent_ids', left_pad=self.left_pad_target, pad_idx=-2
        )
        target_sent_ids = target_sent_ids.index_select(0, sort_order)

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths
            },
            'target': target,
        }
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
        batch['net_input']['source_sent_ids'] = source_sent_ids
        batch['net_input']['target_sent_ids'] = target_sent_ids
        return batch

    def collater(self, samples):
        return self._collate(
            samples,
            pad_idx=self.vocab.pad(),
            eos_idx=self.vocab.eos(),
            segment_label=self.lang_id,
        )

    def get_dummy_batch(
        self,
        num_tokens,
        max_positions,
        tgt_len=128
    ):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            tgt_len = min(tgt_len, max_positions)
        source = self.vocab.dummy_sentence(tgt_len)
        target = self.vocab.dummy_sentence(tgt_len)
        bsz = max(num_tokens // tgt_len, 1)
        return self.collater([
            {
                'id': i,
                'source': source,
                'target': target,
                'output': target,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False) and getattr(self.src, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)

    def size(self, index):
        return (self.sizes[index], int(round(self.sizes[index] * self.ratio)))

    def mask_word(self, w):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(self.vocab.nspecial+1, len(self.vocab))
        else:
            return w

    def random_word(self, w, pred_probs):
        cands = [
            self.vocab.mask_index,
            np.random.randint(self.vocab.nspecial+1, len(self.vocab)),
            w
        ]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_pitch(self, w):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(self.pitch_start, self.duration_start)
        else:
            return w

    def random_pitch(self, w, pred_probs):
        cands = [
            self.vocab.mask_index,
            np.random.randint(self.pitch_start, self.duration_start),
            w
        ]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_duration(self, w):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index
        elif p >= 0.1:
            return np.random.randint(self.duration_start, len(self.vocab))
        else:
            return w

    def random_duration(self, w, pred_probs):
        cands = [
            self.vocab.mask_index,
            np.random.randint(self.duration_start, len(self.vocab)),
            w
        ]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_pitch_duration(self, pit, dur):
        p = np.random.random()
        if p >= 0.2:
            return self.vocab.mask_index, self.vocab.mask_index
        elif p >= 0.1:
            ret_pit = np.random.randint(self.pitch_start, self.duration_start)
            ret_dur = np.random.randint(self.duration_start, len(self.vocab))
            return ret_pit, ret_dur
        else:
            return pit, dur

    def random_pitch_duration(self, pit, dur, pred_probs):
        rnd_pit = np.random.randint(self.pitch_start, self.duration_start)
        rnd_dur = np.random.randint(self.duration_start, len(self.vocab))
        cands = [
            (self.vocab.mask_index, self.vocab.mask_index),
            (rnd_pit, rnd_dur),
            (pit, dur)
        ]
        prob = torch.multinomial(self.pred_probs, 1, replacement=True)
        return cands[prob]

    def mask_interval(self, start, end):
        # not include end
        mask_length = round((end - start) * self.ratio)
        mask_length = max(1, mask_length)
        mask_start = self.mask_start(start, end - mask_length)
        return mask_start, mask_length

    def mask_start(self, start, end):
        p = np.random.random()
        if p >= 0.8:
            return start
        elif p >= 0.6:
            return end
        else:
            return np.random.randint(start, end + 1)

    def mask_interval_align(self, start, end):
        # not include end
        mask_length = round((end-start) * self.ratio)
        if mask_length % 2 != 0:
            mask_length -= 1
        mask_length = max(2, mask_length)
        mask_start = self.mask_start(start, end - mask_length)
        if mask_start % 2 != 0:
            mask_start -= 1
        return mask_start, mask_length
