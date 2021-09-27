# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from collections import OrderedDict
import os
import torch

from fairseq.data import (
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    NoisingDataset,
    RoundRobinZipDatasets,
    MonolingualDataset,
    TokenBlockDataset,
    data_utils,
    indexed_dataset,
)

from fairseq.data.dictionary import Dictionary
from fairseq import options, checkpoint_utils, tokenizer
from fairseq.models import FairseqMultiModel
from fairseq.sequence_generator import SequenceGenerator

from fairseq.tasks import register_task, FairseqTask
from fairseq.tasks.semisupervised_translation import parse_lambda_config

from .music_mass_dataset import MusicMassDataset
from .music_mt_dataset import MusicMtDataset


def _get_mass_dataset_key(lang_pair):
    return "mass:" + lang_pair


def _get_mt_dataset_key(lang_pair):
    return "" + lang_pair


class MaskedLMDictionary(Dictionary):
    """
    Dictionary for Masked Language Modelling tasks. This extends Dictionary by
    adding the mask symbol.
    """
    def __init__(
        self,
        pad='<pad>',
        eos='</s>',
        unk='<unk>',
        mask='<mask>',
    ):
        super().__init__()
        self.mask_word = mask
        self.mask_index = self.add_symbol(mask)
        self.nspecial = len(self.symbols)

    def mask(self):
        """Helper to get index of mask symbol"""
        return self.mask_index


@register_task('xmasked_seq2seq')
class XMassTranslationTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--langs', default=None, metavar='LANGS',
                            help='comma-separated list of languages in tasks: en,de,fr')
        parser.add_argument('--source-langs', default=None, metavar='LANGS',
                            help='comma-separated list of source languages: en,fr')
        parser.add_argument('--target-langs', default=None, metavar='LANGS',
                            help='comma-separated list of target languages: en,fr')
        parser.add_argument('--valid-lang-pairs', default='', metavar='LANG-PAIRS',
                            help='comma-separated list of language pairs: en-en, zh-zh')

        parser.add_argument('--mass_steps', default='', metavar='LANG-PAIRS',
                            help='mass for monolingual data (en-en,zh-zh)')
        parser.add_argument('--mt_steps', default='', metavar='LANG-PAIRS',
                            help='supervised machine translation data (en-zh,zh-en)')

        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')

        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')

        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')

        parser.add_argument('--lm-bias', action='store_true',
                            help='append language model bias')

        parser.add_argument('--word_mask', default=0.25, type=float, metavar='RATIO',
                            help='The mask ratio')
        parser.add_argument('--word_mask_keep_rand', default="0.8,0.1,0.1", type=str,
                            help='Word prediction proability')

        parser.add_argument('--reload-checkpoint', type=str, default=None,
                            help="pre-trained checkpoint")

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.training = training
        self.langs = list(dicts.keys())

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = MaskedLMDictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training = cls.prepare(args, **kwargs)
        return cls(args, dicts, training)

    @classmethod
    def prepare(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        s = args.word_mask_keep_rand.split(',')
        s = [float(x) for x in s]
        setattr(args, 'pred_probs', torch.FloatTensor([s[0], s[1], s[2]]))

        args.langs = sorted(args.langs.split(','))
        args.source_langs = sorted(args.source_langs.split(','))
        args.target_langs = sorted(args.target_langs.split(','))

        for lang in args.source_langs:
            assert lang in args.langs
        for lang in args.target_langs:
            assert lang in args.langs

        args.mass_steps = [s for s in args.mass_steps.split(',') if len(s) > 0]
        args.mt_steps = [s for s in args.mt_steps.split(',') if len(s) > 0]

        mono_langs = [
            lang_pair.split('-')[0]
            for lang_pair in args.mass_steps
            if len(lang_pair) > 0
        ]

        mono_lang_pairs = []
        for lang in mono_langs:
            mono_lang_pairs.append('{}-{}'.format(lang, lang))
        setattr(args, 'mono_lang_pairs', mono_lang_pairs)

        args.para_lang_pairs = list(set([
            '-'.join(sorted(lang_pair.split('-')))
            for lang_pair in set(args.mt_steps) if
            len(lang_pair) > 0
        ]))

        args.valid_lang_pairs = [s for s in args.valid_lang_pairs.split(',') if len(s) > 0]

        for lang_pair in args.mono_lang_pairs:
            src, tgt = lang_pair.split('-')
            assert src in args.source_langs and tgt in args.target_langs

        for lang_pair in args.valid_lang_pairs:
            src, tgt = lang_pair.split('-')
            assert src in args.source_langs and tgt in args.target_langs

        if args.source_lang is not None:
            assert args.source_lang in args.source_langs

        if args.target_lang is not None:
            assert args.target_lang in args.target_langs

        langs_id = {}
        ids_lang = {}
        for i, v in enumerate(args.langs):
            langs_id[v] = i
            ids_lang[i] = v
        setattr(args, 'langs_id', langs_id)
        setattr(args, 'ids_lang', ids_lang)

        # If provide source_lang and target_lang, we will switch to translation
        if args.source_lang is not None and args.target_lang is not None:
            setattr(args, 'eval_lang_pair', '{}-{}'.format(args.source_lang, args.target_lang))
            training = False
        else:
            if len(args.para_lang_pairs) > 0:
                required_para = [s for s in set(args.mt_steps)]
                setattr(args, 'eval_lang_pair', required_para[0])
            else:
                setattr(args, 'eval_lang_pair', args.mono_lang_pairs[0])
            training = True
        setattr(args, 'n_lang', len(langs_id))
        setattr(args, 'eval_para', True if len(args.para_lang_pairs) > 0 else False)

        dicts = OrderedDict()
        for lang in args.langs:
            dicts[lang] = MaskedLMDictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[args.langs[0]].pad()
                assert dicts[lang].eos() == dicts[args.langs[0]].eos()
                assert dicts[lang].unk() == dicts[args.langs[0]].unk()
                assert dicts[lang].mask() == dicts[args.langs[0]].mask()
            print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        return dicts, training

    def load_dataset(self, split, **kwargs):

        def load_indexed_dataset(path, dictionary):
            return data_utils.load_indexed_dataset(
                path, dictionary, self.args.dataset_impl,
            )

        def split_exists(split, lang):
            filename = os.path.join(self.args.data, '{}.{}'.format(split, lang))
            return indexed_dataset.dataset_exists(filename, impl=self.args.dataset_impl)

        def split_para_exists(split, key, lang):
            filename = os.path.join(self.args.data, '{}.{}.{}'.format(split, key, lang))
            return indexed_dataset.dataset_exists(filename, impl=self.args.dataset_impl)

        src_mono_datasets = {}
        for lang_pair in self.args.mono_lang_pairs:
            lang = lang_pair.split('-')[0]

            if split_exists(split, lang):
                prefix = os.path.join(self.args.data, '{}.{}'.format(split, lang))
            else:
                raise FileNotFoundError('Not Found available {} dataset for ({}) lang'.format(split, lang))

            src_mono_datasets[lang_pair] = load_indexed_dataset(prefix, self.dicts[lang])
            print('| monolingual {}-{}: {} examples'.format(split, lang, len(src_mono_datasets[lang_pair])))

        src_para_datasets = {}
        for lang_pair in self.args.para_lang_pairs:
            src, tgt = lang_pair.split('-')
            key = '-'.join(sorted([src, tgt]))
            if not split_para_exists(split, key, src):
                raise FileNotFoundError('Not Found available {}-{} para dataset for ({}) lang'.format(split, key, src))
            if not split_para_exists(split, key, tgt):
                raise FileNotFoundError('Not Found available {}-{} para dataset for ({}) lang'.format(split, key, tgt))

            prefix = os.path.join(self.args.data, '{}.{}'.format(split, key))
            if '{}.{}'.format(key, src) not in src_para_datasets:
                src_para_datasets[key + '.' + src] = load_indexed_dataset(prefix + '.' + src, self.dicts[src])
            if '{}.{}'.format(key, tgt) not in src_para_datasets:
                src_para_datasets[key + '.' + tgt] = load_indexed_dataset(prefix + '.' + tgt, self.dicts[tgt])

            print('| bilingual {} {}-{}.{}: {} examples'.format(
                split, src, tgt, src, len(src_para_datasets[key + '.' + src])
            ))
            print('| bilingual {} {}-{}.{}: {} examples'.format(
                split, src, tgt, tgt, len(src_para_datasets[key + '.' + tgt])
            ))

        mt_para_dataset = {}
        for lang_pair in self.args.mt_steps:
            src, tgt = lang_pair.split('-')
            key = '-'.join(sorted([src, tgt]))
            src_key = key + '.' + src
            tgt_key = key + '.' + tgt
            src_dataset = src_para_datasets[src_key]
            tgt_dataset = src_para_datasets[tgt_key]
            src_id, tgt_id = self.args.langs_id[src], self.args.langs_id[tgt]

            mt_para_dataset[lang_pair] = MusicMtDataset(
                src_dataset, src_dataset.sizes,
                tgt_dataset, tgt_dataset.sizes,
                self.dicts[src], self.dicts[tgt],
                src_id, tgt_id,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                src_lang=src,
                tgt_lang=tgt,
            )

        eval_para_dataset = {}
        if split != 'train':
            for lang_pair in self.args.valid_lang_pairs:
                src, tgt = lang_pair.split('-')
                src_id, tgt_id = self.args.langs_id[src], self.args.langs_id[tgt]
                if src == tgt:
                    src_key = src + '-' + tgt
                    tgt_key = src + '-' + tgt
                    src_dataset = src_mono_datasets[src_key]
                    tgt_dataset = src_mono_datasets[tgt_key]
                else:
                    key = '-'.join(sorted([src, tgt]))
                    src_key = key + '.' + src
                    tgt_key = key + '.' + tgt
                    src_dataset = src_para_datasets[src_key]
                    tgt_dataset = src_para_datasets[tgt_key]
                eval_para_dataset[lang_pair] = MusicMtDataset(
                    src_dataset, src_dataset.sizes,
                    tgt_dataset, tgt_dataset.sizes,
                    self.dicts[src], self.dicts[tgt],
                    src_id, tgt_id,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    src_lang=src,
                    tgt_lang=tgt,
                )

        mass_mono_datasets = {}
        if split == 'train':
            for lang_pair in self.args.mass_steps:
                src_dataset = src_mono_datasets[lang_pair]
                lang = lang_pair.split('-')[0]

                mass_mono_dataset = MusicMassDataset(
                    src_dataset, src_dataset.sizes, self.dicts[lang],
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    max_target_positions=self.args.max_target_positions,
                    shuffle=True,
                    lang_id=self.args.langs_id[lang],
                    ratio=self.args.word_mask,
                    pred_probs=self.args.pred_probs,
                    lang=lang
                )
                mass_mono_datasets[lang_pair] = mass_mono_dataset

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (_get_mt_dataset_key(lang_pair), mt_para_dataset[lang_pair])
                for lang_pair in mt_para_dataset.keys()
            ] + [
                (_get_mass_dataset_key(lang_pair), mass_mono_datasets[lang_pair])
                for lang_pair in mass_mono_datasets.keys()
            ] + [
                (_get_mt_dataset_key(lang_pair), eval_para_dataset[lang_pair])
                for lang_pair in eval_para_dataset.keys()
            ]),
            eval_key=None if self.training else self.args.eval_lang_pair
        )

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)
        if args.reload_checkpoint is not None:
            filename = args.reload_checkpoint
            if os.path.exists(filename):
                state = checkpoint_utils.load_checkpoint_to_cpu(filename)
                model.load_state_dict(state['model'], strict=False)
        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        def forward_backward(model, samples, logging_output_key, lang_pair, weight=1.0):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            src_key, tgt_key = lang_pair.split('-')
            samples['net_input']['src_key'] = src_key
            samples['net_input']['tgt_key'] = tgt_key

            loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0

            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            agg_logging_output[logging_output_key] = logging_output

        for lang_pair in self.args.mt_steps:
            sample_key = lang_pair
            forward_backward(model, sample[sample_key], sample_key, lang_pair)

        for lang_pair in self.args.mass_steps:
            sample_key = _get_mass_dataset_key(lang_pair)
            forward_backward(model, sample[sample_key], sample_key, lang_pair)

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

            for lang_pair in self.args.valid_lang_pairs:
                sample_key = lang_pair

                if sample_key not in sample or sample[sample_key] is None or len(sample[sample_key]) == 0:
                    continue

                src_key, tgt_key = lang_pair.split('-')
                sample[sample_key]['net_input']['src_key'] = src_key
                sample[sample_key]['net_input']['tgt_key'] = tgt_key

                loss, sample_size, logging_output = criterion(model, sample[sample_key])

                agg_loss += loss.data.item()
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None,
    ):

        for model in models:
            model.source_lang = self.args.source_lang
            model.target_lang = self.args.target_lang

        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
            )

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        from .song_sequence_generator import SongSequenceGenerator
        return super().build_generator(
            models,
            args,
            SongSequenceGenerator,
            extra_gen_cls_kwargs,
        )

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        logging_output_keys = {
            key
            for logging_output in logging_outputs
            for key in logging_output
        }

        agg_logging_outputs = {
            key: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(key, {}) for logging_output in logging_outputs
            ])
            for key in logging_output_keys
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
            flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')

        return flat_logging_output

    def max_positions(self):
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for key in next(iter(self.datasets.values())).datasets.keys()
        ])

    @property
    def source_dictionary(self):
        return self.dicts[self.args.eval_lang_pair.split('-')[0]]

    @property
    def target_dictionary(self):
        return self.dicts[self.args.eval_lang_pair.split('-')[1]]

    @classmethod
    def load_dictionary(cls, filename):
        return MaskedLMDictionary.load(filename)
