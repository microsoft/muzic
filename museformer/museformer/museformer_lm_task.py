import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from fairseq import utils
from fairseq.tasks.language_modeling import LanguageModelingTask
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from omegaconf import II
from fairseq.data import data_utils
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.tasks import register_task

from .dictionary.compound_dictionary import CompoundDictionary
from .tools import arg_tools
from .datasets.extended_wrapper_dataset import ExtendedWrapperDataset
from .datasets.remove_short_size_samples_dataset import MayRemoveShortSizeSamplesDataset
from .datasets.chunk_sequence_dataset_2 import ChunkSequenceDataset as ChunkSequenceDataset2
from .datasets.prefix_tag_dataset import PrefixTagDataset
from .datasets.music_monolingual_dataset_2 import MusicMonolingualDataset as MusicMonolingualDataset2
from .datasets.remove_long_size_samples_dataset import MayRemoveLongSizeSamplesDataset
from .datasets.truncate_music_dataset_2 import TruncateMusicDataset as TruncateMusicDataset2
from .datasets.post_process_dataset_2 import PostProcessDataset as PostProcessDataset2
from .datasets.add_beat_dataset import AddBeatDataset
from .sequence_generator import MuseformerSequenceGenerator

logger = logging.getLogger(__name__)


@dataclass
class MuseformerLanguageModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    seed: int = II("params.common.seed")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "params.dataset.dataset_impl"
    )


@register_task('museformer_language_modeling', dataclass=MuseformerLanguageModelingConfig)
class MuseformerLanguageModelingTask(LanguageModelingTask):
    @classmethod
    def add_args(cls, parser):
        super(MuseformerLanguageModelingTask, cls).add_args(parser)

        # Basic
        parser.add_argument('--eob-token', default='b-1')
        parser.add_argument('--eoc-token', type=arg_tools.str_to_type_with_specific_word_as_none(str, 'None'),
                            default='e-1')

        parser.add_argument('--chunking-scheme', choices=('bar_aware', 'fixed'), default='bar_aware')
        parser.add_argument('--fixed-chunking-length', type=int, default=None)

        parser.add_argument('--max-size-train', type=int)
        parser.add_argument('--max-size-valid', type=int)
        parser.add_argument('--max-size-test', type=int)

        parser.add_argument('--truncate-train', type=int)
        parser.add_argument('--truncate-valid', type=int)
        parser.add_argument('--truncate-test', type=int)

        parser.add_argument('--take-bos-as-bar', type=arg_tools.str_bool_with_default_error, default=False)
        parser.add_argument('--beat-mask-ts', type=arg_tools.str_bool_with_default_error, default=False)

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dictionary = CompoundDictionary.load(os.path.join(paths[0], "dict.txt"))
            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                raise NotImplementedError
                # output_dictionary = TruncatedDictionary(
                #     dictionary, args.output_dictionary_size
                # )
        return (dictionary, output_dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = ExtendedWrapperDataset(dataset)

        dataset = MayRemoveShortSizeSamplesDataset(dataset, 2)  # Delete empty samples

        assert self.args.eob_token in self.dictionary
        eob_index = self.dictionary.index(self.args.eob_token)
        eoc_index = None

        data_name = self.args.data.replace('\\', '/')
        data_name = data_name[:-1] if self.args.data.endswith('/') else data_name
        data_name = data_name.split('/')[-1]

        take_bos_as_bar = getattr(self.args, 'take_bos_as_bar', False)

        dataset = MusicMonolingualDataset2(dataset)

        dataset = ChunkSequenceDataset2(
            dataset, self.source_dictionary,
            eob_index, eoc_index,
            chunking_scheme=self.args.chunking_scheme,
            chunking_length=getattr(self.args, 'fixed_chunking_length', None),
            dataset_name=split,
            cache_data_label=data_name,
            cache_sequence=True,
            offset=1 if take_bos_as_bar else 0,
            take_bos_as_bar=take_bos_as_bar,
            bos_index=self.source_dictionary.eos_index
        )

        dataset = AddBeatDataset(dataset, self.source_dictionary, cache_data_label=data_name, dataset_name=split,
                                 mask_ts_instead_of_tempo=self.args.beat_mask_ts)

        max_size_split = getattr(self.args, 'max_size_%s' % split, None)
        dataset = MayRemoveLongSizeSamplesDataset(dataset, max_size_split)

        truncate_length = getattr(self.args, 'truncate_%s' % split, None)
        dataset = TruncateMusicDataset2(dataset, truncate_length)

        dataset = PrefixTagDataset(dataset, 0 if take_bos_as_bar or self.args.chunking_scheme != 'bar_aware' else 1)

        dataset = PostProcessDataset2(dataset)

        self.datasets[split] = dataset

        logger.info('loaded %d samples for %s' % (len(self.datasets[split]), split))

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if seq_gen_cls is None:
            seq_gen_cls = MuseformerSequenceGenerator
        return super().build_generator(models, args, seq_gen_cls=seq_gen_cls, extra_gen_cls_kwargs=extra_gen_cls_kwargs)
