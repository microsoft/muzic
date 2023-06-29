from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
import numpy as np
from fairseq.data import data_utils
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from fairseq.tasks import register_task

import logging
from .linear import transformer_lm

from fairseq import search

from fairseq.models.transformer_lm import (
    TransformerLanguageModel,
    TransformerLanguageModelConfig,
    base_lm_architecture,
    transformer_lm_gpt,
    DEFAULT_MAX_TARGET_POSITIONS
)

from fairseq.models import (
    register_model,
    register_model_architecture,
)
import torch



logger = logging.getLogger(__name__)




class CommandDataset(BaseWrapperDataset):
    def __init__(self, dataset, command_data, args = None):
        super().__init__(dataset)
        self._sizes = self.dataset.sizes.copy()
        self.command_data = command_data # need change to np.mmap
        self.args = args

    def __getitem__(self, index):
        sample = self.dataset[index]
        assert len(sample["source"]) <= self.args.truncated_length + 2, f"The maximum length exceeds {self.args.truncated_length}. Please resample the dataset."
        return {
            "id": index,
            "source": sample["source"],
            "target": sample["target"],
            "command": torch.from_numpy(np.array(self.command_data[index])).to(sample["source"].device)
        }


    @property
    def sizes(self):
        return self._sizes

    def size(self, index):
        return self._sizes[index]

    def num_tokens(self, index):
        return self._sizes[index]

    def filter_indices_by_size(self, indices, max_sizes):
        """
        Filter a list of sample indices. Remove those that are longer than
        specified in *max_sizes*.

        WARNING: don't update, override method in child classes

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        if isinstance(max_sizes, float) or isinstance(max_sizes, int):
            if hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray):
                ignored = indices[self.sizes[indices] > max_sizes].tolist()
                indices = indices[self.sizes[indices] <= max_sizes]
            elif (
                hasattr(self, "sizes")
                and isinstance(self.sizes, list)
                and len(self.sizes) == 1
            ):
                ignored = indices[self.sizes[0][indices] > max_sizes].tolist()
                indices = indices[self.sizes[0][indices] <= max_sizes]
            else:
                indices, ignored = data_utils._filter_by_size_dynamic(
                    indices, self.size, max_sizes
                )
        else:
            indices, ignored = data_utils._filter_by_size_dynamic(
                indices, self.size, max_sizes
            )
        if len(ignored) > 0:
            print(self.sizes)
            print(ignored)
            print(max_sizes)
        return indices, ignored

    def collater(self, samples):
        # samples: list->dict, just as __get_item__(index) rets
        # print("samples type:", type(samples))

        return self.collate_helper(samples, self.dataset.vocab.pad(), self.dataset.vocab.eos())
    def collate_helper(self, samples, pad_idx, eos_idx):
        if len(samples) == 0:
            return {}
        def merge(key, is_list=False):
            if is_list:
                res = []
                for i in range(len(samples[0][key])):
                    res.append(
                        data_utils.collate_tokens(
                            [s[key][i] for s in samples],
                            pad_idx,
                            eos_idx,
                            left_pad=False,
                        )
                    )
                return res
            else:
                return data_utils.collate_tokens(
                    [s[key] for s in samples],
                    pad_idx,
                    eos_idx,
                    left_pad=False,
                )

        src_tokens = merge("source")
        if samples[0]["command"] is not None:
            command_tokens = merge("command")
        else:
            command_tokens = None
        if samples[0]["target"] is not None:
            is_target_list = isinstance(samples[0]["target"], list)
            target = merge("target", is_target_list)
        else:
            target = src_tokens

        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "nsentences": len(samples),
            "ntokens": sum(len(s["source"]) for s in samples),
            "net_input": {
                "src_tokens": src_tokens,
                "command_input": command_tokens,
                "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
            },
            "target": target,
        }


@register_task("language_modeling_control", dataclass=LanguageModelingConfig)
class LanguageModelingTaskWithControl(LanguageModelingTask):
    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument("--command_in_dim", type=int)
        parser.add_argument("--command_out_dim", type=int)
        parser.add_argument("--truncated_length", type=int, default=8192)
        parser.add_argument("--feature_num", type=int, default=3)
        parser.add_argument("--control_mode", type=str)
        parser.add_argument("--command_path", type=str)
        parser.add_argument("--bucket_num", type=int)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)
        command_dataset = np.load(f"{self.args.command_path}/{split}_command.npy", mmap_mode="r")
        assert command_dataset.shape[0] == len(self.datasets[split]), f"error command sample num for {split}!"
        assert command_dataset.shape[1] == self.args.feature_num, "command feature_num isn't the same as args feature_num"
        logger.info(f'Load CommandSourceTargetDataset for {split} from {self.args.command_path}, truncated length: {self.args.truncated_length}')
        self.datasets[split] = CommandDataset(self.datasets[split], command_dataset, self.args)


    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        split = None
        if 'train' in self.datasets and dataset == self.datasets['train']:
            split = 'train'
        elif 'valid' in self.datasets and dataset == self.datasets['valid']:
            split = 'valid'
        elif 'test' in self.datasets and dataset == self.datasets['test']:
            split = 'test'

        max_positions_split = getattr(self.args, 'max_positions_%s' % split, None)
        if max_positions_split is None:
            max_positions_split = getattr(self.args, 'truncate_%s' % split, None)
        if max_positions_split is not None:
            max_positions = max_positions_split
        logger.info('Using max_positions limit (%d) for %s' % (max_positions,
                                                               split if split is not None else 'unknown'))

        return super().get_batch_iterator(
            dataset,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            data_buffer_size=data_buffer_size,
            disable_iterator_cache=disable_iterator_cache
        )
    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from .command_seq_generator import CommandSequenceGenerator

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                raise ImportError("SequenceGeneratorWithAlignment is not allowed!")
                # seq_gen_cls = SequenceGeneratorWithAlignment
            else:
                seq_gen_cls = CommandSequenceGenerator
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )










