from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
import numpy as np
from fairseq.data import data_utils
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from fairseq.tasks import register_task

import logging


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
import sys
sys.path.append("..")


from midi_data_extractor.attribute_unit import convert_value_dict_into_unit_dict
from midiprocessor import MidiEncoder
import msgpack

logger = logging.getLogger(__name__)


def mask_attributes(value_dict):
    random_pool = list(range(1, 5))*5 + list(range(6, 9))*3 + list(range(9, len(value_dict.keys())))*2
    chosen_num = np.random.choice(random_pool)
    chosen_attributes = np.random.choice(list(value_dict.keys()), chosen_num, replace=False)
    return chosen_attributes


def get_id(one_hot_vector):
    result = None
    for idx, item in enumerate(one_hot_vector):
        if item == 1:
            if result is not None:
                raise ValueError("This vector is not one-hot!")
            result = idx
    return result


def get_input_command_token_v3(dataset, command_dict, unit_dict):
    input_command_token = []
    if dataset.args.command_mask_prob <= -1:
        chosen_keys = mask_attributes(command_dict["values"])
        for key in dataset.key_order:
            cur_key_vector = unit_dict[key].get_vector(use=True)

            if key in chosen_keys:
                if isinstance(cur_key_vector[0], int):  # 普通分类属性
                    i = get_id(cur_key_vector)
                    input_command_token.append(f"{key}_{i}")
                elif isinstance(cur_key_vector[0], (list, tuple)):
                    true_pos = []
                    NA_pos = []
                    for i in range(len(cur_key_vector)):
                        if cur_key_vector[i][0] == 1:
                            true_pos.append(i)
                        else:
                            NA_pos.append(i)
                    # if len(true_pos) <= 0:
                    #     # for S4, no genre label
                    #     assert key == "S4", f"error true pos for {key} of {index}!"
                    chosen_num = np.random.randint(min(1, len(true_pos)),
                                                   len(true_pos) + 1)  # choose 1 pos at least
                    chosen_true_pos = np.random.choice(true_pos, chosen_num, replace=False)

                    chosen_num = np.random.randint(min(1, len(NA_pos)), len(NA_pos) + 1)
                    chosen_false_pos = np.random.choice(NA_pos, chosen_num, replace=False)
                    for m, fine_vec in enumerate(cur_key_vector):
                        if m in chosen_true_pos:
                            i = get_id(fine_vec) # yes
                        elif m in chosen_false_pos:
                            i = len(fine_vec) - 2 # yes, no, NA --- 0,1,2
                        else:
                            i = len(fine_vec) - 1  # This attribute is not chosen or is NA, set to NA token
                        input_command_token.append(f"{key}_{m}_{i}")
                else:
                    raise ValueError("cur_key_vector: %s   type: %s" % (str(cur_key_vector), type(cur_key_vector)))
            else:
                if isinstance(cur_key_vector[0], int):  # 普通分类属性
                    i = len(cur_key_vector) - 1  # the last one always corresponds to NA
                    input_command_token.append(f"{key}_{i}")
                elif isinstance(cur_key_vector[0], (list, tuple)):
                    for m, fine_vec in enumerate(cur_key_vector):
                        i = len(fine_vec) - 1  # the last one always corresponds to NA
                        input_command_token.append(f"{key}_{m}_{i}")
                else:
                    raise ValueError("cur_key_vector: %s   type: %s" % (str(cur_key_vector), type(cur_key_vector)))
    else:
        for key in dataset.key_order:
            cur_key_vector = unit_dict[key].get_vector(use=True)
            if isinstance(cur_key_vector[0], int):  # 普通分类属性
                if np.random.rand() < dataset.args.command_mask_prob:
                    i = len(cur_key_vector) - 1  # the last one always corresponds to NA
                else:
                    i = get_id(cur_key_vector)
                input_command_token.append(f"{key}_{i}")
            elif isinstance(cur_key_vector[0], (list, tuple)):
                for m, fine_vec in enumerate(cur_key_vector):
                    if np.random.rand() < dataset.args.command_mask_prob:
                        i = len(fine_vec) - 1  # the last one always corresponds to NA
                    else:
                        i = get_id(fine_vec)
                    input_command_token.append(f"{key}_{m}_{i}")
            else:
                raise ValueError("cur_key_vector: %s   type: %s" % (str(cur_key_vector), type(cur_key_vector)))
    return input_command_token

class CommandDataset(BaseWrapperDataset):
    def __init__(self, dataset, command_data, args = None):
        super().__init__(dataset)
        self._sizes = self.dataset.sizes.copy()
        self.command_data = command_data # need change to np.mmap
        self.args = args
        self.command_length_step = []

        # self.key_vec_length = [14, 28, 5, 3, 3, 6, 3, 5, 8, 3, 10, 12, 13, 15, 5]
        self.midi_encoder = MidiEncoder("REMIGEN")
        # self.default_command_dict = None
        self.pad_token_id = self.dataset.vocab.pad()
        self.sep_token_id = self.dataset.vocab.index("<sep>")
        if self.args.command_mask_prob <= -1:
            logger.info("Using step mask!")
        else:
            logger.info("Using prob mask!")

        self.key_order = ['I1s2', 'I4', 'C1', 'R1', 'R3', 'S2s1', 'S4', 'B1s1', 'TS1s1', 'K1', 'T1s1', 'P4', 'ST1',
                          'EM1', 'TM1']

        self.key_has_NA = []
        self.multi_hot_attributes = ["I1s2", "S4"]
        self.get_input_command_token = get_input_command_token_v3
        self.key_index = dict(zip(self.key_order, range(len(self.key_order))))

    def __getitem__(self, index):
        sample = self.dataset[index]
        command_dict = self.command_data[index]
        if len(sample["source"]) > self.args.truncated_length + 2: # for <s> and </s> token
            if not self.args.padding_to_max_length:
                raise ValueError(f"Sample length is greater than {self.args.truncated_length}!")
            else:
                sample["source"] = sample["source"][:self.args.truncated_length]
                sample["target"] = sample["target"][:self.args.truncated_length]

        unit_dict = convert_value_dict_into_unit_dict(command_dict["values"], self.midi_encoder)



        input_command_token = self.get_input_command_token(self, command_dict, unit_dict)
        input_command = []
        for word in input_command_token:
            input_command.append(self.dataset.tgt_vocab.index(word))
        input_command.append(self.sep_token_id)
        input_command = torch.tensor(input_command, dtype = torch.int64)
        # padding the target
        sample["source"] = torch.cat([sample["source"][0:1], input_command, sample["source"][1:]], dim = 0)
        pad_vector = torch.tensor(np.zeros(len(input_command)).astype(np.int64) + self.pad_token_id).to(sample["target"].device)
        sample["target"] = torch.cat([pad_vector, sample["target"]], dim = 0)

        if self.args.padding_to_max_length:
            if len(sample["source"]) < self.args.truncated_length:
                pad_vector = torch.tensor(np.zeros(self.args.truncated_length + 2 - len(sample["source"])).astype(np.int64) + self.pad_token_id).to(sample["target"].device)
                sample["source"] = torch.cat([sample["source"], pad_vector], dim=0)
                sample["target"] = torch.cat([sample["target"], pad_vector], dim=0)
        return {
            "id": index,
            "source": sample["source"],
            "target": sample["target"],
            "sep_pos": len(input_command)
        }



    def dynamic_mask(self, command_input):
        # random set N/A for command_input
        return command_input

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
        if samples[0]["sep_pos"] is not None:
            sep_pos = [samples[j]["sep_pos"] for j in range(len(samples))]
        else:
            sep_pos = None
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
                "sep_pos": sep_pos,
                "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
            },
            "target": target,
        }


@register_task("language_modeling_control", dataclass=LanguageModelingConfig)
class LanguageModelingTaskWithControl(LanguageModelingTask):
    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument("--truncated_length", type=int, default=5868)
        parser.add_argument("--padding_to_max_length", type=int, default=0)
        parser.add_argument("--command_path", type=str)
        parser.add_argument("--command_embed_dim", type = int)
        parser.add_argument("--command_mask_prob", type=float, default=0.4)
        parser.add_argument("--is_inference", type = bool, default = False)



    def load_dataset(self, split, epoch=1, combine=False, **kwargs):

        super().load_dataset(split, epoch=epoch, combine=combine, **kwargs)


        logger.info(f'Load CommandSourceTargetDataset for {split} from {self.args.command_path}, truncated length: {self.args.truncated_length}, mask_prob:{self.args.command_mask_prob}')
        command_dataset = np.load(f"{self.args.command_path}/{split}_command.npy", allow_pickle=True)
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

        from command_seq_generator import CommandSequenceGenerator

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










