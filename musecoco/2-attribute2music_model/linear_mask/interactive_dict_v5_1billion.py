#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import fileinput
import json, pickle
import logging
import math
import os
import sys
sys.path.append("..")
import time
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
from midiprocessor import MidiDecoder, MidiEncoder
from midi_data_extractor.attribute_unit import convert_value_dict_into_unit_dict
import shutil, random
from fairseq_cli import interactive
from data_process.util import key_order, key_has_NA
import A2M_task_new
from linear import transformer_lm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")

Batch = namedtuple("Batch", "ids src_tokens src_lengths constraints")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_remi_token(remi_tokens):
    if remi_tokens[0] in ["Q1", "Q2", "Q3", "Q4", "None"]:
        remi_tokens = remi_tokens[1:]
    if remi_tokens[-1][0] == "o":
        remi_tokens[-1] = "b-1"
    elif remi_tokens[-1][0] == "p" or remi_tokens[-1][0] == "d":
        i = len(remi_tokens) - 1
        while remi_tokens[i][0] != "o":
            i -= 1
        remi_tokens = remi_tokens[:i]
        remi_tokens.append("b-1")
    elif remi_tokens[-1][0] == "v":
        remi_tokens.append("b-1")
    elif remi_tokens[-1][0] == "s-9":
        remi_tokens = remi_tokens[:-1]
    return remi_tokens


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if args.constraints:
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]

    if args.constraints:
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None

    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints=constraints_tensor
        ),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            constraints=constraints,
        )
from A2M_task_new import get_id
def convert_vector_to_token(input):
    key_order = ['I1s2', 'I4', 'C1', 'R1', 'R3', 'S2s1', 'S4', 'B1s1', 'TS1s1', 'K1', 'T1s1', 'P4', 'ST1',
                              'EM1', 'TM1']
    multi_hot_attributes = ["I1s2", "S4"]
    attribute_tokens = []
    for key in key_order:
        if key not in input.keys():
            if key == "I4":
                attribute_tokens.append(f"{key}_28") # for I4 and ST1, set to NA
            elif key == "ST1":
                attribute_tokens.append(f"{key}_14")  # for I4 and ST1, set to NA
            elif key == "C1":
                attribute_tokens.append(f"{key}_4")
            else:
                raise ValueError(f"error key={key} that does not exist.")
        else:
            attri_vector = input[key]
            if key in multi_hot_attributes:
                for i in range(len(attri_vector)):
                    attribute_tokens.append(f"{key}_{i}_{get_id(attri_vector[i])}")
            else:
                attribute_tokens.append(f"{key}_{get_id(attri_vector)}")
    return attribute_tokens
def attributes(args):
    start_time = time.time()
    total_translate_time = 0

    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.batch_size is None:
        args.batch_size = 1

    assert (
            not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
            not args.batch_size or args.batch_size <= args.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation_control
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    for model in models:
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)
        model.decoder.args.is_inference = True

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Initialize generator
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if args.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if args.buffer_size > 1:
        logger.info("Sentence buffer size: %s", args.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0

    # for inputs in buffered_read(args.input, args.buffer_size):
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    midi_decoder = MidiDecoder("REMIGEN2")

    # test_command = np.load("../Text2Music_data/v2.1_20230218/full_0218_filter_by_5866/infer_command_balanced.npy",
    #                        allow_pickle=True).item()
    # test_command = np.load(args.ctrl_command_path, allow_pickle=True).item()

    # test_command = json.load(open(args.ctrl_command_path, "r"))
    if args.use_gold_labels:
        with open(args.save_root + "/Using_gold_labels!.txt", "w") as check_input:
            pass
    else:
        with open(args.save_root + "/Using_pred_labels!.txt", "w") as check_input:
            pass
    test_command = pickle.load(open(args.ctrl_command_path, "rb"))
    if args.start is None:
        args.start = 0
        args.end = len(test_command)
    else:
        args.start = min(max(args.start, 0), len(test_command))
        args.end = min(max(args.end, 0), len(test_command))

    gen_command_list = []
    for j in range(args.need_num):
        for i in range(args.start, args.end):
            if args.use_gold_labels:
                pred_labels = test_command[i]["gold_labels"]
            else:
                pred_labels = test_command[i]["pred_labels"]
            attribute_tokens = convert_vector_to_token(pred_labels)
            # for key in key_order:
            #     if key not in pred_labels.keys():
            #         continue
            #     if key in key_has_NA and pred_labels[key][-1] == 1:
            #         continue
            #     for j in range(len(pred_labels[key])):
            #         if pred_labels[key][j] == 1:
            #             attribute_tokens.append(f"{key}_{j}")
            test_command[i]["infer_command_tokens"] = attribute_tokens
            gen_command_list.append([test_command[i]["infer_command_tokens"], f"{i}", j, test_command[i]])

    steps = len(gen_command_list) // args.batch_size
    print(f"Starts to generate {args.start} to {args.end} of {len(gen_command_list)} samples in {steps + 1} batch steps!")


    for batch_step in range(steps + 1):
        infer_list = gen_command_list[batch_step*args.batch_size:(batch_step+1)*args.batch_size]
        infer_command_token = [g[0] for g in infer_list]
        # assert infer_command.shape[1] == 133, f"error feature dim for {gen_key}!"
        if len(infer_list) == 0:
            continue
        # with open(save_root + f"/{command_index}/text_description.txt", "w") as text_output:
        #     text_output.write(text_description[command_index])

        if os.path.exists(save_root + f"/{infer_list[-1][1]}/remi/{infer_list[-1][2]}.txt"):
            print(f"Skip the {batch_step}-th batch since has been generated!")
            continue

        # start_tokens = [f""]
        start_tokens = []
        sep_pos = []
        for attribute_prefix in infer_command_token:
            start_tokens.append(" ".join(attribute_prefix) + " <sep>")
            sep_pos.append(len(attribute_prefix)) # notice that <sep> pos is len(attribute_prefix) in this sequence
        sep_pos = np.array(sep_pos)
        for inputs in [start_tokens]:  # "" for none prefix input
            results = []
            for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                bsz = batch.src_tokens.size(0)
                src_tokens = batch.src_tokens
                src_lengths = batch.src_lengths
                constraints = batch.constraints

                if use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()
                    if constraints is not None:
                        constraints = constraints.cuda()

                sample = {
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": src_lengths,
                        "sep_pos": sep_pos,
                    },
                }
                translate_start_time = time.time()
                translations = task.inference_step(
                    generator, models, sample, constraints=constraints
                )
                translate_time = time.time() - translate_start_time
                total_translate_time += translate_time
                list_constraints = [[] for _ in range(bsz)]
                if args.constraints:
                    list_constraints = [unpack_constraints(c) for c in constraints]

                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                    constraints = list_constraints[i]
                    results.append(
                        (
                            start_id + id,
                            src_tokens_i,
                            hypos,
                            {
                                "constraints": constraints,
                                "time": translate_time / len(translations),
                                "translation_shape":len(translations),
                            },
                        )
                    )

            # sort output to match input order
            for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                # Process top predictions
                for hypo in hypos[: min(len(hypos), args.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo["tokens"].int().cpu(),
                        src_str=src_str,
                        alignment=hypo["alignment"],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                    )

                    os.makedirs(save_root + f"/{infer_list[id_][1]}", exist_ok=True)
                    if not os.path.exists(save_root + f"/{infer_list[id_][1]}/infer_command.json"):
                        with open(save_root + f"/{infer_list[id_][1]}/infer_command.json", "w") as f:
                            json.dump(infer_list[id_][-1], f)
                    save_id = infer_list[id_][2]

                    os.makedirs(save_root + f"/{infer_list[id_][1]}/remi", exist_ok=True)
                    with open(save_root + f"/{infer_list[id_][1]}/remi/{save_id}.txt", "w") as f:
                        f.write(hypo_str)
                    remi_token = hypo_str.split(" ")[sep_pos[id_] + 1:]
                    print(f"batch:{batch_step} save_id:{save_id} over with length {len(hypo_str.split(' '))}; "
                          f"Average translation time:{info['time']} seconds; Remi seq length: {len(remi_token)}; Batch size:{args.batch_size}; \
                          Translation shape:{info['translation_shape']}.")
                    os.makedirs(save_root + f"/{infer_list[id_][1]}/midi", exist_ok=True)
                    try:
                        midi_obj = midi_decoder.decode_from_token_str_list(remi_token)
                        midi_obj.dump(save_root + f"/{infer_list[id_][1]}/midi/{save_id}.mid")
                    except:
                        pass


def cli_main():
    parser = options.get_interactive_generation_parser()
    parser.add_argument("--save_root", type=str)
    parser.add_argument("--need_num", type=int, default=32)
    parser.add_argument("--ctrl_command_path", type=str, default="")
    parser.add_argument("--start", type = int, default=None)
    parser.add_argument("--end", type = int, default=None)
    parser.add_argument("--use_gold_labels", type = int, default=0)
    args = options.parse_args_and_arch(parser)
    attributes(args)

    # if args.ctrl_command_path != "None":
    #     attributes(args)
    # else:
    #     label_embedding(args)


if __name__ == "__main__":
    seed_everything(2024) # 2023
    cli_main()

