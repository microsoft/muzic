#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import fileinput
import logging
import math
import os
import sys
import time
from collections import namedtuple

import numpy as np
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output
import linear_decoder.controlled_task
from MidiProcessor.midiprocessor import MidiDecoder
import shutil, random
from fairseq_cli import interactive

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

def attribute_generate(args):
    assert args.tgt_emotion in [1, 2, 3, 4], f"Error emotion index for {args.tgt_emotion}."
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

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

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
    command_list = np.load(args.ctrl_command_path)

    # for inputs in buffered_read(args.input, args.buffer_size):
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    midi_decoder = MidiDecoder("REMIGEN2")


    for command_index in [args.tgt_emotion - 1]:
        os.makedirs(save_root + f"/midi", exist_ok=True)
        os.makedirs(save_root + f"/remi", exist_ok=True)
        sample_scores = {}
        for gen_times in range(1000):
            if len(os.listdir(save_root + f"/midi")) > args.need_num:
                break
            if os.path.exists(save_root + f"/remi/{gen_times*args.batch_size}.txt"):
                print(f"command_id: {command_index} sample:{gen_times*args.batch_size} already exists. Skip this batch!")
                continue
            start_tokens = [""]
            for inputs in [start_tokens*args.batch_size]: # "" for none prefix input
                results = []
                for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                    bsz = batch.src_tokens.size(0)
                    src_tokens = batch.src_tokens
                    src_lengths = batch.src_lengths
                    constraints = batch.constraints
                    command_input = []
                    for i in range(args.batch_size):
                        command_input.append(command_list[command_index])
                    command_input = torch.tensor(command_input)
    
                    if use_cuda:
                        src_tokens = src_tokens.cuda()
                        src_lengths = src_lengths.cuda()
                        command_input = command_input.cuda()
                        if constraints is not None:
                            constraints = constraints.cuda()
    
                    sample = {
                        "net_input": {
                            "src_tokens": src_tokens,
                            "src_lengths": src_lengths,
                            "command_input":command_input,
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
                                },
                            )
                        )
    
                # sort output to match input order
                for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                        # print("S-{}\t{}".format(id_, src_str))
                        # print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
                        # for constraint in info["constraints"]:
                        #     print(
                        #         "C-{}\t{}".format(
                        #             id_, tgt_dict.string(constraint, args.remove_bpe)
                        #         )
                        #     )
    
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
                        detok_hypo_str = decode_fn(hypo_str)
                        score = hypo["score"] / math.log(2)  # convert to base 2
                        command_id = command_index
                        save_id = id_ + gen_times*args.batch_size
                        sample_scores[save_id] = score.detach().cpu().numpy().item()
                        print(f"command_id: {command_id} sample:{save_id} over with length {len(hypo_str.split(' '))}")
                        with open(save_root + f"/remi/{save_id}.txt", "w") as f:
                            f.write(hypo_str)
                        remi_token = hypo_str.split(" ")
                        # try:
                        midi_obj = midi_decoder.decode_from_token_str_list(remi_token)
                        midi_obj.dump(save_root + f"/midi/{save_id}.mid")
                        # except:
                        #     pass

def cli_main():
    parser = options.get_interactive_generation_parser()
    parser.add_argument("--ctrl_command_path", type=str)
    parser.add_argument("--save_root", type=str)
    parser.add_argument("--need_num", type=int, default=50)
    parser.add_argument("--tgt_emotion", type=int)
    args = options.parse_args_and_arch(parser)
    attribute_generate(args)

    # if args.ctrl_command_path != "None":
    #     attributes(args)
    # else:
    #     label_embedding(args)


if __name__ == "__main__":
    seed_everything(2023)
    cli_main()
'''
../StyleCtrlData/train_data/1026_EMO/data-bin
--task
language_modeling_control
--path
checkpoints/controlled/checkpoint_best.pt
--ctrl_command_path
../StyleCtrlData/train_data/1026_EMO/bucket2_command_thres_EMO/emotion_rank_100/EMOPIA_inference_nearest.npy
--save_root
generation
--max-len-b
500
--sampling
--beam
1
--sampling-topk
8
--buffer-size
8
--batch-size
8
'''
