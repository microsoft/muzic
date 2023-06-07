#!/usr/bin/env python


import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
from functools import partial

import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
    BertTokenizerFast,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    is_torch_tpu_available
)
from model import BertForAttributModel
from data_collator import default_data_collator
from transformers.trainer_utils import get_last_checkpoint
import math, torch, json
from copy import deepcopy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the test data."}
    )
    attributes: Optional[str] = field(
        default=None, metadata={"help": "A json file containing attributes."}
    )
    num_labels: Optional[str] = field(
        default=None, metadata={"help": "A json file containing the value number of each attribute."}
    )

    def __post_init__(self):
        if (self.train_file or self.validation_file or self.test_file) is None:
            raise ValueError("Need either a training/validation/test file.")
        elif self.train_file and self.validation_file:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension=="json", "`train_file` should be a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (json) as `train_file`."
        
        if self.test_file:
            test_extension = self.test_file.split(".")[-1]
            assert (
                test_extension=="json"
            ), "`test_file` should have the extension `json`."

        if self.attributes is None:
            raise ValueError("Need a attribute file.")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: 
    data_files = {}
    if training_args.do_train:
        if data_args.train_file and data_args.validation_file:
            data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        else:
            raise ValueError("Need a train file and a validation file for `do_train`.")

    if training_args.do_eval:
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file
        else:
            raise ValueError("Need a `validation_file` for `do_eval`.")
    if training_args.do_predict:
        if data_args.test_file is not None:
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == 'json'
            ), "`test_file` should have the extension `json`"
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need a test file for `do_predict`.")
    if not (training_args.do_train or training_args.do_eval or training_args.do_predict):
        raise ValueError("Need an action in [`do_train`, `do_eval`, `do_predict`].")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")


    # Loading a dataset from local json files
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )


    # Attribute values / labels
    attributes = json.load(open(data_args.attributes, 'r'))
    num_labels = OrderedDict()
    if training_args.do_train or training_args.do_eval:
        assert len(attributes) == len(raw_datasets[list(raw_datasets.keys())[0]][0]['labels']), "The attribute labels are not corresponding to the dataset."
        for idx in range(len(attributes)):
            num_labels[attributes[idx]] = len(raw_datasets[list(raw_datasets.keys())[0]][0]['labels'][idx])
        json.dump(num_labels, open("num_labels.json", "w"))
    if training_args.do_predict and not training_args.do_train and not training_args.do_eval:
        if data_args.num_labels:
            num_labels = json.load(open(data_args.num_labels))
        else:
            raise ValueError("Need `num_label`.")

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = BertConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = BertForAttributModel.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        num_labels = num_labels,
        tokenizer = tokenizer,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    

    # multi CLSes
    special_tokens = [f"[unused{i}]" for i in range(len(num_labels))]
    if training_args.do_train:
        tokenizer.add_special_tokens({"additional_special_tokens":special_tokens})
        model.resize_token_embeddings(len(tokenizer))

    model.tokenizer = tokenizer
    
    # Preprocessing the raw_datasets
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples, attributes):
        # Tokenize the texts
        result = tokenizer(examples['text'], 
                           padding=padding, 
                           max_length=max_seq_length, 
                           truncation=True)
        if 'labels' in examples:
            for idx in range(len(examples['labels'])):
                att_value = OrderedDict()
                for order, att in enumerate(attributes):
                    att_value[att] = examples['labels'][idx][order].index(1)
                examples['labels'][idx] = deepcopy(att_value)
            result['labels']= examples['labels']
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            partial(preprocess_function, attributes=attributes),
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions
        labels = p.label_ids
        acc = {}
        for k,v in preds.items():
            pred = np.argmax(preds[k], axis=1)
            acc[k] = accuracy_score(labels[k], pred)

        return {"acc": acc}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
                                
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(5, 0.01)]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.save_model()  # Saves the tokenizer too for easy upload

        try:
            perplexity = math.exp(metrics["train_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["train_perplexity"] = perplexity

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["eval_perplexity"] = perplexity
        
        print(f"***** Eval metrics *****")
        print(metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        result_output = {}
        softmaxprobs = {}
        for k,v in predictions.items():
            pred = np.argmax(predictions[k], axis=1)
            softmax = torch.nn.Softmax(dim=1)
            softmaxprobs[k] = softmax(torch.from_numpy(predictions[k])).tolist()
            result_output[k] = np.zeros(predictions[k].shape, dtype=np.int8)
            for p in range(len(pred)):
                result_output[k][p, pred[p]] = 1
            result_output[k] = result_output[k].tolist()
        json.dump(result_output, open(os.path.join(training_args.output_dir, f"predict_attributes.json"),'w'))
        json.dump(softmaxprobs, open(os.path.join(training_args.output_dir, f"softmax_probs.json"),'w'))

if __name__ == "__main__":
    main()
