#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the reader model on top of the retriever results
"""

import argparse
import collections
import glob
import json
import logging
import os
from collections import defaultdict
from typing import List

import numpy as np
import torch

from dpr.data.qa_validation import exact_match_score
from dpr.data.reader_data import (
    ReaderSample,
    ReaderPassage,
    get_best_spans,
    SpanPrediction,
    convert_retriever_results,
    preprocess_retriever_data,
)
from dpr.models import init_reader_components
from dpr.models.reader import create_reader_input, ReaderBatch, compute_loss
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    set_seed,
    add_training_params,
    add_reader_preprocessing_params,
    set_encoder_params_from_state,
    get_encoder_params_state,
    add_tokenizer_params,
    print_args,
)
from dpr.utils.data_utils import (
    ShardedDataIterator,
    read_serialized_data_from_files,
    Tensorizer,
)
from dpr.utils.model_utils import (
    get_schedule_linear,
    load_states_from_checkpoint,
    move_to_device,
    CheckpointState,
    get_model_file,
    setup_for_distributed_mode,
    get_model_obj,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

ReaderQuestionPredictions = collections.namedtuple(
    "ReaderQuestionPredictions", ["id", "predictions", "gold_answers"]
)


class ReaderTrainer(object):
    def __init__(self, args):
        self.args = args

        self.shard_id = 0
        self.distributed_factor = 1

        logger.info("***** Initializing components for prediction *****")

        model_file = get_model_file(self.args, self.args.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, reader, optimizer = init_reader_components(
            args.encoder_model_type, args
        )

        reader, optimizer = setup_for_distributed_mode(
            reader,
            optimizer,
            args.device,
            args.n_gpu,
            args.local_rank,
            args.fp16,
            args.fp16_opt_level,
        )
        self.reader = reader
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        if saved_state:
            self._load_saved_state(saved_state)

    def get_data_iterator(
        self,
        path: str,
        batch_size: int,
        is_train: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
    ) -> ShardedDataIterator:
        data_files = glob.glob(path)
        logger.info("Data files: %s", data_files)
        if not data_files:
            raise RuntimeError("No Data files found")
        preprocessed_data_files = self._get_preprocessed_files(data_files, is_train)
        data = read_serialized_data_from_files(preprocessed_data_files)

        iterator = ShardedDataIterator(
            data,
            shard_id=self.shard_id,
            num_shards=self.distributed_factor,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            offset=offset,
        )

        # apply deserialization hook
        iterator.apply(lambda sample: sample.on_deserialize())
        return iterator

    def validate(self, question, passages):
        args = self.args
        self.reader.eval()

        # XXX top_docs
        samples = [{
          "question": question, 
          "answers": [],
          "ctxs": [ {"id": id, "text": item["text"], "title": item["title"]} for id, item 
              in enumerate(passages) ]
        }]

        # Disabling padding is crucial, since otherwise question and title eats the whole sequence
        self.tensorizer.set_pad_to_max(False)
        iterator = preprocess_retriever_data(samples, None, tensorizer=self.tensorizer, is_train_set=False)
        samples = list(iterator)
        self.tensorizer.set_pad_to_max(True)

        input = create_reader_input(
            self.tensorizer.get_pad_id(),
            samples,
            # XXX passages per question
            args.passages_per_question_predict,
            args.sequence_length,
            args.max_n_answers,
            is_train=False,
            shuffle=False,
        )

        input = ReaderBatch(**move_to_device(input._asdict(), args.device))
        attn_mask = self.tensorizer.get_attn_mask(input.input_ids)

        with torch.no_grad():
            start_logits, end_logits, relevance_logits = self.reader(
                input.input_ids, attn_mask
            )

        all_results = self._get_best_prediction(
            start_logits,
            end_logits,
            relevance_logits,
            samples,
        )


        print(all_results[0].id)
        best = list(all_results[0].predictions.values())[0]
        print(best.prediction_text)
        print("Span %.2f" % best.span_score)
        print("Relevance %.2f" % best.relevance_score)
        return 0

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)
        self.start_epoch = epoch
        self.start_batch = offset

        model_to_load = get_model_obj(self.reader)
        if saved_state.model_dict:
            logger.info("Loading model weights from saved state ...")
            model_to_load.load_state_dict(saved_state.model_dict)

        logger.info("Loading saved optimizer state ...")
        if saved_state.optimizer_dict:
            self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler_state = saved_state.scheduler_dict

    def _get_best_prediction(
        self,
        start_logits,
        end_logits,
        relevance_logits,
        samples_batch: List[ReaderSample],
    ) -> List[ReaderQuestionPredictions]:

        args = self.args
        max_answer_length = args.max_answer_length
        questions_num, passages_per_question = relevance_logits.size()

        _, idxs = torch.sort(
            relevance_logits,
            dim=1,
            descending=True,
        )

        batch_results = []
        for q in range(questions_num):
            sample = samples_batch[q]

            non_empty_passages_num = len(sample.passages)
            nbest = []
            for p in range(passages_per_question):
                passage_idx = idxs[q, p].item()
                if (
                    passage_idx >= non_empty_passages_num
                ):  # empty passage selected, skip
                    continue
                reader_passage = sample.passages[passage_idx]
                sequence_ids = reader_passage.sequence_ids
                sequence_len = sequence_ids.size(0)
                # assuming question & title information is at the beginning of the sequence
                passage_offset = reader_passage.passage_offset

                p_start_logits = start_logits[q, passage_idx].tolist()[
                    passage_offset:sequence_len
                ]
                p_end_logits = end_logits[q, passage_idx].tolist()[
                    passage_offset:sequence_len
                ]

                ctx_ids = sequence_ids.tolist()[passage_offset:]
                best_spans = get_best_spans(
                    self.tensorizer,
                    p_start_logits,
                    p_end_logits,
                    ctx_ids,
                    max_answer_length,
                    passage_idx,
                    relevance_logits[q, passage_idx].item(),
                    top_spans=10,
                )
                nbest.extend(best_spans)
                if len(nbest) > 0:
                    break

            if len(nbest) == 0:
                predictions = {
                    passages_per_question: SpanPrediction("", -1, -1, -1, "")
                }
            else:
                predictions = {passages_per_question: nbest[0]}
            batch_results.append(
                ReaderQuestionPredictions(sample.question, predictions, sample.answers)
            )
        return batch_results

def main():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    add_reader_preprocessing_params(parser)

    # reader specific params
    parser.add_argument(
        "--max_n_answers",
        default=10,
        type=int,
        help="Max amount of answer spans to marginalize per single passage",
    )
    parser.add_argument(
        "--passages_per_question",
        type=int,
        default=2,
        help="Total amount of positive and negative passages per question",
    )
    parser.add_argument(
        "--passages_per_question_predict",
        type=int,
        default=50,
        help="Total amount of positive and negative passages per question for evaluation",
    )
    parser.add_argument(
        "--max_answer_length",
        default=10,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument("--checkpoint_file_name", type=str, default="dpr_reader")
    parser.add_argument(
        "--prediction_results_file",
        type=str,
        help="path to a file to write prediction results to",
    )

    # training parameters
    parser.add_argument(
        "--eval_step",
        default=2000,
        type=int,
        help="batch steps to run validation and save checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints will be written to",
    )

    parser.add_argument(
        "--fully_resumable",
        action="store_true",
        help="Enables resumable mode by specifying global step dependent random seed before shuffling "
        "in-batch data",
    )

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    setup_args_gpu(args)
    set_seed(args)
    print_args(args)

    trainer = ReaderTrainer(args)
    question = "Kto był pierwszym królem Polski?"
    passages = [
        {
          "text": "Mieszko Pierwszy był pierwszym władcą Polski.", 
          "title": "Mieszko I"
        }, {
          "text": "Bolesław Chrobry był pierwszym królem Polski.", 
          "title": "Bolesław Chrobry"
        }
    ]
    trainer.validate(question, passages)


if __name__ == "__main__":
    main()
