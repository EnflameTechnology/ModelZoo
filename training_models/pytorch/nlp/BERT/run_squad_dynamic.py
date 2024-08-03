# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function
import os
os.environ["ENFLAME_PT_EVALUATE_TENSOR_NEEDED"]="true"
import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
import json
import tops_models.common_utils as common_utils
from tops_models.logger import tops_logger
from tops_models.timers import Timers
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
try:
    import apex
    from apex import amp
    APEX_IS_AVAILABLE = True
except ImportError:
    APEX_IS_AVAILABLE = False

from schedulers import LinearWarmUpScheduler
from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import modeling
from optimization import BertAdam, warmup_linear
from tokenization import (BasicTokenizer, BertTokenizer, whitespace_tokenize)
from utils import is_main_process
import time
import datetime

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = tops_logger()
class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            '''
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            '''
            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def get_answers(examples, features, results, args):
    predictions = collections.defaultdict(list) #it is possible that one example corresponds to multiple features
    Prediction = collections.namedtuple('Prediction', ['text', 'start_logit', 'end_logit'])

    if args.version_2_with_negative:
        null_vals = collections.defaultdict(lambda: (float("inf"),0,0))
    for ex, feat, result in match_results(examples, features, results):
        start_indices = _get_best_indices(result.start_logits, args.n_best_size)
        end_indices = _get_best_indices(result.end_logits, args.n_best_size)
        prelim_predictions = get_valid_prelim_predictions(start_indices, end_indices, feat, result, args)
        prelim_predictions = sorted(
                            prelim_predictions,
                            key=lambda x: (x.start_logit + x.end_logit),
                            reverse=True)
        if args.version_2_with_negative:
            score = result.start_logits[0] + result.end_logits[0]
            if score < null_vals[ex.qas_id][0]:
                null_vals[ex.qas_id] = (score, result.start_logits[0], result.end_logits[0])

        curr_predictions = []
        seen_predictions = []
        for pred in prelim_predictions:
            if len(curr_predictions) == args.n_best_size:
                break
            if pred.start_index > 0:  # this is a non-null prediction TODO: this probably is irrelevant
                final_text = get_answer_text(ex, feat, pred, args)
                if final_text in seen_predictions:
                    continue
            else:
                final_text = ""

            seen_predictions.append(final_text)
            curr_predictions.append(Prediction(final_text, pred.start_logit, pred.end_logit))
        predictions[ex.qas_id] += curr_predictions

    #Add empty prediction
    if args.version_2_with_negative:
        for qas_id in predictions.keys():
            predictions[qas_id].append(Prediction('',
                                                  null_vals[ex.qas_id][1],
                                                  null_vals[ex.qas_id][2]))


    nbest_answers = collections.defaultdict(list)
    answers = {}
    for qas_id, preds in predictions.items():
        nbest = sorted(
                preds,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)[:args.n_best_size]

        # In very rare edge cases we could only have single null prediction.
        # So we just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(Prediction(text="empty", start_logit=0.0, end_logit=0.0))

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry and entry.text:
                best_non_null_entry = entry
        probs = _compute_softmax(total_scores)
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_answers[qas_id].append(output)
        if args.version_2_with_negative:
            score_diff = null_vals[qas_id][0] - best_non_null_entry.start_logit - best_non_null_entry.end_logit
            if score_diff > args.null_score_diff_threshold:
                answers[qas_id] = ""
            else:
                answers[qas_id] = best_non_null_entry.text
        else:
            answers[qas_id] = nbest_answers[qas_id][0]['text']

    return answers, nbest_answers

def get_answer_text(example, feature, pred, args):
    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
    orig_doc_start = feature.token_to_orig_map[pred.start_index]
    orig_doc_end = feature.token_to_orig_map[pred.end_index]
    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)

    final_text = get_final_text(tok_text, orig_text, args.do_lower_case, args.verbose_logging)
    return final_text

def get_valid_prelim_predictions(start_indices, end_indices, feature, result, args):

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit"])
    prelim_predictions = []
    for start_index in start_indices:
        for end_index in end_indices:
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index not in feature.token_to_orig_map:
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if not feature.token_is_max_context.get(start_index, False):
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > args.max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=result.start_logits[start_index],
                    end_logit=result.end_logits[end_index]))
    return prelim_predictions

def match_results(examples, features, results):
    unique_f_ids = set([f.unique_id for f in features])
    unique_r_ids = set([r.unique_id for r in results])
    matching_ids = unique_f_ids & unique_r_ids
    features = [f for f in features if f.unique_id in matching_ids]
    results = [r for r in results if r.unique_id in matching_ids]
    features.sort(key=lambda x: x.unique_id)
    results.sort(key=lambda x: x.unique_id)

    for f, r in zip(features, results): #original code assumes strict ordering of examples. TODO: rewrite this
        yield examples[f.example_index], f, r

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indices(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indices = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indices.append(index_and_score[i][0])
    return best_indices


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        required=True,
                        help="The checkpoint file from pretraining")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus or gcus")
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--vocab_file',
                        type=str, default=None, required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument('--json_summary', type=str, default="logger.json",
                        help='If provided, the json summary will be written to'
                             'the specified file.')
    parser.add_argument("--eval_script",
                        help="Script to evaluate squad predictions",
                        default="evaluate.py",
                        type=str)
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to use evaluate accuracy of predictions")
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--disable-progress-bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument("--skip_cache",
                        default=False,
                        action='store_true',
                        help="Whether to cache train features")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Location to cache train feaures. Will default to the dataset directory")
    parser.add_argument("--device", default='gpu', type=str, help="Device to run ,gpu or gcu.")
    parser.add_argument("--use_apex",
                        default=False,
                        action='store_true',
                        help="Whether to use apex")
    parser.add_argument('--skip_steps', default=0, type=int,help='steps to skip while computing fps')
    parser.add_argument('--print_freq', default=10, type=int,help='print frequency')
    args = parser.parse_args()
    return args
def chunk_list(datas, chunksize):
    """Split list into the chucks

    Params:
        datas     (list): data that want to split into the chunk
        chunksize (int) : how much maximum data in each chunks

    Returns:
        chunks (obj): the chunk of list
    """

    for i in range(0, len(datas), chunksize):
        yield datas[i:i + chunksize]

def main():
    args = parse_args()
    timers = Timers()
    timers('interval time').init(args.train_batch_size,args.skip_steps, args.print_freq)
    args.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1
    args.distributed = args.local_rank != -1
    tb_writer = SummaryWriter('runs/tensorboard/dtu_{}'.format(args.local_rank if args.local_rank >=0 else 0))
    info_dict = common_utils.tops_models_collection(args.device)
    if  args.device.lower() == "gpu":
      import torch.distributed as dist
      from torch.utils.data.distributed import DistributedSampler
      if args.local_rank == -1 or args.no_cuda:
          device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
          n_device = torch.cuda.device_count()
      else:
          torch.cuda.set_device(args.local_rank)
          device = torch.device("cuda", args.local_rank)
          # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
          dist.init_process_group(backend='nccl', init_method='env://')
          n_device = 1
    else:
      import torch_gcu

      import torch_gcu.distributed as dist
      from torch_gcu.utils.data.distributed import DistributedSampler
      device = torch_gcu.gcu_device(args.local_rank if args.local_rank >=0 else 0)
      if args.distributed:
        dist.init_process_group(backend='eccl', init_method='env://')
      n_device = 1

    print("device: {} n_device: {}, distributed training: {}, amp training: {}".format(
                                device, n_device, args.distributed, args.amp))
    logger.info("Config: {}".format([str(args)]))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.max_steps > 0  and args.skip_steps > args.max_steps:
        raise ValueError("max_steps should be large than skip_steps")
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger.info("SEED: {}".format(args.seed))

    if args.device.lower() == "gpu" and n_device > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and os.listdir(args.output_dir)!=['logfile.txt']:
        print("WARNING: Output directory {} already exists and is not empty.".format(args.output_dir), os.listdir(args.output_dir))
    if not os.path.exists(args.output_dir) and is_main_process(dist):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer(args.vocab_file, do_lower_case=args.do_lower_case, max_len=512) # for bert large
    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.do_train:
        train_examples = read_squad_examples(
            input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.distributed:
            num_train_optimization_steps = num_train_optimization_steps // dist.get_world_size()

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForQuestionAnswering(config)
    #model = modeling.BertForQuestionAnswering.from_pretrained(args.bert_model,
    #             cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))
    logger.info("loading_checkpoint: {}".format(True))
    model.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu')["model"], strict=False)
    logger.info("loaded_checkpoint: {}".format(True))
    model.to(device)
    num_weights = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info("model_weights_num: {}".format(num_weights))

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.do_train:
        if args.fp16 and APEX_IS_AVAILABLE and args.use_apex:
            try:
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False)

            if args.loss_scale == 0:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False,
                                                      loss_scale="dynamic")
            else:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=False, loss_scale=args.loss_scale)
            if args.do_train:
                scheduler = LinearWarmUpScheduler(optimizer, warmup=args.warmup_proportion, total_steps=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                    lr=args.learning_rate,
                                    warmup=args.warmup_proportion,
                                    t_total=num_train_optimization_steps)
    if args.distributed and APEX_IS_AVAILABLE and args.use_apex and args.device.lower() == 'gpu':
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.device.lower() == 'gcu' and args.distributed:
        from torch_gcu.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters = True)
    elif n_device > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    if args.do_train:
        if args.cache_dir is None:
            cached_train_features_file = args.train_file + '_{0}_{1}_{2}_{3}'.format(
                list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
                str(args.max_query_length))
        else:
            cached_train_features_file = args.cache_dir.strip('/') + '/' + args.train_file.split('/')[-1] + '_{0}_{1}_{2}_{3}'.format(
                list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
                str(args.max_query_length))

        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)

            if not args.skip_cache and is_main_process(dist):
                logger.info("Cached_train features_file: {}".format(cached_train_features_file))
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        logger.info("train_start: {}".format(True))
        logger.info("training_samples: {}".format(len(train_examples)))
        logger.info("training_features: {}".format(len(train_features)))
        logger.info("train_batch_size: {}".format(args.train_batch_size))
        logger.info("steps: {}".format(num_train_optimization_steps))
        '''
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data, drop_last=True)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size * n_device, num_workers = args.workers)
        '''
        model.train()
        if APEX_IS_AVAILABLE and args.use_apex:
            from apex.multi_tensor_apply import multi_tensor_applier
            class GradientClipper:
                """
                Clips gradient norm of an iterable of parameters.
                """
                def __init__(self, max_grad_norm):
                    self.max_norm = max_grad_norm
                    if multi_tensor_applier.available:
                        import amp_C
                        self._overflow_buf = torch.cuda.IntTensor([0])
                        self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
                        self.multi_tensor_scale = amp_C.multi_tensor_scale
                    else:
                        raise RuntimeError('Gradient clipping requires cuda extensions')

                def step(self, parameters):
                    l = [p.grad for p in parameters if p.grad is not None]
                    total_norm, _ = multi_tensor_applier(self.multi_tensor_l2norm, self._overflow_buf, [l], False)
                    total_norm = total_norm.item()
                    if (total_norm == float('inf')): return
                    clip_coef = self.max_norm / (total_norm + 1e-6)
                    if clip_coef < 1:
                        multi_tensor_applier(self.multi_tensor_scale, self._overflow_buf, [l, l], clip_coef)
            gradClipper = GradientClipper(max_grad_norm=1.0)
        final_loss = None
        if args.amp:
            if args.device.lower() == "gpu":
                scaler = torch.cuda.amp.GradScaler()
                autocast = torch.cuda.amp.autocast
            else:
                scaler = torch_gcu.amp.GradScaler(use_zero_grad=True)
                autocast = torch_gcu.amp.autocast
        print("Start training")
        start_time = time.time()
        for epoch in range(int(args.num_train_epochs)):
            batchs = chunk_list(train_features,args.train_batch_size)
            for step, ebatch in enumerate(batchs):
                aa=[]
                aa=[e for e in ebatch]
                bb=[len(e.input_ids) for e in aa]
                max_len = max(bb)
                print('maxbatchlen_train',max_len)
                for e in aa:
                    while len(e.input_ids) < max_len:
                        e.input_ids.append(0)
                        e.input_mask.append(0)
                        e.segment_ids.append(0)
                input_ids = torch.tensor([f.input_ids for f in aa], dtype=torch.long).to(device)
                input_mask = torch.tensor([f.input_mask for f in aa], dtype=torch.long).to(device)
                segment_ids = torch.tensor([f.segment_ids for f in aa], dtype=torch.long).to(device)
                start_positions = torch.tensor([f.start_position for f in aa], dtype=torch.long).to(device)
                end_positions = torch.tensor([f.end_position for f in aa], dtype=torch.long).to(device)
                # Terminate early for benchmarking
                # Terminate early for benchmarking
                if args.max_steps > 0 and global_step > args.max_steps:
                    break
                if step == 0:
                    timers('interval time').start()
                optimizer.zero_grad(True)
                if args.amp:
                    with autocast():
                        #if n_device == 1:
                        #    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                        #input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                        start_logits, end_logits = model(input_ids, segment_ids, input_mask)
                        # If we are on multi-GPU, split add a dimension
                        if len(start_positions.size()) > 1:
                            start_positions = start_positions.squeeze(-1)
                        if len(end_positions.size()) > 1:
                            end_positions = end_positions.squeeze(-1)
                        # sometimes the start/end positions are outside our model inputs, we ignore these terms
                        ignored_index = start_logits.size(1)
                        start_positions.clamp_(0, ignored_index)
                        end_positions.clamp_(0, ignored_index)

                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                        start_loss = loss_fct(start_logits, start_positions)
                        end_loss = loss_fct(end_logits, end_positions)
                        loss = (start_loss + end_loss) / 2
                        if n_device > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm(model.parameters(),1.0)

                else:
                    #if n_device == 1:
                    #    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                    #input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    start_logits, end_logits = model(input_ids, segment_ids, input_mask)
                    # If we are on multi-GPU, split add a dimension
                    if len(start_positions.size()) > 1:
                        start_positions = start_positions.squeeze(-1)
                    if len(end_positions.size()) > 1:
                        end_positions = end_positions.squeeze(-1)
                    # sometimes the start/end positions are outside our model inputs, we ignore these terms
                    ignored_index = start_logits.size(1)
                    start_positions.clamp_(0, ignored_index)
                    end_positions.clamp_(0, ignored_index)

                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                    start_loss = loss_fct(start_logits, start_positions)
                    end_loss = loss_fct(end_logits, end_positions)
                    loss = (start_loss + end_loss) / 2
                    if n_device > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    loss.backward()

                # gradient clipping
                if APEX_IS_AVAILABLE and args.use_apex:
                    gradClipper.step(amp.master_params(optimizer))

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if not APEX_IS_AVAILABLE:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if args.device.lower() == "gpu":
                        if args.amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    else:
                        if args.amp:
                            scaler.step(optimizer)
                            scaler.update()
                            torch_gcu.unlazy(torch_gcu.collect_amp_training_params(model, optimizer, scaler)+[loss,start_positions,end_positions])
                        else:
                            torch_gcu.optimizer_step(optimizer, [loss, start_positions, end_positions],
                                                     mode=torch_gcu.JitRunMode.ASYNC if args.distributed else torch_gcu.JitRunMode.SAFEASYNC,
                                                     model=model)
                    global_step += 1
                if (step+1) % args.print_freq == 0:
                    final_loss = loss.item()
                    timers('interval time').elapsed()
                    train_fps_step = timers('interval time').step_fps_list[-1]
                    logger.info("step_loss: {},learning_rate: {},step_fps: {}".format(final_loss, optimizer.param_groups[0]['lr'],train_fps_step))
                    tb_writer.add_scalar('loss', final_loss, global_step)
                    tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        train_info = OrderedDict(
          [('model', 'bert'),
          ('train_batch_size', args.train_batch_size),
          ('epochs', args.num_train_epochs),
          ('training_step_per_epoch', len(train_features)),
          ('device', args.device),
          ('skip_steps', args.skip_steps),
          ('train_fps_mean', timers('interval time').calc_fps()),
          ('train_fps_min', timers('interval time').min_fps()),
          ('train_fps_max', timers('interval time').max_fps()),
          ('training_time', total_time_str)
          ])
        info_dict.update(train_info)
    if args.do_train and is_main_process(dist) and not args.skip_checkpoint:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, modeling.WEIGHTS_NAME)
        torch.save({"model":model_to_save.state_dict()}, output_model_file)
        output_config_file = os.path.join(args.output_dir, modeling.CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    if args.do_predict and (args.local_rank == -1 or is_main_process(dist)):

        if not args.do_train and args.fp16:
            model.half()

        eval_examples = read_squad_examples(
            input_file=args.predict_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)

        logger.info("infer_start: {}".format(True))
        logger.info("eval_samples: {}".format(len(eval_examples)))
        logger.info("eval_features: {}".format(len(eval_features)))
        logger.info("predict_batch_size: {}".format(args.predict_batch_size))

        infer_start = time.time()
        model.eval()
        all_results = []
        logger.info("eval_start: {}".format(True))
        batchs = chunk_list(eval_features,args.predict_batch_size)
        for step, ebatch in enumerate(batchs):
            aa=[]
            aa=[e for e in ebatch]
            bb=[len(e.input_ids) for e in aa]
            max_len = max(bb)
            print('maxbatchlen_eval',max_len)
            for e in aa:
                while len(e.input_ids) < max_len:
                    e.input_ids.append(0)
                    e.input_mask.append(0)
                    e.segment_ids.append(0)
            input_ids = torch.tensor([f.input_ids for f in aa], dtype=torch.long).to(device)
            input_mask = torch.tensor([f.input_mask for f in aa], dtype=torch.long).to(device)
            segment_ids = torch.tensor([f.segment_ids for f in aa], dtype=torch.long).to(device)
            example_indices = torch.arange(start=step*args.predict_batch_size,end=step*args.predict_batch_size+input_ids.size(0),dtype=torch.long)
            #print('eval_hudson',input_ids.size(0))
            #if len(all_results) % 1000 == 0:
            #    logger.info("sample_number: {}".format(len(all_results)))
            with torch.no_grad():
                batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
            if(args.device.lower() != "gpu"):
                torch_gcu.unlazy([batch_start_logits, batch_end_logits])
                batch_start_logits_cpu = batch_start_logits.detach().cpu().tolist()
                batch_end_logits_cpu = batch_end_logits.detach().cpu().tolist()
            for i, example_index in enumerate(example_indices):
                if(args.device.lower() != "gpu"):
                    start_logits = batch_start_logits_cpu[i]
                    end_logits = batch_end_logits_cpu[i]
                else:
                    start_logits = batch_start_logits[i].detach().cpu().tolist()
                    end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))
        predict_info = collections.OrderedDict([('predict_num_examples', len(eval_features)),
                                            ('predict_batch_size', args.predict_batch_size)])
        info_dict.update(predict_info)
        time_to_infer = time.time() - infer_start
        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")

        answers, nbest_answers = get_answers(eval_examples, eval_features, all_results, args)
        with open(output_prediction_file, "w") as f:
            f.write(json.dumps(answers, indent=4) + "\n")
        with open(output_nbest_file, "w") as f:
            f.write(json.dumps(nbest_answers, indent=4) + "\n")

        # output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
        # write_predictions(eval_examples, eval_features, all_results,
        #                   args.n_best_size, args.max_answer_length,
        #                   args.do_lower_case, output_prediction_file,
        #                   output_nbest_file, output_null_log_odds_file, args.verbose_logging,
        #                   args.version_2_with_negative, args.null_score_diff_threshold)

        if args.do_eval and is_main_process(dist):
            import sys
            import subprocess
            eval_out = subprocess.check_output([sys.executable, args.eval_script,
                                              args.predict_file, args.output_dir + "/predictions.json"])
            scores = str(eval_out).strip()
            exact_match = float(scores.split(":")[1].split(",")[0])
            f1 = float(scores.split(":")[2].split("}")[0])
            result_info = OrderedDict([('exact_match', exact_match),('f1', f1)])
            info_dict.update(result_info)
    if args.do_train:
        device_count = n_device
        if dist.is_initialized():
            device_count = dist.get_world_size()

        if args.max_steps == -1:
            logger.info("e2e_train_time: {},training_sequences_per_second: {},final_loss: {}".format(
              total_time,(len(train_features) * args.num_train_epochs - args.skip_steps*args.train_batch_size)/ total_time,final_loss))
        else:
            logger.info("e2e_train_time: {},training_sequences_per_second: {},final_loss: {}".format(
              total_time,(args.train_batch_size * args.gradient_accumulation_steps \
                                              * (args.max_steps - args.skip_steps) * device_count) / total_time,final_loss))
    if args.do_predict and is_main_process(dist):
        logger.info("e2e_inference_time: {},inference_sequences_per_second: {}".format(time_to_infer,len(eval_features) / time_to_infer))
    if args.do_eval and is_main_process(dist):
        logger.info("exact_match: {},F1: {}".format(exact_match,f1))
    json_report = json.dumps(info_dict, indent=4, sort_keys=False)
    logger.info("Final report:\n{}".format(json_report))
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
