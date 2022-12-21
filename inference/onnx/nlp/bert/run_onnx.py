# Author: shan.zhu@enflame-tech.com
# Some of the part of codes are from The Google AI Language Team
# Copyright 2018 The Google AI Language Team Authors.
#
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

"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import, division, print_function

import collections
import json
import math
import os
import time
import sys
import subprocess
import numpy as np
import six
import tensorflow as tf
import logging
import onnxruntime as rt

import tokenization
from create_squad_data import *

from collections import OrderedDict
from common.logger import final_report


def set_logging(logging_file):
  """set logging info"""
  _logger = logging.getLogger('BERT SQuaD task inference')
  _logger.setLevel(logging.DEBUG)
  #define file handler and set formatter
  file_handler=logging.FileHandler(logging_file)
  formatter=logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
  file_handler.setFormatter(formatter)
  # add handler to _logger
  _logger.addHandler(file_handler)
  return _logger


flags = tf.flags
FLAGS = None

def extract_run_squad_flags():
  """set flags"""
  ## Required parameters
  flags.DEFINE_string(
          "onnx_path",'./bert_base-squad-nvidia-op13-fp32-N.onnx',
          "The path of onnx model to be inferenced "
          )

  flags.DEFINE_string("model_ver",'nvidia',
          "The version of onnx to use the matching input")

  flags.DEFINE_string('device','cpu',
          "The device for onnx running")

  flags.DEFINE_string(
      "bert_config_file", None,
      "The config json file corresponding to the pre-trained BERT model. "
      "This specifies the model architecture.")

  flags.DEFINE_string("vocab_file", None,
                      "The vocabulary file that the BERT model was trained on.")

  flags.DEFINE_string(
      "output_dir", None,
      "The output directory where the model checkpoints will be written.")


  flags.DEFINE_string('logging_file',None,
          "The output logging files to save current logging")

  ## Other parameters
  flags.DEFINE_string(
      "predict_file", None,
      "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

  flags.DEFINE_string(
      "eval_script", None,
      "SQuAD evaluate.py file to compute f1 and exact_match E.g., evaluate-v1.1.py")

  flags.DEFINE_bool(
      "do_lower_case", True,
      "Whether to lower case the input text. Should be True for uncased "
      "models and False for cased models.")

  flags.DEFINE_integer(
      "max_seq_length", 384,
      "The maximum total input sequence length after WordPiece tokenization. "
      "Sequences longer than this will be truncated, and sequences shorter "
      "than this will be padded.")

  flags.DEFINE_integer(
      "doc_stride", 128,
      "When splitting up a long document into chunks, how much stride to "
      "take between chunks.")

  flags.DEFINE_integer(
      "max_query_length", 64,
      "The maximum number of tokens for the question. Questions longer than "
      "this will be truncated to this length.")


  flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")


  flags.DEFINE_integer("predict_batch_size", 8,
                       "Total batch size for predictions.")


  flags.DEFINE_integer("iterations_per_loop", 1000,
                       "How many steps to make in each estimator call.")

  flags.DEFINE_integer(
      "n_best_size", 20,
      "The total number of n-best predictions to generate in the "
      "nbest_predictions.json output file.")

  flags.DEFINE_integer(
      "max_answer_length", 30,
      "The maximum length of an answer that can be generated. This is needed "
      "because the start and end predictions are not conditioned on one another.")


  flags.DEFINE_bool(
      "verbose_logging", False,
      "If true, all of the warnings related to data processing will be printed. "
      "A number of warnings are expected for a normal SQuAD evaluation.")

  flags.DEFINE_bool(
      "version_2_with_negative", False,
      "If true, the SQuAD examples contain some that do not have an answer.")

  flags.DEFINE_float(
      "null_score_diff_threshold", 0.0,
      "If null_score - best_non_null is greater than the threshold predict null.")

  flags.DEFINE_integer("num_eval_iterations", None,
                       "How many eval iterations to run - performs inference on subset")


  return flags.FLAGS

def inference_with_onnx(onnx_path, features, batch_size, all_result):
  """inference of onnx model with onnxruntime"""
  #backend to use
  if  FLAGS.device=='cpu':
      EP_list = ['CPUExecutionProvider']
      options = [{}]
  elif FLAGS.device=='gpu':
      EP_list = ['CUDAExecutionProvider']
      options = [{}]
  elif FLAGS.device=='gcu':
      print("this is gcu set!")
      EP_list = ['TopsInferenceExecutionProvider']
      options = [{}]
  session = rt.InferenceSession(onnx_path, providers=EP_list, provider_options=options)
  
  idx = 0
  N = len(features)
  all_result = None
  _logger.info('inference with onnx start')

  try:
    while True:
      a_feature = features[idx:idx+batch_size]
      inputs = {"unique_ids": np.array([feature.unique_id for feature in a_feature]),
                "input_ids": np.array([feature.input_ids for feature in a_feature]),
                "input_mask": np.array([feature.input_mask for feature in a_feature]),
                "segment_ids": np.array([feature.segment_ids for feature in a_feature])}
      if idx >= N: break
      idx += batch_size
      if FLAGS.model_ver=='nvidia':
          data = {"input_ids:0": inputs['input_ids'].astype(np.int32),
                  "input_mask:0": inputs['input_mask'].astype(np.int32),
                  "segment_ids:0": inputs['segment_ids'].astype(np.int32)}
          output_name = ["unstack:0", "unstack:1"]
      elif FLAGS.model_ver=='google':
          data = {"input_ids_2:0": inputs['input_ids'].astype(np.int32),
                  "input_mask_2:0": inputs['input_mask'].astype(np.int32),
                  "segment_ids_2:0": inputs['segment_ids'].astype(np.int32)}
          output_name = ["unstack:0", "unstack:1"]
      elif FLAGS.model_ver=='huggingface':
          data = {"input_ids": inputs['input_ids'].astype(np.int64),
                  "attention_mask": inputs['input_mask'].astype(np.int64),
                  "token_type_ids": inputs['segment_ids'].astype(np.int64)}
          output_name = ["start_logits", "end_logits"]
      elif FLAGS.model_ver=='zoo':
          data = {"unique_ids_raw_output___9:0":np.arange(256,dtype=np.int64),
                  "input_ids:0": inputs['input_ids'].astype(np.int64),
                  "input_mask:0": inputs['input_mask'].astype(np.int64),
                  "segment_ids:0": inputs['segment_ids'].astype(np.int64)}
          output_name = ["unstack:0", "unstack:1"]
      elif FLAGS.model_ver=='mlperf':
          data = {"input_ids": inputs['input_ids'].astype(np.int64),
                  "input_mask": inputs['input_mask'].astype(np.int64),
                  "segment_ids": inputs['segment_ids'].astype(np.int64)}
          output_name = ["output_start_logits", "output_end_logits"]
      else:
          print(f'The model version {FLAGS.model_ver} is invalid. Please check.')
      result = {}
      result['unique_ids'] = inputs['unique_ids']
      res = session.run(output_name, data)
      result['start_logits'] = res[0]
      result['end_logits'] = res[1]
      if not all_result:
        all_result = result
      else:
        all_result['unique_ids'] = np.append(all_result['unique_ids'],result['unique_ids'])
        all_result['start_logits'] = np.concatenate((all_result['start_logits'],result['start_logits']),axis=0)
        all_result['end_logits'] = np.concatenate((all_result['end_logits'],result['end_logits']),axis=0)
      print(f'The {idx+1}-th infer is done!')
  except tf.errors.OutOfRangeError:pass
  return all_result

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def get_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length,
  do_lower_case, version_2_with_negative, verbose_logging):
  """Get final predictions"""

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
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
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))

    if version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
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

        final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True
      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    if not version_2_with_negative:
      all_predictions[example.qas_id] = nbest_json[0]["text"]
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff

      try:
        null_score_diff_threshold = FLAGS.null_score_diff_threshold
      except:
        null_score_diff_threshold = 0.0
      if score_diff > null_score_diff_threshold:
        all_predictions[example.qas_id] = ""
      else:
        all_predictions[example.qas_id] = best_non_null_entry.text

    all_nbest_json[example.qas_id] = nbest_json
  return all_predictions, all_nbest_json, scores_diff_json

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      version_2_with_negative, verbose_logging):
  """Write final predictions to the json file and log-odds of null if needed."""

  _logger.info("Writing predictions to: %s" % (output_prediction_file))
  _logger.info("Writing nbest to: %s" % (output_nbest_file))

  all_predictions, all_nbest_json, scores_diff_json = get_predictions(all_examples, all_features,
    all_results, n_best_size, max_answer_length, do_lower_case, version_2_with_negative, verbose_logging)

  with tf.io.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

  with tf.io.gfile.GFile(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

  if version_2_with_negative:
    with tf.io.gfile.GFile(output_null_log_odds_file, "w") as writer:
      writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging):
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
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if verbose_logging:
      tf.compat.v1.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if verbose_logging:
      _logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if verbose_logging:
      _logger.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if verbose_logging:
      _logger.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


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


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_predict:
    raise ValueError(" `do_predict`must be True.")

  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  tf.io.gfile.makedirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  master_process = True
  if master_process:
      _logger.info("***** Configuaration *****")
      for key in FLAGS.__flags.keys():
          _logger.info('  {}: {}'.format(key, getattr(FLAGS, key)))
      _logger.info("**************************")

  if FLAGS.do_predict and master_process:
    eval_examples = read_squad_examples(
        input_file=FLAGS.predict_file, is_training=False,
        version_2_with_negative=FLAGS.version_2_with_negative)

    # Perform evaluation on subset, useful for profiling
    if FLAGS.num_eval_iterations is not None:
        eval_examples = eval_examples[:FLAGS.num_eval_iterations*FLAGS.predict_batch_size]

    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature,
        verbose_logging=FLAGS.verbose_logging)

    _logger.info("***** Running predictions *****")
    _logger.info("  Num orig examples = %d", len(eval_examples))
    _logger.info("  Num split examples = %d", len(eval_features))
    _logger.info("  Batch size = %d", FLAGS.predict_batch_size)
    _logger.info("  PATH of ONNX model is %s"%( FLAGS.onnx_path))

    ONNX_PATH=FLAGS.onnx_path
    all_result_onnx={}
    eval_start_time = time.time()
    all_result_onnx=inference_with_onnx(ONNX_PATH,eval_features,FLAGS.predict_batch_size,all_result_onnx)
    eval_time_elapsed = time.time() - eval_start_time
    results_onnx=[]
    idx=0
    for idx in range(len(all_result_onnx['unique_ids'])):
      if idx % 1000 == 0:
        _logger.info("Processing example: %d" % (idx))
      unique_id_onnx = int(all_result_onnx["unique_ids"][idx])
      start_logits_onnx = [float(x) for x in all_result_onnx["start_logits"][idx].flat]
      end_logits_onnx = [float(x) for x in all_result_onnx["end_logits"][idx].flat]
      results_onnx.append(
          RawResult(
              unique_id=unique_id_onnx,
              start_logits=start_logits_onnx,
              end_logits=end_logits_onnx))

    eval_time_elapsed = time.time() - eval_start_time

    _logger.info("-----------------------------")
    _logger.info("Total Inference Time = %0.2f for batchsize = %d", eval_time_elapsed,
                    FLAGS.predict_batch_size)
    _logger.info("Total Inference FPS = %0.4f",  len(eval_features)/eval_time_elapsed)
    _logger.info("Summary Inference Statistics")
    _logger.info("Batch size = %d", FLAGS.predict_batch_size)
    _logger.info("Sequence Length = %d", FLAGS.max_seq_length)
    _logger.info("-----------------------------")

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    write_predictions(eval_examples, eval_features, results_onnx,
                      FLAGS.n_best_size, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      FLAGS.version_2_with_negative, FLAGS.verbose_logging)

    if FLAGS.eval_script:
      eval_out = subprocess.check_output([sys.executable, FLAGS.eval_script,
                                        FLAGS.predict_file, output_prediction_file])
      scores = str(eval_out).strip()
      exact_match = float(scores.split(":")[1].split(",")[0])
      f1 = float(scores.split(":")[2].split("}")[0])
      _logger.info("===============inference accuracy=================")
      _logger.info("exact_match is: {} and f1 is {}".format(exact_match, f1))
      print(str(eval_out))

    # formatted json output
    runtime_info = OrderedDict(
      [('model', 'bert'),
        ('device', FLAGS.device),
        ('dataset', 'squad v1.1'),
        ('exact_match', format(exact_match, '.5f')),
        ('f1', format(f1, '.5f'))
        ])
    final_report(_logger, runtime_info)

if __name__ == "__main__":
  FLAGS = extract_run_squad_flags()
  _logger=set_logging(FLAGS.logging_file)
  tf.app.run()