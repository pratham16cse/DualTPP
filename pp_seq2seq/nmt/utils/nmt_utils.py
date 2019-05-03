# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Utility functions specifically for NMT."""
from __future__ import print_function

import codecs
import time
import numpy as np
import tensorflow as tf

from ..utils import evaluation_utils
from ..utils import misc_utils as utils

__all__ = ["decode_and_evaluate", "get_translation"]


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_mark_file,
                        trans_time_file,
                        ref_mark_file,
                        ref_time_file,
                        metrics,
                        subword_option,
                        beam_width,
                        tgt_eos,
                        num_translations_per_input=1,
                        decode=True,
                        infer_mode="greedy",
                        decode_mark=False,
                        decode_time=False):
  """Decode a test set and compute a score according to the evaluation task."""
  # Decode
  if decode:
    utils.print_out("  decoding to output %s and %s" % (trans_mark_file, trans_time_file))

    start_time = time.time()
    num_sentences = 0
    with codecs.getwriter("utf-8")(
        tf.gfile.GFile(trans_mark_file, mode="wb")) as trans_mark_f, \
        codecs.getwriter("utf-8")(
        tf.gfile.GFile(trans_time_file, mode="wb")) as trans_time_f:
      trans_mark_f.write("")  # Write empty string to ensure file is created.
      trans_time_f.write("")  # Write empty string to ensure file is created.

      if infer_mode == "greedy":
        num_translations_per_input = 1
      elif infer_mode == "beam_search":
        num_translations_per_input = min(num_translations_per_input, beam_width)

      while True:
        try:
          mark_outputs, time_outputs, _ = model.decode(sess)
          if infer_mode != "beam_search":
            mark_outputs = np.expand_dims(mark_outputs, 0)
            time_outputs = np.expand_dims(time_outputs, 0)

          batch_size = mark_outputs.shape[1]
          num_sentences += batch_size

          for sent_id in range(batch_size):
            for beam_id in range(num_translations_per_input):
              mark_text, time_text = get_translation(
                  mark_outputs[beam_id],
                  time_outputs[beam_id],
                  sent_id,
                  tgt_eos=tgt_eos,
                  subword_option=subword_option)
              trans_mark_f.write((mark_text + b"\n").decode("utf-8"))
              trans_time_f.write((time_text + b"\n").decode("utf-8"))
        except tf.errors.OutOfRangeError:
          utils.print_time(
              "  done, num sentences %d, num translations per input %d" %
              (num_sentences, num_translations_per_input), start_time)
          break

  # Evaluation
  evaluation_scores = {}
  if ref_mark_file and tf.gfile.Exists(trans_mark_file):
    for metric in metrics:
      score, mark_score, time_score = evaluation_utils.evaluate(
          ref_mark_file,
          trans_mark_file,
          ref_time_file,
          trans_time_file,
          metric,
          decode_mark,
          decode_time,
          subword_option=subword_option)
      evaluation_scores[metric] = score
      mark_metric, time_metric = metric.split('_')
      utils.print_out("  %s %s: %f = %f(%s) + %f(%s)" \
              % (name, metric, score, mark_score, mark_metric, time_score, time_metric))

  return evaluation_scores


def get_translation(mark_outputs, time_outputs, sent_id, tgt_eos, subword_option):
  """Given batch decoding outputs, select a sentence and turn to text."""
  if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
  # Select a sentence
  mark_output = mark_outputs[sent_id, :].tolist()
  time_output = time_outputs[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in mark_output:
    mark_output = mark_output[:mark_output.index(tgt_eos)]
    #time_output = time_output[:time_output.index(tgt_eos)] #TODO need to change this

  if subword_option == "bpe":  # BPE
    mark_text = utils.format_bpe_text(mark_output)
    time_text = utils.format_bpe_text(time_output)
  elif subword_option == "spm":  # SPM
    mark_text = utils.format_spm_text(mark_output)
    time_text = utils.format_spm_text(time_output)
  else:
    mark_text = utils.format_text(mark_output)
    time_text = utils.format_float(time_output)

  return mark_text, time_text
