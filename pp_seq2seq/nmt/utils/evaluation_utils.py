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

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess
import numpy as np
import tensorflow as tf

from ..scripts import bleu
from ..scripts import rouge


__all__ = ["evaluate"]


def evaluate(ref_mark_file, trans_mark_file,
             ref_time_file, trans_time_file,
             metric, decode_mark, decode_time,
             subword_option=None):
  """Pick a pair of metrics and evaluate depending on task."""
  #`metric` is a underscore separated pair of 
  # mark and time metrics
  mark_metric, time_metric = metric.split('_')
  evaluation_score = 0.0
  mark_score, time_score = 0.0, 0.0
  # BLEU scores for translation task
  if mark_metric.lower() == "bleu":
    evaluation_score += _bleu(ref_file, trans_file,
                             subword_option=subword_option)
  # ROUGE scores for summarization tasks
  elif mark_metric.lower() == "rouge":
    mark_score = _rouge(ref_mark_file, trans_mark_file,
                              subword_option=subword_option)
  elif mark_metric.lower() == "accuracy":
    mark_score = _accuracy(ref_mark_file, trans_mark_file)
  elif mark_metric.lower() == "percenterror":
    mark_score = _percenterror(ref_mark_file, trans_mark_file)
  elif mark_metric.lower() == "word_accuracy":
    mark_score = _word_accuracy(ref_mark_file, trans_mark_file)
  else:
    raise ValueError("Unknown mark_metric %s" % mark_metric)
  evaluation_score += mark_score

  if time_metric.lower() == "rmse":
    time_score = _rmse(ref_time_file, trans_time_file)
  else:
    raise ValueError("Unknown time_metric %s" % time_metric)
  evaluation_score += time_score

  return evaluation_score, mark_score, time_score


def _clean(sentence, subword_option):
  """Clean and handle BPE or SPM outputs."""
  sentence = sentence.strip()

  # BPE
  if subword_option == "bpe":
    sentence = re.sub("@@ ", "", sentence)

  # SPM
  elif subword_option == "spm":
    sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

  return sentence


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, subword_option=None):
  """Compute BLEU scores and handling BPE."""
  max_order = 4
  smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(reference_filename, "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = _clean(reference, subword_option)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=None)
      translations.append(line.split(" "))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)
  return 100 * bleu_score


def _rouge(ref_file, summarization_file, subword_option=None):
  """Compute ROUGE scores and handling BPE."""

  references = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
    for line in fh:
      references.append(_clean(line, subword_option))

  hypotheses = []
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(summarization_file, "rb")) as fh:
    for line in fh:
      hypotheses.append(_clean(line, subword_option=None))

  rouge_score_map = rouge.rouge(hypotheses, references)
  return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file):
  """Compute accuracy, each line contains a label."""

  with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
    with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
      count = 0.0
      match = 0.0
      for pred in pred_fh:
        label = label_fh.readline().strip()
        pred = pred.strip()
        if label == pred:
          match += 1
        count += 1
  return 100 * match / count

def _percenterror(label_file, pred_file):
  return 100.0 - _accuracy(label_file, pred_file)

def _word_accuracy(label_file, pred_file):
  """Compute accuracy on per word basis."""

  with codecs.getreader("utf-8")(tf.gfile.GFile(label_file, "rb")) as label_fh:
    with codecs.getreader("utf-8")(tf.gfile.GFile(pred_file, "rb")) as pred_fh:
      total_acc, total_count = 0., 0.
      for sentence in label_fh:
        labels = sentence.strip().split(" ")
        preds = pred_fh.readline().strip().split(" ")
        match = 0.0
        for pos in range(min(len(labels), len(preds))):
          label = labels[pos]
          pred = preds[pos]
          if label == pred:
            match += 1
        total_acc += 100 * match / max(len(labels), len(preds))
        total_count += 1
  return total_acc / total_count


def _moses_bleu(multi_bleu_script, tgt_test, trans_file, subword_option=None):
  """Compute BLEU scores using Moses multi-bleu.perl script."""

  # TODO(thangluong): perform rewrite using python
  # BPE
  if subword_option == "bpe":
    debpe_tgt_test = tgt_test + ".debpe"
    if not os.path.exists(debpe_tgt_test):
      # TODO(thangluong): not use shell=True, can be a security hazard
      subprocess.call("cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
      subprocess.call("sed s/@@ //g %s" % (debpe_tgt_test),
                      shell=True)
    tgt_test = debpe_tgt_test
  elif subword_option == "spm":
    despm_tgt_test = tgt_test + ".despm"
    if not os.path.exists(despm_tgt_test):
      subprocess.call("cp %s %s" % (tgt_test, despm_tgt_test))
      subprocess.call("sed s/ //g %s" % (despm_tgt_test))
      subprocess.call(u"sed s/^\u2581/g %s" % (despm_tgt_test))
      subprocess.call(u"sed s/\u2581/ /g %s" % (despm_tgt_test))
    tgt_test = despm_tgt_test
  cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

  # subprocess
  # TODO(thangluong): not use shell=True, can be a security hazard
  bleu_output = subprocess.check_output(cmd, shell=True)

  # extract BLEU score
  m = re.search("BLEU = (.+?),", bleu_output)
  bleu_score = float(m.group(1))

  return bleu_score


def _rmse(ref_file, trans_file):
  with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as ref_fh:
    with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as trans_fh:
      count = 0.0
      err = 0.0
      for trans in trans_fh:
        ref = [float(r) for r in ref_fh.readline().strip().split()]
        trans = [float(t) for t in trans.strip().split()]
        err += np.sum((np.array(ref)-np.array(trans))**2*1.0)
        count += 1
  err = np.sqrt(err/count)
  return err
