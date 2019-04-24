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
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

from ..utils import vocab_utils


__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source_mark", "source_time",
                            "target_mark_input", "target_time_input",
                            "target_mark_output", "target_time_output",
                            "source_sequence_length", "target_sequence_length"))):
  pass


def get_infer_iterator(src_mark_dataset,
                       src_time_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       src_max_len=None,
                       use_char_encode=False):
  if use_char_encode:
    src_eos_id = vocab_utils.EOS_CHAR_ID
  else:
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_dataset = tf.data.Dataset.zip((src_mark_dataset, src_time_dataset))
  src_dataset = src_dataset.map(lambda src_m, src_t: (tf.string_split([src_m]).values,
                                                      tf.string_to_number(
                                                          tf.string_split([src_t]).values)))

  if src_max_len:
    src_dataset = src_dataset.map(lambda src_m, src_t: (src_m[:src_max_len],
                                                        src_t[:src_max_len]))

  if use_char_encode:
    # Convert the word strings to character ids
    src_dataset = src_dataset.map(
        lambda src_m, src_t: (tf.reshape(vocab_utils.tokens_to_bytes(src_m), [-1]),
                              src_t))
  else:
    # Convert the word strings to ids
    src_dataset = src_dataset.map(
        lambda src_m, src_t: (tf.cast(src_vocab_table.lookup(src_m), tf.int32),
                              src_t))

  # Add in the word counts.
  if use_char_encode:
    src_dataset = src_dataset.map(
        lambda src_m, src_t: (src_m, src_t,
                     tf.to_int32(
                         tf.size(src_m) / vocab_utils.DEFAULT_CHAR_MAXLEN)))
  else:
    src_dataset = src_dataset.map(lambda src_m, src_t: (src_m, src_t, tf.size(src_m)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src_m
            tf.TensorShape([None]),  # src_t
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src_m
            0.0,  # src_t
            0),  # src_len -- unused
          drop_remainder=True)

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_m_ids, src_t_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source_mark=src_m_ids,
      source_time=src_t_ids,
      target_mark_input=None,
      target_time_input=None,
      target_mark_output=None,
      target_time_output=None,
      source_sequence_length=src_seq_len,
      target_sequence_length=None)


def get_iterator(src_event_dataset,
                 src_time_dataset,
                 tgt_event_dataset,
                 tgt_time_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True,
                 use_char_encode=False):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000

  if use_char_encode:
    src_eos_id = vocab_utils.EOS_CHAR_ID
  else:
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_event_dataset,
                                         src_time_dataset,
                                         tgt_event_dataset,
                                         tgt_time_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src_m, src_t, tgt_m, tgt_t: (
          tf.string_split([src_m]).values,
          tf.string_to_number(tf.string_split([src_t]).values),
          tf.string_split([tgt_m]).values,
          tf.string_to_number(tf.string_split([tgt_t]).values)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src_m, src_t, tgt_m, tgt_t: tf.logical_and(tf.size(src_m) > 0,
                                                        tf.size(tgt_m) > 0))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_m, src_t, tgt_m, tgt_t: (src_m[:src_max_len],
                                            src_t[:src_max_len],
                                            tgt_m, tgt_t),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_m, src_t, tgt_m, tgt_t: (src_m, src_t,
                                            tgt_m[:tgt_max_len],
                                            tgt_t[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  if use_char_encode:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_m, src_t, tgt_m, tgt_t: (tf.reshape(vocab_utils.tokens_to_bytes(src_m), [-1]),
                                            src_t,
                                            tf.cast(tgt_vocab_table.lookup(tgt_m), tf.int32),
                                            tgt_t),
        num_parallel_calls=num_parallel_calls)
  else:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_m, src_t, tgt_m, tgt_t: (tf.cast(src_vocab_table.lookup(src_m), tf.int32),
                                            src_t,
                                            tf.cast(tgt_vocab_table.lookup(tgt_m), tf.int32),
                                            tgt_t),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src_m, src_t, tgt_m, tgt_t: (src_m, src_t,
                        tf.concat(([src_m[-1]], tgt_m), 0),
                        tf.concat((tgt_m, [tgt_eos_id]), 0),
                        tf.concat(([src_t[-1]], tgt_t), 0),
                        tf.concat((tgt_t, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  if use_char_encode:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_m, src_t, tgt_m_in, tgt_m_out, tgt_t_in, tgt_t_out: (
            src_m, src_t, tgt_m_in, tgt_m_out, tgt_t_in, tgt_t_out,
            tf.to_int32(tf.size(src) / vocab_utils.DEFAULT_CHAR_MAXLEN),
            tf.size(tgt_m_in)),
        num_parallel_calls=num_parallel_calls)
  else:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_m, src_t, tgt_m_in, tgt_m_out, tgt_t_in, tgt_t_out: (
            src_m, src_t, tgt_m_in, tgt_m_out, tgt_t_in, tgt_t_out,
            tf.size(src_m), tf.size(tgt_m_in)),
        num_parallel_calls=num_parallel_calls)

  src_tgt_dataset = src_tgt_dataset.prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src_m
            tf.TensorShape([None]),  # src_t
            tf.TensorShape([None]),  # tgt_m_input
            tf.TensorShape([None]),  # tgt_m_output
            tf.TensorShape([None]),  # tgt_t_input
            tf.TensorShape([None]),  # tgt_t_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src_m
            0.0,  # src_t
            tgt_eos_id,  # tgt_m_input
            tgt_eos_id,  # tgt_m_output
            0.0,  # tgt_t_input
            0.0,  # tgt_t_output
            0,  # src_len -- unused
            0),  # tgt_len -- unused
        drop_remainder=True)

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, unused_4, unused_5, unused_6,
                 src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_m_ids, src_t_ids,
   tgt_m_input_ids, tgt_m_output_ids,
   tgt_t_input_ids, tgt_t_output_ids,
   src_seq_len, tgt_seq_len) = (batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source_mark=src_m_ids,
      source_time=src_t_ids,
      target_mark_input=tgt_m_input_ids,
      target_time_input=tgt_t_input_ids,
      target_mark_output=tgt_m_output_ids,
      target_time_output=tgt_t_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)
