"""A library of customized helpers for use with SamplingDecoders.
   These helper classes are written to work with real-valued outputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq import Helper

import tensorflow as tf

__all__ = [
    "TimeTrainingHelper",
    "TimeGreedyHelper",
    #"SampleEmbeddingHelper",
    #"CustomHelper",
    #"ScheduledEmbeddingTrainingHelper",
    #"ScheduledOutputTrainingHelper",
    #"InferenceHelper",
]

_transpose_batch_time = decoder._transpose_batch_time  # pylint: disable=protected-access


def _unstack_ta(inp):
  return tensor_array_ops.TensorArray(
      dtype=inp.dtype, size=array_ops.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)


class TimeTrainingHelper(Helper):
  """A helper for use during training.  Only reads real-valued inputs.

  Returned sample_ids are real-valued outputs obtained from RNN state.
  """

  def __init__(self, inputs, sequence_length, time_major=False, name=None):
    """Initializer.

    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.

    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    with ops.name_scope(name, "TimeTrainingHelper", [inputs, sequence_length]):
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      self._inputs = inputs
      if not time_major:
        inputs = nest.map_structure(_transpose_batch_time, inputs)

      self._input_tas = nest.map_structure(_unstack_ta, inputs)
      self._sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if self._sequence_length.get_shape().ndims != 1:
        raise ValueError(
            "Expected sequence_length to be a vector, but received shape: %s" %
            self._sequence_length.get_shape())

      self._zero_inputs = nest.map_structure(
          lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

      self._batch_size = array_ops.size(sequence_length)

  @property
  def inputs(self):
    return self._inputs

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tensor_shape.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return dtypes.float32

  def initialize(self, name=None):
    with ops.name_scope(name, "TimeTrainingHelperInitialize"):
      finished = math_ops.equal(0, self._sequence_length)
      all_finished = math_ops.reduce_all(finished)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
      return (finished, next_inputs)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    with ops.name_scope(name, "TimeTrainingHelperSample", [time, outputs]):
      sample_ids = outputs
      return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for TimeTrainingHelper."""
    with ops.name_scope(name, "TimeTrainingHelperNextInputs",
                        [time, outputs, state]):
      next_time = time + 1
      finished = (next_time >= self._sequence_length)
      all_finished = math_ops.reduce_all(finished)
      def read_from_ta(inp):
        return inp.read(next_time)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: nest.map_structure(read_from_ta, self._input_tas))
      return (finished, next_inputs, state)


class TimeGreedyHelper(Helper):
  """A helper for use during inference.

  Uses the real-valued output (of last layer) to get the next input.
  """

  def __init__(self, start_tokens, end_token):
    """Initializer.

    Args:
      start_tokens: `float32` vector shaped `[batch_size]`, the start tokens.
      end_token: `float32` scalar, the token that marks end of decoding.

    Raises:
      ValueError: if `start_tokens` is not a 1D tensor or `end_token` is not a
        scalar.
    """

    self._start_tokens = ops.convert_to_tensor(
        start_tokens, dtype=dtypes.float32, name="start_tokens")
    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.float32, name="end_token")
    if self._start_tokens.get_shape().ndims != 1:
      raise ValueError("start_tokens must be a vector")
    self._batch_size = array_ops.size(start_tokens)
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    self._start_inputs = tf.expand_dims(self._start_tokens, axis=-1)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tensor_shape.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return dtypes.float32

  def initialize(self, name=None):
    finished = array_ops.tile([False], [self._batch_size])
    return (finished, self._start_inputs)

  def sample(self, time, outputs, state, name=None):
    """sample for TimeGreedyHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    sample_ids = tf.squeeze(outputs, axis=-1)
    return sample_ids

  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """next_inputs_fn for GreedyEmbeddingHelper."""
    print(sample_ids.get_shape())
    del time, outputs  # unused by next_inputs_fn
    finished = math_ops.equal(sample_ids, self._end_token)
    all_finished = math_ops.reduce_all(finished)
    next_inputs = control_flow_ops.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda: self._start_inputs,
        lambda: tf.expand_dims(sample_ids, axis=-1))
    print(next_inputs.get_shape())
    return (finished, next_inputs, state)

