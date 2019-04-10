"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

import tensorflow as tf

__all__ = [
    "BasicDecoderOutput",
    "MyBasicDecoder",
]


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("mark_rnn_output",
                                                  "time_rnn_output",
                                                  "mark_sample_id",
                                                  "time_sample_val"))):
  pass


class MyBasicDecoder(decoder.Decoder):
  """Basic sampling decoder."""

  def __init__(self, cell, initial_state,
               mark_helper=None, time_helper=None,
               output_mark_layer=None, output_time_layer=None):
    """Initialize MyBasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      mark_helper: A `Helper` instance.
      time_helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_mark_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result or sampling.
      output_time_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `<mark/time>_helper` or `output_<mark/time>_layer` 
        have an incorrect type.
    """
    rnn_cell_impl.assert_like_rnncell("cell", cell)
    if mark_helper is None and time_helper is None:
      raise Exception("Both mark_helper and time_helper cannot be None")
    if output_mark_layer is not None:
      assert mark_helper is not None
    if output_time_layer is not None:
      assert time_helper is not None
    if mark_helper is not None and time_helper is not None:
      oml = output_mark_layer is not None
      otl = output_time_layer is not None
      assert oml == otl
    if mark_helper is not None and not isinstance(mark_helper, helper_py.Helper):
      raise TypeError("mark_helper must be a Helper, received: %s" % type(mark_helper))
    if time_helper is not None and not isinstance(time_helper, helper_py.Helper):
      raise TypeError("time_helper must be a Helper, received: %s" % type(time_helper))
    if (output_mark_layer is not None
        and not isinstance(output_mark_layer, layers_base.Layer)):
      raise TypeError(
          "output_mark_layer must be a Layer, received: %s" % type(output_mark_layer))
    if (output_time_layer is not None
        and not isinstance(output_time_layer, layers_base.Layer)):
      raise TypeError(
          "output_time_layer must be a Layer, received: %s" % type(output_time_layer))
    self._cell = cell
    self._mark_helper = mark_helper
    self._time_helper = time_helper
    self._initial_state = initial_state
    self._output_mark_layer = output_mark_layer
    self._output_time_layer = output_time_layer

  @property
  def batch_size(self):
    return self._mark_helper.batch_size

  def _mark_rnn_output_size(self):
    size = self._cell.output_size
    if self._output_mark_layer is None:
      return size
    else: 
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_mark_layer.compute_output_shape(
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  def _time_rnn_output_size(self):
    size = self._cell.output_size
    if self._output_time_layer is None:
      return size
    else: 
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_time_layer.compute_output_shape(
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        mark_rnn_output=self._mark_rnn_output_size(),
        time_rnn_output=self._time_rnn_output_size(),
        mark_sample_id=self._mark_helper.sample_ids_shape if self._mark_helper else self._time_helper.sample_ids_shape,
        time_sample_val=self._time_helper.sample_ids_shape if self._time_helper else self._mark_helper.sample_ids_shape)

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and the sample_ids_dtype from the helper.
    dtype = nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._mark_rnn_output_size()),
        nest.map_structure(lambda _: dtype, self._time_rnn_output_size()),
        self._mark_helper.sample_ids_dtype if self._mark_helper else self._time_helper.sample_ids_dtype,
        self._time_helper.sample_ids_dtype if self._time_helper else self._mark_helper.sample_ids_dtype)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    if self._mark_helper is not None:
        (mark_finished, mark_inputs) = self._mark_helper.initialize()
    if self._time_helper is not None:
        (time_finished, time_inputs) = self._time_helper.initialize()
    
    #print(mark_inputs.get_shape().ndims, time_inputs.get_shape().ndims)
    if self._mark_helper is not None and self._time_helper is not None:
        finished = tf.logical_and(mark_finished, time_finished)
        #print(mark_inputs.get_shape().ndims, time_inputs.get_shape().ndims)
        inputs = tf.concat([mark_inputs, time_inputs], axis=-1)
    elif self._mark_helper is not None:
        finished, inputs = mark_finished, mark_inputs
    elif self._time_helper is not None:
        finished, inputs = time_finished, time_inputs
   
    return (finished, inputs) + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_mark_layer is not None:
        cell_mark_outputs = self._output_mark_layer(cell_outputs)
      else: cell_mark_outputs = cell_outputs
      if self._output_time_layer is not None:
        cell_time_outputs = self._output_time_layer(cell_outputs)
      else: cell_time_outputs = cell_outputs

      if self._mark_helper is not None:
        mark_sample_ids = self._mark_helper.sample(
            time=time,
            outputs=cell_mark_outputs,
            state=cell_state)
        (mark_finished, mark_next_inputs, mark_next_state) = self._mark_helper.next_inputs(
            time=time,
            outputs=cell_mark_outputs,
            state=cell_state,
            sample_ids=mark_sample_ids)

      if self._time_helper is not None:
        time_sample_ids = self._time_helper.sample(
            time=time,
            outputs=cell_time_outputs,
            state=cell_state)
        (time_finished, time_next_inputs, time_next_state) = self._time_helper.next_inputs(
            time=time,
            outputs=cell_time_outputs,
            state=cell_state,
            sample_ids=time_sample_ids)

      if self._mark_helper is None:
        mark_sample_ids = tf.zeros_like(time_sample_ids)
      if self._time_helper is None:
        time_sample_ids = tf.zeros_like(mark_sample_ids)

      if self._mark_helper is not None and self._time_helper is not None:
        finished = tf.logical_and(mark_finished, time_finished)
        next_inputs = tf.concat([mark_next_inputs, time_next_inputs], axis=-1)
        assert mark_next_state == time_next_state #TODO make sure this works
        next_state = mark_next_state
      elif self._mark_helper is not None:
        finished, next_inputs, next_state = mark_finished, mark_next_inputs, mark_next_state
      elif self._time_helper is not None:
        finished, next_inputs, next_state = time_finished, time_next_inputs, time_next_state


    outputs = BasicDecoderOutput(cell_mark_outputs, cell_time_outputs,
                                 mark_sample_ids, time_sample_ids)
    return (outputs, next_state, next_inputs, finished)
