"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import numpy as np

import tensorflow as tf

from . import model
from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import vocab_utils

from . import my_basic_decoder
from . import my_helper

utils.check_tensorflow_version()

__all__ = ["GreedyTppModel"]

class GreedyTppModel(model.Model):

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    self.tgt_max_len_infer = hparams.tgt_max_len_infer

    super(GreedyTppModel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope,
        extra_args=extra_args)

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss_tuple, final_context_state, mark_sample_id, time_sample_val),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: loss = the total loss / batch_size.
        final_context_state: the final state of decoder RNN.
        mark_sample_id: sampling indices.
        time_sample_val: sampled time

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# Creating %s graph ..." % self.mode)

    # Projection
    if not self.extract_encoder_layers:
      with tf.variable_scope(scope or "build_network"):
        with tf.variable_scope("decoder/output_projection"):
          if self.decode_mark:
            self.output_mark_layer = tf.layers.Dense(
                self.tgt_vocab_size, use_bias=False, name="output_mark_projection")
          else: self.output_mark_layer = None
          if self.decode_time:
            self.output_time_layer = tf.layers.Dense(
                3, activation=None, name="output_time_projection")
          else: self.output_time_layer = None

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=self.dtype):
      # Encoder
      if hparams.language_model:  # no encoder for language modeling
        utils.print_out("  language modeling: no encoder")
        self.encoder_outputs = None
        encoder_state = None
      else:
        self.encoder_outputs, encoder_state = self._build_encoder(hparams)

      # Skip decoder if extracting only encoder layers
      if self.extract_encoder_layers:
        return

      ## Decoder
      logits, decoder_cell_outputs, time_pred, \
      mark_sample_id, time_sample_val, \
      final_context_state = (
          self._build_decoder(self.encoder_outputs, encoder_state, hparams))

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                   self.num_gpus)):
          mark_loss, time_loss = self._compute_loss(logits, decoder_cell_outputs, time_pred)
      else:
        mark_loss, time_loss = tf.constant(0.0), tf.constant(0.0)

      return logits, mark_loss, time_loss, final_context_state, mark_sample_id, time_sample_val



  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    #tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
    #                     tf.int32)
    #tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
    #                     tf.int32)
    tgt_sos_id = -1
    tgt_eos_id = -1
    iterator = self.iterator

    # maximum_iteration: The maximum decoding steps.
    maximum_iterations = self._get_infer_maximum_iterations(
        hparams, iterator.source_sequence_length)

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)

      # Optional ops depends on which mode we are in and which loss function we
      # are using.
      logits = tf.no_op()
      decoder_cell_outputs = None

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # decoder_emb_inp: [max_time, batch_size, num_units]
        target_mark_input = iterator.target_mark_input
        target_time_input = iterator.target_time_input
        if self.time_major:
          target_mark_input = tf.transpose(target_mark_input)
          target_time_input = tf.transpose(target_time_input)
        if target_time_input.get_shape().ndims == 2:
            target_time_input = tf.expand_dims(target_time_input, axis=2)
        decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, target_mark_input)

        # Helpers
        if self.decode_mark:
            mark_helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, iterator.target_sequence_length,
                time_major=self.time_major)
        else: mark_helper = None
        if self.decode_time:
            time_helper = my_helper.TimeTrainingHelper(
                target_time_input, iterator.target_sequence_length,
                time_major=self.time_major)
        else: time_helper = None
        print(decoder_emb_inp.get_shape(), target_time_input.get_shape())

        # Decoder
        my_decoder = my_basic_decoder.MyBasicDecoder(
            cell,
            decoder_initial_state,
            mark_helper, time_helper,)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        mark_sample_id = outputs.mark_sample_id if self.decode_mark else outputs.time_sample_val
        time_sample_val = outputs.time_sample_val if self.decode_time else outputs.mark_sample_id

        if self.num_sampled_softmax > 0:
          # Note: this is required when using sampled_softmax_loss.
          decoder_mark_cell_outputs = outputs.mark_rnn_output

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        num_layers = self.num_decoder_layers
        num_gpus = self.num_gpus
        device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
        # Colocate output layer with the last RNN cell if there is no extra GPU
        # available. Otherwise, put last layer on a separate GPU.
        with tf.device(model_helper.get_device_str(device_id, num_gpus)):
          logits = self.output_mark_layer(outputs.mark_rnn_output) if self.decode_mark else None
          time_pred = self.output_time_layer(outputs.time_rnn_output) if self.decode_time else None

        if self.num_sampled_softmax > 0:
          logits = tf.no_op()  # unused when using sampled softmax loss.

      ## Inference
      else:
        infer_mode = hparams.infer_mode
        #mark_start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        mark_start_tokens = iterator.source_mark[:self.batch_size, -1]
        #time_start_tokens = tf.fill([self.batch_size], 0.0)
        time_start_tokens = iterator.source_time[:self.batch_size, -1]
        if time_start_tokens.get_shape().ndims == 2:
            time_start_tokens= tf.expand_dims(time_start_tokens, axis=2)
        mark_end_token = tgt_eos_id
        time_end_token = 0
        utils.print_out(
            "  decoder: infer_mode=%sbeam_width=%d, "
            "length_penalty=%f, coverage_penalty=%f"
            % (infer_mode, hparams.beam_width, hparams.length_penalty_weight,
               hparams.coverage_penalty_weight))

        if infer_mode == "beam_search":
          beam_width = hparams.beam_width
          length_penalty_weight = hparams.length_penalty_weight
          coverage_penalty_weight = hparams.coverage_penalty_weight

          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_mark_layer,
              length_penalty_weight=length_penalty_weight,
              coverage_penalty_weight=coverage_penalty_weight)
        elif infer_mode == "sample":
          # Helper
          sampling_temperature = hparams.sampling_temperature
          assert sampling_temperature > 0.0, (
              "sampling_temperature must greater than 0.0 when using sample"
              " decoder.")
          helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token,
              softmax_temperature=sampling_temperature,
              seed=self.random_seed)
        elif infer_mode == "greedy":
          if self.decode_mark:
            mark_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding_decoder, mark_start_tokens, mark_end_token)
          else: mark_helper = None
          if self.decode_time:
            time_helper = my_helper.TimeGreedyPointProcessHelper(
                time_start_tokens, time_end_token)
          else: time_helper = None
        else:
          raise ValueError("Unknown infer_mode '%s'", infer_mode)

        if infer_mode != "beam_search":
          my_decoder = my_basic_decoder.MyBasicDecoder(
              cell,
              decoder_initial_state,
              mark_helper, time_helper,
              output_mark_layer=self.output_mark_layer if self.decode_mark else None,  # applied per timestep
              output_time_layer=self.output_time_layer if self.decode_time else None,  # applied per timestep
          )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if infer_mode == "beam_search":
          sample_id = outputs.predicted_ids
        else:
          if self.decode_mark:
            logits = outputs.mark_rnn_output
            mark_sample_id = outputs.mark_sample_id
          else: logits, mark_sample_id = tf.no_op(), tf.no_op()
          if self.decode_time:
            time_pred = outputs.time_rnn_output
            time_sample_val = outputs.time_sample_val
          else: time_pred, time_sample_val = tf.no_op(), tf.no_op()

    return logits, decoder_cell_outputs, time_pred, mark_sample_id, time_sample_val, final_context_state

  def _time_loss(self, time_pred, target_time_output):
    last_target_time_input = self.iterator.source_time[:self.batch_size, -1:]
    if self.time_major:
      last_target_time_input = tf.transpose(last_target_time_input)
      target_time_output_minus_1 = tf.concat([last_target_time_input, target_time_output[:-1, :]], axis=0)
    else:
      target_time_output_minus_1 = tf.concat([last_target_time_input, target_time_output[:, :-1]], axis=1)
    gap = target_time_output - target_time_output_minus_1

    lambda_0, w, gamma = tf.split(time_pred, 3, axis=-1)
    lambda_0 = tf.squeeze(lambda_0, axis=-1)
    w = tf.squeeze(w, axis=-1)
    gamma = tf.squeeze(gamma, axis=-1)
    ln_lambda_star = lambda_0 + w * gap + gamma * target_time_output_minus_1
    lambda_star = tf.exp(tf.minimum(10.0, ln_lambda_star))
    ln_f_star = -(1.0) \
              * ln_lambda_star \
              + 1.0/w * tf.exp(tf.minimum(10.0, lambda_0)) \
              + 1.0/w * lambda_star

    return ln_f_star
