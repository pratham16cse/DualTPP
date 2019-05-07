"""Seq2seq based temporal point process model with predicting
   full distribution over time variable."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import numpy as np

import tensorflow as tf

from . import model
from .model import InferOutputTuple
from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import vocab_utils

from . import my_basic_decoder
from . import my_helper
from scipy.integrate import quad

__all__ = ["VAEmodel"]


class VAEmodel(model.Model):

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):

    if not hasattr(hparams, "tgt_max_len_infer"):
      raise Exception('VAEmodel requires tgt_max_len_infer attribute to be set.')
    self.tgt_max_len_infer = hparams.tgt_max_len_infer
    self.infer_learning_rate = hparams.infer_learning_rate
    self.z_dim = 5
    self.padder = True
    self.initial_state_flag = False

    super(VAEmodel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope,
        extra_args=extra_args)


  def build_graph(self, hparams, scope=None):
    #TODO Edit docstring
    """
    Creates a VAEtpp graph with dynamic RNN decoder API.
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
          # if self.decode_mark:
          self.output_mark_layer = tf.layers.Dense(
                self.tgt_vocab_size, use_bias=False, name="output_mark_projection")
          # else: self.output_mark_layer = None
          # if self.decode_time:
          self.output_time_layer = tf.layers.Dense(
                1, activation=tf.nn.softplus, name="output_time_projection")
          # else: self.output_time_layer = None

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=self.dtype):
      # Encoder
      if hparams.language_model:  # no encoder for language modeling
        utils.print_out("  language modeling: no encoder")
        self.encoder_outputs = None
        encoder_state = None
      else:
        self.encoder_outputs, encoder_state = self._build_encoder(hparams)

      self.mu, self.sigma, self.latent_vector_q_min = self.sampling(self.encoder_outputs, name='encoder_q')

      distribution_a = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
      # print(type(encoder_state), encoder_state)
      # print(type(self.latent_vector_q_min), self.latent_vector_q_min)
      # print('-----------------------------------')

      self.test_encoder_state = encoder_state

      decoder_initial_state_input = encoder_state
      if self.initial_state_flag:
        decoder_initial_state_input = self.latent_vector_q_min

      #TODO: latent vector to be generated in c,t pairs of layer size

      # Skip decoder if extracting only encoder layers
      if self.extract_encoder_layers:
        return

      ## Decoder
      logits, decoder_cell_outputs, _, \
      mark_sample_id, _, \
      final_context_state, self.decoder_outputs, self.decoder_outputs_d  = (
          self._build_decoder(self.encoder_outputs, encoder_state, hparams))

      #---- Computing negative log joint distribution (train and infer) ----#
      neg_ln_joint_distribution_train, neg_ln_joint_distribution_infer, gap_pred, time_pred \
              = self.f_star(self.encoder_outputs, self.decoder_outputs, self.latent_vector_q_min)

      neg_ln_joint_distribution_infer = tf.reduce_sum(neg_ln_joint_distribution_infer, axis=0 if self.time_major else 1)
      self.time_optimizer = tf.train.AdamOptimizer(self.infer_learning_rate).minimize(neg_ln_joint_distribution_infer, var_list=[gap_pred])
      self.gap_pred = gap_pred

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:

        self.mu_d, self.sigma_d, self.latent_vector_q_dot = self.sampling(self.decoder_outputs_d, name='decoder_q')

        # distribution_b = tf.distributions.Normal(tf.zeros_like(self.mu), tf.ones_like(self.sigma))

        distribution_b = tf.distributions.Normal(loc=self.mu_d, scale=self.sigma_d)

        # self.decoder_outputs = tf.clip_by_value(self.decoder_outputs, 1e-8, 1 - 1e-8)
        # KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1, 1)
        # KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + self.sigma - tf.log(1e-8 + self.sigma) - 1, 1)

        KL_divergence = tf.distributions.kl_divergence(distribution_a, distribution_b)
        KL_divergence = tf.reduce_sum(KL_divergence, 1)
        self.KL_divergence = tf.reduce_mean(KL_divergence)

        
        with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                   self.num_gpus)):
          mark_loss, time_loss = self._compute_loss(logits, decoder_cell_outputs, time_pred, neg_ln_joint_distribution_train)
          mark_loss = mark_loss + self.KL_divergence

      else:
        
        self.mu_d, self.sigma_d, self.latent_vector_q_dot = self.sampling(self.decoder_outputs, name='decoder_q')        

        mark_loss, time_loss = tf.constant(0.0), tf.constant(0.0)

      #TODO: KL Divergence should be returned

      #Based on --decode_time and --decode_mark
      # self.ELBO = mark_loss + time_loss - self.KL_divergence
      # self.loss = -self.ELBO

      return logits, mark_loss, time_loss, final_context_state, mark_sample_id, time_pred

  def _build_encoder_from_sequence(self, hparams, sequence_mark, sequence_time, sequence_length):
    """Build an encoder from a sequence.

    Args:
      hparams: hyperparameters.
      sequence_mark: tensor with input mark sequence data.
      sequence_time: tensor with input time sequence data.
      sequence_length: tensor with length of the input sequence.

    Returns:
      encoder_outputs: RNN encoder outputs.
      encoder_state: RNN encoder state.

    Raises:
      ValueError: if encoder_type is neither "uni" nor "bi".
    """
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers

    if self.time_major:
      sequence_mark = tf.transpose(sequence_mark)
      sequence_time = tf.transpose(sequence_time)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype

      self.encoder_emb_inp = self.encoder_emb_lookup_fn(
          self.embedding_encoder, sequence_mark)

      # Encoder_outputs: [max_time, batch_size, num_units]
      rnn_input = tf.concat([self.encoder_emb_inp, tf.expand_dims(sequence_time, axis=2)], axis=2)
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell(hparams, num_layers,
                                        num_residual_layers)

        #TODO: rnn_input should be given in each step.
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            #self.encoder_emb_inp,
            rnn_input,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=self.time_major,
            swap_memory=True)

        # print(encoder_outputs, 'encoder_outputs')
        # print(encoder_state, 'encoder_state')
        # print('#########################')

      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                #inputs=self.encoder_emb_inp,
                inputs=rnn_input,
                sequence_length=sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    # Use the top layer for now
    self.encoder_state_list = [encoder_outputs]
    
    return encoder_outputs, encoder_state

  def _build_encoder(self, hparams):
    """Build encoder from source."""
    utils.print_out("# Build a basic encoder")
    return self._build_encoder_from_sequence(hparams,
                                             self.iterator.source_mark,
                                             self.iterator.source_time,
                                             self.iterator.source_sequence_length)

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major,
        swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

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
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)
    iterator = self.iterator

    # maximum_iteration: The maximum decoding steps.
    maximum_iterations = self._get_infer_maximum_iterations(
        hparams, iterator.source_sequence_length)

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:

      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)

      cell_d, decoder_initial_state_d = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)

      #TODO: decoder_initial_state should be 128/129 depending on consider_time value
      # consider_time = True if self.decode_time else False
      # consider_mark = True if self.decode_mark else False
      
      consider_time = False
      consider_mark = True
      consider_time_d = True
      consider_mark_d = True

      # Optional ops depends on which mode we are in and which loss function we
      # are using.
      logits = tf.no_op()
      decoder_cell_outputs = None
      latent_vector_out_q = None

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
        if consider_mark:
            mark_helper = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, iterator.target_sequence_length,
                time_major=self.time_major)
        else: mark_helper = None

        if consider_time:
            time_helper = my_helper.TimeTrainingHelper(
                target_time_input, iterator.target_sequence_length,
                time_major=self.time_major)
        else: time_helper = None
        
        if consider_mark_d:
            mark_helper_d = tf.contrib.seq2seq.TrainingHelper(
                decoder_emb_inp, iterator.target_sequence_length,
                time_major=self.time_major)
        else: mark_helper_d = None

        if consider_time_d:
            time_helper_d = my_helper.TimeTrainingHelper(
                target_time_input, iterator.target_sequence_length,
                time_major=self.time_major)
        else: time_helper_d = None

        print(decoder_emb_inp.get_shape(), target_time_input.get_shape())

        # print('mark_helper', tf.shape(mark_helper))
        # print('time_helper', tf.shape(time_helper))

        # Decoder

        my_decoder = my_basic_decoder.MyBasicDecoder(
            cell,
            decoder_initial_state,
            mark_helper, time_helper,
            consider_time=consider_time,
            consider_mark=consider_mark)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        # d_Decoder
        my_decoder_d = my_basic_decoder.MyBasicDecoder(
            cell_d,
            decoder_initial_state_d,
            mark_helper_d, time_helper_d, 
            consider_time=consider_time_d,
            consider_mark=consider_mark_d)

        # d_Dynamic decoding
        outputs_d, final_context_state_d, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder_d,
            output_time_major=self.time_major,
            swap_memory=True
            )

        latent_vector_out_q = outputs_d.cell_output

        #TODO: do check this
        mark_sample_id = outputs.mark_sample_id# if self.decode_mark else outputs.time_sample_val
        time_sample_val = outputs.time_sample_val# if self.decode_time else outputs.mark_sample_id

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
          logits = self.output_mark_layer(outputs.mark_rnn_output)# if self.decode_mark else None
          time_pred = self.output_time_layer(outputs.time_rnn_output)# if self.decode_time else None

        if self.num_sampled_softmax > 0:
          logits = tf.no_op()  # unused when using sampled softmax loss.


      ## Inference
      else:
        infer_mode = hparams.infer_mode
        # mark_start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        mark_start_tokens = iterator.source_mark[:self.batch_size, -1]
        # time_start_tokens = tf.fill([self.batch_size], 0.0)
        time_start_tokens = iterator.source_mark[:self.batch_size, -1]
        # mark_start_tokens = tf.to_float(mark_start_tokens)
        time_start_tokens = tf.to_float(time_start_tokens)



        print('Start and End Tokens', mark_start_tokens, time_start_tokens)

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
          if consider_mark:
            mark_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                  self.embedding_decoder, mark_start_tokens, mark_end_token)
          else: mark_helper = None
          if consider_time:
            time_helper = my_helper.TimeGreedyHelper(
                  time_start_tokens, time_end_token)
          else: time_helper = None

          if consider_mark:
            mark_helper_d = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                  self.embedding_decoder, mark_start_tokens, mark_end_token)
          else: mark_helper_d = None

          if consider_time:
            time_helper_d = my_helper.TimeGreedyHelper(
                  time_start_tokens, time_end_token)
          else: time_helper_d = None

        else:
          raise ValueError("Unknown infer_mode '%s'", infer_mode)

        if infer_mode != "beam_search":
          my_decoder = my_basic_decoder.MyBasicDecoder(
              cell,
              decoder_initial_state,
              mark_helper, time_helper,
              output_mark_layer=self.output_mark_layer,# if self.decode_mark else None,  # applied per timestep
              output_time_layer=self.output_time_layer,# if self.decode_time else None,  # applied per timestep
              consider_time=consider_time,
              consider_mark=consider_mark
          )

          # d_Decoder
          my_decoder_d = my_basic_decoder.MyBasicDecoder(
              cell_d,
              decoder_initial_state_d,
              mark_helper_d, time_helper_d,
              consider_time=consider_time,
              consider_mark=consider_mark)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        # d_Dynamic decoding
        outputs_d, final_context_state_d, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder_d,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True
            )

        if infer_mode == "beam_search":
          sample_id = outputs.predicted_ids
        else:
          # if self.decode_mark:
          logits = outputs.mark_rnn_output
          mark_sample_id = outputs.mark_sample_id
          # else: logits, mark_sample_id = tf.no_op(), tf.no_op()
          # if self.decode_time:
          time_pred = outputs.time_rnn_output
          time_sample_val = outputs.time_sample_val
          # else: time_pred, time_sample_val = tf.no_op(), tf.no_op()

    return logits, decoder_cell_outputs, time_pred, mark_sample_id, time_sample_val, final_context_state, outputs.cell_output, latent_vector_out_q

  def sampling(self, rnn_outputs, name=""):
    #TODO: Need Help for dimension

    # latent_vector_out = rnn_outputs[-1]
    latent_vector_out = rnn_outputs[-1, :, :] if self.time_major else rnn_outputs[:, -1, :]

    # print(latent_vector_out)
    # print(tf.shape(latent_vector_out))
    # print('---------------------------------')

    if self.padder:
      # W_mu = tf.Variable(tf.random_normal([128, 128]))
      # W_std = tf.Variable(tf.random_normal([128, 128]))
      # b_mu = tf.Variable(tf.random_normal([128]))
      # b_std = tf.Variable(tf.random_normal([128]))

      #TODO: Check why adding bias not converges.
      # mean = tf.matmul(latent_vector_out, W_mu)
      # stddev = 1e-6 + tf.nn.softplus(tf.matmul(latent_vector_out, W_std))

      mean = tf.layers.dense(latent_vector_out, self.num_units)
      stddev = 1e-6 + tf.nn.softplus(tf.layers.dense(latent_vector_out, self.num_units))

    else:
      self.z_dim = latent_vector_out.get_shape()
      print(self.z_dim)
      self.z_dim = self.z_dim[1]//2
      # The mean parameter is unconstrained
      mean = latent_vector_out[:, :self.z_dim]
      # The standard deviation must be positive.
      # Parameterize with a softplus and add a small epsilon for numerical stability
      stddev = 1e-6 + tf.nn.softplus(latent_vector_out[:, self.z_dim:])

    latent_vector = tf.add (mean , tf.matmul(stddev , tf.random_normal([stddev.get_shape().as_list()[1], stddev.get_shape().as_list()[1]], mean=0., stddev=1., dtype=tf.float32)), name=name)

    return mean, stddev, latent_vector

  def f_star(self, encoder_outputs=None, decoder_outputs=None, latent_z_hat=None):
    print('Inside f_star.........', self.mode, self.batch_size)
    if self.time_major:
      pred_shape = (self.tgt_max_len_infer, self.batch_size)
    else:
      pred_shape = (self.batch_size, self.tgt_max_len_infer)
    gap_pred = tf.get_variable('gap_pred', initializer=tf.random_normal(pred_shape))
    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      target_time_output = self.iterator.target_time_output
      target_time_output = tf.transpose(target_time_output) if self.time_major else target_time_output
    else:
      target_time_output = tf.zeros_like(gap_pred) #In infer mode, true target is irrelevant

    encoder_outputs_ph = self.encoder_outputs_ph = tf.placeholder(tf.float32, shape=[None, None, self.num_units])
    decoder_outputs_ph = self.decoder_outputs_ph = tf.placeholder(tf.float32, shape=[None, None, self.num_units])

    lambda_0_layer = tf.layers.Dense(1, name='lambda_layer')
    w_layer = tf.layers.Dense(1, name='w_layer')
    gamma_layer = tf.layers.Dense(1, name='gamma_layer')

    latent_z_hat = tf.expand_dims(latent_z_hat, 0 if self.time_major else 1)

    #---- Computing joint likelihood for inference ----#
    h_m = encoder_outputs_ph[-1:, :, :] if self.time_major else encoder_outputs_ph[:, -1:, :]
    dec_len = tf.shape(decoder_outputs_ph)[0] if self.time_major else tf.shape(decoder_outputs_ph)[1]
    h_m = tf.tile(h_m, [dec_len, 1, 1] if self.time_major else [1, dec_len, 1])
    z_hat = tf.tile(latent_z_hat, [dec_len, 1, 1] if self.time_major else [1, dec_len, 1])
    inputs = tf.concat([h_m, decoder_outputs_ph, z_hat], axis=-1)
    # inputs = tf.concat([h_m, decoder_outputs_ph], axis=-1)
    self.lambda_0 = tf.squeeze(lambda_0_layer(inputs), axis=-1)
    self.w = tf.squeeze(w_layer(inputs), axis=-1)
    self.gamma = tf.squeeze(gamma_layer(inputs), axis=-1)


    #lambda_star = tf.exp(tf.minimum(10.0, self.lambda_0 + self.w * gap_pred))
    ln_lambda_star = self.lambda_0 + self.w * gap_pred
    lambda_star = tf.exp(tf.minimum(10.0, ln_lambda_star))
    neg_ln_joint_distribution_infer = (-1.0) \
                              * (ln_lambda_star \
                              + (1.0/self.w) * tf.exp(tf.minimum(10.0, self.lambda_0)) \
                              - (1.0/self.w) * lambda_star)

    #---- Computing joint likelihood on true outputs ----#
    h_m = encoder_outputs[-1:, :, :] if self.time_major else encoder_outputs[:, -1:, :]
    dec_len = tf.shape(decoder_outputs)[0] if self.time_major else tf.shape(decoder_outputs)[1]
    h_m = tf.tile(h_m, [dec_len, 1, 1] if self.time_major else [1, dec_len, 1])
    z_hat = tf.tile(latent_z_hat, [dec_len, 1, 1] if self.time_major else [1, dec_len, 1])
    inputs = tf.concat([h_m, decoder_outputs, z_hat], axis=-1)
    # print('inputs', inputs)
    # inputs = tf.concat([h_m, decoder_outputs], axis=-1)
    self.lambda_0 = tf.squeeze(lambda_0_layer(inputs), axis=-1)
    self.w = tf.squeeze(w_layer(inputs), axis=-1)
    self.gamma = tf.squeeze(gamma_layer(inputs), axis=-1)

    last_time_input = self.iterator.source_time[:self.batch_size, -1:]
    last_time_input = tf.transpose(last_time_input) if self.time_major else last_time_input
    target_time_output_concat = tf.concat([last_time_input, target_time_output], axis=0 if self.time_major else 1)
    if self.time_major:
      target_gap = target_time_output_concat[1:, :] - target_time_output_concat[:-1, :]
    else:
      target_gap = target_time_output_concat[:, 1:] - target_time_output_concat[:, :-1]
    #target_gap = tf.Print(target_gap, [target_gap], message='Printing target_gap')

    ln_lambda_star = self.lambda_0 + self.w * target_gap
    lambda_star = tf.exp(tf.minimum(10.0, ln_lambda_star))
    neg_ln_joint_distribution_train = (-1.0) \
                              * (ln_lambda_star \
                              + (1.0/self.w) * tf.exp(tf.minimum(10.0, self.lambda_0))
                              - (1.0/self.w) * lambda_star)
    neg_ln_joint_distribution_train = tf.maximum(neg_ln_joint_distribution_train, tf.log(-1e-6))

    # f_star_val = tf.exp(tf.minimum(10.0, (-1.0) * neg_ln_joint_distribution_infer))

    #---- Obtain predicted time from gaps ----#
    gap_pred_pos = tf.nn.softplus(gap_pred)
    time_pred = last_time_input + tf.cumsum(gap_pred_pos, axis=0 if self.time_major else 1)

    return neg_ln_joint_distribution_train, neg_ln_joint_distribution_infer, gap_pred, time_pred

  def quad_func(self, t, f_star_time):
    """This is the t * f(t) function calculating the mean time to next event,
    given c, w."""
    return t * f_star_time

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    output_tuple = InferOutputTuple(infer_logits=self.infer_logits,
                                    infer_time=self.infer_time,
                                    infer_summary=self.infer_summary,
                                    mark_sample_id=self.mark_sample_id,
                                    time_sample_val=self.time_sample_val,
                                    sample_words=self.sample_words,
                                    sample_times=self.sample_times)

    output_tuple_ret, encoder_outputs_ret, decoder_outputs_ret = sess.run([output_tuple,
                                                                           self.encoder_outputs,
                                                                           self.decoder_outputs])
    

    # encoder_outputs_ret, decoder_outputs_ret = sess.run([self.encoder_outputs,
    #                                                     self.decoder_outputs])

    # output_tuple_ret = sess.run(output_tuple,
    #                            feed_dict={self.encoder_outputs_ph:encoder_outputs_ret,
    #                            self.decoder_outputs_ph:decoder_outputs_ret})

    output_tuple_ret = output_tuple_ret._replace(infer_time=self.infer_time)

    # sess.run(tf.initialize_variables([self.gap_pred]))
    # for i in range(10):
    #     try:
    #       sess.run(self.time_optimizer, feed_dict={self.encoder_outputs_ph:encoder_outputs_ret,
    #                                              self.decoder_outputs_ph:decoder_outputs_ret})
    #     except:
    #       print('An exception occured')
    #print(output_tuple_ret.sample_words.shape, output_tuple_ret.sample_times.shape)



    # ans = []
    # for time_f_star in output_tuple_ret[6]:
    #   lst = []
    #   for f_star_time in time_f_star:
    #     args = (f_star_time)
    #     val, _err = quad(self.quad_func, 0, np.inf, args=args)
    #     lst.append(val)
    #   ans.append(lst)

    # output_tuple_ret = output_tuple_ret._replace(time_sample_val=np.array(ans))
    # output_tuple_ret = output_tuple_ret._replace(sample_times=np.array(ans))

    return output_tuple_ret
  

  def _softmax_cross_entropy_loss(
      self, logits, decoder_cell_outputs, labels):
    """Compute softmax loss or sampled softmax loss."""
    if self.num_sampled_softmax > 0:

      is_sequence = (decoder_cell_outputs.shape.ndims == 3)

      if is_sequence:
        labels = tf.reshape(labels, [-1, 1])
        inputs = tf.reshape(decoder_cell_outputs, [-1, self.num_units])

      crossent = tf.nn.sampled_softmax_loss(
          weights=tf.transpose(self.output_mark_layer.kernel),
          biases=self.output_mark_layer.bias or tf.zeros([self.tgt_vocab_size]),
          labels=labels,
          inputs=inputs,
          num_sampled=self.num_sampled_softmax,
          num_classes=self.tgt_vocab_size,
          partition_strategy="div",
          seed=self.random_seed)

      if is_sequence:
        if self.time_major:
          crossent = tf.reshape(crossent, [-1, self.batch_size])
        else:
          crossent = tf.reshape(crossent, [self.batch_size, -1])

    else:
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)

    return crossent

  def _time_loss(self, time_pred, target_time_output):
    return tf.squared_difference(target_time_output, tf.squeeze(time_pred, axis=-1))

  def _compute_loss(self, logits, decoder_cell_outputs, time_pred, neg_ln_joint_distribution):
    """Compute optimization loss."""
    target_mark_output = self.iterator.target_mark_output
    target_time_output = self.iterator.target_time_output
    if self.time_major:
      target_mark_output = tf.transpose(target_mark_output)
      target_time_output = tf.transpose(target_time_output)
    max_time = self.get_max_time(target_mark_output)

    crossent = self._softmax_cross_entropy_loss(
        logits, decoder_cell_outputs, target_mark_output)
    timeloss = neg_ln_joint_distribution

    target_weights = tf.sequence_mask(
        self.iterator.target_sequence_length, max_time, dtype=self.dtype)
    if self.time_major:
      target_weights = tf.transpose(target_weights)

    mark_loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.batch_size)
    time_loss = tf.reduce_sum(
            timeloss * target_weights) / tf.to_float(self.batch_size)

    return mark_loss, time_loss

  def decode(self, sess):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    output_tuple = self.infer(sess)
    sample_words = output_tuple.sample_words
    sample_times = output_tuple.sample_times
    infer_summary = output_tuple.infer_summary

    # make sure outputs is of shape [batch_size, time] or [beam_width,
    # batch_size, time] when using beam search.
    if self.time_major:
      sample_words = sample_words.transpose()
      sample_times = sample_times.transpose()
    elif sample_words.ndim == 3:
      # beam search output in [batch_size, time, beam_width] shape.
      sample_words = sample_words.transpose([2, 0, 1])
    return sample_words, sample_times, infer_summary

  def _set_train_or_infer(self, res, reverse_target_vocab_table, hparams):
    """Set up training and inference."""
    res_1, res_2 = res[1], res[2]
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_mark_loss, self.train_time_loss = res_1, res_2
      self.train_loss = res_1 + res_2
      self.word_count = tf.reduce_sum(
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_mark_loss, self.eval_time_loss = res_1, res_2
      self.eval_loss = res_1 + res_2
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, _, self.final_context_state, \
      self.mark_sample_id, self.infer_time = res
      self.sample_words = reverse_target_vocab_table.lookup(
          tf.to_int64(self.mark_sample_id))
      self.time_sample_val = self.infer_time   # These variables are used to ensure
      self.sample_times = self.time_sample_val # analogy between mark and time

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(
          self.iterator.target_sequence_length)

    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrange for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)
      else:
        raise ValueError("Unknown optimizer type %s" % hparams.optimizer)

      # Gradients
      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

      clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.grad_norm_summary = grad_norm_summary
      self.grad_norm = grad_norm

      self.update = opt.apply_gradients(
          zip(clipped_grads, params), global_step=self.global_step)

      # Summary
      self.train_summary = self._get_train_summary()
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Print trainable variables
    utils.print_out("# Trainable variables")
    utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
    for param in params:
      utils.print_out(" %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))
