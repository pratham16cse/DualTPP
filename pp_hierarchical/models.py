from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import ipdb
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import reader_rmtpp


one_by = tf.math.reciprocal_no_nan
ETH = 10.0

class InverseTransformSampling(layers.Layer):
  """Uses (D, WT) to sample E[f*(g)], expected gap before next event."""

  def call(self, inputs):
    D, WT = inputs
    u = tf.ones_like(D) * tf.range(0.0, 1.0, 1/500)
    c = -tf.exp(D)
    val = one_by(WT) * tf.math.log(WT * one_by(c) * tf.math.log(1.0 - u) + 1)
    val = tf.reduce_mean(val, axis=-1, keepdims=True)
    return val

class NegativeLogLikelihood(tf.keras.losses.Loss):
    def __init__(self, D, WT,
                 reduction=keras.losses.Reduction.AUTO,
                 name='negative_log_likelihood'):
        super(NegativeLogLikelihood, self).__init__(reduction=reduction,
                                                    name=name)

        self.D = D
        self.WT = WT

    def call(self, gaps_true, gaps_pred):
        log_lambda_ = (self.D + (gaps_true * self.WT))
        lambda_ = tf.exp(tf.minimum(ETH, log_lambda_), name='lambda_')
        log_f_star = (log_lambda_
                      + one_by(self.WT) * tf.exp(tf.minimum(ETH, self.D))
                      - one_by(self.WT) * lambda_)

        return -log_f_star

class RMTPP(tf.keras.Model):
    def __init__(self, num_categories,
                 embed_size,
                 hidden_layer_size,
                 name='RMTPP',
                 use_marks=True,
                 use_intensity=True,
                 use_time_embed=True,
                 **kwargs):
        super(RMTPP, self).__init__(name=name, **kwargs)
        self.use_marks = use_marks
        self.use_intensity = use_intensity
        self.use_time_embed = use_time_embed
        time_embed_size = 10

        if self.use_marks:
            self.embedding_layer = layers.Embedding(num_categories+1, embed_size,
                                                    mask_zero=False,
                                                    name='marks_embedding')
        if self.use_time_embed:
            self.time_feature_embedding_layer = layers.Embedding(25, time_embed_size,
                                                    mask_zero=False,
                                                    name='time_feature_embedding')
        else:
            self.time_features_layer = layers.Dense(time_embed_size, 
                                                    activation='sigmoid',
                                                    name='time_features_layer')

        self.rnn_layer = layers.GRU(hidden_layer_size, return_sequences=True,
                                    return_state=True, stateful=True,
                                    name='GRU_Layer')
        self.D_layer = layers.Dense(1, name='D_layer')
        if self.use_marks:
            self.marks_output_layer = layers.Dense(num_categories,
                                                   activation='softmax',
                                                   name='marks_output_layer')
        if self.use_intensity:
            self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')
            self.gaps_output_layer = InverseTransformSampling()

    def process_inputs(self, gaps, mask, time_features, marks=None):
        if gaps.ndim == 2:
            gaps = tf.expand_dims(gaps, axis=1)
        if mask.ndim == 1:
            mask = tf.expand_dims(mask, axis=1)
        if time_features.ndim == 2:
            time_features = tf.expand_dims(time_features, axis=1)
        if marks is not None and marks.ndim == 2:
            marks = tf.expand_dims(marks, axis=1)

        return gaps, mask, time_features, marks

    def process_output(self, marks_logits, gaps_pred, D, WT):
        if marks_logits is not None and marks_logits.shape[1] == 1:
            marks_logits = tf.squeeze(marks_logits, axis=1)
        if gaps_pred.shape[1] == 1:
            gaps_pred = tf.squeeze(gaps_pred, axis=1)
        if D.shape[1] == 1:
            D = tf.squeeze(D, axis=1)
        if WT.shape[1] == 1:
            WT = tf.squeeze(WT, axis=1)

        return marks_logits, gaps_pred, D, WT

    def call(self, gaps, mask, time_features, marks=None, initial_state=None):
        ''' Forward pass of the RMTPP model'''

        self.gaps = gaps
        self.mask = mask
        self.time_features = time_features
        self.marks = marks
        self.initial_state = initial_state
        self.gaps, self.mask, self.time_features, self.marks \
                = self.process_inputs(self.gaps, self.mask, self.time_features,
                                      marks=self.marks)
        # Gather input for the rnn
        if self.use_marks:
            self.marks_embd = self.embedding_layer(self.marks)

        if self.use_time_embed:
            self.time_features = tf.round(self.time_features)
            self.time_features = tf.squeeze(self.time_features, axis=2)
            self.time_features = self.time_feature_embedding_layer(self.time_features)
        else:
            self.time_features = self.time_features_layer(self.time_features)
        if self.use_marks:
            rnn_inputs = tf.concat([self.marks_embd, self.gaps, self.time_features], axis=-1)
        else:
            rnn_inputs = tf.concat([self.gaps, self.time_features], axis=-1)

        self.hidden_states, self.final_state \
                = self.rnn_layer(rnn_inputs,
                                 initial_state=self.initial_state,
                                 mask=self.mask)

        # Generate D, WT, and gaps_pred
        self.D = self.D_layer(self.hidden_states)

        if self.use_marks:
            self.marks_logits = self.marks_output_layer(self.hidden_states)
        else:
            self.marks_logits = None
        if self.use_intensity:
            self.WT = self.WT_layer(self.hidden_states)
            self.gaps_pred = self.gaps_output_layer((self.D, self.WT))
        else:
            self.gaps_pred = tf.nn.softplus(self.D)
            self.WT = tf.zeros_like(self.D)

        # Apply mask on outputs
        if self.use_marks:
            self.marks_logits = self.marks_logits * self.mask
        self.gaps_pred = self.gaps_pred * tf.expand_dims(self.mask, axis=-1)
        self.D = self.D * tf.expand_dims(self.mask, axis=-1)
        if self.use_intensity:
            self.WT = self.WT * tf.expand_dims(self.mask, axis=-1)

        self.marks_logits, self.gaps_pred, self.D, self.WT \
                = self.process_output(self.marks_logits, self.gaps_pred,
                                      self.D, self.WT)

        return self.marks_logits, self.gaps_pred, self.D, self.WT


class FeedForward(layers.Layer):
    def __init__(self,
                 hidden_layer_size,
                 name='FeedForward',
                 **kwargs):
        super(FeedForward, self).__init__(name=name, **kwargs)
        self.st_l1 = layers.Dense(hidden_layer_size,
                                 activation=tf.sigmoid,
                                 name='st_l1')
        self.st_l2 = layers.Dense(hidden_layer_size,
                                 activation=tf.sigmoid,
                                 name='st_l2')
    def call(self, inputs):
        st_1 = self.st_l1(inputs)
        st_2 = self.st_l2(st_1)

        return st_2 #TODO Play around with this network

class HierarchicalRNN(tf.keras.Model):
    def __init__(self,
                 num_categories,
                 embed_size,
                 hidden_layer_size,
                 name='HierarchicalRNN',
                 use_marks=True,
                 use_intensity=True,
                 use_time_embed=True,
                 **kwargs):
        super(HierarchicalRNN, self).__init__(name=name, **kwargs)
        self.use_marks = use_marks
        self.use_intensity = use_intensity
        self.l1_rnn = RMTPP(num_categories, embed_size, hidden_layer_size,
                            use_marks=use_marks,
                            use_intensity=use_intensity,
                            use_time_embed=use_time_embed)
        self.l2_rnn = RMTPP(num_categories, embed_size, hidden_layer_size,
                            use_marks=use_marks,
                            use_intensity=use_intensity,
                            use_time_embed=use_time_embed)
        self.ff = FeedForward(hidden_layer_size)

    def call(self, l2_gaps, l2_mask, l2_time_features,
             l1_gaps=None, l1_mask=None, l1_time_features=None,
             l2_marks=None, l1_marks=None, debug=False):

        # Gather input for the rnn
        if self.use_marks:
            self.l1_marks_embd = self.l1_rnn.embedding_layer(l1_marks)
        self.l1_gaps = l1_gaps
        if self.use_marks:
            l1_rnn_inputs = tf.concat([self.l1_marks_embd, self.l1_gaps], axis=-1)
        else:
            l1_rnn_inputs = self.l1_gaps

        # Forward pass for the compound-event rnn
        self.l2_marks_logits, self.l2_gaps_pred, self.l2_D, self.l2_WT \
                 = self.l2_rnn(l2_gaps, l2_mask, l2_time_features, marks=l2_marks)

        # Transform compound-hidden-states using FF
        self.state_transformed = self.ff(self.l2_rnn.hidden_states)

        # For each transformed compound-hidden-state,
        # predict next decoder_length simple events.
        if l1_gaps is not None:
            self.l1_D, self.l1_WT = list(), list()
            self.l1_marks_logits, self.l1_gaps_pred = list(), list()
            l1_mask = tf.tile(tf.expand_dims(l1_mask, axis=-1), [1, 1, l1_gaps.shape[2]])
            for idx in range(self.state_transformed.shape[1]):
                l1_m_logits, l1_g_pred, l1_D, l1_WT \
                        = self.l1_rnn(l1_gaps[:, idx],
                                      l1_mask[:, idx],
                                      l1_time_features[:, idx],
                                      initial_state=self.state_transformed[:, idx])
                self.l1_D.append(l1_D)
                self.l1_WT.append(l1_WT)
                self.l1_marks_logits.append(l1_m_logits)
                self.l1_gaps_pred.append(l1_g_pred)
                self.l1_rnn.reset_states() #TODO What happens if this line is removed?

            self.l1_D = tf.stack(self.l1_D, axis=1)
            self.l1_WT = tf.stack(self.l1_WT, axis=1)
            self.l1_gaps_pred = tf.stack(self.l1_gaps_pred, axis=1)
            if self.use_marks:
                self.l1_marks_logits = tf.stack(self.l1_marks_logits, axis=1)
            else:
                self.l1_marks_logits = None
        else:
            self.l1_marks_logits, self.l1_gaps_pred = None, None
            self.l2_D, self.l2_WT = None, None

        return (self.l2_marks_logits, self.l2_gaps_pred, self.l2_D, self.l2_WT,
                self.l1_marks_logits, self.l1_gaps_pred, self.l1_D, self.l1_WT)

    def reset_states(self):
        self.l1_rnn.reset_states()
        self.l2_rnn.reset_states()

class SimulateRMTPP:
    def simulate(self, model, times, gaps_in, block_begin_ts, t_b_plus,
                 decoder_length, normalizers, marks_in=None):
    
        marks_logits, gaps_pred = list(), list()
        normalizer_d, normalizer_a = normalizers
        normalizer_d = tf.squeeze(normalizer_d, axis=1)
        normalizer_a = tf.squeeze(normalizer_a, axis=1)
        if gaps_in.shape[1] == 1:
            gaps_in = tf.squeeze(gaps_in, axis=-1)
        if times.shape[1] == 1:
            times = tf.squeeze(times, axis=-1)
        times_pred = list()
        last_gaps_pred_unnorm = (gaps_in - normalizer_a) * normalizer_d
        last_times_pred = times + last_gaps_pred_unnorm
        times_pred.append(last_times_pred)
        time_features = reader_rmtpp.get_time_features_for_data((last_times_pred))
        N = len(gaps_in)
        pred_idxes = -1.0 * np.ones(N)
        begin_idxes, end_idxes = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
        simul_step = 0
        mask = np.squeeze(np.ones_like(gaps_in), axis=-1)
        while any(times_pred[-1]<t_b_plus) or any(pred_idxes<decoder_length):

            step_marks_logits, step_gaps_pred, _, _ \
                    = model(gaps_in, tf.constant(mask),
                            time_features,
                            marks=marks_in)

            if marks_in is not None:
                step_marks_pred = tf.argmax(step_marks_logits, axis=-1) + 1
            else:
                step_marks_pred = None
    
            #print('Simul step:', simul_step, tf.squeeze(tf.squeeze(step_gaps_pred, axis=1), axis=-1))
            marks_in, gaps_in = step_marks_pred, step_gaps_pred
            marks_logits.append(step_marks_logits)
            last_gaps_pred_unnorm = (step_gaps_pred - normalizer_a) * normalizer_d
            last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
            gaps_pred.append(last_gaps_pred_unnorm)
            times_pred.append(last_times_pred)
            time_features = reader_rmtpp.get_time_features_for_data((last_times_pred))

            for ex_id, (t_pred, t_b) in enumerate(zip(times_pred[-1], t_b_plus)):
                if t_pred > t_b:
                    pred_idxes[ex_id] += 1
                    if pred_idxes[ex_id] == 0:
                        begin_idxes[ex_id] = simul_step
                    if pred_idxes[ex_id] == decoder_length:
                        end_idxes[ex_id] = simul_step
                        mask[ex_id] = 0.
    
            simul_step += 1
    
        if marks_in is not None:
            marks_logits = tf.squeeze(tf.stack(marks_logits, axis=1), axis=2)
            marks_logits = [m_l[b_idx:e_idx] for m_l, b_idx, e_idx in \
                                zip(marks_logits, begin_idxes, end_idxes)]
            marks_logits = tf.stack(marks_logits, axis=0)
        gaps_pred = tf.stack(gaps_pred, axis=1)
        gaps_pred = [t_l[b_idx:e_idx] for t_l, b_idx, e_idx in \
                            zip(gaps_pred, begin_idxes, end_idxes)]
        gaps_pred = tf.stack(gaps_pred, axis=0)
    
        return marks_logits, gaps_pred


class SimulateHierarchicalRNN:
    def simulate(self, model, c_times_in, c_gaps_pred, c_seq_lens,
                 block_begin_ts,
                 t_b_plus, c_t_b_plus,
                 decoder_length, c_normalizers, normalizers):

        c_normalizer_d, c_normalizer_a = c_normalizers
        c_normalizer_d = tf.squeeze(c_normalizer_d, axis=1)
        c_normalizer_a = tf.squeeze(c_normalizer_a, axis=1)
        normalizer_d, normalizer_a = normalizers
        normalizer_d = tf.squeeze(normalizer_d, axis=1)
        normalizer_a = tf.squeeze(normalizer_a, axis=1)
        # ----- Start: Simulation  Layer 2 RNN ----- #

        l2_hidden_states = list()
        l2_gaps_pred = list()
        l2_times_pred = list()
        # TODO Make sure that second last l2_gaps_pred, l2_times_pred and l2_hidden_states are being tracked by l2_idxes
        N = len(c_gaps_pred)
        l2_idxes = np.ones(N, dtype=int) * -1

        l2_second_last_gaps_pred = tf.gather(c_gaps_pred, c_seq_lens-2, batch_dims=1)
        l2_second_last_gaps_pred = tf.squeeze(l2_second_last_gaps_pred, axis=1)
        l2_second_last_gaps_pred_unnorm = l2_second_last_gaps_pred * c_normalizer_d
        #l2_second_last_times_in = tf.gather(c_times_in, c_seq_lens-2, batch_dims=1)
        #l2_second_last_times_pred = tf.squeeze(l2_second_last_times_in + l2_second_last_gaps_pred_unnorm, axis=-1)
        #l2_times_pred.append(l2_second_last_times_pred)

        l2_last_gaps_pred = tf.gather(c_gaps_pred, c_seq_lens-1, batch_dims=1)
        l2_last_gaps_pred = tf.squeeze(l2_last_gaps_pred, axis=1)
        l2_last_gaps_pred_unnorm = l2_last_gaps_pred * c_normalizer_d
        l2_last_times_in = tf.gather(c_times_in, c_seq_lens-1, batch_dims=1)
        l2_last_times_in = tf.squeeze(l2_last_times_in, axis=1)
        l2_last_times_pred = l2_last_times_in + l2_last_gaps_pred_unnorm
        l2_times_pred.append(l2_last_times_pred)

        l2_hidden_states.append(model.l2_rnn.hidden_states[:, -2])
        l2_gaps_pred.append(l2_second_last_gaps_pred_unnorm)
        l2_gaps_inputs = l2_last_gaps_pred
        simul_step = 0
        l2_mask = np.ones_like(l2_gaps_inputs)

        time_features = reader_rmtpp.get_time_features_for_data((l2_times_pred[-1]))
        while any(l2_times_pred[-1]<c_t_b_plus):

            #print('layer 2 simul_step:', simul_step)

            prev_hidden_state = model.l2_rnn.hidden_states[:, -1]

            #(_, step_l2_gaps_pred, _, _, _, _, _, _) \
            #        = model(None, l2_gaps=l2_gaps_inputs)
            _, step_l2_gaps_pred, _, _ \
                     = model.l2_rnn(l2_gaps_inputs, tf.constant(l2_mask), time_features)

            l2_hidden_states.append(prev_hidden_state)
            l2_gaps_pred.append(l2_gaps_inputs * c_normalizer_d)
            l2_gaps_inputs = step_l2_gaps_pred
            step_l2_gaps_pred_unnorm =  step_l2_gaps_pred * c_normalizer_d
            l2_last_times_pred = l2_times_pred[-1] + step_l2_gaps_pred_unnorm
            l2_times_pred.append(l2_last_times_pred)
            time_features = reader_rmtpp.get_time_features_for_data((l2_times_pred[-1]))

            for ex_id, (l2_t_pred, c_t_b) in enumerate(zip(l2_times_pred[-1], c_t_b_plus)):
                if l2_t_pred >= c_t_b and l2_idxes[ex_id] == -1:
                    l2_idxes[ex_id] = len(l2_hidden_states)-2
                    l2_mask[ex_id] = 0.

            simul_step += 1

        #l2_gaps_pred = tf.stack(l2_gaps_pred, axis=0)
        l2_gaps_pred = tf.concat(l2_gaps_pred, axis=1)
        self.all_l2_gaps_pred = l2_gaps_pred # all predicted gaps from start to end of simulation
        l2_idxes = tf.expand_dims(l2_idxes, axis=-1)
        l2_gaps_pred = tf.gather(l2_gaps_pred, l2_idxes, batch_dims=1)
        all_l2_times_pred = tf.concat(l2_times_pred, axis=1)
        l2_times_pred = tf.gather(all_l2_times_pred, l2_idxes, batch_dims=1)
        #ipdb.set_trace()


        l1_gaps_pred = list()
        l1_times_pred = list()
        l1_times_pred.append(l2_times_pred)
        #ipdb.set_trace()
        l1_begin_idxes, l1_end_idxes =  np.zeros(N, dtype=int), np.zeros(N, dtype=int)
        pred_idxes = -1.0 * np.ones(N)
        simul_step = 0

        #ipdb.set_trace()
        l1_gaps_inputs = l2_gaps_pred / 10.0 #TODO Can we do better here?
        l1_gaps_inputs = l1_gaps_inputs / normalizer_d
        l1_mask = np.ones_like(l1_gaps_inputs)

        # Transform hidden state of layer 2 rnn using ff
        l1_rnn_init_state =  model.ff(l2_hidden_states[-2])

        model.l1_rnn.reset_states()
        while any(l1_times_pred[-1]<t_b_plus) or any(pred_idxes<decoder_length):

            # print('layer 1 simul_step:', simul_step)

            #(_, _, _, _, _, step_l1_gaps_pred, _, _) \
            #        = model(None, l1_gaps=tf.expand_dims(l1_gaps_inputs, axis=-1), debug=False)
            _, step_l1_gaps_pred, _, _ \
                    = model.l1_rnn(l1_gaps_inputs, tf.constant(l1_mask), time_features,
                                  initial_state=l1_rnn_init_state)
            #step_l1_gaps_pred = tf.expand_dims(step_l1_gaps_pred, axis=1)
            #ipdb.set_trace()

            # print('step_l1_gaps_pred', step_l1_gaps_pred)
            step_l1_gaps_pred_unnorm = step_l1_gaps_pred * normalizer_d
            l1_gaps_pred.append(step_l1_gaps_pred_unnorm)
            l1_gaps_inputs = step_l1_gaps_pred
            l1_last_times_pred = l1_times_pred[-1] + step_l1_gaps_pred
            l1_times_pred.append(l1_last_times_pred)
            l1_rnn_init_state = model.l1_rnn.hidden_states[:, -1]

            for ex_id, (l1_t_pred, t_b) in enumerate(zip(l1_times_pred[-1], t_b_plus)):
                if l1_t_pred > t_b:
                    pred_idxes[ex_id] += 1
                    if pred_idxes[ex_id] == 0:
                        l1_begin_idxes[ex_id] = simul_step
                    if pred_idxes[ex_id] == decoder_length:
                        l1_end_idxes[ex_id] = simul_step
                        l1_mask[ex_id] = 0.

            simul_step += 1

        l1_gaps_pred = tf.concat(l1_gaps_pred, axis=1)
        self.all_l1_gaps_pred = l1_gaps_pred
        l1_idxes = tf.expand_dims(tf.constant(l1_begin_idxes, dtype=tf.int32), axis=-1) + tf.expand_dims(tf.range(decoder_length, dtype=tf.int32), axis=0)
        l1_gaps_pred = tf.gather(l1_gaps_pred, l1_idxes, batch_dims=1)
        #ipdb.set_trace()
        #l1_gaps_pred = [t_l[b_idx:e_idx] for t_l, b_idx, e_idx in \
        #                    zip(l1_gaps_pred, l1_begin_idxes, l1_end_idxes)]
        #l1_gaps_pred = tf.stack(l1_gaps_pred, axis=0)
        #l1_gaps_pred = tf.squeeze(l1_gaps_pred, axis=-1)
        #ipdb.set_trace()

        return l1_gaps_pred
