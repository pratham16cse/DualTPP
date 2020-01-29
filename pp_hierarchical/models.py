from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ETH = 10.0

class InverseTransformSampling(layers.Layer):
  """Uses (D, WT) to sample E[f*(g)], expected gap before next event."""

  def call(self, inputs):
    D, WT = inputs
    u = tf.ones_like(D) * tf.range(0.0, 1.0, 1/500)
    c = -tf.exp(D)
    val = (1.0/WT) * tf.math.log((WT/c) * tf.math.log(1.0 - u) + 1)
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
                      + (1.0 / self.WT) * tf.exp(tf.minimum(ETH, self.D))
                      - (1.0 / self.WT) * lambda_)

        return -log_f_star

#class MaskedDense(tf.keras.layers.Layer):
#
#    def call(self, inputs, mask=None):

class RMTPP(tf.keras.Model):
    def __init__(self,
                 num_categories,
                 embed_size,
                 hidden_layer_size,
                 name='RMTPP',
                 use_marks=True,
                 **kwargs):
        super(RMTPP, self).__init__(name=name, **kwargs)
        self.use_marks = use_marks
        if self.use_marks:
            self.embedding_layer = layers.Embedding(num_categories+1, embed_size,
                                                    mask_zero=False,
                                                    name='marks_embedding')
        self.rnn_layer = layers.GRU(hidden_layer_size, return_sequences=True,
                                    return_state=True, stateful=True,
                                    name='GRU_Layer')
        self.D_layer = layers.Dense(1, name='D_layer')
        self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')
        if self.use_marks:
            self.marks_output_layer = layers.Dense(num_categories,
                                                   activation='softmax',
                                                   name='marks_output_layer')
        self.gaps_output_layer = InverseTransformSampling()

    def call(self, gaps, marks=None, initial_state=None):
        if self.use_marks:
            self.marks_embd = self.embedding_layer(marks)
            mask = self.embedding_layer.compute_mask(marks)
        else:
            mask = tf.not_equal(gaps, 0.0)
        self.gaps = gaps
        if self.use_marks:
            rnn_inputs = tf.concat([self.marks_embd, self.gaps], axis=-1)
        else:
            rnn_inputs = self.gaps
        self.hidden_states, self.final_state \
                = self.rnn_layer(rnn_inputs,
                                 initial_state=initial_state,
                                 mask=mask)
        self.D = self.D_layer(self.hidden_states)
        self.WT = self.WT_layer(self.hidden_states)
        if self.use_marks:
            self.marks_logits = self.marks_output_layer(self.hidden_states)
        else:
            self.marks_logits = None
        self.gaps_pred = self.gaps_output_layer((self.D, self.WT))

        return self.marks_logits, self.gaps_pred, self.D, self.WT

class HierarchicalRNN(tf.keras.Model):
    def __init__(self,
                 num_categories,
                 embed_size,
                 hidden_layer_size,
                 name='HierarchicalRNN',
                 use_marks=True,
                 **kwargs):
        super(HierarchicalRNN, self).__init__(name=name, **kwargs)
        self.use_marks = use_marks
        self.l1_rnn = RMTPP(num_categories, embed_size, hidden_layer_size)
        self.l2_rnn = RMTPP(num_categories, embed_size, hidden_layer_size)
        self.state_transform_l1 = layers.Dense(hidden_layer_size,
                                               activation=tf.sigmoid,
                                               name='state_transform_l1')
        self.state_transform_l2 = layers.Dense(hidden_layer_size,
                                               activation=tf.tanh,
                                               name='state_transform_l2')

    def call(self, l1_gaps, l2_gaps, l1_marks=None, l2_marks=None):
        if self.use_marks:
            self.l1_marks_embd = self.l1_rnn.embedding_layer(l1_marks)
            l1_mask = self.l1_rnn.embedding_layer.compute_mask(l1_marks)
        else:
            l1_mask = tf.not_equal(l1_gaps, 0.0)
        self.l1_gaps = l1_gaps
        if self.use_marks:
            l1_rnn_inputs = tf.concat([self.l1_marks_embd, self.l1_gaps], axis=-1)
        else:
            l1_rnn_inputs = self.l1_gaps

        self.l2_marks_logits, self.l2_gaps_pred, self.l2_D, self.l2_WT \
                 = self.l2_rnn(l2_gaps, l2_marks)
        state_transform_1 = self.state_transform_l1(self.l2_rnn.final_state)
        state_transform_2 = self.state_transform_l2(state_transform_1)
        self.l1_marks_logits, self.l1_gaps_pred, self.l1_D, self.l1_WT \
                = self.l1_rnn(l1_gaps, l1_marks, initial_state=state_transform_2)

        return self.l1_marks_logits, self.l1_gaps_pred, self.l1_D, self.l1_WT


def simulate(model, gaps, last_input_ts, t_b_plus, decoder_length, marks=None):
    marks_logits, gaps_pred = list(), list()
    marks_inputs, gaps_inputs = marks, gaps
    cum_gaps_pred = tf.squeeze(gaps, axis=1)
    offset = t_b_plus - tf.squeeze(last_input_ts, axis=1)
    N = len(gaps)
    simul_step = 0
    pred_idxes = -1.0 * np.ones(N)
    begin_idxes, end_idxes = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
    simul_step = 0
    while any(cum_gaps_pred<offset) or any(pred_idxes<decoder_length):
        #print('Simul step:', simul_step)
        step_marks_logits, step_gaps_pred, _, _ = model(gaps_inputs, marks_inputs)
        if marks is not None:
            step_marks_pred = tf.argmax(step_marks_logits, axis=-1) + 1
        else:
            step_marks_pred = None

        marks_logits.append(step_marks_logits)
        gaps_pred.append(step_gaps_pred)
        cum_gaps_pred += tf.squeeze(step_gaps_pred, axis=1)

        marks_inputs, gaps_inputs = step_marks_pred, step_gaps_pred
        simul_step += 1

        for ex_id, (cum_ts, off) in enumerate(zip(cum_gaps_pred, offset)):
            if cum_ts > off:
                pred_idxes[ex_id] += 1
                if pred_idxes[ex_id] == 0:
                    begin_idxes[ex_id] = simul_step
                if pred_idxes[ex_id] == decoder_length:
                    end_idxes[ex_id] = simul_step

    if marks is not None:
        marks_logits = tf.squeeze(tf.stack(marks_logits, axis=1), axis=2)
        marks_logits = [m_l[b_idx:e_idx] for m_l, b_idx, e_idx in \
                            zip(marks_logits, begin_idxes, end_idxes)]
        marks_logits = tf.stack(marks_logits, axis=0)
    gaps_pred = tf.squeeze(tf.stack(gaps_pred, axis=1), axis=2)
    gaps_pred = [t_l[b_idx:e_idx] for t_l, b_idx, e_idx in \
                        zip(gaps_pred, begin_idxes, end_idxes)]
    gaps_pred = tf.stack(gaps_pred, axis=0)

    return marks_logits, gaps_pred
