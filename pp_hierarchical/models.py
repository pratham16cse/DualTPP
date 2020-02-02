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
                 use_intensity=True,
                 **kwargs):
        super(RMTPP, self).__init__(name=name, **kwargs)
        self.use_marks = use_marks
        self.use_intensity = use_intensity

        if self.use_marks:
            self.embedding_layer = layers.Embedding(num_categories+1, embed_size,
                                                    mask_zero=False,
                                                    name='marks_embedding')
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

    def call(self, gaps, mask=None, marks=None, initial_state=None):
        if self.use_marks:
            self.marks_embd = self.embedding_layer(marks)
        #    mask = self.embedding_layer.compute_mask(marks)
        #else:
        #    mask = tf.not_equal(gaps, tf.zeros_like(gaps))
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

        if mask is not None:
            if self.use_marks:
                self.marks_logits = self.marks_logits * mask
            self.gaps_pred = self.gaps_pred * tf.expand_dims(mask, axis=-1)
            self.D = self.D * tf.expand_dims(mask, axis=-1)
            if self.use_intensity:
                self.WT = self.WT * mask

        return self.marks_logits, self.gaps_pred, self.D, self.WT

class HierarchicalRNN(tf.keras.Model):
    def __init__(self,
                 num_categories,
                 embed_size,
                 hidden_layer_size,
                 name='HierarchicalRNN',
                 use_marks=True,
                 use_intensity=True,
                 **kwargs):
        super(HierarchicalRNN, self).__init__(name=name, **kwargs)
        self.use_marks = use_marks
        self.use_intensity = use_intensity
        self.l1_rnn = RMTPP(num_categories, embed_size, hidden_layer_size,
                            use_marks=use_marks,
                            use_intensity=use_intensity)
        self.l2_rnn = RMTPP(num_categories, embed_size, hidden_layer_size,
                            use_marks=use_marks,
                            use_intensity=use_intensity)
        self.state_transform_l1 = layers.Dense(hidden_layer_size,
                                               activation=tf.sigmoid,
                                               name='state_transform_l1')
        self.state_transform_l2 = layers.Dense(hidden_layer_size,
                                               activation=tf.tanh,
                                               name='state_transform_l2')

    def call(self, inputs, l2_gaps=None, l1_gaps=None,
             l2_mask=None, l1_mask=None,
             l2_marks=None, l1_marks=None):
        if self.use_marks:
            self.l1_marks_embd = self.l1_rnn.embedding_layer(l1_marks)
        #TODO Feed mask externally
        #    l1_mask = self.l1_rnn.embedding_layer.compute_mask(l1_marks)
        #else:
        #    l1_mask = tf.not_equal(l1_gaps, tf.zeros_like(l1_gaps))
        self.l1_gaps = l1_gaps
        if self.use_marks:
            l1_rnn_inputs = tf.concat([self.l1_marks_embd, self.l1_gaps], axis=-1)
        else:
            l1_rnn_inputs = self.l1_gaps

        if l2_gaps is not None:
            self.l2_marks_logits, self.l2_gaps_pred, self.l2_D, self.l2_WT \
                     = self.l2_rnn(l2_gaps, mask=l2_mask, marks=l2_marks)

        self.state_transform_1 = self.state_transform_l1(self.l2_rnn.hidden_states)
        self.state_transformed = self.state_transform_l2(self.state_transform_1)

        if l1_gaps is not None:
            self.l1_D, self.l1_WT = list(), list()
            self.l1_marks_logits, self.l1_gaps_pred = list(), list()
            for idx in range(self.state_transformed.shape[1]):
                l1_m_logits, l1_g_pred, l1_D, l1_WT \
                        = self.l1_rnn(l1_gaps[:, idx],
                                      initial_state=self.state_transformed[:, idx])
                self.l1_D.append(l1_D)
                self.l1_WT.append(l1_WT)
                self.l1_marks_logits.append(l1_m_logits)
                self.l1_gaps_pred.append(l1_g_pred)

            self.l1_D = tf.stack(self.l1_D, axis=1)
            self.l1_WT = tf.stack(self.l1_WT, axis=1)
            self.l1_gaps_pred = tf.stack(self.l1_gaps_pred, axis=1)
            if self.use_marks:
                self.l1_marks_logits = tf.stack(self.l1_marks_logits, axis=1)
            else:
                self.l1_marks_logits = None

        return (self.l2_marks_logits, self.l2_gaps_pred, self.l2_D, self.l2_WT,
                self.l1_marks_logits, self.l1_gaps_pred, self.l1_D, self.l1_WT)

    def reset_states(self):
        self.l1_rnn.reset_states()
        self.l2_rnn.reset_states()

def simulate_rmtpp_tb_compare(model, times, gaps_in, block_begin_ts, t_b_plus,
                   decoder_length, marks_in=None):

    marks_logits, gaps_pred = list(), list()
    times_pred = list()
    last_times_pred = tf.squeeze(times + gaps_in, axis=-1)
    times_pred.append(last_times_pred)
    N = len(gaps_in)
    pred_idxes = -1.0 * np.ones(N)
    begin_idxes, end_idxes = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
    simul_step = 0
    while any(times_pred[-1]<t_b_plus) or any(pred_idxes<decoder_length):
        step_marks_logits, step_gaps_pred, _, _ = model(gaps_in, marks_in)
        if marks_in is not None:
            step_marks_pred = tf.argmax(step_marks_logits, axis=-1) + 1
        else:
            step_marks_pred = None

        #print('Simul step:', simul_step, tf.squeeze(tf.squeeze(step_gaps_pred, axis=1), axis=-1))
        gaps_pred.append(step_gaps_pred)
        marks_logits.append(step_marks_logits)
        last_times_pred = times_pred[-1] + tf.squeeze(step_gaps_pred, axis=-1)
        times_pred.append(last_times_pred)

        marks_in, gaps_in = step_marks_pred, step_gaps_pred

        for ex_id, (t_pred, t_b) in enumerate(zip(times_pred[-1], t_b_plus)):
            if t_pred > t_b:
                pred_idxes[ex_id] += 1
                if pred_idxes[ex_id] == 0:
                    begin_idxes[ex_id] = simul_step
                if pred_idxes[ex_id] == decoder_length:
                    end_idxes[ex_id] = simul_step

        simul_step += 1

    if marks_in is not None:
        marks_logits = tf.squeeze(tf.stack(marks_logits, axis=1), axis=2)
        marks_logits = [m_l[b_idx:e_idx] for m_l, b_idx, e_idx in \
                            zip(marks_logits, begin_idxes, end_idxes)]
        marks_logits = tf.stack(marks_logits, axis=0)
    gaps_pred = tf.squeeze(tf.stack(gaps_pred, axis=1), axis=2)
    gaps_pred = [t_l[b_idx:e_idx] for t_l, b_idx, e_idx in \
                        zip(gaps_pred, begin_idxes, end_idxes)]
    gaps_pred = tf.stack(gaps_pred, axis=0)

    return marks_logits, gaps_pred


def simulate_rmtpp_offset_compare(model, gaps, block_begin_ts, t_b_plus,
                   decoder_length, marks=None):
    marks_logits, gaps_pred = list(), list()
    marks_inputs, gaps_inputs = marks, gaps
    cum_gaps_pred = tf.squeeze(gaps, axis=1)
    offset = t_b_plus - block_begin_ts
    N = len(gaps)
    simul_step = 0
    pred_idxes = -1.0 * np.ones(N)
    begin_idxes, end_idxes = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
    simul_step = 0
    while any(cum_gaps_pred<offset) or any(pred_idxes<decoder_length):
        step_marks_logits, step_gaps_pred, _, _ = model(gaps_inputs, marks_inputs)
        if marks is not None:
            step_marks_pred = tf.argmax(step_marks_logits, axis=-1) + 1
        else:
            step_marks_pred = None

        #print('Simul step:', simul_step, tf.squeeze(tf.squeeze(step_gaps_pred, axis=1), axis=-1))
        gaps_pred.append(step_gaps_pred)
        marks_logits.append(step_marks_logits)
        cum_gaps_pred += tf.squeeze(step_gaps_pred, axis=1)

        marks_inputs, gaps_inputs = step_marks_pred, step_gaps_pred
        simul_step += 1

        for ex_id, (cum_ts, off) in enumerate(zip(cum_gaps_pred, offset)):
            if cum_ts > off:
                pred_idxes[ex_id] += 1
                if pred_idxes[ex_id] == 0:
                    begin_idxes[ex_id] = simul_step-1
                if pred_idxes[ex_id] == decoder_length:
                    end_idxes[ex_id] = simul_step-1

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


class SimulateHierarchicalRNN:
    def simulate(self, model, c_times_in, c_gaps_pred,
                 block_begin_ts,
                 t_b_plus, c_t_b_plus,
                 decoder_length):

        # ----- Start: Simulation of Layer 2 RNN ----- #

        l2_hidden_states = list()
        l2_gaps_pred = list()
        l2_times_pred = list()
        N = len(c_gaps_pred)
        l2_idxes = np.zeros(N, dtype=int)
        l2_last_times_pred = tf.squeeze(c_times_in + c_gaps_pred[:, -1:], axis=-1)
        l2_times_pred.append(l2_last_times_pred)
        l2_hidden_states.append(model.l2_rnn.hidden_states[:, -2])
        l2_gaps_pred.append(c_gaps_pred[:, -2:-1])
        l2_gaps_inputs = c_gaps_pred[:, -1:]
        simul_step = 0
        while any(l2_times_pred[-1]<c_t_b_plus):

            #print('layer 2 simul_step:', simul_step)

            prev_hidden_state = model.l2_rnn.hidden_states[:, -1]

            (_, step_l2_gaps_pred, _, _, _, _, _, _) \
                    = model(None, l2_gaps=l2_gaps_inputs)

            l2_hidden_states.append(prev_hidden_state)
            l2_gaps_pred.append(l2_gaps_inputs)
            l2_gaps_inputs = step_l2_gaps_pred
            l2_last_times_pred = l2_times_pred[-1] + tf.squeeze(step_l2_gaps_pred, axis=-1)
            l2_times_pred.append(l2_last_times_pred)

            for ex_id, (l2_t_pred, c_t_b) in enumerate(zip(l2_times_pred[-1], c_t_b_plus)):
                if l2_t_pred > c_t_b and l2_idxes[ex_id] == -1:
                    l2_idxes[ex_id] = len(l2_hidden_states)-2

            simul_step += 1

        #l2_gaps_pred = tf.stack(l2_gaps_pred, axis=0)
        l2_gaps_pred = tf.squeeze(tf.stack(l2_gaps_pred, axis=1), axis=2)
        self.all_l2_gaps_pred = l2_gaps_pred # all predicted gaps from start to end of simulation
        l2_idxes = tf.expand_dims(l2_idxes, axis=-1)
        l2_gaps_pred = tf.gather(l2_gaps_pred, l2_idxes, batch_dims=1)
        l2_times_pred = tf.squeeze(tf.stack(l2_times_pred, axis=1), axis=-1)
        l2_times_pred = tf.gather(l2_times_pred, l2_idxes, batch_dims=1)


        l1_gaps_pred = list()
        l1_times_pred = list()
        l1_times_pred.append(l2_times_pred)
        l1_begin_idxes, l1_end_idxes =  np.zeros(N, dtype=int), np.zeros(N, dtype=int)
        pred_idxes = -1.0 * np.ones(N)
        l1_gaps_inputs = l2_gaps_pred / 10.0 #TODO Can we do better here?
        #l1_gaps_inputs = tf.expand_dims(l1_gaps_inputs, axis=-1)
        simul_step = 0

        while any(l1_times_pred[-1]<t_b_plus) or any(pred_idxes<decoder_length):

            #print('layer 1 simul_step:', simul_step)

            (_, _, _, _, _, step_l1_gaps_pred, _, _) \
                    = model(None, l1_gaps=tf.expand_dims(l1_gaps_inputs, axis=-1))

            l1_gaps_pred.append(step_l1_gaps_pred)
            l1_gaps_inputs = tf.squeeze(step_l1_gaps_pred, axis=1)
            step_l1_gaps_pred_squeeze = tf.squeeze(tf.squeeze(step_l1_gaps_pred, axis=-1), axis=-1)
            l1_last_times_pred = l1_times_pred[-1] + step_l1_gaps_pred_squeeze
            l1_times_pred.append(l1_last_times_pred)

            for ex_id, (l1_t_pred, t_b) in enumerate(zip(l1_times_pred[-1], t_b_plus)):
                if l1_t_pred > t_b:
                    pred_idxes[ex_id] += 1
                    if pred_idxes[ex_id] == 0:
                        l1_begin_idxes[ex_id] = simul_step
                    if pred_idxes[ex_id] == decoder_length:
                        l1_end_idxes[ex_id] = simul_step

            simul_step += 1

        l1_gaps_pred = tf.squeeze(tf.stack(l1_gaps_pred, axis=1), axis=2)
        l1_gaps_pred = [t_l[b_idx:e_idx] for t_l, b_idx, e_idx in \
                            zip(l1_gaps_pred, l1_begin_idxes, l1_end_idxes)]
        l1_gaps_pred = tf.stack(l1_gaps_pred, axis=0)
        l1_gaps_pred = tf.squeeze(l1_gaps_pred, axis=-1)

        return l1_gaps_pred


def simulate_hierarchicalrnn(model, c_times_in, c_gaps_pred,
                             block_begin_ts,
                             t_b_plus, c_t_b_plus,
                             decoder_length):

    # ----- Start: Simulation of Layer 2 RNN ----- #

    l2_hidden_states = list()
    l2_gaps_pred = list()
    l2_times_pred = list()
    N = len(c_gaps_pred)
    l2_idxes = np.zeros(N, dtype=int) 
    l2_last_times_pred = tf.squeeze(c_times_in + c_gaps_pred[:, -1:], axis=-1)
    l2_times_pred.append(l2_last_times_pred)
    l2_hidden_states.append(model.l2_rnn.hidden_states[:, -2])
    l2_gaps_pred.append(c_gaps_pred[:, -2:-1])
    l2_gaps_inputs = c_gaps_pred[:, -1:]
    simul_step = 0
    while any(l2_times_pred[-1]<c_t_b_plus):

        #print('layer 2 simul_step:', simul_step)

        prev_hidden_state = model.l2_rnn.hidden_states[:, -1]

        (_, step_l2_gaps_pred, _, _, _, _, _, _) \
                = model(None, l2_gaps=l2_gaps_inputs)

        l2_hidden_states.append(prev_hidden_state)
        l2_gaps_pred.append(l2_gaps_inputs)
        l2_gaps_inputs = step_l2_gaps_pred
        l2_last_times_pred = l2_times_pred[-1] + tf.squeeze(step_l2_gaps_pred, axis=-1)
        l2_times_pred.append(l2_last_times_pred)

        for ex_id, (l2_t_pred, c_t_b) in enumerate(zip(l2_times_pred[-1], c_t_b_plus)):
            if l2_t_pred > c_t_b and l2_idxes[ex_id] == -1:
                l2_idxes[ex_id] = len(l2_hidden_states)-2

        simul_step += 1

    #l2_gaps_pred = tf.stack(l2_gaps_pred, axis=0)
    l2_gaps_pred = tf.squeeze(tf.stack(l2_gaps_pred, axis=1), axis=2)
    l2_idxes = tf.expand_dims(l2_idxes, axis=-1)
    l2_gaps_pred = tf.gather(l2_gaps_pred, l2_idxes, batch_dims=1)
    l2_times_pred = tf.squeeze(tf.stack(l2_times_pred, axis=1), axis=-1)
    l2_times_pred = tf.gather(l2_times_pred, l2_idxes, batch_dims=1)


    l1_gaps_pred = list()
    l1_times_pred = list()
    l1_times_pred.append(l2_times_pred)
    l1_begin_idxes, l1_end_idxes =  np.zeros(N, dtype=int), np.zeros(N, dtype=int)
    pred_idxes = -1.0 * np.ones(N)
    l1_gaps_inputs = l2_gaps_pred / 10.0 #TODO Can we do better here?
    #l1_gaps_inputs = tf.expand_dims(l1_gaps_inputs, axis=-1)
    simul_step = 0

    while any(l1_times_pred[-1]<t_b_plus) or any(pred_idxes<decoder_length):

        #print('layer 1 simul_step:', simul_step)

        (_, _, _, _, _, step_l1_gaps_pred, _, _) \
                = model(None, l1_gaps=tf.expand_dims(l1_gaps_inputs, axis=-1))

        l1_gaps_pred.append(step_l1_gaps_pred)
        step_l1_gaps_pred_squeeze = tf.squeeze(tf.squeeze(step_l1_gaps_pred, axis=-1), axis=-1)
        l1_last_times_pred = l1_times_pred[-1] + step_l1_gaps_pred_squeeze
        l1_times_pred.append(l1_last_times_pred)

        for ex_id, (l1_t_pred, t_b) in enumerate(zip(l1_times_pred[-1], t_b_plus)):
            if l1_t_pred > t_b:
                pred_idxes[ex_id] += 1
                if pred_idxes[ex_id] == 0:
                    l1_begin_idxes[ex_id] = simul_step
                if pred_idxes[ex_id] == decoder_length:
                    l1_end_idxes[ex_id] = simul_step

        simul_step += 1

    l1_gaps_pred = tf.squeeze(tf.stack(l1_gaps_pred, axis=1), axis=2)
    self.all_l1_gaps_pred = l1_gaps_pred # all predicted gaps from start to end of simulation
    l1_gaps_pred = [t_l[b_idx:e_idx] for t_l, b_idx, e_idx in \
                        zip(l1_gaps_pred, l1_begin_idxes, l1_end_idxes)]
    l1_gaps_pred = tf.stack(l1_gaps_pred, axis=0)
    l1_gaps_pred = tf.squeeze(l1_gaps_pred, axis=-1)

    return l1_gaps_pred
