from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from reader import read_data, split_data, get_input_output_seqs, create_train_dev_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.random.set_seed(42)

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

#class MaskedDense(tf.keras.layers.Layer):
#
#    def call(self, inputs, mask=None):

class RMTPP(tf.keras.Model):
    def __init__(self,
                 num_categories,
                 embed_size,
                 hidden_layer_size,
                 name='RMTPP',
                 **kwargs):
        super(RMTPP, self).__init__(name=name, **kwargs)
        self.embedding_layer = layers.Embedding(num_categories, embed_size,
                                                mask_zero=True,
                                                name='marks_embedding')
        self.rnn_layer = layers.GRU(hidden_layer_size, return_sequences=True,
                                    stateful=True,
                                    name='GRU_Layer')
        self.D_layer = layers.Dense(1, name='D_layer')
        self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')
        self.marks_output_layer = layers.Dense(num_categories,
                                               activation='softmax',
                                               name='marks_output_layer')
        self.times_output_layer = InverseTransformSampling()

    def call(self, marks, times):
        self.marks_embd = self.embedding_layer(marks)
        mask = self.embedding_layer.compute_mask(marks)
        self.times = times
        rnn_inputs = tf.concat([self.marks_embd, self.times], axis=-1)
        self.hidden_states = self.rnn_layer(rnn_inputs, mask=mask)
        self.D = self.D_layer(self.hidden_states)
        self.WT = self.WT_layer(self.hidden_states)
        self.marks_logits = self.marks_output_layer(self.hidden_states)
        self.times_pred = self.times_output_layer((self.D, self.WT))

        return self.marks_logits, self.times_pred, self.D, self.WT

def simulate(model, marks, times):
    marks_logits, times_pred = list(), list()
    marks_inputs, times_inputs = marks, times
    #TODO Properly set the number of simulation steps
    for simul_step in range(1073):
        print('Simul step:', simul_step)
        step_marks_logits, step_times_pred, _, _ = model(marks_inputs, times_inputs)
        step_marks_pred = tf.argmax(step_marks_logits, axis=-1) + 1

        marks_logits.append(step_marks_logits)
        times_pred.append(step_times_pred)

        marks_inputs, times_inputs = step_marks_pred, step_times_pred

    marks_logits = tf.squeeze(tf.stack(marks_logits, axis=1), axis=2)
    times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)


    return marks_logits, times_pred


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

batch_size = 2
BPTT = 20
# ----- Start: Read and Preprocess data ----- #
marks, times = read_data('testdata.txt')
#marks, times = split_data((marks, times), 7)

max_offset = 2
train_marks, train_times, \
        dev_marks, dev_times, \
        test_marks, test_times \
        = create_train_dev_test_split((marks, times), max_offset)

train_marks_in, train_marks_out, train_times_in, train_times_out \
        = get_input_output_seqs((train_marks, train_times))
#dev_marks_in, dev_marks_out, dev_times_in, dev_times_out \
#        = get_input_output_seqs((dev_marks, dev_times))
#test_marks_in, test_marks_out, test_times_in, test_times_out \
#        = get_input_output_seqs((test_marks, test_times))

train_marks_in = pad_sequences(train_marks_in, padding='post')
train_times_in = pad_sequences(train_times_in, padding='post')
train_marks_out = pad_sequences(train_marks_out, padding='post')
train_times_out = pad_sequences(train_times_out, padding='post')

dev_marks = pad_sequences(dev_marks, padding='post')
dev_times = pad_sequences(dev_times, padding='post')
#dev_marks_in = pad_sequences(dev_marks_in, padding='post')
#dev_times_in = pad_sequences(dev_times_in, padding='post')
#dev_marks_out = pad_sequences(dev_marks_out, padding='post')
#dev_times_out = pad_sequences(dev_times_out, padding='post')

test_marks = pad_sequences(test_marks, padding='post')
test_times = pad_sequences(test_times, padding='post')
#test_marks_in = pad_sequences(test_marks_in, padding='post')
#test_times_in = pad_sequences(test_times_in, padding='post')
#test_marks_out = pad_sequences(test_marks_out, padding='post')
#test_times_out = pad_sequences(test_times_out, padding='post')

train_times_in = tf.expand_dims(tf.cast(train_times_in, tf.float32), axis=-1)
train_times_out = tf.expand_dims(tf.cast(train_times_out, tf.float32), axis=-1)

dev_times = tf.expand_dims(tf.cast(dev_times, tf.float32), axis=-1)
#dev_times_in = tf.expand_dims(tf.cast(dev_times_in, tf.float32), axis=-1)
#dev_times_out = tf.expand_dims(tf.cast(dev_times_out, tf.float32), axis=-1)

test_times = tf.expand_dims(tf.cast(test_times, tf.float32), axis=-1)
#test_times_in = tf.expand_dims(tf.cast(test_times_in, tf.float32), axis=-1)
#test_times_out = tf.expand_dims(tf.cast(test_times_out, tf.float32), axis=-1)

def transpose(m_in, t_in, m_out, t_out):
    return tf.transpose(m_in), tf.transpose(t_in, [1, 0, 2]), \
            tf.transpose(m_out), tf.transpose(t_out, [1, 0, 2])
train_marks_in, train_times_in, train_marks_out, train_times_out \
        = transpose(train_marks_in, train_times_in,
                    train_marks_out, train_times_out)
train_dataset = tf.data.Dataset.from_tensor_slices((train_marks_in,
                                                    train_times_in,
                                                    train_marks_out,
                                                    train_times_out))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BPTT).map(transpose)
# ----- End: Read and Preprocess data ----- #


# Loss function
mark_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
time_loss_fn = tf.keras.losses.MeanAbsoluteError()


# Evaluation metrics
train_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_time_metric = tf.keras.metrics.MeanAbsoluteError()
dev_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
dev_time_metric = tf.keras.metrics.MeanAbsoluteError()
test_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_time_metric = tf.keras.metrics.MeanAbsoluteError()


num_categories = len(np.unique(marks)) + 1
model = RMTPP(num_categories, 8, 32)

optimizer = keras.optimizers.Adam(learning_rate=1e-2)

# Iterate over epochs.
epochs = 100
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (marks_batch_in, times_batch_in, marks_batch_out, times_batch_out) \
                    in enumerate(train_dataset):

        with tf.GradientTape() as tape:

            marks_logits, times_pred, D, WT = model(marks_batch_in, times_batch_in)
            marks_pred = tf.argmax(marks_logits, axis=-1) + 1

            #print(step, marks_batch_in.shape, times_batch_in.shape)
            #print(marks_logits.shape, D.shape, times_pred.shape)
            #print(times_batch_out[0])
            #print(times_pred[0])
            #print(marks_batch_out)
            #print(tf.argmax(marks_logits, axis=-1).numpy())

            # Compute the loss for this minibatch.
            mark_loss = mark_loss_fn(marks_batch_out, marks_logits)
            #time_loss = time_loss_fn(times_batch_out, D)
            time_loss_fn = NegativeLogLikelihood(D, WT)
            time_loss = time_loss_fn(times_batch_out, D)
            loss = mark_loss + time_loss

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # TODO Make sure that padding is considered during evaluation
        train_mark_metric(marks_batch_out, marks_logits)
        train_time_metric(times_batch_out, times_pred)

        # Log every 200 batches.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s %s %s' \
                    % (step, float(loss), float(mark_loss), float(time_loss)))
            print('Seen so far: %s samples' % ((step + 1) * batch_size))

        #model.rnn_layer.reset_states() # Reset RNN state after 
                                       # a sequence is finished

    train_mark_acc = train_mark_metric.result()
    train_time_err = train_time_metric.result()
    print('Training mark acc and time err over epoch: %s, %s' \
            % (float(train_mark_acc), float(train_time_err)))
    train_mark_metric.reset_states()
    train_time_metric.reset_states()

    #dev_marks_logits, dev_times_pred, _, _ = model(dev_marks_in, dev_times_in)
    #test_marks_logits, test_times_pred, _, _ = model(test_marks_in, test_times_in)
    #dev_marks_logits, dev_times_pred, _, _ = model(dev_marks, dev_times)
    #test_marks_logits, test_times_pred, _, _ = model(test_marks, test_times)

    dev_marks_logits, dev_times_pred = simulate(model, marks_pred[:, -1:], times_pred[:, -1:])
    test_marks_logits, test_times_pred = simulate(model, marks_pred[:, -1:], times_pred[:, -1:])

    dev_mark_metric(dev_marks, dev_marks_logits)
    dev_time_metric(dev_times, dev_times_pred)
    test_mark_metric(test_marks, test_marks_logits)
    test_time_metric(test_times, test_times_pred)
    dev_mark_acc = dev_mark_metric.result()
    dev_time_err = dev_time_metric.result()
    test_mark_acc = test_mark_metric.result()
    test_time_err = test_time_metric.result()
    print('Dev mark acc and time err over epoch: %s, %s' \
            % (float(dev_mark_acc), float(dev_time_err)))
    print('Test mark acc and time err over epoch: %s, %s' \
            % (float(test_mark_acc), float(test_time_err)))
    dev_mark_metric.reset_states()
    dev_time_metric.reset_states()
    test_mark_metric.reset_states()
    test_time_metric.reset_states()

    model.rnn_layer.reset_states()
