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
        self.marks_pred = self.marks_output_layer(self.hidden_states)
        self.times_pred = self.times_output_layer((self.D, self.WT))

        return self.marks_pred, self.times_pred, self.D, self.WT


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

train_marks_in = pad_sequences(train_marks_in, padding='post')
train_times_in = pad_sequences(train_times_in, padding='post')
train_marks_out = pad_sequences(train_marks_out, padding='post')
train_times_out = pad_sequences(train_times_out, padding='post')

train_times_in = tf.expand_dims(tf.cast(train_times_in, tf.float32), axis=-1)
train_times_out = tf.expand_dims(tf.cast(train_times_out, tf.float32), axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_marks_in,
                                                    train_times_in,
                                                    train_marks_out,
                                                    train_times_out))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
# ----- End: Read and Preprocess data ----- #


# Loss function
mark_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
time_loss_fn = tf.keras.losses.MeanAbsoluteError()

# Evaluation metrics
train_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_time_metric = tf.keras.metrics.MeanAbsoluteError()


num_categories = len(np.unique(marks))
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

            marks_pred, times_pred, D, WT = model(marks_batch_in, times_batch_in)

            print(marks_pred.shape, D.shape)
            #print(times_batch_out[0])
            #print(times_pred[0])
            print(marks_batch_out)
            print(tf.argmax(marks_pred, axis=-1).numpy())

            # Compute the loss for this minibatch.
            mark_loss = mark_loss_fn(marks_batch_out, marks_pred)
            #time_loss = time_loss_fn(times_batch_out, D)
            time_loss_fn = NegativeLogLikelihood(D, WT)
            time_loss = time_loss_fn(times_batch_out, D)
            loss = mark_loss + time_loss

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # TODO Make sure that padding is considered during evaluation
        train_mark_metric(marks_batch_out, marks_pred)
        train_time_metric(times_batch_out, times_pred)

        # Log every 200 batches.
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s %s %s' \
                    % (step, float(loss), float(mark_loss), float(time_loss)))
            print('Seen so far: %s samples' % ((step + 1) * batch_size))

        model.rnn_layer.reset_states() # Reset RNN state after 
                                       # a sequence is finished

    train_mark_acc = train_mark_metric.result()
    train_time_err = train_time_metric.result()
    print('Training mark acc and time err over epoch: %s, %s' \
            % (float(train_mark_acc), float(train_time_err)))
    train_mark_metric.reset_states()
    train_time_metric.reset_states()
