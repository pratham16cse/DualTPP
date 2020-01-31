from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_right

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(42)

#from reader_hierarchical import get_preprocessed_data, transpose, \
#        read_data, split_data, \
#        get_train_input_output, create_train_dev_test_split, \
#        get_gaps, get_dev_test_input_output

import reader_hierarchical

import models
                    
epochs = 100
patience = 10

batch_size = 2
BPTT = 20
block_size = 1
max_offset = 1
block_size_sec = 3600.0 * block_size
max_offset_sec = 3600.0 * max_offset
decoder_length = 5
use_marks = False
use_intensity = False
data = reader_hierarchical.get_preprocessed_data(block_size, decoder_length)
num_categories = data['num_categories']
num_sequences = data['num_sequences']

c_train_dataset = data['c_train_dataset']

c_dev_dataset = data['c_dev_dataset']
c_dev_seq_lens = data['c_dev_seq_lens']
dev_marks_out = data['dev_marks_out']
dev_gaps_out = data['dev_gaps_out']
dev_times_out = data['dev_times_out']
dev_t_b_plus = data['dev_begin_tss'] + max_offset_sec
dev_begin_tss = data['dev_begin_tss']
dev_offsets = tf.random.uniform(shape=(num_sequences, 1)) * 3600.
#dev_t_b_plus = data['dev_begin_tss'] + max_offset_sec
dev_t_b_plus = data['dev_begin_tss'] + dev_offsets
print(dev_offsets)
print(tf.squeeze(dev_times_out, axis=-1).numpy().tolist()[0])
dev_times_out_indices = [bisect_right(dev_t_out, t_b) for dev_t_out, t_b \
                            in zip(dev_times_out, dev_t_b_plus)]
dev_times_out_indices = tf.expand_dims(dev_times_out_indices, axis=-1)
dev_times_out_indices \
        = (dev_times_out_indices-1) \
        + tf.expand_dims(tf.range(decoder_length), axis=0)
print(dev_times_out_indices)
dev_gaps_out = tf.gather(dev_gaps_out, dev_times_out_indices, batch_dims=1)


c_test_dataset = data['c_test_dataset']
test_marks_out = data['test_marks_out']
test_gaps_out = data['test_gaps_out']
test_times_out = data['test_times_out']
test_t_b_plus = data['test_begin_tss'] + max_offset_sec
c_test_seq_lens = data['c_test_seq_lens']
test_begin_tss = data['test_begin_tss']
test_offsets = tf.random.uniform(shape=(num_sequences, 1)) * 3600.
#test_t_b_plus = data['test_begin_tss'] + max_offset_sec
test_t_b_plus = data['test_begin_tss'] + test_offsets
print(test_offsets)
print(tf.squeeze(test_times_out, axis=-1).numpy().tolist()[0])
test_times_out_indices = [bisect_right(test_t_out, t_b) for test_t_out, t_b \
                            in zip(test_times_out, test_t_b_plus)]
test_times_out_indices = tf.expand_dims(test_times_out_indices, axis=-1)
test_times_out_indices \
        = (test_times_out_indices-1) \
        + tf.expand_dims(tf.range(decoder_length), axis=0)
print(test_times_out_indices)
test_gaps_out = tf.gather(test_gaps_out, test_times_out_indices, batch_dims=1)

c_train_dataset = c_train_dataset.batch(BPTT, drop_remainder=False).map(reader_hierarchical.transpose)
c_dev_dataset = c_dev_dataset.batch(num_sequences)
c_test_dataset = c_test_dataset.batch(num_sequences)

# Loss function
mark_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
if not use_intensity:
    c_gap_loss_fn = tf.keras.losses.MeanSquaredError()
    gap_loss_fn = tf.keras.losses.MeanSquaredError()

# Evaluation metrics
train_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_gap_metric = tf.keras.metrics.MeanAbsoluteError()
dev_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
dev_gap_metric = tf.keras.metrics.MeanAbsoluteError()
test_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_gap_metric = tf.keras.metrics.MeanAbsoluteError()

model = models.HierarchicalRNN(num_categories, 8, 32, use_marks=use_marks,
                               use_intensity=use_intensity)

optimizer = keras.optimizers.Adam(learning_rate=1e-2)

# Iterate over epochs.
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (c_marks_batch_in, c_gaps_batch_in, c_times_batch_in, c_seqmask_batch_in,
               c_marks_batch_out, c_gaps_batch_out, c_times_batch_out, c_seqmask_batch_out,
               gaps_batch_in, times_batch_in, seqmask_batch_in,
               gaps_batch_out, times_batch_out, seqmask_batch_out) \
                       in enumerate(c_train_dataset):

        with tf.GradientTape() as tape:

            (l2_marks_logits, l2_gaps_pred, l2_D, l2_WT,
             l1_marks_logits, l1_gaps_pred, l1_D, l1_WT) \
                    = model(None, l2_gaps=c_gaps_batch_in,
                            l1_gaps=gaps_batch_in,
                            l2_marks=c_marks_batch_in)
            #marks_pred = tf.argmax(marks_logits, axis=-1) + 1

            #print(step, marks_batch_in.shape, gaps_batch_in.shape)
            #print(marks_logits.shape, D.shape, gaps_pred.shape)
            #print(gaps_batch_out[0])
            #print(gaps_pred[0])
            #print(marks_batch_out)
            #print(tf.argmax(marks_logits, axis=-1).numpy())

            # Compute the loss for this minibatch.
            if use_marks:
                mark_loss = mark_loss_fn(marks_batch_out, marks_logits)
            else:
                mark_loss = 0.0
            if use_intensity:
                c_gap_loss_fn = models.NegativeLogLikelihood(l2_D, l2_WT)
                gap_loss_fn = models.NegativeLogLikelihood(l1_D, l1_WT)
            c_gap_loss = c_gap_loss_fn(c_gaps_batch_out, l2_gaps_pred)
            gap_loss = gap_loss_fn(gaps_batch_out, l1_gaps_pred)
            loss = mark_loss + c_gap_loss + gap_loss

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # TODO Make sure that padding is considered during evaluation
        if use_marks:
            train_mark_metric(marks_batch_out, marks_logits)
        train_gap_metric(gaps_batch_out, l1_gaps_pred)

        # Log every 200 batches.
        if step % 200 == 0:
        #    print(tf.squeeze(c_gaps_batch_out, axis=-1))
        #    print(tf.squeeze(l2_gaps_pred, axis=-1))
            print('Training loss (for one batch) at step %s: %s %s %s %s' \
                    % (step, float(loss), float(mark_loss), float(c_gap_loss), float(gap_loss)))
            print('Seen so far: %s samples' % ((step + 1) * batch_size))

        model.reset_states() # Reset RNN state after 
                             # a sequence is finished

    if use_marks:
        train_mark_acc = train_mark_metric.result()
        train_mark_metric.reset_states()
    else:
        train_mark_acc = 0.0
    train_gap_err = train_gap_metric.result()
    train_gap_metric.reset_states()
    print('Training mark acc and gap err over epoch: %s, %s' \
            % (float(train_mark_acc), float(train_gap_err)))

    if epoch > patience:

        for dev_step, (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in, c_dev_seqmask_in) \
                in enumerate(c_dev_dataset):

            (dev_l2_marks_logits, dev_l2_gaps_pred, _, _,
             dev_l1_marks_logits, dev_l1_gaps_pred, _, _) \
                    = model(c_dev_gaps_in)
            #dev_marks_logits, dev_gaps_pred, _, _ = model(c_dev_gaps_in, c_dev_marks_in)
            if use_marks:
                dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
                dev_marks_pred_last = dev_marks_pred[:, -1:]
            else:
                dev_marks_pred_last = None
            dev_gaps_pred \
                    = models.simulate_hierarchicalrnn(model,
                                      dev_l2_gaps_pred[:, -1:],
                                      dev_begin_tss,
                                      dev_t_b_plus,
                                      decoder_length)
        model.reset_states()

        for test_step, (c_test_marks_in, c_test_gaps_in, c_test_times_in, c_test_seqmask_in) \
                in enumerate(c_test_dataset):

            (test_l2_marks_logits, test_l2_gaps_pred, _, _,
             test_l1_marks_logits, test_l1_gaps_pred, _, _) \
                    = model(c_test_gaps_in)
            #test_marks_logits, test_gaps_pred, _, _ = model(c_test_gaps_in, c_test_marks_in)
            if use_marks:
                test_marks_pred = tf.argmax(test_marks_logits, axis=-1) + 1
                test_marks_pred_last = test_marks_pred[:, -1:]
            else:
                test_marks_pred_last = None
            last_test_input_ts = tf.gather(c_test_times_in, c_test_seq_lens-1, batch_dims=1)
            test_gaps_pred \
                    = models.simulate_hierarchicalrnn(model,
                                      test_l2_gaps_pred[:, -1:],
                                      test_begin_tss,
                                      test_t_b_plus,
                                      decoder_length)
        model.reset_states()

        #print(dev_marks_out, 'dev_marks_out')
        #print(np.argmax(dev_marks_logits, axis=-1), 'dev_marks_preds')

        if use_marks:
            dev_mark_metric(dev_marks_out, dev_marks_logits)
            test_mark_metric(test_marks_out, test_marks_logits)
            dev_mark_acc = dev_mark_metric.result()
            test_mark_acc = test_mark_metric.result()
            dev_mark_metric.reset_states()
            test_mark_metric.reset_states()
        else:
            dev_mark_acc, test_mark_acc = 0.0, 0.0

        print('\ndev_gaps_pred')
        print(tf.squeeze(dev_gaps_pred[:, 1:], axis=-1))
        print('\ndev_gaps_out')
        print(tf.squeeze(dev_gaps_out[:, 1:], axis=-1))

        dev_gap_metric(dev_gaps_out[:, 1:], dev_gaps_pred[:, 1:])
        test_gap_metric(test_gaps_out[:, 1:], test_gaps_pred[:, 1:])
        dev_gap_err = dev_gap_metric.result()
        test_gap_err = test_gap_metric.result()
        dev_gap_metric.reset_states()
        test_gap_metric.reset_states()
        print('Dev mark acc and gap err over epoch: %s, %s' \
                % (float(dev_mark_acc), float(dev_gap_err)))
        print('Test mark acc and gap err over epoch: %s, %s' \
                % (float(test_mark_acc), float(test_gap_err)))

        model.reset_states()
