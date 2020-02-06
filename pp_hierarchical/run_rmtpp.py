from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import math
from bisect import bisect_right
import os, sys
import ipdb
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(42)

#from reader_rmtpp import get_preprocessed_data, transpose, \
#        read_data, split_data, \
#        get_train_input_output, create_train_dev_test_split, \
#        get_gaps, get_dev_test_input_output

import reader_rmtpp

import models
                    
epochs = 100
patience = 10

batch_size = 2
BPTT = 20
block_size = 1 # Number of hours in a block
block_size_sec = 3600.0 * block_size
decoder_length = 5
use_marks = False
use_intensity = True

data = reader_rmtpp.get_preprocessed_data(block_size, decoder_length)
num_categories = data['num_categories']
num_sequences = data['num_sequences']
initial_timestamp = data['initial_timestamp']
train_dataset = data['train_dataset']

best_dev_gap_error = np.inf
best_test_gap_error = np.inf
best_dev_mark_acc = np.inf
best_test_mark_acc = np.inf
best_epoch = 0

# ----- Start: Load dev_dataset ----- #
# dynamic_block_size = data['dev_normalizer_d']
# dynamic_block_size = round(dynamic_block_size.numpy().tolist()[0][0]//6)
# dynamic_block_size = max(1, dynamic_block_size)
# print('block_size', dynamic_block_size)

dev_dataset = data['dev_dataset']
dev_seq_lens = data['dev_seq_lens']
dev_seq_lens_in = tf.cast(tf.reduce_sum(data['dev_seqmask_in'], axis=-1), tf.int32)
dev_seq_lens_out = tf.cast(tf.reduce_sum(data['dev_seqmask_out'], axis=-1), tf.int32)
dev_marks_out = data['dev_marks_out']
dev_gaps_out = data['dev_gaps_out']
dev_times_out = data['dev_times_out']
dev_begin_tss = data['dev_begin_tss']
dev_offsets = tf.random.uniform(shape=(num_sequences, 1)) * 3600. * block_size
dev_t_b_plus = dev_begin_tss + dev_offsets

#print('\n dev_begin_tss')
#for d in dev_begin_tss.numpy().tolist():
#    print('{0:.15f}'.format(d[0]))
#print('\n dev_offsets')
#for d in dev_offsets.numpy().tolist():
#    print('{0:.15f}'.format(d[0]))
#print('\n dev_t_b_plus')
#for d in dev_t_b_plus.numpy().tolist():
#    print('{0:.15f}'.format(d[0]))

dev_times_out_indices = [bisect_right(dev_t_out, t_b) for dev_t_out, t_b in zip(dev_times_out, dev_t_b_plus)]
dev_times_out_indices = tf.minimum(dev_times_out_indices, dev_seq_lens_out-decoder_length+1)
dev_times_out_indices = tf.expand_dims(dev_times_out_indices, axis=-1)
print('\ndev_seq_lens_out', dev_seq_lens_out)
dev_times_out_indices \
        = (dev_times_out_indices-1) \
        + tf.expand_dims(tf.range(decoder_length), axis=0)
print('\ndev_times_out_indices')
print(dev_times_out_indices)
dev_gaps_out = tf.gather(dev_gaps_out, dev_times_out_indices, batch_dims=1)

# ----- Normalize dev_offsets and dev_t_b_plus ----- #
dev_normalizer_d = data['dev_normalizer_d']
dev_normalizer_a = data['dev_normalizer_a']
print(dev_offsets, dev_normalizer_d)
dev_offsets_sec_norm = dev_offsets/dev_normalizer_d + dev_normalizer_a
dev_t_b_plus = dev_begin_tss + dev_offsets_sec_norm
print('\n dev_begin_tss')
print(dev_begin_tss)
print('\n dev_offsets_sec_norm')
print(dev_offsets_sec_norm)
print('\n dev_t_b_plus')
print(dev_t_b_plus)

# ----- End: Load dev_dataset ----- #

# ----- Start: Load test_dataset ----- #
test_dataset = data['test_dataset']
test_seq_lens = data['test_seq_lens']
test_seq_lens_in = tf.cast(tf.reduce_sum(data['test_seqmask_in'], axis=-1), tf.int32)
test_seq_lens_out = tf.cast(tf.reduce_sum(data['test_seqmask_out'], axis=-1), tf.int32)
test_marks_out = data['test_marks_out']
test_gaps_out = data['test_gaps_out']
test_times_out = data['test_times_out']
test_begin_tss = data['test_begin_tss']
test_offsets = tf.random.uniform(shape=(num_sequences, 1)) * 3600. * block_size
test_t_b_plus = test_begin_tss + test_offsets

#print('\n test_begin_tss')
#for d in test_begin_tss.numpy().tolist():
#    print('{0:.15f}'.format(d[0]))
#print('\n test_offsets')
#for d in test_offsets.numpy().tolist():
#    print('{0:.15f}'.format(d[0]))
#print('\n test_t_b_plus')
#for d in test_t_b_plus.numpy().tolist():
#    print('{0:.15f}'.format(d[0]))

test_times_out_indices = [bisect_right(test_t_out, t_b) for test_t_out, t_b in zip(test_times_out, test_t_b_plus)]
test_times_out_indices = tf.minimum(test_times_out_indices, test_seq_lens_out-decoder_length+1)
test_times_out_indices = tf.expand_dims(test_times_out_indices, axis=-1)
print('\ntest_seq_lens_out', test_seq_lens_out)
test_times_out_indices = (test_times_out_indices-1) + tf.expand_dims(tf.range(decoder_length), axis=0)
print('\ntest_times_out_indices')
print(test_times_out_indices)
test_gaps_out = tf.gather(test_gaps_out, test_times_out_indices, batch_dims=1)

# ----- Normalize test_offsets and test_t_b_plus ----- #
test_normalizer_d = data['test_normalizer_d']
test_normalizer_a = data['test_normalizer_a']
test_offsets_sec_norm = test_offsets/test_normalizer_d + test_normalizer_a
test_t_b_plus = test_begin_tss + test_offsets_sec_norm
print('\n test_begin_tss')
print(test_begin_tss)
print('\n test_offsets_sec_norm')
print(test_offsets_sec_norm)
print('\n test_t_b_plus')
print(test_t_b_plus)

# ----- End: Load test_dataset ----- #

dev_normalizer_d = tf.expand_dims(dev_normalizer_d, axis=1)
dev_normalizer_a = tf.expand_dims(dev_normalizer_a, axis=1)
test_normalizer_d = tf.expand_dims(test_normalizer_d, axis=1)
test_normalizer_a = tf.expand_dims(test_normalizer_a, axis=1)


#train_dataset = train_dataset.batch(BPTT, drop_remainder=True).map(reader_rmtpp.transpose)
train_dataset = train_dataset.batch(BPTT, drop_remainder=False).map(reader_rmtpp.transpose)

dev_dataset = dev_dataset.batch(num_sequences)
test_dataset = test_dataset.batch(num_sequences)

# Loss function
mark_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
if not use_intensity:
    gap_loss_fn = tf.keras.losses.MeanSquaredError()

# Evaluation metrics
train_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_gap_metric = tf.keras.metrics.MeanAbsoluteError()
dev_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
dev_gap_metric = tf.keras.metrics.MeanAbsoluteError()
test_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
test_gap_metric = tf.keras.metrics.MeanAbsoluteError()

model = models.RMTPP(num_categories, 8, 32, use_marks=use_marks,
                     use_intensity=use_intensity)

optimizer = keras.optimizers.Adam(learning_rate=1e-2)

SAVE_DIR = './plots/rmtpp/'
os.makedirs(SAVE_DIR, exist_ok=True)
cntr = 0
cntr = len(next(os.walk(SAVE_DIR))[1])

train_losses = list()
inference_times = list()
# Iterate over epochs.
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (marks_batch_in, gaps_batch_in, times_batch_in, seqmask_batch_in,
               marks_batch_out, gaps_batch_out, times_batch_out, seqmask_batch_out, train_time_feature) \
                       in enumerate(train_dataset):

        with tf.GradientTape() as tape:

            marks_logits, gaps_pred, D, WT = model(gaps_batch_in, 
                                                   seqmask_batch_in, 
                                                   marks_batch_in, 
                                                   train_time_feature)
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
                gap_loss_fn = models.NegativeLogLikelihood(D, WT)
            gap_loss = gap_loss_fn(gaps_batch_out, gaps_pred)
            loss = mark_loss + gap_loss
            train_losses.append(gap_loss.numpy())


        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # TODO Make sure that padding is considered during evaluation
        if use_marks:
            train_mark_metric(marks_batch_out, marks_logits)
        train_gap_metric(gaps_batch_out, gaps_pred)

        # Log every 200 batches.
        #    print(tf.squeeze(gaps_batch_out, axis=-1))
        #    print(tf.squeeze(gaps_pred, axis=-1))
        print('Training loss (for one batch) at step %s: %s %s %s' \
                % (step, float(loss), float(mark_loss), float(gap_loss)))
        #print('Seen so far: %s events' % ((step + 1) * BPTT))

    model.rnn_layer.reset_states() # Reset RNN state after
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

        for dev_step, (dev_marks_in, dev_gaps_in, dev_times_in, dev_seqmask_in, dev_time_feature) \
                in enumerate(dev_dataset):

            print(dev_gaps_in.shape, dev_seqmask_in.shape, dev_marks_in.shape)
            dev_marks_logits, dev_gaps_pred, _, _ = model(dev_gaps_in,
                                                          dev_seqmask_in,
                                                          dev_marks_in,
                                                          dev_time_feature)
            if use_marks:
                dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
                dev_marks_pred_last = dev_marks_pred[:, -1:]
            else:
                dev_marks_pred_last = None

            last_dev_times_in = tf.gather(dev_times_in,
                                          dev_seq_lens-1,
                                          batch_dims=1)
            dev_simulator = models.SimulateRMTPP()
            dev_marks_logits, dev_gaps_pred \
                    = dev_simulator.simulate(model,
                                            last_dev_times_in,
                                            dev_gaps_pred[:, -1:],
                                            dev_begin_tss,
                                            dev_t_b_plus,
                                            decoder_length,
                                            normalizers=(dev_normalizer_d, dev_normalizer_a),
                                            initial_timestamp=initial_timestamp,
                                            marks_in=dev_marks_pred_last)
        model.rnn_layer.reset_states()

        start_time = time.time()
        for test_step, (test_marks_in, test_gaps_in, test_times_in, test_seqmask_in, test_time_feature) \
                in enumerate(test_dataset):

            test_marks_logits, test_gaps_pred, _, _ = model(test_gaps_in,
                                                            test_seqmask_in,
                                                            test_marks_in,
                                                            test_time_feature)
            if use_marks:
                test_marks_pred = tf.argmax(test_marks_logits, axis=-1) + 1
                test_marks_pred_last = test_marks_pred[:, -1:]
            else:
                test_marks_pred_last = None

            last_test_times_in = tf.gather(test_times_in,
                                          test_seq_lens-1,
                                          batch_dims=1)
            test_simulator = models.SimulateRMTPP()
            test_marks_logits, test_gaps_pred \
                    = test_simulator.simulate(model,
                                              last_test_times_in,
                                              test_gaps_pred[:, -1:],
                                              test_begin_tss,
                                              test_t_b_plus,
                                              decoder_length,
                                              normalizers=(test_normalizer_d, test_normalizer_a),
                                              initial_timestamp=initial_timestamp,
                                              marks_in=test_marks_pred_last)
        model.rnn_layer.reset_states()
        end_time = time.time()
        inference_times.append(end_time-start_time)

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


        dev_gaps_pred = (dev_gaps_pred - dev_normalizer_a) * dev_normalizer_d
        test_gaps_pred = (test_gaps_pred - test_normalizer_a) * test_normalizer_d

        print('\ndev_gaps_pred')
        print(tf.squeeze(dev_gaps_pred[:, 1:], axis=-1))
        print('\ndev_gaps_out')
        print(tf.squeeze(dev_gaps_out[:, 1:], axis=-1))

        end_of_input_seq = dev_seq_lens - 20
        dev_gaps_in_unnorm = data['dev_gaps_in'].numpy()
        dev_gaps_in_unnorm_lst = list()
        for x in range(len(dev_gaps_in_unnorm)):
            dev_gaps_in_unnorm_lst.append(dev_gaps_in_unnorm[x, end_of_input_seq[x][0]:dev_seq_lens[x][0]])
        dev_gaps_in_unnorm = np.array(dev_gaps_in_unnorm_lst)

        idx = 1
        true_gaps_plot = dev_gaps_out.numpy()[idx]
        pred_gaps_plot = dev_gaps_pred.numpy()[idx]
        inp_tru_gaps = dev_gaps_in_unnorm[idx]

        true_gaps_plot = list(inp_tru_gaps) + list(true_gaps_plot)
        pred_gaps_plot = list(inp_tru_gaps) + list(pred_gaps_plot)

        plot_dir = os.path.join(SAVE_DIR,'dev_plots_'+str(cntr))
        # if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        name_plot = os.path.join(plot_dir, 'epoch_' + str(epoch))

        assert len(true_gaps_plot) == len(pred_gaps_plot)

        fig_pred_gaps = plt.figure()
        ax1 = fig_pred_gaps.add_subplot(111)
        ax1.plot(list(range(1, len(pred_gaps_plot)+1)), pred_gaps_plot, 'r*-', label='Pred gaps')
        ax1.plot(list(range(1, len(true_gaps_plot)+1)), true_gaps_plot, 'bo-', label='True gaps')
        ax1.plot([BPTT-0.5, BPTT-0.5],
                 [0, max(np.concatenate([true_gaps_plot, pred_gaps_plot]))],
                 'g-')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Gaps')
        plt.grid()

        plt.savefig(name_plot+'.png')
        plt.close()

        dev_gap_metric(dev_gaps_out[:, 1:], dev_gaps_pred[:, 1:])
        test_gap_metric(test_gaps_out[:, 1:], test_gaps_pred[:, 1:])

        dev_gap_err = dev_gap_metric.result()
        test_gap_err = test_gap_metric.result()
        dev_gap_metric.reset_states()
        test_gap_metric.reset_states()

        if dev_gap_err < best_dev_gap_error:
            best_dev_gap_error = dev_gap_err
            best_test_gap_error = test_gap_err
            best_dev_mark_acc = dev_mark_acc
            best_test_mark_acc = test_mark_acc
            best_epoch = epoch + 1

            best_true_gaps_plot = dev_gaps_out.numpy()
            best_pred_gaps_plot = dev_gaps_pred.numpy()
            best_inp_tru_gaps = dev_gaps_in_unnorm

        print('Dev mark acc and gap err over epoch: %s, %s' \
                % (float(dev_mark_acc), float(dev_gap_err)))
        print('Test mark acc and gap err over epoch: %s, %s' \
                % (float(test_mark_acc), float(test_gap_err)))

        model.rnn_layer.reset_states()

print('Best Dev mark acc and gap err: %s, %s' \
        % (float(best_dev_mark_acc), float(best_dev_gap_error)))
print('Best Test mark acc and gap err: %s, %s' \
        % (float(best_test_mark_acc), float(best_test_gap_error)))
print('Best epoch:', best_epoch)

SAVE_DIR = './joint_plots/rmtpp/'
os.makedirs(SAVE_DIR, exist_ok=True)
cntr = 0
cntr = len(next(os.walk(SAVE_DIR))[1])

plot_dir = os.path.join(SAVE_DIR,'dev_plots_'+str(cntr))
os.makedirs(plot_dir, exist_ok=True)

for idx in range(len(best_inp_tru_gaps)):

    name_plot = os.path.join(plot_dir, 'seq_' + str(idx))

    true_gaps_plot = list(best_inp_tru_gaps[idx]) + list(best_true_gaps_plot[idx])
    pred_gaps_plot = list(best_inp_tru_gaps[idx]) + list(best_pred_gaps_plot[idx])
    assert len(true_gaps_plot) == len(pred_gaps_plot)

    fig_pred_gaps = plt.figure()
    ax1 = fig_pred_gaps.add_subplot(111)
    ax1.plot(list(range(1, len(pred_gaps_plot)+1)), pred_gaps_plot, 'r*-', label='Pred gaps')
    ax1.plot(list(range(1, len(true_gaps_plot)+1)), true_gaps_plot, 'bo-', label='True gaps')
    ax1.plot([BPTT-0.5, BPTT-0.5],
             [0, max(np.concatenate([true_gaps_plot, pred_gaps_plot]))],
             'g-')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Gaps')
    plt.grid()

    plt.savefig(name_plot+'.png')
    plt.close()

print('\n train_losses')
print(train_losses)
print('\n average inference time:', np.mean(inference_times))
