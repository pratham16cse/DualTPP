from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_right
import os, sys

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
patience = 20

batch_size = 2
BPTT = 20
block_size = 1
block_size_sec = 3600.0 * block_size
decoder_length = 5
use_marks = False
use_intensity = False
data = reader_hierarchical.get_preprocessed_data(block_size, decoder_length)
num_categories = data['num_categories']
num_sequences = data['num_sequences']

c_train_dataset = data['c_train_dataset']

best_dev_gap_error = np.inf
best_test_gap_error = np.inf
best_dev_mark_acc = np.inf
best_test_mark_acc = np.inf

# ----- Start: Load dev_dataset ----- #
c_dev_dataset = data['c_dev_dataset']
c_dev_seq_lens = data['c_dev_seq_lens']
c_dev_seq_lens_in = tf.cast(tf.reduce_sum(data['c_dev_seqmask_in'], axis=-1), tf.int32)
c_dev_seq_lens_out = tf.cast(tf.reduce_sum(data['c_dev_seqmask_out'], axis=-1), tf.int32)
dev_seq_lens_in = tf.cast(tf.reduce_sum(data['dev_seqmask_in'], axis=-1), tf.int32)
dev_seq_lens_out = tf.cast(tf.reduce_sum(data['dev_seqmask_out'], axis=-1), tf.int32)
dev_marks_out = data['dev_marks_out']
dev_gaps_out = data['dev_gaps_out']
dev_times_out = data['dev_times_out']
dev_begin_tss = data['dev_begin_tss']
dev_offsets = tf.random.uniform(shape=(num_sequences, 1)) * 3600. * block_size # Sampling offsets
dev_t_b_plus = dev_begin_tss + dev_offsets
print(dev_offsets)
print(tf.squeeze(dev_times_out, axis=-1).numpy().tolist()[0])
dev_times_out_indices = [bisect_right(dev_t_out, t_b) for dev_t_out, t_b \
                            in zip(dev_times_out, dev_t_b_plus)]
dev_times_out_indices = tf.minimum(dev_times_out_indices, dev_seq_lens_out-decoder_length+1)
print('\ndev_seq_lens_out', dev_seq_lens_out)
dev_times_out_indices = tf.expand_dims(dev_times_out_indices, axis=-1)
dev_times_out_indices \
        = (dev_times_out_indices-1) \
        + tf.expand_dims(tf.range(decoder_length), axis=0)
print('\ndev_times_out_indices')
print(dev_times_out_indices)
dev_gaps_out = tf.gather(dev_gaps_out, dev_times_out_indices, batch_dims=1)

# ----- Normalize dev_offsets and dev_t_b_plus ----- #
c_dev_normalizer_d = data['c_dev_normalizer_d']
c_dev_normalizer_a = data['c_dev_normalizer_a']
c_dev_offsets_sec_norm = dev_offsets/c_dev_normalizer_d + c_dev_normalizer_a
c_dev_t_b_plus = dev_begin_tss + c_dev_offsets_sec_norm

dev_normalizer_d = data['dev_normalizer_d']
dev_normalizer_a = data['dev_normalizer_a']
dev_offsets_sec_norm = dev_offsets/dev_normalizer_d + dev_normalizer_a
dev_t_b_plus = dev_begin_tss + dev_offsets_sec_norm

# ----- End: Load dev_dataset ----- #

# ----- Start: Load test_dataset ----- #
c_test_dataset = data['c_test_dataset']
c_test_seq_lens = data['c_test_seq_lens']
c_test_seq_lens_in = tf.cast(tf.reduce_sum(data['c_test_seqmask_in'], axis=-1), tf.int32)
c_test_seq_lens_out = tf.cast(tf.reduce_sum(data['c_test_seqmask_out'], axis=-1), tf.int32)
test_seq_lens_in = tf.cast(tf.reduce_sum(data['test_seqmask_in'], axis=-1), tf.int32)
test_seq_lens_out = tf.cast(tf.reduce_sum(data['test_seqmask_out'], axis=-1), tf.int32)
test_marks_out = data['test_marks_out']
test_gaps_out = data['test_gaps_out']
test_times_out = data['test_times_out']
test_begin_tss = data['test_begin_tss']
test_offsets = tf.random.uniform(shape=(num_sequences, 1)) * 3600. * block_size # Sampling offsets
test_t_b_plus = test_begin_tss + test_offsets
print(test_offsets)
print(tf.squeeze(test_times_out, axis=-1).numpy().tolist()[0])
test_times_out_indices = [bisect_right(test_t_out, t_b) for test_t_out, t_b \
                            in zip(test_times_out, test_t_b_plus)]
test_times_out_indices = tf.minimum(test_times_out_indices, test_seq_lens_out-decoder_length+1)
print('\ntest_seq_lens_out', test_seq_lens_out)
test_times_out_indices = tf.expand_dims(test_times_out_indices, axis=-1)
test_times_out_indices \
        = (test_times_out_indices-1) \
        + tf.expand_dims(tf.range(decoder_length), axis=0)
print('\ntest_times_out_indices')
print(test_times_out_indices)
test_gaps_out = tf.gather(test_gaps_out, test_times_out_indices, batch_dims=1)

# ----- Normalize test_offsets and test_t_b_plus ----- #
c_test_normalizer_d = data['c_test_normalizer_d']
c_test_normalizer_a = data['c_test_normalizer_a']
c_test_offsets_sec_norm = test_offsets/c_test_normalizer_d + c_test_normalizer_a
c_test_t_b_plus = test_begin_tss + c_test_offsets_sec_norm

test_normalizer_d = data['test_normalizer_d']
test_normalizer_a = data['test_normalizer_a']
test_offsets_sec_norm = test_offsets/test_normalizer_d + test_normalizer_a
test_t_b_plus = test_begin_tss + test_offsets_sec_norm

# ----- End: Load test_dataset ----- #

tile_shape = dev_gaps_out.get_shape().as_list()
tile_shape[0] = tile_shape[2] = 1
dev_normalizer_d = tf.tile(tf.expand_dims(dev_normalizer_d, axis=1), tile_shape)
dev_normalizer_a = tf.tile(tf.expand_dims(dev_normalizer_a, axis=1), tile_shape)
tile_shape = test_gaps_out.get_shape().as_list()
tile_shape[0] = tile_shape[2] = 1
test_normalizer_d = tf.tile(tf.expand_dims(test_normalizer_d, axis=1), tile_shape)
test_normalizer_a = tf.tile(tf.expand_dims(test_normalizer_a, axis=1), tile_shape)

#c_train_dataset = c_train_dataset.batch(BPTT, drop_remainder=True).map(reader_hierarchical.transpose)
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

SAVE_DIR = './plots/hierarchical/'
os.makedirs(SAVE_DIR, exist_ok=True)
cntr = 0
cntr = len(next(os.walk(SAVE_DIR))[1])
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
                            l2_mask=c_seqmask_batch_in,
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
                    = model(None, l2_gaps=c_dev_gaps_in, l2_mask=c_dev_seqmask_in)
            #dev_marks_logits, dev_gaps_pred, _, _ = model(c_dev_gaps_in, c_dev_marks_in)
            if use_marks:
                dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
                dev_marks_pred_last = dev_marks_pred[:, -1:]
            else:
                dev_marks_pred_last = None

            last_c_dev_times_in = tf.gather(c_dev_times_in,
                                            c_dev_seq_lens-1,
                                            batch_dims=1)
            print('Inputs shape:', c_dev_gaps_in.shape)
            dev_gaps_pred \
                    = models.simulate_hierarchicalrnn(model,
                                                      last_c_dev_times_in,
                                                      dev_l2_gaps_pred,
                                                      dev_begin_tss,
                                                      dev_t_b_plus,
                                                      c_dev_t_b_plus,
                                                      decoder_length)
        model.reset_states()

        for test_step, (c_test_marks_in, c_test_gaps_in, c_test_times_in, c_test_seqmask_in) \
                in enumerate(c_test_dataset):

            print('Inputs shape:', c_test_gaps_in.shape)
            (test_l2_marks_logits, test_l2_gaps_pred, _, _,
             test_l1_marks_logits, test_l1_gaps_pred, _, _) \
                    = model(None, l2_gaps=c_test_gaps_in, l2_mask=c_test_seqmask_in)
            #test_marks_logits, test_gaps_pred, _, _ = model(c_test_gaps_in, c_test_marks_in)
            if use_marks:
                test_marks_pred = tf.argmax(test_marks_logits, axis=-1) + 1
                test_marks_pred_last = test_marks_pred[:, -1:]
            else:
                test_marks_pred_last = None

            last_c_test_times_in = tf.gather(c_test_times_in,
                                             c_test_seq_lens-1,
                                             batch_dims=1)
            test_gaps_pred \
                    = models.simulate_hierarchicalrnn(model,
                                                      last_c_test_times_in,
                                                      test_l2_gaps_pred,
                                                      test_begin_tss,
                                                      test_t_b_plus,
                                                      c_test_t_b_plus,
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

        dev_gaps_pred = (dev_gaps_pred - dev_normalizer_a) * dev_normalizer_d
        test_gaps_pred = (test_gaps_pred - test_normalizer_a) * test_normalizer_d

        print('\ndev_gaps_pred')
        print(tf.squeeze(dev_gaps_pred[:, 1:], axis=-1))
        print('\ndev_gaps_out')
        print(tf.squeeze(dev_gaps_out[:, 1:], axis=-1))

        dev_gaps_in_unnorm = data['dev_gaps_in'][:, -20:]

        idx = 1
        true_gaps_plot = dev_gaps_out.numpy()[idx]
        pred_gaps_plot = dev_gaps_pred.numpy()[idx]
        inp_tru_gaps = dev_gaps_in_unnorm.numpy()[idx]

        true_gaps_plot = list(inp_tru_gaps) + list(true_gaps_plot)
        pred_gaps_plot = list(inp_tru_gaps) + list(pred_gaps_plot)

        plot_dir = os.path.join(SAVE_DIR,'dev_plots_'+str(cntr))
        # if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        name_plot = os.path.join(plot_dir, 'epoch_' + str(epoch))

        assert len(true_gaps_plot) == len(pred_gaps_plot)

        fig_pred_gaps = plt.figure()
        ax1 = fig_pred_gaps.add_subplot(111)
        ax1.scatter(list(range(1, len(pred_gaps_plot)+1)), pred_gaps_plot, c='r', label='Pred gaps')
        ax1.scatter(list(range(1, len(true_gaps_plot)+1)), true_gaps_plot, c='b', label='True gaps')
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
            
            best_true_gaps_plot = dev_gaps_out.numpy()
            best_pred_gaps_plot = dev_gaps_pred.numpy()
            best_inp_tru_gaps = dev_gaps_in_unnorm.numpy()

        print('Dev mark acc and gap err over epoch: %s, %s' \
                % (float(dev_mark_acc), float(dev_gap_err)))
        print('Test mark acc and gap err over epoch: %s, %s' \
                % (float(test_mark_acc), float(test_gap_err)))

        model.reset_states()

print('Best Dev mark acc and gap err over epoch: %s, %s' \
        % (float(best_dev_mark_acc), float(best_dev_gap_error)))
print('Best Test mark acc and gap err over epoch: %s, %s' \
        % (float(best_test_mark_acc), float(best_test_gap_error)))

SAVE_DIR = './joint_plots/hierarchical/'
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

    print(true_gaps_plot)
    print(len(true_gaps_plot))

    fig_pred_gaps = plt.figure()
    ax1 = fig_pred_gaps.add_subplot(111)
    ax1.scatter(list(range(1, len(pred_gaps_plot)+1)), pred_gaps_plot, c='r', label='Pred gaps')
    ax1.scatter(list(range(1, len(true_gaps_plot)+1)), true_gaps_plot, c='b', label='True gaps')
    ax1.plot([BPTT-0.5, BPTT-0.5],
             [0, max(np.concatenate([true_gaps_plot, pred_gaps_plot]))],
             'g-')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Gaps')
    plt.grid()

    plt.savefig(name_plot+'.png')
    plt.close()