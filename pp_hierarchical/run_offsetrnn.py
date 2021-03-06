from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
import math
from bisect import bisect_right
import os, sys
#import ipdb
import time
from dtw import dtw
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import reader_offsetrnn
import reader_rmtpp
import models


#epochs = 100
#patience = 20
#
#BPTT = 20
#block_size = 1 # Number of hours in a block
#decoder_length = 5
#use_marks = False
#use_intensity = True


def compute_actual_event_in_range(t_b_plus, t_e_plus, times_out):
    times_out_indices_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(times_out, t_b_plus)]
    times_out_indices_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(times_out, t_e_plus)]

    actual_event_count = [times_out_indices_te[idx] - times_out_indices_tb[idx] for idx in range(len(t_b_plus))]
    times_out_indices = [times_out_indices_tb[idx] + tf.range(times_out_indices_te[idx]-times_out_indices_tb[idx]) for idx in range(len(t_b_plus))]
    actual_times_out = [tf.gather(times_out[idx], times_out_indices[idx], batch_dims=0).numpy() for idx in range(len(t_b_plus))]
    actual_gaps_out = [actual_times_out[idx][1:]-actual_times_out[idx][:-1] for idx in range(len(t_b_plus))]

    return actual_event_count, actual_times_out, actual_gaps_out

def DTW(time_preds, time_true):

    clipped_time_true = time_true#[:, :seq_limit]

    euclidean_norm = lambda x, y: np.abs(x - y)
    distance = 0
    for time_preds_, clipped_time_true_ in zip(time_preds, clipped_time_true):
        #TODO This is not right way
        if np.shape(time_preds_)[0] == 0:
            time_preds_ = np.array([0.0]).reshape(-1, 1)
        if np.shape(clipped_time_true_)[0] == 0:
            clipped_time_true_ = np.array([0.0]).reshape(-1, 1)
        d, cost_matrix, acc_cost_matrix, path = dtw(time_preds_, clipped_time_true_, dist=euclidean_norm)
        distance += d
    distance = distance / len(clipped_time_true)

    return distance

def run(args):

    if not args.training_mode:
        args.epochs = 1

    tf.random.set_seed(args.seed)
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path

    epochs = args.epochs
    patience = args.patience
    BPTT = args.bptt
    block_size = args.block_size
    decoder_length = args.decoder_length
    use_marks = args.use_marks
    use_intensity = args.use_intensity
    normalization = args.normalization

    hidden_layer_size = args.hidden_layer_size

    block_size_sec = 3600.0 * block_size


    data = reader_offsetrnn.get_preprocessed_data(dataset_name, dataset_path, block_size,
                                                  decoder_length, normalization)
    num_categories = data['num_categories']
    num_sequences = data['num_sequences']

    train_dataset = data['train_dataset']

    # ----- Start: Load dev_dataset ----- #
    dev_dataset = data['dev_dataset']
    dev_seq_lens = data['dev_seq_lens']
    dev_seq_lens_in = tf.cast(tf.reduce_sum(data['dev_seqmask_in'], axis=-1), tf.int32)
    dev_seq_lens_out = tf.cast(tf.reduce_sum(data['dev_seqmask_out'], axis=-1), tf.int32)
    dev_marks_out = data['dev_marks_out']
    dev_gaps_out = data['dev_gaps_out']
    dev_times_out = data['dev_times_out']
    dev_begin_tss = data['dev_begin_tss']
    dev_offsets = tf.random.uniform(shape=(num_sequences, 1)) * 3600. * block_size
    #if args.training_mode:
    #    dev_offsets = tf.zeros_like(dev_offsets)
    dev_t_b_plus = dev_begin_tss + dev_offsets
    sample_hours = 5
    dev_offsets_t_e = tf.random.uniform(shape=(num_sequences, 1)) * 60. * sample_hours # Sampling offsets for t_e_+
    dev_t_e_plus = dev_t_b_plus + dev_offsets_t_e
    event_bw_range_tb_te = compute_actual_event_in_range(dev_t_b_plus, dev_t_e_plus, dev_times_out)

    if args.verbose:
        print('\n dev_begin_tss')
        for d in dev_begin_tss.numpy().tolist():
            print('{0:.15f}'.format(d[0]))
        print('\n dev_offsets')
        for d in dev_offsets.numpy().tolist():
            print('{0:.15f}'.format(d[0]))
        print('\n dev_t_b_plus')
        for d in dev_t_b_plus.numpy().tolist():
            print('{0:.15f}'.format(d[0]))

    dev_times_out_indices = [bisect_right(dev_t_out, t_b) for dev_t_out, t_b in zip(dev_times_out, dev_t_b_plus)]
    dev_times_out_indices = tf.minimum(dev_times_out_indices, dev_seq_lens_out-decoder_length+1)
    dev_times_out_indices = tf.expand_dims(dev_times_out_indices, axis=-1)
    print(dev_times_out_indices)
    print(dev_seq_lens_out)
    dev_times_out_indices \
            = (dev_times_out_indices-1) \
            + tf.expand_dims(tf.range(decoder_length), axis=0)
    print(dev_times_out_indices)
    dev_gaps_out = tf.gather(dev_gaps_out, dev_times_out_indices, batch_dims=1)

    if args.verbose:
        print('\ndev_times_out_indices')
        print(dev_times_out_indices)

    # ----- Normalize dev_offsets and dev_t_b_plus ----- #
    dev_normalizer_d = data['dev_normalizer_d']
    dev_normalizer_a = data['dev_normalizer_a']
    print(dev_offsets, dev_normalizer_d)
    dev_offsets_sec_norm = dev_offsets/dev_normalizer_d + dev_normalizer_a
    #dev_t_b_plus = dev_begin_tss + dev_offsets_sec_norm

    if args.verbose:
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
    if args.training_mode:
        test_offsets = tf.zeros_like(test_offsets)
    test_t_b_plus = test_begin_tss + test_offsets

    dev_offsets_sec_norm_t_e = dev_offsets_t_e/dev_normalizer_d + dev_normalizer_a
    dev_t_e_plus = dev_t_b_plus + dev_offsets_sec_norm_t_e

    if args.verbose:
        print('\n test_begin_tss')
        for d in test_begin_tss.numpy().tolist():
            print('{0:.15f}'.format(d[0]))
        print('\n test_offsets')
        for d in test_offsets.numpy().tolist():
            print('{0:.15f}'.format(d[0]))
        print('\n test_t_b_plus')
        for d in test_t_b_plus.numpy().tolist():
            print('{0:.15f}'.format(d[0]))


    test_times_out_indices = [bisect_right(test_t_out, t_b) for test_t_out, t_b in zip(test_times_out, test_t_b_plus)]
    test_times_out_indices = tf.minimum(test_times_out_indices, test_seq_lens_out-decoder_length+1)
    test_times_out_indices = tf.expand_dims(test_times_out_indices, axis=-1)
    test_times_out_indices \
            = (test_times_out_indices-1) \
            + tf.expand_dims(tf.range(decoder_length), axis=0)
    test_gaps_out = tf.gather(test_gaps_out, test_times_out_indices, batch_dims=1)

    if args.verbose:
        print('\ntest_times_out_indices')
        print(test_times_out_indices)

    # ----- Normalize test_offsets and test_t_b_plus ----- #
    test_normalizer_d = data['test_normalizer_d']
    test_normalizer_a = data['test_normalizer_a']
    print(test_offsets, test_normalizer_d)
    test_offsets_sec_norm = test_offsets/test_normalizer_d + test_normalizer_a
    #test_t_b_plus = test_begin_tss + test_offsets_sec_norm

    if args.verbose:
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
    train_dataset = train_dataset.batch(BPTT, drop_remainder=False).map(reader_offsetrnn.transpose)

    dev_dataset = dev_dataset.batch(num_sequences)
    test_dataset = test_dataset.batch(num_sequences)

    # Loss function
    mark_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if not use_intensity:
        gap_loss_fn_next = tf.keras.losses.MeanSquaredError()
        gap_loss_fn_off = tf.keras.losses.MeanSquaredError()

    # Evaluation metrics
    train_mark_metric_next = tf.keras.metrics.SparseCategoricalAccuracy()
    train_gap_metric_next = tf.keras.metrics.MeanAbsoluteError()
    train_mark_metric_off = tf.keras.metrics.SparseCategoricalAccuracy()
    train_gap_metric_off = tf.keras.metrics.MeanAbsoluteError()
    dev_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    dev_gap_metric = tf.keras.metrics.MeanAbsoluteError()
    test_mark_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_gap_metric = tf.keras.metrics.MeanAbsoluteError()

    model = models.OffsetRNN(num_categories, 8, hidden_layer_size,
                         use_marks=use_marks,
                         use_intensity=use_intensity)

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

    global_step = 0

    # Create checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               model=model,
                               optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join(args.ckpt_dir, 'ckpts'),
                                         max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # Create summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.output_dir, 'logs', current_time + '/train')
    dev_log_dir = os.path.join(args.output_dir, 'logs', current_time + '/dev')
    test_log_dir = os.path.join(args.output_dir, 'logs', current_time + '/test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    dev_summary_writer = tf.summary.create_file_writer(dev_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    best_dev_gap_error = np.inf
    best_test_gap_error = np.inf
    best_dev_mark_acc = np.inf
    best_test_mark_acc = np.inf
    best_epoch = 0

    train_losses = list()
    inference_times = list()
    # Iterate over epochs.
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        if args.training_mode:
            for step, (marks_batch_in, gaps_batch_in, times_batch_in, seqmask_batch_in,
                       marks_batch_out_next, gaps_batch_out_next, times_batch_out_next, seqmask_batch_out_next,
                       marks_batch_out_off, gaps_batch_out_off, times_batch_out_off, seqmask_batch_out_off,
                       train_time_features, train_offsets) \
                               in enumerate(train_dataset):

                with tf.GradientTape() as tape:

                    train_offsets = tf.expand_dims(train_offsets, axis=-1)
                    train_offsets_unnorm = train_offsets * (3600. * block_size)
                    train_time_offset_features \
                            = reader_rmtpp.get_time_features_for_data((times_batch_in + train_offsets_unnorm))

                    (marks_logits_next, gaps_pred_next, D_next, WT_next,
                     marks_logits_off, gaps_pred_off, D_off, WT_off) \
                            = model(gaps_batch_in,
                                    seqmask_batch_in,
                                    train_time_features,
                                    offsets=train_offsets,
                                    offset_features=train_time_offset_features,
                                    marks=marks_batch_in)

                    # Compute the loss for this minibatch.
                    if use_marks:
                        mark_loss_next = mark_loss_fn(marks_batch_out_next, marks_logits_next)
                        mark_loss_off = mark_loss_fn(marks_batch_out_off, marks_logits_off)
                    else:
                        mark_loss_next, mark_loss_off = 0.0, 0.0
                    if use_intensity:
                        gap_loss_fn_next = models.NegativeLogLikelihood(D_next, WT_next)
                        gap_loss_fn_off = models.NegativeLogLikelihood(D_off, WT_off)
                    gap_loss_next = gap_loss_fn_next(gaps_batch_out_next, gaps_pred_next)
                    gap_loss_off = gap_loss_fn_off(gaps_batch_out_off, gaps_pred_off)
                    loss = mark_loss_next + gap_loss_next + mark_loss_off + gap_loss_off
                    train_losses.append(loss.numpy())
                    with train_summary_writer.as_default():
                        tf.summary.scalar('gap_loss_next', gap_loss_next, step=global_step)
                        tf.summary.scalar('mark_loss_next', mark_loss_next, step=global_step)
                        tf.summary.scalar('gap_loss_off', gap_loss_off, step=global_step)
                        tf.summary.scalar('mark_loss_off', mark_loss_off, step=global_step)
                        tf.summary.scalar('loss', loss, step=global_step)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # TODO Make sure that padding is considered during evaluation
                if use_marks:
                    train_mark_metric_next(marks_batch_out_next, marks_logits_next)
                    train_mark_metric_off(marks_batch_out_off, marks_logits_off)
                train_gap_metric_next(gaps_batch_out_next, gaps_pred_next)
                train_gap_metric_off(gaps_batch_out_off, gaps_pred_off)

                print('Training loss (for one batch) at step %s: %s %s %s %s %s' \
                        % (step, float(loss), float(mark_loss_next), float(gap_loss_next),
                           float(mark_loss_off), float(gap_loss_off)))

                global_step += 1

            model.rnn_layer.reset_states() # Reset RNN state after
                                           # a sequence is finished

            if use_marks:
                train_mark_acc_next = train_mark_metric_next.result()
                train_mark_metric_next.reset_states()
            else:
                train_mark_acc_next = 0.0
            train_gap_err_next = train_gap_metric_next.result()
            train_gap_metric_next.reset_states()
            print('Training mark acc and gap err over epoch: %s, %s' \
                    % (float(train_mark_acc_next), float(train_gap_err_next)))

        if epoch > patience-1 or args.training_mode==0.0:

            for dev_step, (dev_marks_in, dev_gaps_in,
                           dev_times_in, dev_seqmask_in,
                           dev_time_feature) \
                    in enumerate(dev_dataset):

                dev_offsets_expand = tf.expand_dims(dev_offsets, axis=-1)
                dev_offsets_norm = dev_offsets_expand /  (3600. * block_size)
                dev_offsets_norm = tf.tile(dev_offsets_norm,
                                           [1, dev_gaps_in.shape[1], 1])
                print(dev_gaps_in.shape, dev_seqmask_in.shape, dev_marks_in.shape)
                dev_time_offset_features \
                        = reader_rmtpp.get_time_features_for_data((dev_times_in + dev_offsets_expand))

                (_, _, _, _,
                 dev_marks_logits_off, dev_gaps_pred_off, _, _ ) \
                         = model(dev_gaps_in,
                                 dev_seqmask_in,
                                 dev_time_feature,
                                 offsets=dev_offsets_norm,
                                 offset_features=dev_time_offset_features,
                                 marks=dev_marks_in)
                if use_marks:
                    dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
                    dev_marks_pred_last = dev_marks_pred[:, -1:]
                else:
                    dev_marks_pred_last = None

                last_dev_gaps_pred = tf.gather(dev_gaps_pred_off,
                                               dev_seq_lens-1,
                                               batch_dims=1)
                last_dev_gaps_pred = tf.squeeze(last_dev_gaps_pred, axis=1)
                last_dev_times_in = tf.gather(dev_times_in,
                                              dev_seq_lens-1,
                                              batch_dims=1)
                last_dev_times_in = tf.squeeze(last_dev_times_in, axis=1)
                last_dev_times_in += dev_offsets
                dev_simulator = models.SimulateOffsetRNN()


                # Query 1
                dev_marks_logits, dev_gaps_pred, _, _, _, \
                        = dev_simulator.simulate(model,
                                                last_dev_times_in,
                                                last_dev_gaps_pred,
                                                dev_begin_tss,
                                                dev_t_b_plus,
                                                decoder_length,
                                                normalizers=(dev_normalizer_d, dev_normalizer_a),
                                                marks_in=dev_marks_pred_last)

                # #Query 2 and 3
                # _, _, last_dev_time_pred, last_dev_gaps_pred, _, _ \
                #         = dev_simulator.simulate(model,
                #                                 last_dev_times_in,
                #                                 last_dev_gaps_pred,
                #                                 dev_begin_tss,
                #                                 dev_t_b_plus,
                #                                 0,
                #                                 normalizers=(dev_normalizer_d, dev_normalizer_a),
                #                                 initial_timestamp=initial_timestamp,
                #                                 marks_in=dev_marks_pred_last)

                # _, _, _, _, all_times_pred, simul_count \
                #         = dev_simulator.simulate(model,
                #                                 last_dev_time_pred,
                #                                 last_dev_gaps_pred,
                #                                 dev_begin_tss,
                #                                 dev_t_e_plus,
                #                                 0,
                #                                 normalizers=(dev_normalizer_d, dev_normalizer_a),
                #                                 initial_timestamp=initial_timestamp,
                #                                 marks_in=None)

                # # print('Events between t_b_plus and t_e_plus are at:', all_times_pred.numpy().tolist())
                # # print('Number of events between t_b_plus and t_e_plus are:', simul_count)

                # total_number_of_events_in_range = np.array(simul_count)
                # actual_event_count = np.array(event_bw_range_tb_te[0])
                # error_in_event_count = np.mean((total_number_of_events_in_range-actual_event_count)**2)

                # print('total_number_of_events_in_range', total_number_of_events_in_range)
                # print('Actual count of events:', actual_event_count)
                # # print('dev_t_b_plus', dev_t_b_plus)
                # # print('dev_t_e_plus', dev_t_e_plus)

                # actual_dev_event_in_range = event_bw_range_tb_te[1]
                # actual_dev_gaps_in_range = event_bw_range_tb_te[2]
                # dev_gaps_pred = dev_simulator.all_gaps_pred

                # dev_gaps_pred_in_range = (dev_gaps_pred[:, 1:] - dev_normalizer_a) * dev_normalizer_d
                # dev_gaps_pred_in_range = dev_gaps_pred_in_range.numpy()

                # dev_gaps_pred_in_range = [np.trim_zeros(dev_gaps_pred_in_range[idx], 'b') for idx in range(len(dev_t_b_plus))]
                # dtw_cost_in_range = DTW(dev_gaps_pred_in_range, actual_dev_gaps_in_range)

                # # print('actual_dev_event_in_range', actual_dev_event_in_range)
                # # print('dev_gaps_pred_in_range', dev_gaps_pred_in_range)
                # # print('actual_dev_gaps_in_range', actual_dev_gaps_in_range)
                # print('dtw_cost_in_range', dtw_cost_in_range)
                # print('error_in_event_count', error_in_event_count)

                #Query 4
                # dev_marks_logits, dev_gaps_pred, _, _ = model(dev_gaps_in,
                #                                           dev_seqmask_in,
                #                                           dev_marks_in,
                #                                           dev_time_feature)
                # if use_marks:
                #     dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
                #     dev_marks_pred_last = dev_marks_pred[:, -1:]
                # else:
                #     dev_marks_pred_last = None

                # last_dev_times_in = tf.gather(dev_times_in,
                #                               dev_seq_lens-1,
                #                               batch_dims=1)
                # last_dev_gaps_pred = tf.gather(dev_gaps_pred,
                #                                dev_seq_lens-1,
                #                                batch_dims=1)

                # last_dev_time_inp_pivot = last_dev_time_pred
                # last_dev_gaps_inp_pivot = last_dev_gaps_pred
                # while cnt>10:
                    
                #     dev_marks_logits, _, _, _, _, simul_count \
                #             = dev_simulator.simulate(model,
                #                                     last_dev_times_in,
                #                                     last_dev_gaps_pred,
                #                                     dev_begin_tss,
                #                                     dev_t_b_plus,
                #                                     decoder_length,
                #                                     normalizers=(dev_normalizer_d, dev_normalizer_a),
                #                                     initial_timestamp=initial_timestamp,
                #                                     marks_in=dev_marks_pred_last)



                #     cnt-=1


            model.rnn_layer.reset_states()

            start_time = time.time()
            for test_step, (test_marks_in, test_gaps_in,
                           test_times_in, test_seqmask_in,
                           test_time_feature) \
                    in enumerate(test_dataset):

                test_offsets_expand = tf.expand_dims(test_offsets, axis=-1)
                test_offsets_norm = test_offsets_expand /  (3600. * block_size)
                test_offsets_norm = tf.tile(test_offsets_norm,
                                           [1, test_gaps_in.shape[1], 1])
                print(test_gaps_in.shape, test_seqmask_in.shape, test_marks_in.shape)
                test_time_offset_features \
                        = reader_rmtpp.get_time_features_for_data((test_times_in + test_offsets_expand))

                (_, _, _, _,
                 test_marks_logits_off, test_gaps_pred_off, _, _ ) \
                         = model(test_gaps_in,
                                 test_seqmask_in,
                                 test_time_feature,
                                 offsets=test_offsets_norm,
                                 offset_features=test_time_offset_features,
                                 marks=test_marks_in)
                if use_marks:
                    test_marks_pred = tf.argmax(test_marks_logits, axis=-1) + 1
                    test_marks_pred_last = test_marks_pred[:, -1:]
                else:
                    test_marks_pred_last = None

                last_test_gaps_pred = tf.gather(test_gaps_pred_off,
                                               test_seq_lens-1,
                                               batch_dims=1)
                last_test_gaps_pred = tf.squeeze(last_test_gaps_pred, axis=1)
                last_test_times_in = tf.gather(test_times_in,
                                              test_seq_lens-1,
                                              batch_dims=1)
                last_test_times_in = tf.squeeze(last_test_times_in, axis=1)
                last_test_times_in += test_offsets
                test_simulator = models.SimulateOffsetRNN()


                # Query 1
                test_marks_logits, test_gaps_pred, _, _, _, \
                        = test_simulator.simulate(model,
                                                last_test_times_in,
                                                last_test_gaps_pred,
                                                test_begin_tss,
                                                test_t_b_plus,
                                                decoder_length,
                                                normalizers=(test_normalizer_d, test_normalizer_a),
                                                marks_in=test_marks_pred_last)
            model.rnn_layer.reset_states()
            end_time = time.time()
            inference_times.append(end_time-start_time)


###################################################################################
            if use_marks:
                dev_mark_metric(dev_marks_out, dev_marks_logits)
                test_mark_metric(test_marks_out, test_marks_logits)
                dev_mark_acc = dev_mark_metric.result()
                test_mark_acc = test_mark_metric.result()
                dev_mark_metric.reset_states()
                test_mark_metric.reset_states()
            else:
                dev_mark_acc, test_mark_acc = 0.0, 0.0


            #dev_gaps_pred = (dev_gaps_pred - dev_normalizer_a) * dev_normalizer_d
            #test_gaps_pred = (test_gaps_pred - test_normalizer_a) * test_normalizer_d

            dev_gap_metric(dev_gaps_out[:, 1:], dev_gaps_pred[:, 1:])
            test_gap_metric(test_gaps_out[:, 1:], test_gaps_pred[:, 1:])

            dev_gap_err = dev_gap_metric.result()
            test_gap_err = test_gap_metric.result()
            with dev_summary_writer.as_default():
                tf.summary.scalar('dev_gap_err', dev_gap_err, step=epoch)
            with test_summary_writer.as_default():
                tf.summary.scalar('test_gap_err', test_gap_err, step=epoch)
                #TODO Add marks summary later

            dev_gap_metric.reset_states()
            test_gap_metric.reset_states()

            if args.verbose:
                print('\ndev_gaps_pred')
                print(tf.squeeze(dev_gaps_pred[:, 1:], axis=-1))
                print('\ndev_gaps_out')
                print(tf.squeeze(dev_gaps_out[:, 1:], axis=-1))

            if args.generate_plots:
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

                plot_dir = os.path.join(args.output_dir, 'plots', 'dev_plots')
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

            if dev_gap_err < best_dev_gap_error:
                best_dev_gap_error = dev_gap_err
                best_test_gap_error = test_gap_err
                best_dev_mark_acc = dev_mark_acc
                best_test_mark_acc = test_mark_acc
                if args.training_mode:
                    best_epoch = epoch + 1

                    save_path = manager.save()
                    print("Saved checkpoint for epoch %s" % (epoch))

                if args.generate_plots:
                    best_true_gaps_plot = dev_gaps_out.numpy()
                    best_pred_gaps_plot = dev_gaps_pred.numpy()
                    best_inp_tru_gaps = dev_gaps_in_unnorm

            print('Dev mark acc and gap err over epoch: %s, %s' \
                    % (float(dev_mark_acc), float(dev_gap_err)))
            print('Test mark acc and gap err over epoch: %s, %s' \
                    % (float(test_mark_acc), float(test_gap_err)))

    print('Best Dev mark acc and gap err: %s, %s' \
            % (float(best_dev_mark_acc), float(best_dev_gap_error)))
    print('Best Test mark acc and gap err: %s, %s' \
            % (float(best_test_mark_acc), float(best_test_gap_error)))
    print('Best epoch:', best_epoch)

    if args.generate_plots and args.training_mode==0.0:
        plot_dir = os.path.join(args.output_dir, 'joint_plots', 'dev_plots')
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

    if args.verbose:
        print('\n train_losses')
        print(train_losses)
        print('\n average inference time:', np.mean(inference_times))


    return {
            'best_dev_gap_error': float(best_dev_gap_error.numpy()),
            'best_test_gap_error': float(best_test_gap_error.numpy()),
            'best_epoch': best_epoch,
            'average_inference_time': np.mean(inference_times),
           }
