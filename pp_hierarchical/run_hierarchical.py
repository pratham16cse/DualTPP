from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_right
import os, sys
import ipdb
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import reader_hierarchical
import models

#epochs = 100
#patience = 20
#
#batch_size = 2
#BPTT = 20
#block_size = 1
#decoder_length = 5
#use_marks = False
#use_intensity = True


def run(args):
    tf.random.set_seed(args.seed)
    dataset_path = args.dataset_path

    epochs = args.epochs
    patience = args.patience
    BPTT = args.bptt
    block_size = args.block_size
    decoder_length = args.decoder_length
    use_marks = args.use_marks
    use_intensity = args.use_intensity
    normalization = args.normalization
    compound_event_size = args.compound_event_size


    data = reader_hierarchical.get_preprocessed_data(dataset_path, block_size,
                                                     decoder_length,
                                                     normalization,
                                                     compound_event_size)
    num_categories = data['num_categories']
    num_sequences = data['num_sequences']

    c_train_dataset = data['c_train_dataset']
    c_train_normalizer_d = data['c_train_normalizer_d']
    c_train_normalizer_a = data['c_train_normalizer_a']
    train_normalizer_d = data['train_normalizer_d']
    train_normalizer_a = data['train_normalizer_a']

    best_dev_gap_error = np.inf
    best_test_gap_error = np.inf
    best_dev_mark_acc = np.inf
    best_test_mark_acc = np.inf
    best_epoch = 0

    # ----- Start: Load dev_dataset ----- #
    c_dev_dataset = data['c_dev_dataset']
    c_dev_seq_lens = data['c_dev_seq_lens']
    dev_seq_lens = data['dev_seq_lens']
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
    dev_times_out_indices = [bisect_right(dev_t_out, t_b) for dev_t_out, t_b \
                                in zip(dev_times_out, dev_t_b_plus)]
    dev_times_out_indices = tf.minimum(dev_times_out_indices, dev_seq_lens_out-decoder_length+1)
    dev_times_out_indices = tf.expand_dims(dev_times_out_indices, axis=-1)
    dev_times_out_indices \
            = (dev_times_out_indices-1) \
            + tf.expand_dims(tf.range(decoder_length), axis=0)
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
    test_seq_lens = data['test_seq_lens']
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
    test_times_out_indices = [bisect_right(test_t_out, t_b) for test_t_out, t_b \
                                in zip(test_times_out, test_t_b_plus)]
    test_times_out_indices = tf.minimum(test_times_out_indices, test_seq_lens_out-decoder_length+1)
    test_times_out_indices = tf.expand_dims(test_times_out_indices, axis=-1)
    test_times_out_indices \
            = (test_times_out_indices-1) \
            + tf.expand_dims(tf.range(decoder_length), axis=0)
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

    c_train_gap_metric = tf.keras.metrics.MeanAbsoluteError()

    model = models.HierarchicalRNN(num_categories, 8, 32, use_marks=use_marks,
                                   use_intensity=use_intensity)

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Create checkpoint manager
    ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                               model=model,
                               optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt,
                                         os.path.join(args.output_dir, 'ckpts'),
                                         max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train_c_losses = list()
    train_losses = list()
    inference_times = list()
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
                        = model(None, c_gaps_batch_in, c_seqmask_batch_in,
                                gaps_batch_in, seqmask_batch_in)

                # Apply mask on l1_gaps_pred
                l1_gaps_pred = l1_gaps_pred * tf.expand_dims(tf.expand_dims(c_seqmask_batch_in, axis=-1), axis=-1)
                #TODO Compute MASKED-losses manually instead of using tf helper functions

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
                train_losses.append(gap_loss.numpy())
                train_c_losses.append(c_gap_loss.numpy())

            c_train_gap_metric(c_gaps_batch_out, l2_gaps_pred)
            c_train_gap_err = c_train_gap_metric.result()

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # TODO Make sure that padding is considered during evaluation
            if use_marks:
                train_mark_metric(marks_batch_out, marks_logits)
            train_gap_metric(gaps_batch_out, l1_gaps_pred)

            print('Training loss (for one batch) at step %s: %s %s %s %s' \
                    % (step, float(loss), float(mark_loss), float(c_gap_loss), float(gap_loss)))

            # ----- Training nowcasting plots for layer 2 and layer 1 ----- #
            # For testdata:
            #   c_train_normalizer_d: 66.709526
            #   train_normalizer_d: 6.667143
            # For sin data:
            #   c_train_normalizer_d: 104.655846
            #   train_normalizer_d: 10.457432
            #c_gaps_batch_out_unnorm = (c_gaps_batch_out) * 104.655846
            #l2_gaps_pred_unnorm = (l2_gaps_pred) * 104.655846
            #gaps_batch_out_unnorm = (gaps_batch_out) * 10.457432
            #l1_gaps_pred_unnorm = (l1_gaps_pred) * 10.457432

            #if epoch > patience-1:
            #    print('\nc_train_batch_gaps_out')
            #    print(tf.squeeze(c_gaps_batch_out_unnorm[0], axis=-1))
            #    print('\nc_train_batch_gaps_pred')
            #    print(tf.squeeze(l2_gaps_pred_unnorm[0], axis=-1))
            #    plt.plot(tf.squeeze(c_gaps_batch_out_unnorm[0], axis=-1), 'bo-')
            #    plt.plot(tf.squeeze(l2_gaps_pred_unnorm[0], axis=-1), 'r*-')
            #    plot_dir_l2_trn = os.path.join(args.output_dir, 'plots_l2_trn', 'trn_plots')
            #    os.makedirs(plot_dir_l2_trn, exist_ok=True)
            #    name_plot = os.path.join(plot_dir_l2_trn, 'epoch_' + str(epoch))
            #    plt.savefig(name_plot+'.png')
            #    plt.close()

            #    print('\ntrain_batch_gaps_out')
            #    print(tf.squeeze(gaps_batch_out_unnorm[0][0], axis=-1))
            #    print('\ntrain_batch_gaps_pred')
            #    print(tf.squeeze(l1_gaps_pred_unnorm[0][0], axis=-1))
            #    plt.plot(tf.squeeze(gaps_batch_out_unnorm[0][0], axis=-1), 'bo-')
            #    plt.plot(tf.squeeze(l1_gaps_pred_unnorm[0][0], axis=-1), 'r*-')
            #    plot_dir_l1_trn = os.path.join(args.output_dir, 'plots_l1_trn', 'trn_plots')
            #    os.makedirs(plot_dir_l1_trn, exist_ok=True)
            #    name_plot = os.path.join(plot_dir_l1_trn, 'epoch_' + str(epoch))
            #    plt.savefig(name_plot+'.png')
            #    plt.close()

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

        if epoch > patience-1:

            for dev_step, (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in, c_dev_seqmask_in,
                           c_dev_gaps_out, c_dev_times_out, c_dev_seqmask_out) \
                    in enumerate(c_dev_dataset):

                (dev_l2_marks_logits, dev_l2_gaps_pred, _, _,
                 _, _, _, _) \
                        = model(None, c_dev_gaps_in, c_dev_seqmask_in)

                if use_marks:
                    dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
                    dev_marks_pred_last = dev_marks_pred[:, -1:]
                else:
                    dev_marks_pred_last = None
                #ipdb.set_trace()

                dev_simulator = models.SimulateHierarchicalRNN()
                dev_gaps_pred \
                        = dev_simulator.simulate(model,
                                                 c_dev_times_in,
                                                 dev_l2_gaps_pred,
                                                 c_dev_seq_lens,
                                                 dev_begin_tss,
                                                 dev_t_b_plus,
                                                 c_dev_t_b_plus,
                                                 decoder_length)
            model.reset_states()

            start_time = time.time()
            for test_step, (c_test_marks_in, c_test_gaps_in, c_test_times_in, c_test_seqmask_in,
                            c_test_gaps_out, c_dev_times_out, c_test_seqmask_out) \
                    in enumerate(c_test_dataset):

                (test_l2_marks_logits, test_l2_gaps_pred, _, _,
                 _, _, _, _) \
                        = model(None, c_test_gaps_in, c_test_seqmask_in)

                if use_marks:
                    test_marks_pred = tf.argmax(test_marks_logits, axis=-1) + 1
                    test_marks_pred_last = test_marks_pred[:, -1:]
                else:
                    test_marks_pred_last = None

                test_simulator = models.SimulateHierarchicalRNN()
                test_gaps_pred \
                        = test_simulator.simulate(model,
                                                  c_test_times_in,
                                                  test_l2_gaps_pred,
                                                  c_test_seq_lens,
                                                  test_begin_tss,
                                                  test_t_b_plus,
                                                  c_test_t_b_plus,
                                                  decoder_length)
            model.reset_states()
            end_time = time.time()
            inference_times.append(end_time-start_time)

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

            dev_gap_metric(dev_gaps_out[:, 1:], dev_gaps_pred[:, 1:])
            test_gap_metric(test_gaps_out[:, 1:], test_gaps_pred[:, 1:])
            dev_gap_err = dev_gap_metric.result()
            test_gap_err = test_gap_metric.result()
            dev_gap_metric.reset_states()
            test_gap_metric.reset_states()

            if args.verbose:
                print('\ndev_gaps_pred')
                print(tf.squeeze(dev_gaps_pred[:, 1:], axis=-1))
                print('\ndev_gaps_out')
                print(tf.squeeze(dev_gaps_out[:, 1:], axis=-1))

            if args.generate_plots:
                # ----- Dev nowcasting plots for layer 2 ----- #
                plot_dir_l2 = os.path.join(args.output_dir, 'plots_l2', 'dev_plots')
                os.makedirs(plot_dir_l2, exist_ok=True)
                name_plot = os.path.join(plot_dir_l2, 'epoch_' + str(epoch))

                all_c_dev_gaps_pred = dev_simulator.all_l2_gaps_pred
                all_c_dev_gaps_pred = (all_c_dev_gaps_pred) * tf.expand_dims(c_dev_normalizer_d, axis=1)

                plt.plot(tf.squeeze(c_dev_gaps_out[1], axis=-1), 'bo-')
                plt.plot(tf.squeeze(all_c_dev_gaps_pred[1][1:len(c_dev_gaps_out[1])+1], axis=-1), 'r*-')
                plt.savefig(name_plot+'.png')
                plt.close()

                end_of_input_seq = dev_seq_lens - 20
                dev_gaps_in_unnorm = data['dev_gaps_in'].numpy()
                dev_gaps_in_unnorm_lst = list()
                for x in range(len(dev_gaps_in_unnorm)):
                    dev_gaps_in_unnorm_lst.append(dev_gaps_in_unnorm[x, end_of_input_seq[x][0]:dev_seq_lens[x][0]])
                dev_gaps_in_unnorm = np.array(dev_gaps_in_unnorm_lst)


                plot_dir = os.path.join(args.output_dir, 'plots', 'dev_plots')
                os.makedirs(plot_dir, exist_ok=True)
                idx = 1
                true_gaps_plot = dev_gaps_out.numpy()[idx]
                pred_gaps_plot = dev_gaps_pred.numpy()[idx]
                inp_tru_gaps = dev_gaps_in_unnorm[idx]

                true_gaps_plot = list(inp_tru_gaps) + list(true_gaps_plot)
                pred_gaps_plot = list(inp_tru_gaps) + list(pred_gaps_plot)

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

        print('l2_train gap err over epoch: %s' \
                % (float(c_train_gap_err)))
        c_train_gap_metric.reset_states()

    print('Best Dev mark acc, gap err: %s, %s' \
            % (float(best_dev_mark_acc), float(best_dev_gap_error)))
    print('Best Test mark acc and gap err: %s, %s' \
            % (float(best_test_mark_acc), float(best_test_gap_error)))
    print('Best epoch:', best_epoch)

    if args.generate_plots:
        plot_dir = os.path.join(args.output_dir, 'joint_plots', 'dev_plots_')
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
        print('\n train_c_losses')
        print(train_c_losses)
        print('\n average infernece time:', np.mean(inference_times))
