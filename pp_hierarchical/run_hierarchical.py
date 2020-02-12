from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import matplotlib.pyplot as plt
import numpy as np
from bisect import bisect_right
import os, sys
#import ipdb
import time
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import reader_hierarchical
import models
from dtw import dtw
#epochs = 100
#patience = 20
#
#batch_size = 2
#BPTT = 20
#block_size = 1
#decoder_length = 5
#use_marks = False
#use_intensity = True



def process_query_1(model, arguments, params, extra_args):
    (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in, c_dev_seqmask_in,
    c_dev_gaps_out, c_dev_times_out, c_dev_seqmask_out, c_dev_time_feature) = arguments

    (use_marks) = params

    (c_dev_seq_lens, c_dev_t_b_plus, c_dev_t_e_plus, dev_t_b_plus, 
        dev_t_e_plus, decoder_length, dev_begin_tss) = extra_args

    (dev_l2_marks_logits, dev_l2_gaps_pred, _, _,_, _, _, _) \
            = model(c_dev_gaps_in, c_dev_seqmask_in, c_dev_time_feature)

    if use_marks:
        dev_marks_pred = tf.argmax(dev_l2_marks_logits, axis=-1) + 1
        dev_marks_pred_last = dev_marks_pred[:, -1:]
    else:
        dev_marks_pred_last = None

    # ipdb.set_trace()
    # #TODO: Dev sequence lens cant be 1 but it is one in some case like in delhi dataset.
    # if any(c_dev_seq_lens==1):
    #     second_last_gaps_pred = None
    # else:
    #     second_last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-2, batch_dims=1)

    second_last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-2, batch_dims=1)
    last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-1, batch_dims=1)
    last_times_in = tf.gather(c_dev_times_in, c_dev_seq_lens-1, batch_dims=1)

    dev_simulator = models.SimulateHierarchicalRNN()
    all_l2_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, _, _, before_tb_hidden_state, _ \
                                   = dev_simulator.simulator(model.l2_rnn,
                                     last_times_in,
                                     last_gaps_pred,
                                     c_dev_seq_lens,
                                     dev_begin_tss,
                                     c_dev_t_b_plus,
                                     c_dev_t_b_plus,
                                     decoder_length,
                                     2,
                                     second_last_gaps_pred,)
    
    #TODO Process l2 gaps to l1  gaps
    # before_tb_gaps_pred

    last_gaps_pred = before_tb_gaps_pred / 10.0
    l1_rnn_init_state =  model.ff(before_tb_hidden_state)

    all_l1_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, dev_gaps_pred, _, _, simulation_count \
                                   = dev_simulator.simulator(model.l1_rnn,
                                     last_times_pred,
                                     last_gaps_pred,
                                     0,
                                     dev_begin_tss,
                                     dev_t_b_plus,
                                     dev_t_b_plus,
                                     decoder_length,
                                     1,
                                     initial_state=l1_rnn_init_state)

    return (dev_gaps_pred, all_l2_dev_gaps_pred)

def process_query_2(model, arguments, params, extra_args):
    (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in, c_dev_seqmask_in,
    c_dev_gaps_out, c_dev_times_out, c_dev_seqmask_out) = arguments

    (use_marks) = params

    (c_dev_seq_lens, c_dev_t_b_plus, c_dev_t_e_plus, dev_t_b_plus, 
        dev_t_e_plus, decoder_length, dev_begin_tss) = extra_args

    (dev_l2_marks_logits, dev_l2_gaps_pred, _, _,_, _, _, _) \
            = model(None, c_dev_gaps_in, c_dev_seqmask_in)

    if use_marks:
        dev_marks_pred = tf.argmax(dev_l2_marks_logits, axis=-1) + 1
        dev_marks_pred_last = dev_marks_pred[:, -1:]
    else:
        dev_marks_pred_last = None

    second_last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-2, batch_dims=1)
    last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-1, batch_dims=1)
    last_times_in = tf.gather(c_dev_times_in, c_dev_seq_lens-1, batch_dims=1)

    dev_simulator = models.SimulateHierarchicalRNN()
    all_l2_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, _, _, before_tb_hidden_state, _ \
                                   = dev_simulator.simulator(model.l2_rnn,
                                     last_times_in,
                                     last_gaps_pred,
                                     c_dev_seq_lens,
                                     dev_begin_tss,
                                     c_dev_t_b_plus,
                                     c_dev_t_b_plus,
                                     decoder_length,
                                     2,
                                     second_last_gaps_pred,)
    
    #TODO Process l2 gaps to l1  gaps
    # before_tb_gaps_pred

    last_gaps_pred = before_tb_gaps_pred / 10.0
    l1_rnn_init_state =  model.ff(before_tb_hidden_state)

    all_l1_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, dev_gaps_pred, _, l1_rnn_init_state, simulation_count \
                                   = dev_simulator.simulator(model.l1_rnn,
                                     last_times_pred,
                                     last_gaps_pred,
                                     0,
                                     dev_begin_tss,
                                     dev_t_b_plus,
                                     dev_t_b_plus,
                                     0,
                                     1,
                                     initial_state=l1_rnn_init_state)

    all_l1_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, dev_gaps_pred, _, _, simulation_count \
                                   = dev_simulator.simulator(model.l1_rnn,
                                     last_times_pred,
                                     before_tb_gaps_pred,
                                     0,
                                     dev_begin_tss,
                                     dev_t_e_plus,
                                     dev_t_e_plus,
                                     0,
                                     1,
                                     initial_state=l1_rnn_init_state)

    all_gaps_pred = tf.squeeze(all_l1_dev_gaps_pred, axis=-1)

    return (all_gaps_pred, dev_simulator.all_times_pred, simulation_count)

# def process_query_3(model, arguments, params, extra_args):
#     (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in, c_dev_seqmask_in,
#     c_dev_gaps_out, c_dev_times_out, c_dev_seqmask_out) = arguments

#     (dev_l2_marks_logits, dev_l2_gaps_pred, _, _,_, _, _, _) \
#             = model(None, c_dev_gaps_in, c_dev_seqmask_in)

#     if use_marks:
#         dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
#         dev_marks_pred_last = dev_marks_pred[:, -1:]
#     else:
#         dev_marks_pred_last = None

#     second_last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-2, batch_dims=1)
#     last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-1, batch_dims=1)
#     last_times_in = tf.gather(c_dev_times_in, c_dev_seq_lens-1, batch_dims=1)

#     total_number_of_events_till_tb = np.zeros(len(last_gaps_pred))
#     total_number_of_events_till_te = np.zeros(len(last_gaps_pred))

#     dev_simulator = models.SimulateHierarchicalRNN()

#     _, last_times_pred, before_tb_gaps_pred, _, _, before_tb_hidden_state, simulation_count \
#                                    = dev_simulator.simulator(model.l2_rnn,
#                                      last_times_in,
#                                      last_gaps_pred,
#                                      c_dev_seq_lens,
#                                      dev_begin_tss,
#                                      c_dev_t_b_plus,
#                                      c_dev_t_b_plus,
#                                      decoder_length,
#                                      2,
#                                      second_last_gaps_pred,)
    
#     _, last_times_pred, before_tb_gaps_pred, _, _, before_tb_hidden_state, simulation_count \
#                                    = dev_simulator.simulator(model.l2_rnn,
#                                      last_times_pred,
#                                      before_tb_gaps_pred,
#                                      c_dev_seq_lens,
#                                      dev_begin_tss,
#                                      c_dev_t_e_plus,
#                                      c_dev_t_e_plus,
#                                      decoder_length,
#                                      2,
#                                      second_last_gaps_pred,)

#     total_number_of_events_in_range = simulation_count * 10

#     #TODO Process l2 gaps to l1  gaps
#     # before_tb_gaps_pred

#     last_gaps_pred = before_tb_gaps_pred / 10.0
#     l1_rnn_init_state =  model.ff(before_tb_hidden_state)

#     all_l1_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, dev_gaps_pred, _, _, simulation_count \
#                                    = dev_simulator.simulator(model.l1_rnn,
#                                      last_times_pred,
#                                      last_gaps_pred,
#                                      0,
#                                      dev_begin_tss,
#                                      dev_t_e_plus,
#                                      dev_t_e_plus,
#                                      decoder_length,
#                                      1,
#                                      initial_state=l1_rnn_init_state)

#     total_number_of_events_in_range += simulation_count

#     # print(all_l1_dev_gaps_pred)
#     # ipdb.set_trace()

# def process_query_4(model, arguments, params, extra_args):
#     (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in, c_dev_seqmask_in,
#     c_dev_gaps_out, c_dev_times_out, c_dev_seqmask_out) = arguments

#     (dev_l2_marks_logits, dev_l2_gaps_pred, _, _,_, _, _, _) \
#             = model(None, c_dev_gaps_in, c_dev_seqmask_in)

#     if use_marks:
#         dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
#         dev_marks_pred_last = dev_marks_pred[:, -1:]
#     else:
#         dev_marks_pred_last = None

#     mask = np.zeros(len(dev_times_out))


#     second_last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-2, batch_dims=1)
#     last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-1, batch_dims=1)
#     last_times_in = tf.gather(c_dev_times_in, c_dev_seq_lens-1, batch_dims=1)

#     total_number_of_events_till_tb = np.zeros(len(last_gaps_pred))
#     total_number_of_events_till_te = np.zeros(len(last_gaps_pred))

#     dev_simulator = models.SimulateHierarchicalRNN()

#     _, last_times_pred, before_tb_gaps_pred, _, _, before_tb_hidden_state, simulation_count \
#                                    = dev_simulator.simulator(model.l2_rnn,
#                                      last_times_in,
#                                      last_gaps_pred,
#                                      c_dev_seq_lens,
#                                      dev_begin_tss,
#                                      c_dev_t_b_plus,
#                                      c_dev_t_b_plus,
#                                      decoder_length,
#                                      2,
#                                      second_last_gaps_pred,)
    
#     _, last_times_pred, before_tb_gaps_pred, _, _, before_tb_hidden_state, simulation_count \
#                                    = dev_simulator.simulator(model.l2_rnn,
#                                      last_times_pred,
#                                      before_tb_gaps_pred,
#                                      c_dev_seq_lens,
#                                      dev_begin_tss,
#                                      c_dev_t_e_plus,
#                                      c_dev_t_e_plus,
#                                      decoder_length,
#                                      2,
#                                      second_last_gaps_pred,)

#     total_number_of_events_in_range = simulation_count * 10

#     #TODO Process l2 gaps to l1  gaps
#     # before_tb_gaps_pred

#     last_gaps_pred = before_tb_gaps_pred / 10.0
#     l1_rnn_init_state =  model.ff(before_tb_hidden_state)

#     all_l1_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, dev_gaps_pred, _, _, simulation_count \
#                                    = dev_simulator.simulator(model.l1_rnn,
#                                      last_times_pred,
#                                      last_gaps_pred,
#                                      0,
#                                      dev_begin_tss,
#                                      dev_t_e_plus,
#                                      dev_t_e_plus,
#                                      decoder_length,
#                                      1,
#                                      initial_state=l1_rnn_init_state)

#     total_number_of_events_in_range += simulation_count

#     # print(all_l1_dev_gaps_pred)
#     # ipdb.set_trace()




def query_processor(query, model, arguments, params, extra_args):
    if query == 1:
            return process_query_1(model, arguments, params, extra_args)
    if query == 2:
            return process_query_2(model, arguments, params, extra_args)
    return None

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
    compound_event_size = args.compound_event_size

    hidden_layer_size = args.hidden_layer_size

    data = reader_hierarchical.get_preprocessed_data(dataset_name, dataset_path, block_size,
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
    if args.training_mode:
        dev_offsets = tf.zeros_like(dev_offsets)
    dev_t_b_plus = dev_begin_tss + dev_offsets
    sample_hours = 5
    dev_offsets_t_e = tf.random.uniform(shape=(num_sequences, 1)) * 60. * sample_hours # Sampling offsets for t_e_+
    dev_t_e_plus = dev_t_b_plus + dev_offsets_t_e
    event_bw_range_tb_te = compute_actual_event_in_range(dev_t_b_plus, dev_t_e_plus, dev_times_out)

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
    c_dev_t_b_plus = dev_begin_tss + dev_offsets

    dev_normalizer_d = data['dev_normalizer_d']
    dev_normalizer_a = data['dev_normalizer_a']
    dev_offsets_sec_norm = dev_offsets/dev_normalizer_d + dev_normalizer_a

    sample_hours = 5
    dev_offsets_t_e = tf.random.uniform(shape=(num_sequences, 1)) * 60. * sample_hours # Sampling offsets for t_e_+
    c_dev_offsets_sec_norm_t_e = dev_offsets_t_e/c_dev_normalizer_d + c_dev_normalizer_a
    c_dev_t_e_plus = dev_t_b_plus + c_dev_offsets_sec_norm_t_e
    dev_offsets_sec_norm_t_e = dev_offsets_t_e/dev_normalizer_d + dev_normalizer_a
    dev_t_e_plus = dev_t_b_plus + dev_offsets_sec_norm_t_e

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
    if args.training_mode:
        test_offsets = tf.zeros_like(test_offsets)
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
    c_test_t_b_plus = test_begin_tss + test_offsets

    test_normalizer_d = data['test_normalizer_d']
    test_normalizer_a = data['test_normalizer_a']
    test_offsets_sec_norm = test_offsets/test_normalizer_d + test_normalizer_a

    # ----- End: Load test_dataset ----- #

    c_dev_normalizer_d = tf.expand_dims(c_dev_normalizer_d, axis=1)
    c_dev_normalizer_a = tf.expand_dims(c_dev_normalizer_a, axis=1)
    c_test_normalizer_d = tf.expand_dims(c_test_normalizer_d, axis=1)
    c_test_normalizer_a = tf.expand_dims(c_test_normalizer_a, axis=1)

    dev_normalizer_d = tf.expand_dims(dev_normalizer_d, axis=1)
    dev_normalizer_a = tf.expand_dims(dev_normalizer_a, axis=1)
    test_normalizer_d = tf.expand_dims(test_normalizer_d, axis=1)
    test_normalizer_a = tf.expand_dims(test_normalizer_a, axis=1)

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

    model = models.HierarchicalRNN(num_categories, 8, hidden_layer_size,
                                   use_marks=use_marks,
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

    # Create summary writers
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(args.output_dir, 'logs', current_time + '/train')
    dev_log_dir = os.path.join(args.output_dir, 'logs', current_time + '/dev')
    test_log_dir = os.path.join(args.output_dir, 'logs', current_time + '/test')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    dev_summary_writer = tf.summary.create_file_writer(dev_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    global_step = 0

    best_dev_gap_error = np.inf
    best_test_gap_error = np.inf
    best_dev_mark_acc = np.inf
    best_test_mark_acc = np.inf
    best_epoch = 0

    train_c_losses = list()
    train_losses = list()
    inference_times = list()
    # Iterate over epochs.
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        if args.training_mode:
            for step, (c_marks_batch_in, c_gaps_batch_in, c_times_batch_in, c_seqmask_batch_in,
                       c_marks_batch_out, c_gaps_batch_out, c_times_batch_out, c_seqmask_batch_out,
                       c_batch_time_feature,
                       gaps_batch_in, times_batch_in, seqmask_batch_in,
                       gaps_batch_out, times_batch_out, seqmask_batch_out,
                       batch_time_feature) \
                               in enumerate(c_train_dataset):

                with tf.GradientTape() as tape:

                    (l2_marks_logits, l2_gaps_pred, l2_D, l2_WT,
                     l1_marks_logits, l1_gaps_pred, l1_D, l1_WT) \
                            = model(c_gaps_batch_in, c_seqmask_batch_in, c_batch_time_feature,
                                    gaps_batch_in, seqmask_batch_in, batch_time_feature)

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
                    with train_summary_writer.as_default():
                        tf.summary.scalar('c_gap_loss', c_gap_loss, step=global_step)
                        tf.summary.scalar('gap_loss', gap_loss, step=global_step)
                        tf.summary.scalar('mark_loss', mark_loss, step=global_step)
                        tf.summary.scalar('loss', loss, step=global_step)

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

                global_step += 1

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

            print('l2_train gap err over epoch: %s' \
                    % (float(c_train_gap_err)))
            c_train_gap_metric.reset_states()

        ############################################################################
        # Starting prediction with simulator

        if epoch > patience-1 or args.training_mode==0.0:

            for dev_step, (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in, c_dev_seqmask_in,
                           c_dev_gaps_out, c_dev_times_out, c_dev_seqmask_out, c_dev_time_feature) \
                    in enumerate(c_dev_dataset):

                arguments = (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in, c_dev_seqmask_in,
                             c_dev_gaps_out, c_dev_times_out, c_dev_seqmask_out, c_dev_time_feature)

                params = (use_marks)

                extra_args = (c_dev_seq_lens, c_dev_t_b_plus, c_dev_t_e_plus,
                              dev_t_b_plus, dev_t_e_plus, decoder_length, dev_begin_tss)

                query = 1

                if query == 1:
                    query_result = query_processor(1, model, arguments, params, extra_args)
                    (dev_gaps_pred, all_l2_dev_gaps_pred) = query_result

                elif query == 2:
                    query_result = query_processor(2, model, arguments, params, extra_args)
                    (dev_gaps_pred, all_times_pred_in_range, total_number_of_events_in_range) = query_result

                    # print('dev_gaps_pred', dev_gaps_pred)
                    # print('all_times_pred_in_range', all_times_pred_in_range)

                    total_number_of_events_in_range = np.array(total_number_of_events_in_range)
                    actual_event_count = np.array(event_bw_range_tb_te[0])
                    error_in_event_count = np.mean((total_number_of_events_in_range-actual_event_count)**2)

                    print('total_number_of_events_in_range', total_number_of_events_in_range)
                    print('Actual count of events:', actual_event_count)
                    # print('dev_t_b_plus', dev_t_b_plus)
                    # print('dev_t_e_plus', dev_t_e_plus)

                    actual_dev_event_in_range = event_bw_range_tb_te[1]
                    actual_dev_gaps_in_range = event_bw_range_tb_te[2]

                    dev_gaps_pred_in_range = (dev_gaps_pred[:, 1:] - dev_normalizer_a) * dev_normalizer_d
                    dev_gaps_pred_in_range = dev_gaps_pred_in_range.numpy()

                    dev_gaps_pred_in_range = [np.trim_zeros(dev_gaps_pred_in_range[idx], 'b') for idx in range(len(dev_t_b_plus))]
                    dtw_cost_in_range = DTW(dev_gaps_pred_in_range, actual_dev_gaps_in_range)

                    # print('actual_dev_event_in_range', actual_dev_event_in_range)
                    # print('dev_gaps_pred_in_range', dev_gaps_pred_in_range)
                    # print('actual_dev_gaps_in_range', actual_dev_gaps_in_range)
                    print('dtw_cost_in_range', dtw_cost_in_range)
                    print('error_in_event_count', error_in_event_count)


                # (dev_l2_marks_logits, dev_l2_gaps_pred, _, _,
                #  _, _, _, _) \
                #         = model(None, c_dev_gaps_in, c_dev_seqmask_in)

                # if use_marks:
                #     dev_marks_pred = tf.argmax(dev_marks_logits, axis=-1) + 1
                #     dev_marks_pred_last = dev_marks_pred[:, -1:]
                # else:
                #     dev_marks_pred_last = None

                # second_last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-2, batch_dims=1)
                # last_gaps_pred = tf.gather(dev_l2_gaps_pred, c_dev_seq_lens-1, batch_dims=1)
                # last_times_in = tf.gather(c_dev_times_in, c_dev_seq_lens-1, batch_dims=1)
        
                # dev_simulator = models.SimulateHierarchicalRNN()
                # all_l2_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, _, _, before_tb_hidden_state, _ \
                #                                = dev_simulator.simulator(model.l2_rnn,
                #                                  last_times_in,
                #                                  last_gaps_pred,
                #                                  c_dev_seq_lens,
                #                                  dev_begin_tss,
                #                                  c_dev_t_b_plus,
                #                                  c_dev_t_b_plus,
                #                                  decoder_length,
                #                                  2,
                #                                  second_last_gaps_pred,)
                
                # #TODO Process l2 gaps to l1  gaps
                # # before_tb_gaps_pred

                # last_gaps_pred = before_tb_gaps_pred / 10.0
                # l1_rnn_init_state =  model.ff(before_tb_hidden_state)

                # all_l1_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, dev_gaps_pred, _, _, simulation_count \
                #                                = dev_simulator.simulator(model.l1_rnn,
                #                                  last_times_pred,
                #                                  last_gaps_pred,
                #                                  0,
                #                                  dev_begin_tss,
                #                                  dev_t_b_plus,
                #                                  dev_t_b_plus,
                #                                  decoder_length,
                #                                  1,
                #                                  initial_state=l1_rnn_init_state)

                # all_l1_dev_gaps_pred, last_times_pred, before_tb_gaps_pred, dev_gaps_pred, _, _, simulation_count \
                #                                = dev_simulator.simulator(model.l1_rnn,
                #                                  last_times_pred,
                #                                  before_tb_gaps_pred,
                #                                  0,
                #                                  dev_begin_tss,
                #                                  dev_t_e_plus,
                #                                  dev_t_e_plus,
                #                                  decoder_length,
                #                                  1,
                #                                  initial_state=l1_rnn_init_state)

                # print(all_l1_dev_gaps_pred)
                # ipdb.set_trace()

            model.reset_states()
            # ipdb.set_trace()

            ############################################################################

            start_time = time.time()
            for test_step, (c_test_marks_in, c_test_gaps_in, c_test_times_in, c_test_seqmask_in,
                            c_test_gaps_out, c_dev_times_out, c_test_seqmask_out, c_test_time_feature) \
                    in enumerate(c_test_dataset):

                (test_l2_marks_logits, test_l2_gaps_pred, _, _,
                 _, _, _, _) \
                        = model(c_test_gaps_in, c_test_seqmask_in, c_test_time_feature)

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
                                                  decoder_length,
                                                  (c_test_normalizer_d, c_test_normalizer_a),
                                                  (test_normalizer_d, test_normalizer_a))
            model.reset_states()
            end_time = time.time()
            inference_times.append(end_time-start_time)

            #############################################################################

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
                # ----- Dev nowcasting plots for layer 2 ----- #
                plot_dir_l2 = os.path.join(args.output_dir, 'plots_l2', 'dev_plots')
                os.makedirs(plot_dir_l2, exist_ok=True)
                name_plot = os.path.join(plot_dir_l2, 'epoch_' + str(epoch))

                # all_c_dev_gaps_pred = tf.squeeze(all_l2_dev_gaps_pred, axis=-1)
                # # all_c_dev_gaps_pred = dev_simulator.all_l2_gaps_pred
                # all_c_dev_gaps_pred = (all_c_dev_gaps_pred) * tf.expand_dims(c_dev_normalizer_d, axis=1)
                all_c_dev_gaps_pred = dev_simulator.all_l2_gaps_pred

                plt.plot(tf.squeeze(c_dev_gaps_out[1], axis=-1), 'bo-')
                plt.plot(all_c_dev_gaps_pred[1][1:], 'r*-')
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
                         [0, max([max(true_gaps_plot), max(pred_gaps_plot)])],
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

    print('Best Dev mark acc, gap err: %s, %s' \
            % (float(best_dev_mark_acc), float(best_dev_gap_error)))
    print('Best Test mark acc and gap err: %s, %s' \
            % (float(best_test_mark_acc), float(best_test_gap_error)))
    print('Best epoch:', best_epoch)

    if args.generate_plots and args.training_mode==0.0:
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
                     [0, max([max(true_gaps_plot), max(pred_gaps_plot)])],
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


    return {
            'best_dev_gap_error': float(best_dev_gap_error.numpy()),
            'best_test_gap_error': float(best_test_gap_error.numpy()),
            'best_epoch': best_epoch,
            'average_inference_time': np.mean(inference_times),
           }
