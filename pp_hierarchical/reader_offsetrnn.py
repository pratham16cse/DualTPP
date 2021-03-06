from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from bisect import bisect_right

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

import reader_rmtpp
import reader_hierarchical

def get_normalized_dataset(data, normalization='average', max_offset=0.0):

    gaps_in, gaps_out_next, gaps_out_off = data

    def _norm_seq(gap_seq_in, gap_seq_out_next, gap_seq_out_off):

        if normalization in ['minmax']:
            max_gap = tf.clip_by_value(tf.math.reduce_max(gap_seq_in), 1.0, np.inf)
            min_gap = tf.clip_by_value(tf.math.reduce_min(gap_seq_in), 1.0, np.inf)
            n_d, n_a = [(max_gap - min_gap)], [(-min_gap/(max_gap - min_gap))]
        elif normalization == 'average_per_seq':
            avg_gap = tf.clip_by_value(tf.math.reduce_mean(gap_seq_in), 1.0, np.inf)
            n_d, n_a = [avg_gap], [0.0]
        elif normalization == 'max_per_seq':
            max_gap = tf.clip_by_value(tf.math.reduce_max(gap_seq_in), 1.0, np.inf)
            n_d, n_a = [max_gap], [0.0]
        elif normalization is None or normalization in ['average', 'max_offset']:
            n_d, n_a = [1.0], [0.0]
        else:
            print('Normalization not found')
            assert False

        avg_gap_norm_in = gap_seq_in/n_d + n_a
        avg_gap_norm_out_next = gap_seq_out_next/n_d + n_a
        avg_gap_norm_out_off = gap_seq_out_off/n_d + n_a

        return avg_gap_norm_in, avg_gap_norm_out_next, avg_gap_norm_out_off, n_d, n_a
    
    avg_gaps_norm_in, avg_gaps_norm_out_next, avg_gaps_norm_out_off, normalizer_d, normalizer_a = list(), list(), list(), list(), list()

    for sequence_in, sequence_out_next, sequence_out_off in zip(gaps_in, gaps_out_next, gaps_out_off):
        avg_gap_norm_in, avg_gap_norm_out_next, avg_gap_norm_out_off, n_d, n_a = _norm_seq(sequence_in, sequence_out_next, sequence_out_off)
        avg_gaps_norm_in.append(avg_gap_norm_in)
        avg_gaps_norm_out_next.append(avg_gap_norm_out_next)
        avg_gaps_norm_out_off.append(avg_gap_norm_out_off)
        normalizer_d.append(n_d)
        normalizer_a.append(n_a)

    if normalization == 'average':
        n_d = tf.reduce_mean(avg_gaps_norm_in)
        avg_gaps_norm_in = [gap_seq/n_d for gap_seq in avg_gaps_norm_in]
        avg_gaps_norm_out_next = [gap_seq/n_d for gap_seq in avg_gaps_norm_out_next]
        avg_gaps_norm_out_off = [gap_seq/n_d for gap_seq in avg_gaps_norm_out_off]
        normalizer_d = np.ones((len(gaps_in), 1)) * n_d

    elif normalization == 'max_offset':
        n_d = [max_offset]
        avg_gaps_norm_in = [gap_seq/n_d for gap_seq in avg_gaps_norm_in]
        avg_gaps_norm_out_next = [gap_seq/n_d for gap_seq in avg_gaps_norm_in]
        avg_gaps_norm_out_off = [gap_seq/n_d for gap_seq in avg_gaps_norm_in]
        normalizer_d = np.ones((len(gaps_in), 1)) * n_d

    avg_gaps_norm_in = tf.stack(avg_gaps_norm_in, axis=0)
    avg_gaps_norm_out_next = tf.stack(avg_gaps_norm_out_next, axis=0)
    avg_gaps_norm_out_off = tf.stack(avg_gaps_norm_out_off, axis=0)

    return avg_gaps_norm_in, avg_gaps_norm_out_next, avg_gaps_norm_out_off, normalizer_d, normalizer_a

def get_train_input_output(data, block_size):
    marks, times = data

    #marks = [np.array(x[1:]) for x in marks]
    marks_in = [np.array(x[1:-1]) for x in marks]
    marks_out_next = [np.array(x[2:]) for x in marks]

    gaps = [np.array(x[1:])-np.array(x[:-1]) for x in times]
    gaps_in = [x[:-1] for x in gaps]
    gaps_out_next = [x[1:] for x in gaps]

    times_in = [np.array(x[1:-1]) for x in times]
    times_out_next = [np.array(x[2:]) for x in times]

    # For each input times timestamp, sample offset between [0, block_size] hours
    # and select first event after t+off_unnorm as the output
    offsets = list()
    times_out_off, gaps_out_off, marks_out_off = list(), list(), list()
    for times_i, times_seq, gaps_seq, marks_seq in zip(times_in, times, gaps, marks):
        offs = list()
        times_o, gaps_o, marks_o = list(), list(), list()
        for t in times_i:
            off = np.random.uniform()
            off_unnorm =  off * 3600. * block_size
            if t+off_unnorm > times_i[-1]:
                out_idx = len(times_i)-1
                off_unnorm = times_i[-1]-t
                off = off_unnorm / (3600. * block_size)
            else:
                out_idx = bisect_right(times_seq, t+off_unnorm)
            times_o.append(times_seq[out_idx])
            gaps_o.append(gaps_seq[out_idx-1])
            marks_o.append(marks_seq[out_idx])
            offs.append(off)
        times_out_off.append(times_o)
        gaps_out_off.append(gaps_o)
        marks_out_off.append(marks_o)
        offsets.append(offs)

    return (marks_in, marks_out_next, marks_out_off,
            gaps_in, gaps_out_next, gaps_out_off,
            times_in, times_out_next, times_out_off, offsets)

def transpose(m_in, g_in, t_in, sm_in,
              m_out_next, g_out_next, t_out_next, sm_out_next,
              m_out_off, g_out_off, t_out_off, sm_out_off,
              time_feature, offsets):
    return tf.transpose(m_in), tf.transpose(g_in, [1, 0, 2]), tf.transpose(t_in, [1, 0, 2]), tf.transpose(sm_in), \
            tf.transpose(m_out_next), tf.transpose(g_out_next, [1, 0, 2]), tf.transpose(t_out_next, [1, 0, 2]), tf.transpose(sm_out_next), \
            tf.transpose(m_out_off), tf.transpose(g_out_off, [1, 0, 2]), tf.transpose(t_out_off, [1, 0, 2]), tf.transpose(sm_out_off), \
            tf.transpose(time_feature, [1, 0, 2]), tf.transpose(offsets)

def get_padded_dataset(data):
    (marks_in, gaps_in, times_in,
     marks_out_next, gaps_out_next, times_out_next,
     marks_out_off, gaps_out_off, times_out_off,
     offsets) = data

    seq_lens = np.expand_dims(np.array([len(s) for s in times_in]), axis=-1)

    marks_in = pad_sequences(marks_in, padding='post')
    gaps_in = pad_sequences(gaps_in, padding='post', dtype='float32')
    times_in = pad_sequences(times_in, padding='post')
    marks_out_next = pad_sequences(marks_out_next, padding='post')
    gaps_out_next = pad_sequences(gaps_out_next, padding='post', dtype='float32')
    times_out_next = pad_sequences(times_out_next, padding='post')
    marks_out_off = pad_sequences(marks_out_off, padding='post')
    gaps_out_off = pad_sequences(gaps_out_off, padding='post', dtype='float32')
    times_out_off = pad_sequences(times_out_off, padding='post')
    offsets = pad_sequences(offsets, padding='post', dtype='float32')

    #print('seq_lens', seq_lens)
    #print('times_in.shape', times_in.shape)

    gaps_in = tf.expand_dims(tf.cast(gaps_in, tf.float32), axis=-1)
    times_in = tf.expand_dims(tf.cast(times_in, tf.float32), axis=-1)
    gaps_out_next = tf.expand_dims(tf.cast(gaps_out_next, tf.float32), axis=-1)
    times_out_next = tf.expand_dims(tf.cast(times_out_next, tf.float32), axis=-1)
    gaps_out_off = tf.expand_dims(tf.cast(gaps_out_off, tf.float32), axis=-1)
    times_out_off = tf.expand_dims(tf.cast(times_out_off, tf.float32), axis=-1)

    return (marks_in, gaps_in, times_in,
            marks_out_next, gaps_out_next, times_out_next,
            marks_out_off, gaps_out_off, times_out_off,
            offsets, seq_lens)

def get_preprocessed_(data, block_size, decoder_length, normalization):
    marks, times = data
    num_categories = len(np.unique(marks))

    (train_marks, train_times,
     dev_marks, dev_times,
     test_marks, test_times,
     dev_begin_tss, test_begin_tss) \
            = reader_rmtpp.create_train_dev_test_split((marks, times), block_size, decoder_length)
    num_sequences = len(train_marks)

    #print(train_times[0].tolist())

    (train_marks_in, train_marks_out_next, train_marks_out_off,
     train_gaps_in, train_gaps_out_next, train_gaps_out_off,
     train_times_in, train_times_out_next, train_times_out_off,
     train_offsets) \
            = get_train_input_output((train_marks, train_times), block_size)
    #print(train_times_in[0].tolist())
    #print(train_times_out[0].tolist())

    train_seqmask_in, _ = reader_rmtpp.get_seq_mask(train_gaps_in)
    train_seqmask_out_next, _ = reader_rmtpp.get_seq_mask(train_gaps_out_next)
    train_seqmask_out_off, _ = reader_rmtpp.get_seq_mask(train_gaps_out_off)

    (train_marks_in, train_gaps_in, train_times_in,
     train_marks_out_next, train_gaps_out_next, train_times_out_next,
     train_marks_out_off, train_gaps_out_off, train_times_out_off,
     train_offsets, train_seq_lens) \
            = get_padded_dataset((train_marks_in, train_gaps_in, train_times_in,
                                  train_marks_out_next, train_gaps_out_next, train_times_out_next,
                                  train_marks_out_off, train_gaps_out_off, train_times_out_off,
                                  train_offsets))

    (train_time_feature) \
            = reader_rmtpp.get_time_features_for_data((train_times_in))

    (train_marks_in, train_gaps_in, train_times_in, train_seqmask_in,
     train_marks_out_next, train_gaps_out_next, train_times_out_next, train_seqmask_out_next,
     train_marks_out_off, train_gaps_out_off, train_times_out_off, train_seqmask_out_off,
     train_time_feature, train_offsets) \
            = transpose(train_marks_in, train_gaps_in, train_times_in, train_seqmask_in,
                        train_marks_out_next, train_gaps_out_next, train_times_out_next, train_seqmask_out_next,
                        train_marks_out_off, train_gaps_out_off, train_times_out_off, train_seqmask_out_off,
                        train_time_feature, train_offsets)

    (train_gaps_in_norm, train_gaps_out_next_norm, train_gaps_out_off_norm,
     train_normalizer_d, train_normalizer_a) \
            = get_normalized_dataset((train_gaps_in, train_gaps_out_next, train_gaps_out_off),
                                     normalization=normalization)

    print('train_time_feature', train_time_feature)
    print('train_times_in', train_times_in)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_marks_in,
                                                        train_gaps_in_norm,
                                                        train_times_in,
                                                        train_seqmask_in,
                                                        train_marks_out_next,
                                                        train_gaps_out_next_norm,
                                                        train_times_out_next,
                                                        train_seqmask_out_next,
                                                        train_marks_out_off,
                                                        train_gaps_out_off_norm,
                                                        train_times_out_off,
                                                        train_seqmask_out_off,
                                                        train_time_feature,
                                                        train_offsets))

    (dev_marks_in, dev_gaps_in, dev_times_in,
     dev_marks_out, dev_gaps_out, dev_times_out,
     test_marks_in, test_gaps_in, test_times_in,
     test_marks_out, test_gaps_out, test_times_out) \
            = reader_rmtpp.get_dev_test_input_output(train_marks, train_times,
                                                     dev_marks, dev_times,
                                                     test_marks, test_times)

    dev_seqmask_in, _ = reader_rmtpp.get_seq_mask(dev_gaps_in)
    dev_seqmask_out, _ = reader_rmtpp.get_seq_mask(dev_gaps_out)

    #dev_marks_out = [d_m[-decoder_length:] for d_m in dev_marks_out]
    #dev_gaps_out = [d_g[-decoder_length:] for d_g in dev_gaps_out]
    #dev_times_out = [d_t[-decoder_length:] for d_t in dev_times_out]
    #test_marks_out = [t_m[-decoder_length:] for t_m in test_marks_out]
    #test_gaps_out = [d_g[-decoder_length:] for d_g in test_gaps_out]
    #test_times_out = [t_t[-decoder_length:] for t_t in test_times_out]

    (dev_marks_in, dev_gaps_in, dev_times_in,
     dev_marks_out, dev_gaps_out, dev_times_out,
     dev_seq_lens) \
            = reader_rmtpp.get_padded_dataset((dev_marks_in, dev_gaps_in, dev_times_in,
                                              dev_marks_out, dev_gaps_out, dev_times_out))

    (dev_time_feature) \
            = reader_rmtpp.get_time_features_for_data((dev_times_in))

    (dev_gaps_in_norm, dev_gaps_out_norm,
     dev_normalizer_d, dev_normalizer_a) \
            = reader_rmtpp.get_normalized_dataset((dev_gaps_in, dev_gaps_out),
                                     normalization=normalization)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_marks_in,
                                                      dev_gaps_in_norm,
                                                      dev_times_in,
                                                      dev_seqmask_in,
                                                      dev_time_feature))

    test_seqmask_in, _ = reader_rmtpp.get_seq_mask(test_gaps_in)
    test_seqmask_out, _ = reader_rmtpp.get_seq_mask(test_gaps_out)

    (test_marks_in, test_gaps_in, test_times_in,
     test_marks_out, test_gaps_out, test_times_out,
     test_seq_lens) \
            = reader_rmtpp.get_padded_dataset((test_marks_in, test_gaps_in, test_times_in,
                                              test_marks_out, test_gaps_out, test_times_out))

    print('test_times_in', test_times_in)
    (test_time_feature) \
            = reader_rmtpp.get_time_features_for_data((test_times_in))

    print('test_time_feature', test_time_feature)

    (test_gaps_in_norm, test_gaps_out_norm,
     test_normalizer_d, test_normalizer_a) \
            = reader_rmtpp.get_normalized_dataset((test_gaps_in, test_gaps_out))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_marks_in,
                                                       test_gaps_in_norm,
                                                       test_times_in,
                                                       test_seqmask_in,
                                                       test_time_feature))

    #print('Train In')
    #print(tf.squeeze(tf.transpose(train_times_in, [1, 0, 2]), axis=-1).numpy().tolist()[0])
    #print('Dev In')
    #print(tf.squeeze(dev_times_in, axis=-1).numpy().tolist()[0])
    #print('\n')
    #print('Dev Begin Timestamp:')
    #print(dev_begin_tss[0])
    #print('\n')
    #print('Dev Out')
    #print(tf.squeeze(dev_times_out, axis=-1).numpy().tolist()[0])
    #print('\n')
    #print('Test In')
    #print(tf.squeeze(test_times_in, axis=-1).numpy().tolist()[0])
    #print('\n')
    #print('Test Begin Timestamp:')
    #print(test_begin_tss[0])
    #print('\n')
    #print('Test Out')
    #print(tf.squeeze(test_times_out, axis=-1).numpy().tolist()[0])

    return {
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'test_dataset': test_dataset,
        'dev_marks_out': dev_marks_out,
        'dev_gaps_in': dev_gaps_in,
        'dev_gaps_out': dev_gaps_out,
        'dev_times_out': dev_times_out,
        'test_marks_out': test_marks_out,
        'test_gaps_out': test_gaps_out,
        'test_times_out': test_times_out,
        'dev_begin_tss': dev_begin_tss,
        'test_begin_tss': test_begin_tss,
        'num_categories': num_categories,
        'num_sequences': num_sequences,
        'train_seq_lens': train_seq_lens,
        'dev_seq_lens': dev_seq_lens,
        'test_seq_lens': test_seq_lens,
        'train_seqmask_in': train_seqmask_in,
        'dev_seqmask_in': dev_seqmask_in,
        'test_seqmask_in': test_seqmask_in,
        'train_seqmask_out_next': train_seqmask_out_next,
        'train_seqmask_out_off': train_seqmask_out_off,
        'dev_seqmask_out': dev_seqmask_out,
        'test_seqmask_out': test_seqmask_out,

        'train_gaps_in_norm': train_gaps_in_norm,
        'train_gaps_out_next_norm': train_gaps_out_next_norm,
        'train_gaps_out_off_norm': train_gaps_out_off_norm,
        'train_normalizer_d': train_normalizer_d,
        'train_normalizer_a': train_normalizer_a,

        'dev_gaps_in_norm': dev_gaps_in_norm,
        'dev_gaps_out_norm': dev_gaps_out_norm,
        'dev_normalizer_d': dev_normalizer_d,
        'dev_normalizer_a': dev_normalizer_a,

        'test_gaps_in_norm': test_gaps_in_norm,
        'test_gaps_out_norm': test_gaps_out_norm,
        'test_normalizer_d': test_normalizer_d,
        'test_normalizer_a': test_normalizer_a,

        }

def get_preprocessed_data(dataset_name, dataset_path, block_size, decoder_length, normalization):
    marks, times = reader_rmtpp.read_data(dataset_name, dataset_path)

    # pre-aggregate K=10 events into single event. The time of aggregated
    # event is represented by the time of occurrence of the K^{th} event
    marks, times, _ = reader_hierarchical.get_compound_events((marks, times), K=10)

    data = get_preprocessed_((marks, times), block_size, decoder_length,
                             normalization)
    return data
