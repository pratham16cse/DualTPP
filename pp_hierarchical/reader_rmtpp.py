from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

def read_data(filename):
    with open(filename, 'r') as f:
        data = list()
        for line in f:
            mark, time = line.strip().split()[:2]
            data.append((int(mark), float(time)))
    data_sorted = sorted(data, key=itemgetter(1))

    marks = np.array([event[0] for event in data_sorted])
    times = np.array([event[1] for event in data_sorted])
    return marks, times

def split_data(data, num_chops):
    marks, times = data
    marks = marks[:len(marks)-len(marks)%num_chops]
    times = times[:len(times)-len(times)%num_chops]
    marks_split = np.array(np.array_split(marks, num_chops))
    times_split = np.array(np.array_split(times, num_chops))
    return marks_split, times_split

def get_normalized_dataset(data, normalization='average', max_offset=0.0):

    gaps_in, gaps_out = data

    def _norm_seq(gap_seq_in, gap_seq_out):

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
        avg_gap_norm_out = gap_seq_out/n_d + n_a

        return avg_gap_norm_in, avg_gap_norm_out, n_d, n_a
    
    avg_gaps_norm_in, avg_gaps_norm_out, normalizer_d, normalizer_a = list(), list(), list(), list()

    for sequence_in, sequence_out in zip(gaps_in, gaps_out):
        avg_gap_norm_in, avg_gap_norm_out, n_d, n_a = _norm_seq(sequence_in, sequence_out)
        avg_gaps_norm_in.append(avg_gap_norm_in)
        avg_gaps_norm_out.append(avg_gap_norm_out)
        normalizer_d.append(n_d)
        normalizer_a.append(n_a)

    if normalization == 'average':
        n_d = tf.reduce_mean(avg_gaps_norm_in)
        avg_gaps_norm_in = [gap_seq/n_d for gap_seq in avg_gaps_norm_in]
        avg_gaps_norm_out = [gap_seq/n_d for gap_seq in avg_gaps_norm_out]
        normalizer_d = np.ones((len(gaps_in), 1)) * n_d

    elif normalization == 'max_offset':
        n_d = [max_offset]
        avg_gaps_norm_in = [gap_seq/n_d for gap_seq in avg_gaps_norm_in]
        avg_gaps_norm_out = [gap_seq/n_d for gap_seq in avg_gaps_norm_in]
        normalizer_d = np.ones((len(gaps_in), 1)) * n_d

    avg_gaps_norm_in = tf.stack(avg_gaps_norm_in, axis=0)
    avg_gaps_norm_out = tf.stack(avg_gaps_norm_out, axis=0)

    return avg_gaps_norm_in, avg_gaps_norm_out, normalizer_d, normalizer_a

def get_hour_of_day_ts(ts):
    ''' Returns timestamp at the beginning of the hour'''

    return datetime.timestamp(pd.Timestamp(ts, unit='s', tz='utc').floor('H')) / 3600.

def get_num_events_per_hour(data):
    marks, times = data
    #print(times)
    times = pd.Series(times)
    #for x in times:
    #    print(x, pd.Timestamp(x, unit='s', tz='utc').floor('H'),
    #          datetime.timestamp(pd.Timestamp(x, unit='s', tz='utc').floor('H')),
    #          get_hour_of_day_ts(x))
    times_grouped = times.groupby(lambda x: pd.Timestamp(times[x], unit='s', tz='utc').floor('H')).agg('count')
    #plt.bar(times_grouped.index, times_grouped.tolist(), width=0.02)
    plt.bar(range(len(times_grouped.index)), times_grouped.values)
    return times_grouped

def get_gaps(times):
    return [np.array(x[1:])-np.array(x[:-1]) for x in times]

def get_dev_test_input_output(train_marks, train_times,
                              dev_marks, dev_times,
                              test_marks, test_times):

    dev_marks_in = [trn_m[1:] for trn_m in train_marks]
    dev_times_in = train_times
    dev_gaps_in = get_gaps(dev_times_in)
    dev_times_in = [trn_t[1:] for trn_t in dev_times_in]
    dev_marks_out = dev_marks
    dev_times_out = [np.concatenate([trn_t[-1:], dev_t[:-1]]) \
                        for trn_t, dev_t in zip(train_times, dev_times)]
    dev_gaps_out = get_gaps(dev_times_out)

    test_marks_in = [np.concatenate([trn_m[1:], dev_m]) \
                        for trn_m, dev_m in zip(train_marks, dev_marks)]
    test_times_in = [np.concatenate([trn_t, dev_t]) \
                        for trn_t, dev_t in zip(train_times, dev_times)]
    test_gaps_in = get_gaps(test_times_in)
    test_times_in = [tst_t[1:] for tst_t in test_times_in]
    test_marks_out = test_marks
    test_times_out = [np.concatenate([dev_t[-1:], tst_t[:-1]]) \
                        for dev_t, tst_t in zip(dev_times, test_times)]
    test_gaps_out = get_gaps(test_times_out)

    print('DevIn:', dev_times_in[0].tolist())
    print('\n')
    print('DevOut', dev_times_out[0].tolist())
    print('\n')
    print('TestIn:', test_times_in[0].tolist())
    print('\n')
    print('TestOut', test_times_out[0].tolist())

    return  (dev_marks_in, dev_gaps_in, dev_times_in,
             dev_marks_out, dev_gaps_out, dev_times_out,
             test_marks_in, test_gaps_in, test_times_in,
             test_marks_out, test_gaps_out, test_times_out)

def get_train_input_output(data):
    marks, times = data

    #marks = [np.array(x[1:]) for x in marks]
    marks_in = [np.array(x[1:-1]) for x in marks]
    marks_out = [np.array(x[2:]) for x in marks]

    gaps = [np.array(x[1:])-np.array(x[:-1]) for x in times]
    gaps_in = [x[:-1] for x in gaps]
    gaps_out = [x[1:] for x in gaps]

    times_in = [np.array(x[1:-1]) for x in times]
    times_out = [np.array(x[2:]) for x in times]

    return marks_in, marks_out, gaps_in, gaps_out, times_in, times_out

def create_train_dev_test_split(data, block_size, decoder_length):
    marks, times = data
    num_events_per_hour = get_num_events_per_hour((marks, times))
    print(num_events_per_hour.index[0])
    train_marks, train_times = list(), list()
    dev_marks, dev_times = list(), list()
    test_marks, test_times = list(), list()
    dev_begin_tss, test_begin_tss = list(), list()

    block_begin_idxes = num_events_per_hour.cumsum()
    num_hrs = len(num_events_per_hour)-len(num_events_per_hour)%(4*block_size)
    for idx in range(0, num_hrs, 4*block_size):
        print(idx, num_hrs)
        train_start_idx = block_begin_idxes[idx-1]+1 if idx>0 else 0
        train_end_idx = block_begin_idxes[idx+(2*block_size-1)]#-decoder_length-1
        train_marks.append(marks[train_start_idx:train_end_idx])
        train_times.append(times[train_start_idx:train_end_idx])

        dev_start_idx = block_begin_idxes[idx+(2*block_size-1)]+1#-decoder_length-1
        dev_end_idx = block_begin_idxes[idx+(3*block_size-1)]#-decoder_length-1
        dev_marks.append(marks[dev_start_idx:dev_end_idx])
        dev_times.append(times[dev_start_idx:dev_end_idx])
        dev_begin_tss.append(get_hour_of_day_ts(times[dev_start_idx]) * 3600.)

        test_start_idx = block_begin_idxes[idx+(3*block_size-1)]+1#-decoder_length-1
        test_end_idx = block_begin_idxes[idx+(4*block_size-1)]
        test_marks.append(marks[test_start_idx:test_end_idx])
        test_times.append(times[test_start_idx:test_end_idx])
        test_begin_tss.append(get_hour_of_day_ts(times[test_start_idx]) * 3600.)

    dev_begin_tss = tf.expand_dims(tf.constant(dev_begin_tss), axis=-1)
    test_begin_tss = tf.expand_dims(tf.constant(test_begin_tss), axis=-1)

    return (train_marks, train_times,
            dev_marks, dev_times,
            test_marks, test_times,
            dev_begin_tss, test_begin_tss)

def transpose(m_in, g_in, t_in, sm_in, m_out, g_out, t_out, sm_out):
    return tf.transpose(m_in), tf.transpose(g_in, [1, 0, 2]), tf.transpose(t_in, [1, 0, 2]), tf.transpose(sm_in), \
            tf.transpose(m_out), tf.transpose(g_out, [1, 0, 2]), tf.transpose(t_out, [1, 0, 2]), tf.transpose(sm_out)

def get_padded_dataset(data):
    marks_in, gaps_in, times_in, marks_out, gaps_out, times_out = data

    seq_lens = np.expand_dims(np.array([len(s) for s in times_in]), axis=-1)

    marks_in = pad_sequences(marks_in, padding='post')
    gaps_in = pad_sequences(gaps_in, padding='post')
    times_in = pad_sequences(times_in, padding='post')
    marks_out = pad_sequences(marks_out, padding='post')
    gaps_out = pad_sequences(gaps_out, padding='post')
    times_out = pad_sequences(times_out, padding='post')

    gaps_in = tf.expand_dims(tf.cast(gaps_in, tf.float32), axis=-1)
    times_in = tf.expand_dims(tf.cast(times_in, tf.float32), axis=-1)
    gaps_out = tf.expand_dims(tf.cast(gaps_out, tf.float32), axis=-1)
    times_out = tf.expand_dims(tf.cast(times_out, tf.float32), axis=-1)

    return marks_in, gaps_in, times_in, marks_out, gaps_out, times_out, seq_lens

def get_seq_mask(sequences):
    seq_lens = [len(s) for s in sequences]
    seq_mask = tf.sequence_mask(seq_lens, dtype=tf.float32)
    return seq_mask, seq_lens

def get_compound_events(data, K=1):
    def most_frequent(arr):
        lst = arr.tolist()
        return max(set(lst), key=lst.count)

    marks, times = data
    c_marks, c_times = list(), list()
    #for m_seq, t_seq in zip(marks, times):
    #    c_t_seq = [t_seq[i:i+K][-1] for i in range(0, len(t_seq), K)]
    #    c_times.append(c_t_seq)
    #    c_m_seq = [most_frequent(m_seq[i:i+K]) for i in range(0, len(m_seq), K)]
    #    c_marks.append(c_m_seq)
    #    #TODO Instead of returning most frequent marker, return the simplex of marks

    c_times = np.array([times[i:i+K][-1] for i in range(0, len(times), K)])
    c_marks = np.array([most_frequent(marks[i:i+K]) for i in range(0, len(marks), K)])
    #TODO Instead of returning most frequent marker, return the simplex of marks

    return c_marks, c_times

def get_preprocessed_(data, block_size, decoder_length):
    marks, times = data
    num_categories = len(np.unique(marks))

    (train_marks, train_times,
     dev_marks, dev_times,
     test_marks, test_times,
     dev_begin_tss, test_begin_tss) \
            = create_train_dev_test_split((marks, times), block_size, decoder_length)
    num_sequences = len(train_marks)

    #print(train_times[0].tolist())

    (train_marks_in, train_marks_out,
     train_gaps_in, train_gaps_out,
     train_times_in, train_times_out) \
            = get_train_input_output((train_marks, train_times))
    #print(train_times_in[0].tolist())
    #print(train_times_out[0].tolist())

    train_seqmask_in, _ = get_seq_mask(train_gaps_in)
    train_seqmask_out, _ = get_seq_mask(train_gaps_out)

    (train_marks_in, train_gaps_in, train_times_in,
     train_marks_out, train_gaps_out, train_times_out,
     train_seq_lens) \
            = get_padded_dataset((train_marks_in, train_gaps_in, train_times_in,
                                  train_marks_out, train_gaps_out, train_times_out))
    (train_marks_in, train_gaps_in, train_times_in, train_seqmask_in,
     train_marks_out, train_gaps_out, train_times_out, train_seqmask_out) \
            = transpose(train_marks_in, train_gaps_in, train_times_in, train_seqmask_in,
                        train_marks_out, train_gaps_out, train_times_out, train_seqmask_out)

    (train_gaps_in_norm, train_gaps_out_norm,
     train_normalizer_d, train_normalizer_a) \
            = get_normalized_dataset((train_gaps_in, train_gaps_out))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_marks_in,
                                                        train_gaps_in_norm,
                                                        train_times_in,
                                                        train_seqmask_in,
                                                        train_marks_out,
                                                        train_gaps_out_norm,
                                                        train_times_out,
                                                        train_seqmask_out))

    (dev_marks_in, dev_gaps_in, dev_times_in,
     dev_marks_out, dev_gaps_out, dev_times_out,
     test_marks_in, test_gaps_in, test_times_in,
     test_marks_out, test_gaps_out, test_times_out) \
            = get_dev_test_input_output(train_marks, train_times,
                                        dev_marks, dev_times,
                                        test_marks, test_times)

    dev_seqmask_in, _ = get_seq_mask(dev_gaps_in)
    dev_seqmask_out, _ = get_seq_mask(dev_gaps_out)

    #dev_marks_out = [d_m[-decoder_length:] for d_m in dev_marks_out]
    #dev_gaps_out = [d_g[-decoder_length:] for d_g in dev_gaps_out]
    #dev_times_out = [d_t[-decoder_length:] for d_t in dev_times_out]
    #test_marks_out = [t_m[-decoder_length:] for t_m in test_marks_out]
    #test_gaps_out = [d_g[-decoder_length:] for d_g in test_gaps_out]
    #test_times_out = [t_t[-decoder_length:] for t_t in test_times_out]

    (dev_marks_in, dev_gaps_in, dev_times_in,
     dev_marks_out, dev_gaps_out, dev_times_out,
     dev_seq_lens) \
            = get_padded_dataset((dev_marks_in, dev_gaps_in, dev_times_in,
                                  dev_marks_out, dev_gaps_out, dev_times_out))

    (dev_gaps_in_norm, dev_gaps_out_norm,
     dev_normalizer_d, dev_normalizer_a) \
            = get_normalized_dataset((dev_gaps_in, dev_gaps_out))

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_marks_in,
                                                      dev_gaps_in_norm,
                                                      dev_times_in,
                                                      dev_seqmask_in))

    test_seqmask_in, _ = get_seq_mask(test_gaps_in)
    test_seqmask_out, _ = get_seq_mask(test_gaps_out)

    (test_marks_in, test_gaps_in, test_times_in,
     test_marks_out, test_gaps_out, test_times_out,
     test_seq_lens) \
            = get_padded_dataset((test_marks_in, test_gaps_in, test_times_in,
                                  test_marks_out, test_gaps_out, test_times_out))

    (test_gaps_in_norm, test_gaps_out_norm,
     test_normalizer_d, test_normalizer_a) \
            = get_normalized_dataset((test_gaps_in, test_gaps_out))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_marks_in,
                                                       test_gaps_in_norm,
                                                       test_times_in,
                                                       test_seqmask_in))

    print('Train In')
    print(tf.squeeze(tf.transpose(train_times_in, [1, 0, 2]), axis=-1).numpy().tolist()[0])
    print('Dev In')
    print(tf.squeeze(dev_times_in, axis=-1).numpy().tolist()[0])
    print('\n')
    print('Dev Begin Timestamp:')
    print(dev_begin_tss[0])
    print('\n')
    print('Dev Out')
    print(tf.squeeze(dev_times_out, axis=-1).numpy().tolist()[0])
    print('\n')
    print('Test In')
    print(tf.squeeze(test_times_in, axis=-1).numpy().tolist()[0])
    print('\n')
    print('Test Begin Timestamp:')
    print(test_begin_tss[0])
    print('\n')
    print('Test Out')
    print(tf.squeeze(test_times_out, axis=-1).numpy().tolist()[0])

    return {
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'test_dataset': test_dataset,
        'dev_marks_out': dev_marks_out,
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

        'train_gaps_in_norm': train_gaps_in_norm,
        'train_gaps_out_norm': train_gaps_out_norm,
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

def get_preprocessed_data(block_size, decoder_length):
    marks, times = read_data('testdata.txt')
    #marks, times = split_data((marks, times), 7)
    
    block_size_sec = block_size * 3600.0

    data = get_preprocessed_((marks, times), block_size, decoder_length)

    # ----- Start: create compound events ----- #
    #c_train_times_in = get_compound_times(train_times_in, K=10)
    #c_dev_times_in = get_compound_times(dev_times_in, K=10)
    #c_test_times_in = get_compound_times(test_times_in, K=10)
    #c_marks, c_times = get_compound_events((marks, times), K=10)
    #data_level_2 = get_preprocessed_level((c_marks, c_times), 2, block_size, decoder_length)

    #assert data_level_1['num_sequences'] == data_level_2['num_sequences']
    # ----- End: create compound events ----- #

    return data



def main():
    for dataset in ['Delhi']:#['barca', 'Delhi', 'jaya', 'Movie', 'Fight', 'Verdict', 'Trump']:
        filename = '../pp_seq2seq/data/DataSetForSeq2SeqPP/'+dataset+'.txt'
        marks, times = read_data(filename)
        num_chops = 1
        #marks, times = split_data((marks, times), num_chops)
        num_events_per_hour = get_num_events_per_hour((marks, times))
        print('Number of hours spanned by '+dataset, len(num_events_per_hour))
        #get_best_num_chops((marks, times))
        #get_best_block_size((marks, times))
        train_marks, train_times, \
                dev_marks, dev_times, \
                test_marks, test_times \
                = create_train_dev_test_split((marks, times), 8)

        print(len(train_marks), len(train_times))
        print(len(dev_marks), len(dev_times))
        print(len(test_marks), len(test_times))

        for tr_seq, dev_seq, test_seq in zip(train_times, dev_times, test_times):
            print(len(tr_seq), len(dev_seq), len(test_seq))

if __name__ == '__main__':
    main()

