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

import reader_rmtpp

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

def get_num_events_per_hour(data):
    marks, times = data
    #print(times)
    times = pd.Series(times)
    times_grouped = times.groupby(lambda x: pd.Timestamp(times[x], unit='s').floor('H')).agg('count')
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

    return  (dev_marks_in, dev_gaps_in, dev_times_in,
             dev_marks_out, dev_gaps_out, dev_times_out,
             test_marks_in, test_gaps_in, test_times_in,
             test_marks_out, test_gaps_out, test_times_out)

def get_train_input_output(c_data, data, dec_len):
    c_marks, c_times, l1_idxes = c_data
    marks, times = data

    #marks = [np.array(x[1:]) for x in marks]
    c_marks_in = [np.array(x[1:-1]) for x in c_marks]
    c_marks_out = [np.array(x[2:]) for x in c_marks]

    c_gaps = [np.array(x[1:])-np.array(x[:-1]) for x in c_times]
    c_gaps_in = [x[:-1] for x in c_gaps]
    c_gaps_out = [x[1:] for x in c_gaps]

    c_times_in = [np.array(x[1:-1]) for x in c_times]
    c_times_out = [np.array(x[2:]) for x in c_times]

    gaps = np.array(times[1:]) - np.array(times[:-1])
    gaps_in = [[np.array(gaps[idx:idx+dec_len]) for idx in idxes[:-2]] for idxes in l1_idxes]
    times_in = [[np.array(times[idx+1:idx+1+dec_len]) for idx in idxes[1:-1]] for idxes in l1_idxes]
    gaps_out = [[np.array(gaps[idx:idx+dec_len]) for idx in idxes[1:-1]] for idxes in l1_idxes]
    times_out = [[np.array(times[idx+1:idx+1+dec_len]) for idx in idxes[2:]] for idxes in l1_idxes]

    #print(len(c_marks_in), len(c_marks_out),
    #      len(c_gaps_in), len(c_gaps_out),
    #      len(c_times_in), len(c_times_out),
    #      len(gaps_out), len(times_out))

    return (c_marks_in, c_marks_out,
            c_gaps_in, c_gaps_out,
            c_times_in, c_times_out,
            gaps_in, times_in,
            gaps_out, times_out)

def create_train_dev_test_split(data, block_size, decoder_length):
    marks, times, l1_idxes = data
    num_events_per_hour = get_num_events_per_hour((marks, times))
    print(num_events_per_hour.index[0])
    train_marks, train_times, train_l1_idxes = list(), list(), list()
    dev_marks, dev_times, dev_l1_idxes = list(), list(), list()
    test_marks, test_times, test_l1_idxes = list(), list(), list()
    dev_begin_tss, test_begin_tss = list(), list()

    block_begin_idxes = num_events_per_hour.cumsum()
    num_hrs = len(num_events_per_hour)-len(num_events_per_hour)%(4*block_size)
    for idx in range(0, num_hrs, 4*block_size):
        print(idx, num_hrs)
        train_start_idx = block_begin_idxes[idx-1]+1 if idx>0 else 0
        train_end_idx = block_begin_idxes[idx+(2*block_size-1)]-decoder_length-1
        train_marks.append(marks[train_start_idx:train_end_idx])
        train_times.append(times[train_start_idx:train_end_idx])
        train_l1_idxes.append(l1_idxes[train_start_idx:train_end_idx])

        dev_start_idx = block_begin_idxes[idx+(2*block_size-1)]+1-decoder_length-1
        dev_end_idx = block_begin_idxes[idx+(3*block_size-1)]-decoder_length-1
        dev_marks.append(marks[dev_start_idx:dev_end_idx])
        dev_times.append(times[dev_start_idx:dev_end_idx])
        dev_l1_idxes.append(l1_idxes[dev_start_idx:dev_end_idx])
        dev_begin_tss.append(times[dev_start_idx])

        test_start_idx = block_begin_idxes[idx+(3*block_size-1)]+1-decoder_length-1
        test_end_idx = block_begin_idxes[idx+(4*block_size-1)]
        test_marks.append(marks[test_start_idx:test_end_idx])
        test_times.append(times[test_start_idx:test_end_idx])
        test_l1_idxes.append(l1_idxes[test_start_idx:test_end_idx])
        test_begin_tss.append(times[test_start_idx])

    dev_begin_tss = np.expand_dims(np.array(dev_begin_tss), axis=-1)
    test_begin_tss = np.expand_dims(np.array(test_begin_tss), axis=-1)

    return (train_marks, train_times, train_l1_idxes,
            dev_marks, dev_times, dev_l1_idxes,
            test_marks, test_times, test_l1_idxes,
            dev_begin_tss, test_begin_tss)

def transpose(c_m_in, c_g_in, c_t_in, c_m_out, c_g_out, c_t_out, g_in, t_in, g_out, t_out):
    print(c_m_in.shape, c_g_in.shape, c_t_in.shape,
          c_m_out.shape, c_g_out.shape, c_t_out.shape,
          g_out.shape, t_out.shape)
    return (tf.transpose(c_m_in), tf.transpose(c_g_in, [1, 0, 2]), tf.transpose(c_t_in, [1, 0, 2]),
            tf.transpose(c_m_out), tf.transpose(c_g_out, [1, 0, 2]), tf.transpose(c_t_out, [1, 0, 2]),
            tf.transpose(g_in, [1, 0, 2, 3]), tf.transpose(t_in, [1, 0, 2, 3]),
            tf.transpose(g_out, [1, 0, 2, 3]), tf.transpose(t_out, [1, 0, 2, 3]))

def get_padded_dataset(data):
    (c_marks_in, c_gaps_in, c_times_in,
     c_marks_out, c_gaps_out, c_times_out,
     gaps_in, times_in, gaps_out, times_out) = data

    c_seq_lens = np.expand_dims(np.array([len(s) for s in c_times_in]), axis=-1)

    c_marks_in = pad_sequences(c_marks_in, padding='post')
    c_gaps_in = pad_sequences(c_gaps_in, padding='post')
    c_times_in = pad_sequences(c_times_in, padding='post')
    c_marks_out = pad_sequences(c_marks_out, padding='post')
    c_gaps_out = pad_sequences(c_gaps_out, padding='post')
    c_times_out = pad_sequences(c_times_out, padding='post')

    gaps_in = tf.expand_dims(tf.cast(pad_sequences(gaps_in, padding='post'), tf.float32), axis=-1)
    times_in = tf.expand_dims(tf.cast(pad_sequences(times_in, padding='post'), tf.float32), axis=-1)
    gaps_out = tf.expand_dims(tf.cast(pad_sequences(gaps_out, padding='post'), tf.float32), axis=-1)
    times_out = tf.expand_dims(tf.cast(pad_sequences(times_out, padding='post'), tf.float32), axis=-1)

    c_times_in = tf.expand_dims(tf.cast(c_times_in, tf.float32), axis=-1)
    c_times_out = tf.expand_dims(tf.cast(c_times_out, tf.float32), axis=-1)
    c_gaps_in = tf.expand_dims(tf.cast(c_gaps_in, tf.float32), axis=-1)
    c_gaps_out = tf.expand_dims(tf.cast(c_gaps_out, tf.float32), axis=-1)

    #TODO Create and return mask also
 
    return (c_marks_in, c_gaps_in, c_times_in,
            c_marks_out, c_gaps_out, c_times_out,
            c_seq_lens,
            gaps_in, times_in, gaps_out, times_out)

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
    level_1_idxes = np.array([i+K-1 for i in range(0, len(marks), K)])
    #TODO Instead of returning most frequent marker, return the simplex of marks

    assert len(c_times) == len(c_marks)
    assert len(c_times) == len(level_1_idxes)

    return c_marks, c_times, level_1_idxes

def get_preprocessed_(c_data, data, block_size, decoder_length):
    c_marks, c_times, level_1_idxes = c_data
    marks, times = data
    num_categories = len(np.unique(marks))

    (c_train_marks, c_train_times, c_train_l1_idxes,
     c_dev_marks, c_dev_times, c_dev_l1_idxes,
     c_test_marks, c_test_times, c_test_l1_idxes,
     dev_begin_tss, test_begin_tss) \
            = create_train_dev_test_split(c_data, block_size, decoder_length)
    num_sequences = len(c_train_marks)
    
    (c_train_marks_in, c_train_marks_out,
     c_train_gaps_in, c_train_gaps_out,
     c_train_times_in, c_train_times_out,
     train_gaps_in, train_times_in,
     train_gaps_out, train_times_out) \
            = get_train_input_output((c_train_marks, c_train_times, c_train_l1_idxes),
                                     (marks, times),
                                     decoder_length)
    (train_marks, train_times,
     dev_marks, dev_times,
     test_marks, test_times,
     dev_begin_tss, test_begin_tss) \
            = reader_rmtpp.create_train_dev_test_split((marks, times),
                                                       block_size,
                                                       decoder_length)

    (c_train_marks_in, c_train_gaps_in, c_train_times_in,
     c_train_marks_out, c_train_gaps_out, c_train_times_out,
     c_train_seq_lens,
     train_gaps_in, train_times_in,
     train_gaps_out, train_times_out) \
            = get_padded_dataset((c_train_marks_in, c_train_gaps_in, c_train_times_in,
                                  c_train_marks_out, c_train_gaps_out, c_train_times_out,
                                  train_gaps_in, train_times_in,
                                  train_gaps_out, train_times_out))
    (c_train_marks_in, c_train_gaps_in, c_train_times_in,
     c_train_marks_out, c_train_gaps_out, c_train_times_out,
     train_gaps_in, train_times_in,
     train_gaps_out, train_times_out) \
            = transpose(c_train_marks_in, c_train_gaps_in, c_train_times_in,
                        c_train_marks_out, c_train_gaps_out, c_train_times_out,
                        train_gaps_in, train_times_in,
                        train_gaps_out, train_times_out)
    c_train_dataset = tf.data.Dataset.from_tensor_slices((c_train_marks_in,
                                                          c_train_gaps_in,
                                                          c_train_times_in,
                                                          c_train_marks_out,
                                                          c_train_gaps_out,
                                                          c_train_times_out,
                                                          train_gaps_in,
                                                          train_times_in,
                                                          train_gaps_out,
                                                          train_times_out))

    (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in,
     c_dev_marks_out, c_dev_gaps_out, c_dev_times_out,
     c_test_marks_in, c_test_gaps_in, c_test_times_in,
     c_test_marks_out, c_test_gaps_out, c_test_times_out) \
            = get_dev_test_input_output(c_train_marks, c_train_times,
                                        c_dev_marks, c_dev_times,
                                        c_test_marks, c_test_times)
    (dev_marks_in, dev_gaps_in, dev_times_in,
     dev_marks_out, dev_gaps_out, dev_times_out,
     test_marks_in, test_gaps_in, test_times_in,
     test_marks_out, test_gaps_out, test_times_out) \
            = reader_rmtpp.get_dev_test_input_output(train_marks, train_times,
                                                     dev_marks, dev_times,
                                                     test_marks, test_times)

    dev_marks_out = [d_m[-decoder_length:] for d_m in dev_marks_out]
    dev_gaps_out = [d_g[-decoder_length:] for d_g in dev_gaps_out]
    dev_times_out = [d_t[-decoder_length:] for d_t in dev_times_out]
    test_marks_out = [t_m[-decoder_length:] for t_m in test_marks_out]
    test_gaps_out = [d_g[-decoder_length:] for d_g in test_gaps_out]
    test_times_out = [t_t[-decoder_length:] for t_t in test_times_out]
    # TODO Create these according to given offset

    (c_dev_marks_in, c_dev_gaps_in, c_dev_times_in,
     dev_marks_out, dev_gaps_out, dev_times_out,
     c_dev_seq_lens, _, _, _, _) \
            = get_padded_dataset((c_dev_marks_in, c_dev_gaps_in, c_dev_times_in,
                                  dev_marks_out, dev_gaps_out, dev_times_out,
                                  dev_gaps_in, dev_times_in,
                                  dev_gaps_out, dev_times_out))
    c_dev_dataset = tf.data.Dataset.from_tensor_slices((c_dev_marks_in,
                                                        c_dev_gaps_in,
                                                        c_dev_times_in))

    (c_test_marks_in, c_test_gaps_in, c_test_times_in,
     test_marks_out, test_gaps_out, test_times_out,
     c_test_seq_lens, _, _, _, _) \
            = get_padded_dataset((c_test_marks_in, c_test_gaps_in, c_test_times_in,
                                  test_marks_out, test_gaps_out, test_times_out,
                                  test_gaps_in, test_gaps_out,
                                  test_gaps_out, test_times_out))
    c_test_dataset = tf.data.Dataset.from_tensor_slices((c_test_marks_in,
                                                         c_test_gaps_in,
                                                         c_test_times_in))

    return {
        'c_train_dataset': c_train_dataset,
        'c_dev_dataset': c_dev_dataset,
        'c_test_dataset': c_test_dataset,
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
        'c_train_seq_lens': c_train_seq_lens,
        'c_dev_seq_lens': c_dev_seq_lens,
        'c_test_seq_lens': c_test_seq_lens,
        }

def get_preprocessed_data(block_size, decoder_length):
    marks, times = read_data('testdata.txt')
    c_marks, c_times, level_1_idxes = get_compound_events((marks, times), K=10)
    #marks, times = split_data((marks, times), 7)
    
    block_size_sec = block_size * 3600.0

    data_hierarchical = get_preprocessed_((c_marks, c_times, level_1_idxes),
                                          (marks, times),
                                          block_size,
                                          decoder_length)

    # ----- Start: create compound events ----- #
    #c_train_times_in = get_compound_times(train_times_in, K=10)
    #c_dev_times_in = get_compound_times(dev_times_in, K=10)
    #c_test_times_in = get_compound_times(test_times_in, K=10)
    #c_marks, c_times = get_compound_events((marks, times), K=10)
    #data_level_2 = get_preprocessed_level((c_marks, c_times),
    #                                      (marks, times), 2,
    #                                      block_size, decoder_length)

    #assert data_level_1['num_sequences'] == data_level_2['num_sequences']
    # ----- End: create compound events ----- #

    return data_hierarchical



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

