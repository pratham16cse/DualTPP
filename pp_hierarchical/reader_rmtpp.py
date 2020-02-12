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


def read_multiseq_data(data_path):
    marks_seqs, times_seqs = dict(), dict()
    data = list()
    with open(data_path, 'r') as f:
        for line in f:
            s_id, m, t = line.strip().split(' ')
            if times_seqs.get(s_id, -1) == -1:
                marks_seqs[s_id] = list()
                times_seqs[s_id] = list()
            else:
                marks_seqs[s_id].append(int(m))
                times_seqs[s_id].append(float(t))

    return list(marks_seqs.values()), list(times_seqs.values())

def read_taxi_dataset(data_path):

    data = pd.read_csv(data_path)

    # 06/19/2018 08:13:00 AM
    # data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'], format = '%m/%d/%Y %I:%M:%S %p')
    data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'], format = '%Y-%m-%d %H:%M:%S')
    data = data[(data['tpep_pickup_datetime'].dt.year == 2019)]
    data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'], format = '%Y-%m-%d %H:%M:%S')
    data = data[(data['tpep_dropoff_datetime'].dt.year == 2019)]
    data['pickup'] = data.tpep_pickup_datetime.values.astype(np.int64) // 10 ** 9
    data['dropoff'] = data.tpep_dropoff_datetime.values.astype(np.int64) // 10 ** 9

    data['Time'] = data['pickup']
    data['loc'] = data['PULocationID']
    data['ID1'] = '1'
    data['ID2'] = '2'

    data1 = pd.DataFrame()
    data1['Time'] = data['Time'].append(data['dropoff'], ignore_index=True)
    data1['loc'] = data['loc'].append(data['DOLocationID'], ignore_index=True)
    data1['ID'] = data['ID1'].append(data['ID2'], ignore_index=True)


    # data['Time'] = data['pickup']
    # data['ID'] = data['PULocationID']
    data = data1

    df = data['Time'].groupby(data['loc'])
    data = data.sort_values(['Time'], ascending=True)

    # print(data)
    from collections import defaultdict
    dlst = defaultdict(list)
    deve = defaultdict(list)


    lst1=data['Time'].values.tolist()
    eve1=data['ID'].values.tolist()
    loc=data['loc'].values.tolist()

    for x in range(len(lst1)):
        dlst[loc[x]].append(lst1[x])
        deve[loc[x]].append(eve1[x])

    # print(dlst)
    # print(deve)

    lst = list()
    eve = list()

    for x in dlst:
        #if len(dlst[x]) > 10*sequence_length:# Need at least one train, dev,
        #                                    # and test chunk from each chunk
        lst.append(dlst[x])

    for x in deve:
        #if len(deve[x]) > 10*sequence_length:
        eve.append(deve[x])


    return lst, eve, lst, eve

def timestampToTime(timestamp):
    #t = time.strftime("%Y %m %d %H %M %S", time.localtime(timestamp)).split(' ')
    t = time.strftime("%m %d %H %M %S", time.gmtime(timestamp)).split(' ')
    t = [int(i) for i in t]
    return t

def getTheHour(timestamp):
    if timestamp == 0:
        return 0
    return timestampToTime(timestamp)[2]

def read_data(dataset_name, filename):
    if dataset_name in ['Taxi']:
        times_seqs, marks_seqs, _, _ = read_taxi_dataset(filename)
    elif dataset_name in ['testdata_multiseq', 'sin_multiseq']:
        marks_seqs, times_seqs = read_multiseq_data(filename)
    else:
        with open(filename, 'r') as f:
            data = list()
            for line in f:
                mark, time = line.strip().split()[:2]
                data.append((int(mark), float(time)))
        data = sorted(data, key=itemgetter(1))
        marks, times = zip(*data)
        marks_seqs, times_seqs = [marks], [times]

    marks_seqs = [np.array(marks) for marks in marks_seqs]
    times_seqs = [np.array(times) for times in times_seqs]
    times_seqs = [times-times[0] for times in times_seqs]
    return marks_seqs, times_seqs

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

def get_time_features(data):

    train_times_in, dev_times_in, test_times_in = data

    train_time_feature = tf.py_function(func=getTheHour, inp=[train_times_in], Tout=tf.float32)
    dev_time_feature = tf.py_function(func=getTheHour, inp=[dev_times_in], Tout=tf.float32)
    test_time_feature = tf.py_function(func=getTheHour, inp=[test_times_in], Tout=tf.float32)

    return train_time_feature, dev_time_feature, test_time_feature

def get_time_features_for_data(data):
    times_in = data
    time_feature_hour = (times_in // 3600) % 24
    time_feature_minute = (times_in // 60) % 60
    time_feature_seconds = (times_in) % 60

    time_feature = (time_feature_hour * 3600.0 
                 + time_feature_minute * 60.0
                 + time_feature_seconds)
    time_feature = time_feature / 3600.0
    return time_feature

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
    plt.close()
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

    #print('DevIn:', dev_times_in[0].tolist())
    #print('\n')
    #print('DevOut', dev_times_out[0].tolist())
    #print('\n')
    #print('TestIn:', test_times_in[0].tolist())
    #print('\n')
    #print('TestOut', test_times_out[0].tolist())

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
    marks_sequences, times_sequences = data

    train_marks, train_times = list(), list()
    dev_marks, dev_times = list(), list()
    test_marks, test_times = list(), list()
    dev_begin_tss, test_begin_tss = list(), list()

    for marks, times in zip(marks_sequences, times_sequences):
        num_events_per_hour = get_num_events_per_hour((marks, times))
        print(num_events_per_hour.index[0])
        block_begin_idxes = num_events_per_hour.cumsum()
        num_hrs = len(num_events_per_hour)-len(num_events_per_hour)%(4*block_size)
        for idx in range(0, num_hrs, 4*block_size):
            print(idx, num_hrs)
            train_start_idx = block_begin_idxes[idx-1] if idx>0 else 0
            train_end_idx = block_begin_idxes[idx+(2*block_size-1)]#-decoder_length-1
            train_marks.append(marks[train_start_idx:train_end_idx])
            train_times.append(times[train_start_idx:train_end_idx])
            print(idx, 'train length:', len(times[train_start_idx:train_end_idx]))

            dev_start_idx = block_begin_idxes[idx+(2*block_size-1)]#-decoder_length-1
            dev_end_idx = block_begin_idxes[idx+(3*block_size-1)]#-decoder_length-1
            dev_marks.append(marks[dev_start_idx:dev_end_idx])
            dev_times.append(times[dev_start_idx:dev_end_idx])
            dev_begin_tss.append(get_hour_of_day_ts(times[dev_start_idx]) * 3600.)
            print(idx, 'dev length:', len(times[dev_start_idx:dev_end_idx]))

            test_start_idx = block_begin_idxes[idx+(3*block_size-1)]#-decoder_length-1
            test_end_idx = block_begin_idxes[idx+(4*block_size-1)]
            test_marks.append(marks[test_start_idx:test_end_idx])
            test_times.append(times[test_start_idx:test_end_idx])
            test_begin_tss.append(get_hour_of_day_ts(times[test_start_idx]) * 3600.)
            print(idx, 'test length:', len(times[test_start_idx:test_end_idx]))

    train_marks = train_marks * 5
    train_times = train_times * 5
    dev_marks = dev_marks * 5
    dev_times = dev_times * 5
    test_marks = test_marks * 5
    test_times = test_times * 5
    dev_begin_tss = dev_begin_tss * 5
    test_begin_tss = test_begin_tss * 5

    dev_begin_tss = tf.expand_dims(tf.constant(dev_begin_tss), axis=-1)
    test_begin_tss = tf.expand_dims(tf.constant(test_begin_tss), axis=-1)


    return (train_marks, train_times,
            dev_marks, dev_times,
            test_marks, test_times,
            dev_begin_tss, test_begin_tss)

def transpose(m_in, g_in, t_in, sm_in, m_out, g_out, t_out, sm_out, time_feature):
    return tf.transpose(m_in), tf.transpose(g_in, [1, 0, 2]), tf.transpose(t_in, [1, 0, 2]), tf.transpose(sm_in), \
            tf.transpose(m_out), tf.transpose(g_out, [1, 0, 2]), tf.transpose(t_out, [1, 0, 2]), tf.transpose(sm_out), \
            tf.transpose(time_feature, [1, 0, 2])

def get_padded_dataset(data):
    marks_in, gaps_in, times_in, marks_out, gaps_out, times_out = data

    seq_lens = np.expand_dims(np.array([len(s) for s in times_in]), axis=-1)

    marks_in = pad_sequences(marks_in, padding='post')
    gaps_in = pad_sequences(gaps_in, padding='post')
    times_in = pad_sequences(times_in, padding='post')
    marks_out = pad_sequences(marks_out, padding='post')
    gaps_out = pad_sequences(gaps_out, padding='post')
    times_out = pad_sequences(times_out, padding='post')

    #print('seq_lens', seq_lens)
    #print('times_in.shape', times_in.shape)

    gaps_in = tf.expand_dims(tf.cast(gaps_in, tf.float32), axis=-1)
    times_in = tf.expand_dims(tf.cast(times_in, tf.float32), axis=-1)
    gaps_out = tf.expand_dims(tf.cast(gaps_out, tf.float32), axis=-1)
    times_out = tf.expand_dims(tf.cast(times_out, tf.float32), axis=-1)

    return marks_in, gaps_in, times_in, marks_out, gaps_out, times_out, seq_lens

def get_seq_mask(sequences):
    seq_lens = [len(s) for s in sequences]
    seq_mask = tf.sequence_mask(seq_lens, dtype=tf.float32)
    return seq_mask, seq_lens

def get_preprocessed_(data, block_size, decoder_length, normalization):
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

    (train_time_feature) \
            = get_time_features_for_data((train_times_in))

    (train_marks_in, train_gaps_in, train_times_in, train_seqmask_in,
     train_marks_out, train_gaps_out, train_times_out, train_seqmask_out, train_time_feature) \
            = transpose(train_marks_in, train_gaps_in, train_times_in, train_seqmask_in,
                        train_marks_out, train_gaps_out, train_times_out, train_seqmask_out,
                        train_time_feature)

    (train_gaps_in_norm, train_gaps_out_norm,
     train_normalizer_d, train_normalizer_a) \
            = get_normalized_dataset((train_gaps_in, train_gaps_out),
                                     normalization=normalization)

    print('train_time_feature', train_time_feature)
    print('train_times_in', train_times_in)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_marks_in,
                                                        train_gaps_in_norm,
                                                        train_times_in,
                                                        train_seqmask_in,
                                                        train_marks_out,
                                                        train_gaps_out_norm,
                                                        train_times_out,
                                                        train_seqmask_out,
                                                        train_time_feature))

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

    (dev_time_feature) \
            = get_time_features_for_data((dev_times_in))

    (dev_gaps_in_norm, dev_gaps_out_norm,
     dev_normalizer_d, dev_normalizer_a) \
            = get_normalized_dataset((dev_gaps_in, dev_gaps_out),
                                     normalization=normalization)

    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_marks_in,
                                                      dev_gaps_in_norm,
                                                      dev_times_in,
                                                      dev_seqmask_in,
                                                      dev_time_feature))

    test_seqmask_in, _ = get_seq_mask(test_gaps_in)
    test_seqmask_out, _ = get_seq_mask(test_gaps_out)

    (test_marks_in, test_gaps_in, test_times_in,
     test_marks_out, test_gaps_out, test_times_out,
     test_seq_lens) \
            = get_padded_dataset((test_marks_in, test_gaps_in, test_times_in,
                                  test_marks_out, test_gaps_out, test_times_out))

    print('test_times_in', test_times_in)
    (test_time_feature) \
            = get_time_features_for_data((test_times_in))

    print('test_time_feature', test_time_feature)

    (test_gaps_in_norm, test_gaps_out_norm,
     test_normalizer_d, test_normalizer_a) \
            = get_normalized_dataset((test_gaps_in, test_gaps_out))

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
        'train_seqmask_out': train_seqmask_out,
        'dev_seqmask_out': dev_seqmask_out,
        'test_seqmask_out': test_seqmask_out,

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

def get_preprocessed_data(dataset_name, dataset_path, block_size, decoder_length, normalization):
    marks, times = read_data(dataset_name, dataset_path)
    data = get_preprocessed_((marks, times), block_size, decoder_length,
                             normalization)
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

