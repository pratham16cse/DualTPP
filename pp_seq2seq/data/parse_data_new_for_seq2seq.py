import sys, os
sys.path.append('../../tf_rmtpp/src/tf_rmtpp/')
import argparse
import pickle
from itertools import chain
import numpy as np
from BatchGenerator import BatchGenerator
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from itertools import chain
from collections import Counter, OrderedDict
from operator import itemgetter
import pandas as pd
from datetime import datetime
import time
from bisect import bisect_left, bisect_right
#from preprocess_rmtpp_data import preprocess

np.random.seed(42)
ONE_DAY_SECS = 3600.0 * 24

def getDatetime(epoch):
    date_string=time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(epoch))
    date= datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
    return date

def timestampTotime(ts):
    t = time.strftime("%Y %m %d %H %M %S", time.localtime(ts)).split(' ')
    #t = time.strftime("%m %d %H %M %S", time.gmtime(ts)).split(' ')
    t = [int(i) for i in t]
    return t

def getHour(epoch):
    return timestampTotime(epoch)[3]

def getBeginOfDayTs(ts):
    dt_obj = timestampTotime(ts)
    return datetime.timestamp(datetime(dt_obj[0], dt_obj[1], dt_obj[2]))

def plotter(sequence, enc_len, dec_len, dataset_name):
    sequence = sequence[0]
    print(len(sequence))
    idxes = np.random.choice(range(len(sequence)), size=100, replace=False)
    for idx in idxes:
        print(idx)
        times_in = np.array(sequence[idx:idx+enc_len])
        print(times_in)
        gaps_in = times_in[1:] - times_in[:-1]
        times_out = np.array(sequence[idx+enc_len:idx+enc_len+dec_len])
        gaps_out = np.concatenate([[times_out[0]-times_in[-1]],
                                    times_out[1:] - times_out[:-1]],
                                  axis=-1)
        plt_gap_seq = np.concatenate([gaps_in, gaps_out], axis=-1)
        plt_time_seq = np.concatenate([times_in, times_out], axis=-1)

        # ----- Plotting gaps with discrete index ----- #
        plt.plot(range(1, len(gaps_in)+1), gaps_in, 'ko-')
        plt.plot(range(enc_len, enc_len+len(gaps_out)), gaps_out, 'bo-')
        #plt.plot(range(enc_len, enc_len+len(gaps_pred)), gaps_pred, 'r*-')
        plt.plot([enc_len-0.5, enc_len-0.5], [0, max(plt_gap_seq)], 'g')
        plt.savefig(os.path.join('plots', dataset_name, str(idx)))
        plt.close()


        # ----- Plotting gaps with time axis ----- #
        #cum_gaps_in = np.cumsum(gaps_in)
        #cum_gaps_out = np.cumsum(gaps_out)
        #cum_gaps_pred = np.cumsum(gaps_pred)
        #c = cum_gaps_in[-1]
        #for g in cum_gaps_in:
        #    plt.plot([g, g], [0, 1], 'ko-')
        #for g in cum_gaps_out:
        #    plt.plot([c+g, c+g], [0, 1], 'bo-')
        #for g in cum_gaps_pred:
        #    plt.plot([c+g, c+g], [0, 1], 'r*-')
        #plt.plot([c,c],[0,2], 'g-')

def parse_single_seq_datasets(dataset_path, keep_classes, num_coarse_seq):
    data = pd.read_csv(dataset_path, usecols= [0, 1], delimiter=' ', header=None, names=['ID', 'Time'])
    data['Time'] = data['Time'].astype(float)
    data = data.sort_values(['Time'], ascending=True)

    lst = []
    eve = []

    lst=data['Time'].values.tolist()
    eve=data['ID'].values.tolist()

    eve2id = dict()
    crnt_id = 1
    for e in eve:
        if eve2id.get(e, -1) == -1:
            eve2id[e] = crnt_id
            crnt_id += 1

    eve = [eve2id[e] for e in eve]

    eve = group_less_active_classes(eve, keep_classes=keep_classes)
    #print(eve)

    total = len(lst)

    print(total, 'total')

    if num_coarse_seq==0:
        time_seqs = [lst]
        event_seqs = [eve]
    else:
        time_seqs = np.array_split(np.array(lst), num_coarse_seq)
        event_seqs = np.array_split(np.array(eve), num_coarse_seq)

    return time_seqs, event_seqs, lst, eve

def parse_neural_hawkes_datasets(dataset_path, keep_classes):
    raise NotImplementedError

def parse_taxi_dataset(data_path, keep_classes, sequence_length):

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
        if len(dlst[x]) > 10*sequence_length:# Need at least one train, dev,
                                            # and test chunk from each chunk
            lst.append(dlst[x])

    for x in deve:
        if len(deve[x]) > 10*sequence_length:
            eve.append(deve[x])

    return lst, eve, lst, eve

def print_dataset_stats(dataset_name, train_data, dev_data, test_data):
    print('\n#----- Dataset_name:{} -----#'.format(dataset_name))

    unique_labels = set([elm['type_event'] for seq in train_data['train'] for elm in seq] \
                      + [elm['type_event'] for seq in dev_data['dev'] for elm in seq] \
                      + [elm['type_event'] for seq in test_data['test'] for elm in seq])

    print('Number of Unique Labels: {}'.format(len(unique_labels)))
    print('# of Event Tokens:')
    train_seq_lens = [len(seq) for seq in train_data['train']]
    dev_seq_lens = [len(seq) for seq in dev_data['dev']]
    test_seq_lens = [len(seq) for seq in test_data['test']]
    num_trn_events = sum(train_seq_lens)
    num_dev_events = sum(dev_seq_lens)
    num_tst_events = sum(test_seq_lens)
    print('train: {}, dev: {}, test: {}'.format(num_trn_events, num_dev_events, num_tst_events))

    print('Sequence Lengths:')
    print('Minimum-- train: {}, dev: {}, test: {}'.format(min(train_seq_lens), min(dev_seq_lens), min(test_seq_lens)))
    print('Average-- train: {}, dev: {}, test: {}'.format(np.mean(train_seq_lens), np.mean(dev_seq_lens), np.mean(test_seq_lens)))
    print('Maximum-- train: {}, dev: {}, test: {}'.format(max(train_seq_lens), max(dev_seq_lens), max(test_seq_lens)))

    allTimes = [elm['time_since_start'] for seq in train_data['train'] for elm in seq] \
             + [elm['time_since_start'] for seq in dev_data['dev'] for elm in seq] \
             + [elm['time_since_start'] for seq in test_data['test'] for elm in seq]
    minTime, avgTime, maxTime = min(allTimes), sum(allTimes)/len(allTimes), max(allTimes)
    print('Time Values stats: minimum:{}, average:{}, maximum:{}'.format(minTime, avgTime, maxTime))


def print_dataset_stats_table_format(dataset_name, train_data, dev_data, test_data):
    print('\n')
    print(dataset_name+' & '),

    unique_labels = set([elm['type_event'] for seq in train_data['train'] for elm in seq] \
                      + [elm['type_event'] for seq in dev_data['dev'] for elm in seq] \
                      + [elm['type_event'] for seq in test_data['test'] for elm in seq])

    print(str(len(unique_labels)) + ' & '),
    train_seq_lens = [len(seq) for seq in train_data['train']]
    dev_seq_lens = [len(seq) for seq in dev_data['dev']]
    test_seq_lens = [len(seq) for seq in test_data['test']]
    all_seq_lens = train_seq_lens + dev_seq_lens + test_seq_lens
    num_trn_events = sum(train_seq_lens)
    num_dev_events = sum(dev_seq_lens)
    num_tst_events = sum(test_seq_lens)
    print(str(num_trn_events)+' & '+str(num_dev_events)+' & '+str(num_tst_events)+' & '),

    print(str(min(all_seq_lens))+' & '+str(np.mean(all_seq_lens))+' & '+str(max(all_seq_lens))+' & '),

    print(str(len(train_seq_lens))+' & '+str(len(dev_seq_lens))+' & '+str(len(test_seq_lens)))

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    #if after - myNumber < myNumber - before:
    #   return after
    #else:
    #   return before
    return before

def create_superclasses(events, num_classes=0):
    if num_classes == 0: # Do not create superclasses
        print('No Superclasses')
        return events
    N = len(events)
    C = len(np.unique(events))
    threshold = int(N / num_classes)

    events_counter = OrderedDict(sorted(Counter(events).items(), key=itemgetter(1), reverse=True))
    #print(events_counter)

    cls2supercls = dict()
    new_super_cls = 0
    removed_classes = dict()
    while len(removed_classes) < C:
        class_cardinality = 0
        new_super_cls += 1
        for class_, freq in events_counter.items():
            if removed_classes.get(class_, -1) == -1:
                if class_cardinality < threshold:
                    cls2supercls[class_] = new_super_cls
                    class_cardinality += freq
                    removed_classes[class_] = True
                else:
                    break

    events_new = [cls2supercls[event] for event in events]

    print(Counter(events_new))

    return events_new

def group_less_active_classes(events, keep_classes=0):
    if keep_classes == 0:
        print('No Superclasses')
        return events

    events_counter = OrderedDict(sorted(Counter(events).items(), key=itemgetter(1), reverse=True))
    #print(events_counter)

    cls2supercls = dict()
    new_super_cls = 0
    removed_classes = dict()
    for i, (class_, _) in enumerate(events_counter.items()):
        if i <= keep_classes:
            new_super_cls += 1
        cls2supercls[class_] = new_super_cls

    events_new = [cls2supercls[event] for event in events]

    print(Counter(events_new))

    return events_new

# def generate_norm_seq(sequences, check=0):
#     gap_seqs = sequences[:,1:]-sequences[:,:-1]
#     avg_gaps = np.average(gap_seqs, axis=1)
#     avg_gaps = np.expand_dims(avg_gaps, axis=1)

#     avg_gaps_norm = gap_seqs/avg_gaps
#     avg_gaps_norm = np.cumsum(avg_gaps_norm[::-1])[::-1]

#     gap_norm_seq = sequences[:,-1:] - avg_gaps_norm
#     gap_norm_seq = np.hstack((gap_norm_seq, sequences[:,-1:]))

#     if check==1:
#         print("sequences[:,:1]")
#         print(sequences[:,:1])

#     return gap_norm_seq

def generate_norm_seq(sequences, encoder_length, check=0):
    gap_seqs = sequences[:,1:]-sequences[:,:-1]
    avg_gaps = np.average(gap_seqs[:,:encoder_length-1], axis=1)
    avg_gaps = np.expand_dims(avg_gaps, axis=1)
    avg_gaps = np.clip(avg_gaps, 1.0, np.inf)

    avg_gaps_norm = gap_seqs/avg_gaps
    avg_gaps_norm = np.cumsum(avg_gaps_norm, axis=1)

    zero_pad = np.zeros(np.shape(sequences[:,:1]))
    # gap_norm_seq = sequences[:,:1] + avg_gaps_norm
    gap_norm_seq = np.hstack((zero_pad, avg_gaps_norm))

    if check==1:
        print("sequences[:,:1]")
        print(sequences[:,:1])

    return gap_norm_seq, avg_gaps

def prune_seqs(time_batch_in, event_batch_in,
               time_batch_out, event_batch_out,
               tsIndices, enc_len,
               num_sample_offsets, sampled_offsets):
    tm_b_in, eve_b_in, tm_b_out, eve_b_out, ts, sampl_off = list(), list(), list(), list(), list(), list()
    for i in range(len(time_batch_in)):
        seq = np.array(time_batch_in[i][:enc_len-1])
        gaps_sum = seq[1:]-seq[:-1]
        if sum(gaps_sum==0.0)<0.25*enc_len:# and sum(gaps_sum)>enc_len:
            tm_b_in.append(time_batch_in[i])
            eve_b_in.append(event_batch_in[i])
            tm_b_out.append(time_batch_out[i])
            eve_b_out.append(event_batch_out[i])
            ts.append(tsIndices[i])
            if num_sample_offsets != 0:
                sampl_off.append(sampled_offsets[i])
        else:
            print('pruned ', tsIndices[i], gaps_sum)

    return tm_b_in, eve_b_in, tm_b_out, eve_b_out, ts, sampl_off


def get_input_output_seqs(time_input_seqs, event_input_seqs, time_seqs, event_seqs,
                          ts_indices, max_offset, num_sample_offsets, decoder_length,
                          is_test=False):

    max_offset_sec = max_offset * 3600.0

    time_input_seqs_new, event_input_seqs_new, time_output_seqs, event_output_seqs, ts_indices_new \
            = list(), list(), list(), list(), list()
    sampled_offsets = list()

    for ts_ind, time_in_seq, event_in_seq in zip(ts_indices, time_input_seqs, event_input_seqs):
        if num_sample_offsets==0:
            start_ind = bisect_right(time_seqs[ts_ind], time_in_seq[-1]) # Index of last encoder input in the sequence
            end_ind = bisect_right(time_seqs[ts_ind], time_in_seq[-1]+max_offset_sec) + decoder_length + 1 # Index at which 
            #print(start_ind, end_ind)
            if end_ind < len(time_seqs[ts_ind]):
                time_out_seq = time_seqs[ts_ind][start_ind:end_ind]
                event_out_seq = event_seqs[ts_ind][start_ind:end_ind]
                time_output_seqs.append(time_out_seq)
                event_output_seqs.append(event_out_seq)
                time_input_seqs_new.append(time_in_seq)
                event_input_seqs_new.append(event_in_seq)
                ts_indices_new.append(ts_ind)
        else:
            for i in range(num_sample_offsets):
                low = 0.0 if not is_test else 0.9 * max_offset_sec
                sample_off = np.random.uniform(low=low, high=max_offset_sec)
                start_ind = bisect_right(time_seqs[ts_ind], time_in_seq[-1] + sample_off) # Index of (last_encoder_input + sampled_offset)
                end_ind = start_ind + decoder_length # start_ind + dec_len
                #print(start_ind, end_ind)
                if end_ind < len(time_seqs[ts_ind]):
                    time_out_seq = time_seqs[ts_ind][start_ind:end_ind]
                    event_out_seq = event_seqs[ts_ind][start_ind:end_ind]
                    time_output_seqs.append(time_out_seq)
                    event_output_seqs.append(event_out_seq)
                    time_input_seqs_new.append(time_in_seq)
                    event_input_seqs_new.append(event_in_seq)
                    ts_indices_new.append(ts_ind)
                    sampled_offsets.append(sample_off)

    return time_input_seqs_new, event_input_seqs_new, time_output_seqs, event_output_seqs, ts_indices_new, sampled_offsets

def preprocess(raw_dataset_name,
               dataset_name,
               all_timestamps,
               all_events,
               train_event_seq, train_time_seq,
               dev_event_seq, dev_time_seq,
               test_event_seq, test_time_seq,
               encoder_length, decoder_length,
               train_step_length=None, dev_step_length=None, test_step_length=None,
               keep_classes=0, num_coarse_seq=0, offset=0.0,
               max_offset=0.0, num_sample_offsets=0):

    sequence_length = encoder_length + decoder_length

    print('Step Lengths:', train_step_length, dev_step_length, test_step_length)

    if train_step_length is None:
        train_step_length = sequence_length
    if dev_step_length is None:
        dev_step_length = decoder_length
    if test_step_length is None:
        test_step_length = decoder_length


    unique_labels = set([lbl for lbl in chain(*(train_event_seq + test_event_seq))])
    #print('Unique Labels:', np.array(sorted(unique_labels)))

    train_event_itr = BatchGenerator(train_event_seq, batchSeqLen=sequence_length, batch_pad=False)
    train_time_itr = BatchGenerator(train_time_seq, batchSeqLen=sequence_length, batch_pad=False)

    dev_event_itr = BatchGenerator(dev_event_seq, batchSeqLen=sequence_length, batch_pad=False)
    dev_time_itr = BatchGenerator(dev_time_seq, batchSeqLen=sequence_length, batch_pad=False)

    test_event_itr = BatchGenerator(test_event_seq, batchSeqLen=sequence_length, batch_pad=False)
    test_time_itr = BatchGenerator(test_time_seq, batchSeqLen=sequence_length, batch_pad=False)

    pp_train_event_in_seq, pp_dev_event_in_seq, pp_test_event_in_seq = list(), list(), list()
    pp_train_time_in_seq, pp_dev_time_in_seq, pp_test_time_in_seq = list(), list(), list()
    pp_train_event_out_seq, pp_dev_event_out_seq, pp_test_event_out_seq = list(), list(), list()
    pp_train_time_out_seq, pp_dev_time_out_seq, pp_test_time_out_seq = list(), list(), list()
    dev_actual_time_in, test_actual_time_in = list(), list()
    dev_actual_time_out, test_actual_time_out = list(), list()

    train_tsIndices, dev_tsIndices, test_tsIndices = list(), list(), list()
    train_sampled_offsets, dev_sampled_offsets, test_sampled_offsets = list(), list(), list()

    print('Creating training data . . .')
    batch_size = len(train_time_seq)
    currItr = train_time_itr.iterFinished
    ctr = 0
    while currItr == train_time_itr.iterFinished:
        ctr += 1
        train_event_batch, _, _, _, _ = train_event_itr.nextBatch(batchSize=batch_size, stepLen=train_step_length)
        train_time_batch, tsIndices, _, _, _ = train_time_itr.nextBatch(batchSize=batch_size, stepLen=train_step_length)
        print(ctr)

        #if ctr == 1: print(train_event_batch.shape)

        if max_offset:
            if num_coarse_seq > 0:
                lookup_time_seq = [all_timestamps] * len(train_time_seq)
                lookup_event_seq = [all_events] * len(train_event_seq)
            else:
                lookup_time_seq = train_time_seq
                lookup_event_seq = train_event_seq
            train_time_batch_in, train_event_batch_in, \
            train_time_batch_out, train_event_batch_out, tsIndices, \
            sampled_offsets \
                    = get_input_output_seqs(train_time_batch[:, :encoder_length].tolist(),
                                            train_event_batch[:, :encoder_length].tolist(),
                                            lookup_time_seq, lookup_event_seq, tsIndices,
                                            max_offset, num_sample_offsets, decoder_length)
            train_time_batch_in, train_event_batch_in, \
            train_time_batch_out, train_event_batch_out, tsIndices, \
            sampled_offsets \
                    = prune_seqs(train_time_batch_in, train_event_batch_in,
                                 train_time_batch_out, train_event_batch_out,
                                 tsIndices, encoder_length,
                                 num_sample_offsets, sampled_offsets)
            pp_train_event_in_seq.extend(train_event_batch_in)
            pp_train_time_in_seq.extend(train_time_batch_in)
            pp_train_event_out_seq.extend(train_event_batch_out)
            pp_train_time_out_seq.extend(train_time_batch_out)
        else:
            pp_train_event_in_seq.extend(train_event_batch[:, :encoder_length].tolist())
            pp_train_time_in_seq.extend(train_time_batch[:, :encoder_length].tolist())
            pp_train_event_out_seq.extend(train_event_batch[:, encoder_length:].tolist())
            pp_train_time_out_seq.extend(train_time_batch[:, encoder_length:].tolist())

        train_tsIndices += tsIndices
        train_sampled_offsets += sampled_offsets

    assert len(train_tsIndices) == len(pp_train_time_in_seq)
    print('Created training data . . .')

    print('Creating dev data . . .')
    batch_size = len(dev_time_seq)
    currItr = dev_time_itr.iterFinished
    ctr = 0
    while currItr == dev_time_itr.iterFinished:
        ctr += 1
        dev_event_batch, _, _, _, _ = dev_event_itr.nextBatch(batchSize=batch_size, stepLen=dev_step_length)
        dev_time_batch, tsIndices, _, _, _ = dev_time_itr.nextBatch(batchSize=batch_size, stepLen=dev_step_length)
        print(ctr)

        #if ctr == 1: print(dev_event_batch.shape)

        if max_offset:
            if num_coarse_seq > 0:
                lookup_time_seq = [all_timestamps for _ in dev_time_seq]
                lookup_event_seq = [all_events for _ in dev_event_seq]
            else:
                lookup_time_seq = dev_time_seq
                lookup_event_seq = dev_event_seq
            dev_time_batch_in, dev_event_batch_in, \
            dev_time_batch_out, dev_event_batch_out, tsIndices, \
            sampled_offsets \
                    = get_input_output_seqs(dev_time_batch[:, :encoder_length].tolist(),
                                            dev_event_batch[:, :encoder_length].tolist(),
                                            lookup_time_seq, lookup_event_seq, tsIndices,
                                            max_offset, num_sample_offsets, decoder_length)
            dev_time_batch_in, dev_event_batch_in, \
            dev_time_batch_out, dev_event_batch_out, tsIndices, \
            sampled_offsets \
                    = prune_seqs(dev_time_batch_in, dev_event_batch_in,
                                 dev_time_batch_out, dev_event_batch_out,
                                 tsIndices, encoder_length,
                                 num_sample_offsets, sampled_offsets)
            pp_dev_event_in_seq.extend(dev_event_batch_in)
            pp_dev_time_in_seq.extend(dev_time_batch_in)
            pp_dev_event_out_seq.extend(dev_event_batch_out)
            pp_dev_time_out_seq.extend(dev_time_batch_out)
        else:
            pp_dev_event_in_seq.extend(dev_event_batch[:, :encoder_length].tolist())
            pp_dev_time_in_seq.extend(dev_time_batch[:, :encoder_length].tolist())
            pp_dev_event_out_seq.extend(dev_event_batch[:, encoder_length:].tolist())
            pp_dev_time_out_seq.extend(dev_time_batch[:, encoder_length:].tolist())

        dev_tsIndices += tsIndices
        dev_sampled_offsets += sampled_offsets

    assert len(dev_tsIndices) == len(pp_dev_time_in_seq)
    print('Created dev data . . .')

    print('Creating test data . . .')
    batch_size = len(test_time_seq)
    currItr = test_time_itr.iterFinished
    ctr = 0
    while currItr == test_time_itr.iterFinished:
        ctr += 1
        test_event_batch, _, _, _, _ = test_event_itr.nextBatch(batchSize=batch_size, stepLen=test_step_length)
        test_time_batch, tsIndices, _, _, _ = test_time_itr.nextBatch(batchSize=batch_size, stepLen=test_step_length)
        print(ctr)

        #if ctr == 1: print(test_event_batch.shape)

        if max_offset:
            if num_coarse_seq > 0:
                lookup_time_seq = [all_timestamps for _ in test_time_seq]
                lookup_event_seq = [all_events for _ in test_event_seq]
            else:
                lookup_time_seq = test_time_seq
                lookup_event_seq = test_event_seq
            test_time_batch_in, test_event_batch_in, \
            test_time_batch_out, test_event_batch_out, tsIndices, \
            sampled_offsets \
                    = get_input_output_seqs(test_time_batch[:, :encoder_length].tolist(),
                                            test_event_batch[:, :encoder_length].tolist(),
                                            lookup_time_seq, lookup_event_seq, tsIndices,
                                            max_offset, num_sample_offsets, decoder_length,
                                            is_test=True)
            test_time_batch_in, test_event_batch_in, \
            test_time_batch_out, test_event_batch_out, tsIndices, \
            sampled_offsets \
                    = prune_seqs(test_time_batch_in, test_event_batch_in,
                                 test_time_batch_out, test_event_batch_out,
                                 tsIndices, encoder_length,
                                 num_sample_offsets, sampled_offsets)
            pp_test_event_in_seq.extend(test_event_batch_in)
            pp_test_time_in_seq.extend(test_time_batch_in)
            pp_test_event_out_seq.extend(test_event_batch_out)
            pp_test_time_out_seq.extend(test_time_batch_out)
        else:
            pp_test_event_in_seq.extend(test_event_batch[:, :encoder_length].tolist())
            pp_test_time_in_seq.extend(test_time_batch[:, :encoder_length].tolist())
            pp_test_event_out_seq.extend(test_event_batch[:, encoder_length:].tolist())
            pp_test_time_out_seq.extend(test_time_batch[:, encoder_length:].tolist())

        test_tsIndices += tsIndices
        test_sampled_offsets += sampled_offsets
    print('Created test data . . .')


    # ----- Start: Create Input-Output sequences using offset ----- #
    #all_timestamps_dict = dict()
    #for ind, ts in enumerate(all_timestamps):
    #    all_timestamps_dict[ts] = ind

    #if offset > 0.0:
    #    offset_sec = offset * 3600.0
    #    def get_offset_time_in_seq(pp_time_in_seq, pp_time_out_seq):
    #        offset_time_in_seq, offset_event_in_seq = list(), list()
    #        last_input_ts_list = [seq[-1] for seq in pp_time_in_seq]
    #        for i, l_ts in enumerate(last_input_ts_list):
    #            print('In offset', i)
    #            closest_ts = take_closest(all_timestamps, max(l_ts-offset_sec, all_timestamps[0]))
    #            closest_ts_ind = all_timestamps_dict[closest_ts]
    #            if closest_ts_ind<encoder_length:
    #                end_ind = encoder_length
    #            else:
    #                end_ind = closest_ts_ind + 1
    #            offset_time_in_seq.append(all_timestamps[end_ind-encoder_length:end_ind])
    #            offset_event_in_seq.append(all_events[end_ind-encoder_length:end_ind])

    #        return offset_time_in_seq, offset_event_in_seq

    #    pp_train_time_in_seq, pp_train_event_in_seq = get_offset_time_in_seq(pp_train_time_in_seq, pp_train_time_out_seq)
    #    pp_dev_time_in_seq, pp_dev_event_in_seq = get_offset_time_in_seq(pp_dev_time_in_seq, pp_dev_time_out_seq)
    #    pp_test_time_in_seq, pp_test_event_in_seq = get_offset_time_in_seq(pp_test_time_in_seq, pp_test_time_out_seq)

    #    assert len(pp_train_time_out_seq) == len(pp_train_time_in_seq)
    #    assert len(pp_dev_time_out_seq) == len(pp_dev_time_in_seq)
    #    assert len(pp_test_time_out_seq) == len(pp_test_time_in_seq)

    # ----- End: Create Input-Output sequences using offset ----- #

    assert len(test_tsIndices) == len(pp_test_time_in_seq)


    def write_to_file(fptr, sequences):
        for seq in sequences:
            for e in seq:
                f.write(str(e) + ' ')
            f.write('\n')

    def write_ts_to_file(fptr, sequence):
        for s in sequence:
            f.write(str(s) + '\n')

    dataset_name = dataset_name if dataset_name[-1]!='/' else dataset_name[:-1]
    dataset_name = dataset_name + '_chop' if num_coarse_seq>0 else dataset_name
    dataset_name = dataset_name + '_off'+str(offset) if offset>0.0 else dataset_name
    dataset_name = dataset_name + '_maxoff'+str(max_offset) if max_offset>0.0 else dataset_name
    dataset_name = dataset_name + '_sampleoff'+str(num_sample_offsets) if num_sample_offsets>0 else dataset_name
    dataset_name = dataset_name + '_' + str(encoder_length) + '_' + str(decoder_length) \
                   + '_' + str(train_step_length) + '_' + str(dev_step_length) + '_' + str(test_step_length) \

    dataset_name = dataset_name + '_' + str(keep_classes)


    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)
    with open(os.path.join(dataset_name, 'train.event.in'), 'w') as f:
        write_to_file(f, pp_train_event_in_seq)
    with open(os.path.join(dataset_name, 'train.event.out'), 'w') as f:
        write_to_file(f, pp_train_event_out_seq)
    with open(os.path.join(dataset_name, 'dev.event.in'), 'w') as f:
        write_to_file(f, pp_dev_event_in_seq)
    with open(os.path.join(dataset_name, 'dev.event.out'), 'w') as f:
        write_to_file(f, pp_dev_event_out_seq)
    with open(os.path.join(dataset_name, 'test.event.in'), 'w') as f:
        write_to_file(f, pp_test_event_in_seq)
    with open(os.path.join(dataset_name, 'test.event.out'), 'w') as f:
        write_to_file(f, pp_test_event_out_seq)
    with open(os.path.join(dataset_name, 'train.time.in'), 'w') as f:
        write_to_file(f, pp_train_time_in_seq)
    with open(os.path.join(dataset_name, 'train.time.out'), 'w') as f:
        write_to_file(f, pp_train_time_out_seq)
    with open(os.path.join(dataset_name, 'dev.time.in'), 'w') as f:
        write_to_file(f, pp_dev_time_in_seq)
    with open(os.path.join(dataset_name, 'dev.time.out'), 'w') as f:
        write_to_file(f, pp_dev_time_out_seq)
    with open(os.path.join(dataset_name, 'test.time.in'), 'w') as f:
        write_to_file(f, pp_test_time_in_seq)
    with open(os.path.join(dataset_name, 'test.time.out'), 'w') as f:
        write_to_file(f, pp_test_time_out_seq)
    with open(os.path.join(dataset_name, 'train.time.indices'), 'w') as f:
        write_ts_to_file(f, train_tsIndices)
    with open(os.path.join(dataset_name, 'dev.time.indices'), 'w') as f:
        write_ts_to_file(f, dev_tsIndices)
    with open(os.path.join(dataset_name, 'test.time.indices'), 'w') as f:
        write_ts_to_file(f, test_tsIndices)
    with open(os.path.join(dataset_name, 'train.offsets'), 'w') as f:
        write_ts_to_file(f, train_sampled_offsets)
    with open(os.path.join(dataset_name, 'dev.offsets'), 'w') as f:
        write_ts_to_file(f, dev_sampled_offsets)
    with open(os.path.join(dataset_name, 'test.offsets'), 'w') as f:
        write_ts_to_file(f, test_sampled_offsets)

    with open(os.path.join(dataset_name, 'labels.in'), 'w') as f:
        for lbl in unique_labels:
            f.write(str(lbl) + '\n')

    # ----- Start: Create sequence of previous-day timestamps for attention ----- #

    def get_attn_time_in_seq(pp_time_in_seq):
        attn_time_in_seq = list()
        last_input_ts_list = [seq[-1] for seq in pp_time_in_seq]
        attn_len = encoder_length
        for i, l_ts in enumerate(last_input_ts_list):
            search_ts = l_ts - ONE_DAY_SECS
            match_idx = bisect_right(all_timestamps, search_ts)

            end_ts = getBeginOfDayTs(l_ts)
            begin_ts = end_ts - ONE_DAY_SECS

            begin_idx = bisect_right(all_timestamps, begin_ts)
            end_idx = bisect_right(all_timestamps, end_ts)

            if begin_idx==0 and end_idx==0:
                end_idx = encoder_length
            print(begin_idx, end_idx)

            assert begin_idx>=0 and end_idx>=0

            attn_time_in_seq.append(all_timestamps[begin_idx:end_idx])

        return attn_time_in_seq

    #attn_train_time_in_seq = get_attn_time_in_seq(pp_train_time_in_seq)
    #attn_dev_time_in_seq = get_attn_time_in_seq(pp_dev_time_in_seq)
    #attn_test_time_in_seq = get_attn_time_in_seq(pp_test_time_in_seq)

    if num_coarse_seq>0:
        attn_train_time_in_seq = [all_timestamps for _ in train_time_seq]
        attn_dev_time_in_seq = [all_timestamps for _ in dev_time_seq]
        attn_test_time_in_seq = [all_timestamps for _ in test_time_seq]
    else:
        attn_train_time_in_seq = train_time_seq
        attn_dev_time_in_seq = train_time_seq
        attn_test_time_in_seq = train_time_seq

    #assert len(attn_train_time_in_seq) == len(pp_train_time_in_seq)
    #assert len(attn_dev_time_in_seq) == len(pp_dev_time_in_seq)
    #assert len(attn_test_time_in_seq) == len(pp_test_time_in_seq)

    with open(os.path.join(dataset_name, 'attn.train.time.in'), 'w') as f:
        write_to_file(f, attn_train_time_in_seq)
    with open(os.path.join(dataset_name, 'attn.dev.time.in'), 'w') as f:
        write_to_file(f, attn_dev_time_in_seq)
    with open(os.path.join(dataset_name, 'attn.test.time.in'), 'w') as f:
        write_to_file(f, attn_test_time_in_seq)

    # ----- End: Create sequence of previous-day timestamps for attention ----- #

    #----------- Gap Statistics Start-------------#
    train_gap_seq, dev_gap_seq, test_gap_seq = list(), list(), list()
    for sequence in train_time_seq:
        gap_sequence = [x-y for x,y in zip(sequence[1:], sequence[:-1])]
        train_gap_seq.append(gap_sequence)
        #print(Counter(sequence))
        #print('------------------------')
    for sequence in dev_time_seq:
        gap_sequence = [x-y for x,y in zip(sequence[1:], sequence[:-1])]
        dev_gap_seq.append(gap_sequence)
    for sequence in test_time_seq:
        gap_sequence = [x-y for x,y in zip(sequence[1:], sequence[:-1])]
        test_gap_seq.append(gap_sequence)

    with open(os.path.join(dataset_name, 'train_gaps'), 'w') as f:
        for i in chain(*train_gap_seq):
            f.write('%s\n' % i)
    with open(os.path.join(dataset_name, 'train_gaps_sorted'), 'w') as f:
        for i in sorted(chain(*train_gap_seq)):
            f.write('%s\n' % i)
    all_train_gaps = [i for i in chain(*train_gap_seq)]
    all_train_gaps_log = [np.log(i) for i in all_train_gaps if np.isfinite(np.log(i))]
    plt.hist(all_train_gaps_log, bins=50)
    plt.title(raw_dataset_name+'__train')
    plt.savefig(os.path.join(dataset_name, 'train_gaps_histogram.png'))
    plt.close()
    plt.hist(all_train_gaps, bins=50)
    plt.title(raw_dataset_name+'__train')
    plt.savefig(os.path.join(dataset_name, 'train_gaps_without_log_histogram.png'))
    plt.close()
    plt.boxplot(all_train_gaps, vert=False)#, flierprops={'markerfacecolor':'g', 'marker':'D'}, showfliers=False)
    plt.title(raw_dataset_name+'__train')
    plt.savefig(os.path.join(dataset_name, 'train_gaps_boxplot.png'))
    plt.close()
    train_gaps_freq = Counter(all_train_gaps)
    #print(train_gaps_freq)
    sorted_train_gaps_freq = sorted(train_gaps_freq.items(), key=itemgetter(0))
    with open(os.path.join(dataset_name, 'train_gaps_freq'), 'w') as f:
        for value, freq in sorted_train_gaps_freq:
            f.write('{}, {}\n'.format(value, freq))
    for value, freq in train_gaps_freq.items():
        train_gaps_freq[value] = freq*1.0/len(all_train_gaps)
    plt.loglog(list(train_gaps_freq.keys()), list(train_gaps_freq.values()), 'ro')
    plt.title(raw_dataset_name+'__train')
    plt.savefig(os.path.join(dataset_name, 'train_gaps_freq.png'))
    plt.close()
    median = np.percentile(all_train_gaps, 50)
    Q1 = np.percentile(all_train_gaps, 25)
    Q3 = np.percentile(all_train_gaps, 75)
    IQR = Q3 - Q1
    print('Median:\t{}'.format(median))
    print('Q1:\t{}'.format(Q1))
    print('Q3:\t{}'.format(Q3))
    print('IQR:\t{}'.format(IQR))
    print('Maximum:\t{}'.format(Q3 + 1.5*IQR))
    print('Minimum:\t{}'.format(Q1 -1.5*IQR))


    with open(os.path.join(dataset_name, 'dev_gaps'), 'w') as f:
        for i in chain(*dev_gap_seq):
            f.write('%s\n' % i)
    with open(os.path.join(dataset_name, 'dev_gaps_sorted'), 'w') as f:
        for i in sorted(chain(*dev_gap_seq)):
            f.write('%s\n' % i)
    all_dev_gaps = [i for i in chain(*dev_gap_seq)]
    all_dev_gaps_log = [np.log(i) for i in all_dev_gaps if np.isfinite(np.log(i))]
    plt.hist(all_dev_gaps_log, bins=50)
    plt.title(raw_dataset_name+'__dev')
    plt.savefig(os.path.join(dataset_name, 'dev_gaps_histogram.png'))
    plt.close()
    plt.boxplot(all_dev_gaps, vert=False)#, flierprops={'markerfacecolor':'g', 'marker':'D'}, showfliers=False)
    plt.title(raw_dataset_name+'__dev')
    plt.savefig(os.path.join(dataset_name, 'dev_gaps_boxplot.png'))
    plt.close()
    dev_gaps_freq = Counter(all_dev_gaps)
    for value, freq in dev_gaps_freq.items():
        dev_gaps_freq[value] = freq*1.0/len(all_dev_gaps)
    plt.loglog(list(dev_gaps_freq.keys()), list(dev_gaps_freq.values()), 'ro')
    plt.title(raw_dataset_name+'__dev')
    plt.savefig(os.path.join(dataset_name, 'dev_gaps_freq.png'))
    plt.close()

    with open(os.path.join(dataset_name, 'test_gaps'), 'w') as f:
        for i in chain(*test_gap_seq):
            f.write('%s\n' % i)
    with open(os.path.join(dataset_name, 'test_gaps_sorted'), 'w') as f:
        for i in sorted(chain(*test_gap_seq)):
            f.write('%s\n' % i)
    all_test_gaps = [i for i in chain(*test_gap_seq)]
    all_test_gaps_log = [np.log(i) for i in all_test_gaps if np.isfinite(np.log(i))]
    plt.hist(all_test_gaps_log, bins=50)
    plt.title(raw_dataset_name+'__test')
    plt.savefig(os.path.join(dataset_name, 'test_gaps_histogram.png'))
    plt.close()
    plt.boxplot(all_test_gaps, vert=False)#, flierprops={'markerfacecolor':'g', 'marker':'D'}, showfliers=False)
    plt.title(raw_dataset_name+'__test')
    plt.savefig(os.path.join(dataset_name, 'test_gaps_boxplot.png'))
    plt.close()
    test_gaps_freq = Counter(all_test_gaps)
    for value, freq in test_gaps_freq.items():
        test_gaps_freq[value] = freq*1.0/len(all_test_gaps)
    plt.loglog(list(test_gaps_freq.keys()), list(test_gaps_freq.values()), 'ro')
    plt.title(raw_dataset_name+'__test')
    plt.savefig(os.path.join(dataset_name, 'test_gaps_freq.png'))
    plt.close()

    #----------- Gap Statistics End-------------#

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset Name")
    parser.add_argument("dataset_path", type=str,
                        help="Path of raw dataset file")
    parser.add_argument("encoder_length", type=int, help="Encoder Length")
    parser.add_argument("decoder_length", type=int, help="Decoder Length")
    parser.add_argument("train_step_length", type=int,
                        help="Step length for train-sequence window")
    parser.add_argument("dev_step_length", type=int,
                        help="Step length for dev-sequence window")
    parser.add_argument("test_step_length", type=int,
                        help="Step length for test-sequence window")
    parser.add_argument("--keep_classes", type=int, default=0,
                        help="Number of top-K classes to retain \
                              from the data, retains all classes \
                              when set to zero")
    parser.add_argument("--num_coarse_seq", type=int, default=0,
                        help="Chop a single-large-seqeunce data into C \
                              sequences of approx equal size, no chopping \
                              when set to zero")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Output Sequence offset by 'offset'-hours")
    parser.add_argument("--max_offset", type=float, default=0.0,
                        help="Output Sequence contains all future timestamps \
                              between [0, max_offset] hours")
    parser.add_argument("--num_sample_offsets", type=int, default=0,
                        help="Sample offsets during parsing itself. If >0, \
                              train/dev/test offsets are sampled and stored \
                              in 'offsets' file.")
    args = parser.parse_args()

    dataset = args.dataset
    dataset_path = args.dataset_path
    encoder_length = args.encoder_length
    decoder_length = args.decoder_length
    train_step_length = args.train_step_length
    dev_step_length = args.dev_step_length
    test_step_length = args.test_step_length
    keep_classes = args.keep_classes
    num_coarse_seq = args.num_coarse_seq
    offset = args.offset
    max_offset = args.max_offset
    num_sample_offsets = args.num_sample_offsets
    sequence_length = encoder_length + decoder_length
    output_path = 'NewDataParsed'

    if dataset in ['barca', 'Delhi', 'jaya', 'Fight',
                   'Movie', 'Verdict',
                   'Trump', 'synthpastattntimefeats']:
        time_seqs, event_seqs, all_timestamps, all_events \
                = parse_single_seq_datasets(dataset_path,
                                            keep_classes,
                                            num_coarse_seq)
    elif dataset in ['data_bookorder', 'data_so', 'data_retweet']:
        time_seqs, event_seqs = parse_neural_hawkes_datasets(dataset_path, keep_classes)
    elif dataset in ['Taxi']:
        time_seqs, event_seqs, all_timestamps, all_events = parse_taxi_dataset(dataset_path, keep_classes, sequence_length)
    else:
        print('Dataset Parser Not Found')
        assert False

    tr_per = 0.7
    de_per = 0.1
    te_per = 0.2

    train_time_seq = []
    dev_time_seq = []
    test_time_seq = []
    train_event_seq = []
    dev_event_seq = []
    test_event_seq = []

    for time_seq, event_seq in zip(time_seqs, event_seqs):
        total_ = len(time_seq)
        train_time_seq.append(time_seq[:int(tr_per*total_)])
        train_event_seq.append(event_seq[:int(tr_per*total_)])

        dev_time_seq.append(time_seq[int(tr_per*total_):int((tr_per+de_per)*total_)])
        dev_event_seq.append(event_seq[int(tr_per*total_):int((tr_per+de_per)*total_)])

        test_time_seq.append(time_seq[int((tr_per+de_per)*total_):])
        test_event_seq.append(event_seq[int((tr_per+de_per)*total_):])

    print(np.shape(train_time_seq))
    print(np.shape(test_time_seq))
    print(np.shape(dev_time_seq))
    print(np.shape(train_event_seq))
    print(np.shape(test_event_seq))
    print(np.shape(dev_event_seq))

    #all_timestamps = time_seqs
    #all_events = time_seqs
    #all_timestamps = [s for seq in train_time_seq for s in seq]
    #all_events = [s for seq in train_event_seq for s in seq]

    #plotter(train_time_seq, encoder_length, decoder_length, dataset)

    preprocess(dataset,
               os.path.join(output_path, dataset),
               all_timestamps,
               all_events,
               train_event_seq, train_time_seq,
               dev_event_seq, dev_time_seq,
               test_event_seq, test_time_seq,
               encoder_length, decoder_length,
               train_step_length, dev_step_length, test_step_length,
               keep_classes, num_coarse_seq, offset,
               max_offset, num_sample_offsets)

if __name__ == '__main__':
    main()
