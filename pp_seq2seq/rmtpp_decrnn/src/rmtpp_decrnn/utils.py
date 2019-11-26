from tensorflow.contrib.keras import preprocessing
from collections import defaultdict, Counter
import itertools
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from itertools import chain
from dtw import dtw
from datetime import datetime
import time

def getDatetime(epoch):
    date_string=time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime(epoch))
    date= datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
    return date

#def getHour(epoch):
#    return getDatetime(epoch).hour

pad_sequences = preprocessing.sequence.pad_sequences
DELTA = 3600.0


def create_dir(dirname):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def change_zero_label(sequences):
    return [[lbl+1 for lbl in sequence] for sequence in sequences]

def generate_norm_seq(timeTrain, timeDev, timeTest, enc_len, normalization, check=0):

    def _generate_norm_seq(sequences, enc_len, normalization, check=0):
        sequences = np.array(sequences)
        gap_seqs = sequences[:, 1:]-sequences[:, :-1]
        #initial_gaps = np.mean(gap_seqs[:, :enc_len-1], axis=1, keepdims=True)
        initial_gaps = np.zeros_like(np.mean(gap_seqs[:, :enc_len-1], axis=1, keepdims=True))
        gap_seqs = np.hstack((initial_gaps, gap_seqs))
        N = len(gap_seqs)
        if normalization == 'minmax':
            max_gaps = np.clip(np.max(gap_seqs[:, :enc_len-1], keepdims=True), 1.0, np.inf)
            min_gaps = np.clip(np.min(gap_seqs[:, :enc_len-1], keepdims=True), 1.0, np.inf)
            normalizer_d = np.ones((N, 1)) * (max_gaps - min_gaps)
            normalizer_a = np.ones((N, 1)) * (-mjn_gaps/(max_gaps - min_gaps))
        elif normalization == 'average':
            avg_gaps = np.clip(np.mean(gap_seqs[:, :enc_len-1], keepdims=True), 1.0, np.inf)
            normalizer_d = np.ones((N, 1)) * avg_gaps
            normalizer_a = np.zeros((N, 1))
        elif normalization == 'average_per_seq':
            avg_gaps = np.clip(np.mean(gap_seqs[:, :enc_len-1], axis=1, keepdims=True), 1.0, np.inf)
            normalizer_d = avg_gaps
            normalizer_a = np.zeros((N, 1))
        elif normalization is None:
            normalizer_d = np.ones((N, 1))
            normalizer_a = np.zeros((N, 1))
        else:
            print('Normalization not found')
            assert False

        avg_gaps_norm = gap_seqs/normalizer_d + normalizer_a
        avg_gaps_norm = np.cumsum(avg_gaps_norm, axis=1)
        #gap_norm_seq = np.hstack((np.zeros((N, 1)), avg_gaps_norm))

        if check==1:
            print("sequences[:,:1]")
            print(sequences[:,:1])

        return avg_gaps_norm, normalizer_d, normalizer_a, initial_gaps

    timeTrain, trainND, trainNA, trainIG = _generate_norm_seq(timeTrain, enc_len, normalization)
    timeDev, devND, devNA, devIG = _generate_norm_seq(timeDev, enc_len, normalization)
    timeTest, testND, testNA, testIG = _generate_norm_seq(timeTest, enc_len, normalization)

    return timeTrain, trainND, trainNA, trainIG, \
            timeDev, devND, devNA, devIG, \
            timeTest, testND, testNA, testIG

def read_seq2seq_data(dataset_path, alg_name, normalization=None, pad=True):
    """Read data from given files and return it as a dictionary."""

    with open(dataset_path+'train.event.in', 'r') as in_file:
        eventTrainIn = [[int(y) for y in x.strip().split()] for x in in_file]
    with open(dataset_path+'train.event.out', 'r') as in_file:
        eventTrainOut = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(dataset_path+'dev.event.in', 'r') as in_file:
        eventDevIn = [[int(y) for y in x.strip().split()] for x in in_file]
    with open(dataset_path+'dev.event.out', 'r') as in_file:
        eventDevOut = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(dataset_path+'test.event.in', 'r') as in_file:
        eventTestIn = [[int(y) for y in x.strip().split()] for x in in_file]
    with open(dataset_path+'test.event.out', 'r') as in_file:
        eventTestOut = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(dataset_path+'train.time.in', 'r') as in_file:
        timeTrainIn = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(dataset_path+'train.time.out', 'r') as in_file:
        timeTrainOut = [[float(y) for y in x.strip().split()] for x in in_file]

    with open(dataset_path+'dev.time.in', 'r') as in_file:
        timeDevIn = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(dataset_path+'dev.time.out', 'r') as in_file:
        timeDevOut = [[float(y) for y in x.strip().split()] for x in in_file]

    with open(dataset_path+'test.time.in', 'r') as in_file:
        timeTestIn = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(dataset_path+'test.time.out', 'r') as in_file:
        timeTestOut = [[float(y) for y in x.strip().split()] for x in in_file]

    with open(dataset_path+'attn.train.time.in', 'r') as in_file:
        attn_timeTrainIn = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(dataset_path+'attn.dev.time.in', 'r') as in_file:
        attn_timeDevIn = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(dataset_path+'attn.test.time.in', 'r') as in_file:
        attn_timeTestIn = [[float(y) for y in x.strip().split()] for x in in_file]

    # Compute Hour-of-day features from data
    getHour = lambda t: t // 3600 % 24
    timeTrainInFeats = [[getHour(s) for s in seq] for seq in timeTrainIn]
    timeDevInFeats = [[getHour(s) for s in seq] for seq in timeDevIn]
    timeTestInFeats = [[getHour(s) for s in seq] for seq in timeTestIn]
    timeTrainOutFeats = [[getHour(s) for s in seq] for seq in timeTrainOut]
    timeDevOutFeats = [[getHour(s) for s in seq] for seq in timeDevOut]
    timeTestOutFeats = [[getHour(s) for s in seq] for seq in timeTestOut]

    assert len(timeTrainIn) == len(eventTrainIn)
    assert len(timeDevIn) == len(eventDevIn)
    assert len(timeTestIn) == len(eventTestIn)
    assert len(timeTrainOut) == len(eventTrainOut)
    assert len(timeDevOut) == len(eventDevOut)
    assert len(timeTestOut) == len(eventTestOut)
    # 0 label is not allowed, if present increment all label values
    for sequence in itertools.chain(eventTrainIn, eventTrainOut, eventDevIn, eventDevOut, eventTestIn, eventTestOut):
        if 0 in sequence:
            eventTrainIn = change_zero_label(eventTrainIn)
            eventTrainOut = change_zero_label(eventTrainOut)
            eventDevIn = change_zero_label(eventDevIn)
            eventDevOut = change_zero_label(eventDevOut)
            eventTestIn = change_zero_label(eventTestIn)
            eventTestOut = change_zero_label(eventTestOut)
            break

    # Combine eventTrainIn and eventTrainOut into one eventTrain
    eventTrain = [in_seq + out_seq for in_seq, out_seq in zip(eventTrainIn, eventTrainOut)]
    # Similarly for time ...
    timeTrain = [in_seq + out_seq for in_seq, out_seq in zip(timeTrainIn, timeTrainOut)]
    timeDev = [in_seq + out_seq for in_seq, out_seq in zip(timeDevIn, timeDevOut)]
    timeTest = [in_seq + out_seq for in_seq, out_seq in zip(timeTestIn, timeTestOut)]
    timeTrainFeats = [in_seq + out_seq for in_seq, out_seq in zip(timeTrainInFeats, timeTrainOutFeats)]
    timeDevFeats = [in_seq + out_seq for in_seq, out_seq in zip(timeDevInFeats, timeDevOutFeats)]
    timeTestFeats = [in_seq + out_seq for in_seq, out_seq in zip(timeTestInFeats, timeTestOutFeats)]

    # nb_samples = len(eventTrain)
    # max_seqlen = max(len(x) for x in eventTrain)
    unique_samples = set()

    for x in eventTrainIn + eventTrainOut + eventDevIn + eventTestIn:
        unique_samples = unique_samples.union(x)

    enc_len = len(timeTrainIn[0])
    devActualTimeIn = np.array(timeDevIn)[:, enc_len-1:enc_len].tolist()
    testActualTimeIn = np.array(timeTestIn)[:, enc_len-1:enc_len].tolist()
    devActualTimeOut, testActualTimeOut = timeDevOut, timeTestOut

    # ----- Normalization by gaps ----- #
    timeTrain, trainND, trainNA, trainIG, \
            timeDev, devND, devNA, devIG, \
            timeTest, testND, testNA, testIG \
            = generate_norm_seq(timeTrain, timeDev, timeTest, enc_len, normalization)

    timeTrainIn, timeTrainOut = timeTrain[:, :enc_len], timeTrain[:, enc_len:]
    timeDevIn, timeDevOut = timeDev[:, :enc_len], timeDev[:, enc_len:]
    timeTestIn, timeTestOut = timeTest[:, :enc_len], timeTest[:, enc_len:]

#    timeTrainIn = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTrainIn]
#    timeTrainOut = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTrainOut]

    timeTrainFeats = np.array(timeTrainFeats)
    timeDevFeats = np.array(timeDevFeats)
    timeTestFeats = np.array(timeTestFeats)
    timeTrainInFeats, timeTrainOutFeats = timeTrainFeats[:, :enc_len], timeTrainFeats[:, enc_len:]
    timeDevInFeats, timeDevOutFeats = timeDevFeats[:, :enc_len], timeDevFeats[:, enc_len:]
    timeTestInFeats, timeTestOutFeats = timeTestFeats[:, :enc_len], timeTestFeats[:, enc_len:]

    if pad:
        train_event_in_seq = pad_sequences(eventTrainIn, padding='post')
        train_event_out_seq = pad_sequences(eventTrainOut, padding='post')
        train_time_in_seq = pad_sequences(timeTrainIn, dtype=float, padding='post')
        train_time_out_seq = pad_sequences(timeTrainOut, dtype=float, padding='post')
        train_time_in_feats = pad_sequences(timeTrainInFeats, dtype=float, padding='post')
        train_time_out_feats = pad_sequences(timeTrainOutFeats, dtype=float, padding='post')
    else:
        train_event_in_seq = eventTrainIn
        train_event_out_seq = eventTrainOut
        train_time_in_seq = timeTrainIn
        train_time_out_seq = timeTrainOut
        train_time_in_feats = timeTrainInFeats
        train_time_out_feats = timeTrainOutFeats

#    timeDevIn = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeDevIn]
#    timeDevOut = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeDevOut]

    if pad:
        dev_event_in_seq = pad_sequences(eventDevIn, padding='post')
        dev_event_out_seq = pad_sequences(eventDevOut, padding='post')
        dev_time_in_seq = pad_sequences(timeDevIn, dtype=float, padding='post')
        dev_time_out_seq = pad_sequences(timeDevOut, dtype=float, padding='post')
        dev_time_in_feats = pad_sequences(timeDevInFeats, dtype=float, padding='post')
        dev_time_out_feats = pad_sequences(timeDevOutFeats, dtype=float, padding='post')
    else:
        dev_event_in_seq = eventDevIn
        dev_event_out_seq = eventDevOut
        dev_time_in_seq = timeDevIn
        dev_time_out_seq = timeDevOut
        dev_time_in_feats = timeDevInFeats
        dev_time_out_feats = timeDevOutFeats

#    timeTestIn = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTestIn]
#    timeTestOut = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTestOut]

    if pad:
        test_event_in_seq = pad_sequences(eventTestIn, padding='post')
        test_event_out_seq = pad_sequences(eventTestOut, padding='post')
        test_time_in_seq = pad_sequences(timeTestIn, dtype=float, padding='post')
        test_time_out_seq = pad_sequences(timeTestOut, dtype=float, padding='post')
        test_time_in_feats = pad_sequences(timeTestInFeats, dtype=float, padding='post')
        test_time_out_feats = pad_sequences(timeTestOutFeats, dtype=float, padding='post')
    else:
        test_event_in_seq = eventTestIn
        test_event_out_seq = eventTestOut
        test_time_in_seq = timeTestIn
        test_time_out_seq = timeTestOut
        test_time_in_feats = timeTestInFeats
        test_time_out_feats = timeTestOutFeats

    # ----- Start: Create coarse sequence by grouping nearby events  ----- #

    def create_coarse_seq(attn_time_seq):
        coarse_gaps_in_seq = list()
        coarse_time_in_feats = list()
        attn_gaps_idxes = list()
        attn_gaps = list()
        attn_time_in_feats = list()
        for sequence in attn_time_seq:
            gaps = list()
            coarse_seq = list()
            begin_idxes = list()
            coarse_feats = list()
            attn_feats = list()
            cnt, total_gap = 0, 0.0
            begin_ts = sequence[0]
            begin_idxes.append(0)
            hod = list()
            for i, ts in enumerate(sequence[:-enc_len]):
                gaps.append((sequence[i] - sequence[i-1]) if i>0 else 0.0)
                attn_feats.append(getHour(ts))
                #if ts > begin_ts + DELTA or i==len(sequence)-1:
                if ts > begin_ts + DELTA or i==len(sequence)-1-enc_len:
                    #print(ts > begin_ts + DELTA, i==len(sequence)-1)
                    #print(begin_ts, ts, cnt, total_gap, hod)
                    if total_gap > 0.0:
                        begin_ts = ts
                        coarse_feats.append(max(hod))
                        avg_gap = total_gap * 1.0/cnt
                        #coarse_seq.append([cnt, avg_gap])
                        coarse_seq.append(avg_gap)
                        cnt, total_gap = 1, 0.0
                        hod = list()
                        #if i<len(sequence)-1-enc_len:
                        begin_idxes.append(i)
                    elif total_gap == 0.0:
                        if i<len(sequence)-1-enc_len:
                            begin_idxes[-1] = i
                            begin_ts = ts
                            hod = list()
                            cnt, total_gap = 1, 0.0
                        #else:
                        #    del begin_idxes[-1]

                else:
                    cnt +=1
                    total_gap += ((ts - sequence[i-1]) if i>0 else 0.0)
                    hod.append(getHour(ts))

            for i, ts in enumerate(sequence[-enc_len:], start=len(sequence)-enc_len):
                gaps.append((sequence[i] - sequence[i-1]) if i>0 else 0.0)
                attn_feats.append(getHour(ts))
                coarse_seq.append((sequence[i] - sequence[i-1]) if i>0 else 0.0)
                coarse_feats.append(getHour(ts))
                if i<len(sequence)-1:
                    begin_idxes.append(i)


            #print(len(begin_idxes), len(coarse_seq))
            #print(begin_idxes[-enc_len], len(gaps), begin_idxes[-enc_len] + 5 < len(gaps))
            assert len(begin_idxes) == len(coarse_seq)
            attn_gaps.append(gaps)
            attn_gaps_idxes.append(begin_idxes)
            coarse_gaps_in_seq.append(coarse_seq)
            coarse_time_in_feats.append(coarse_feats)
            attn_time_in_feats.append(attn_feats)


        #lens = [len(sequence) for sequence in coarse_gaps_in_seq]
        #print('---------------------------------')
        #lens_counts = Counter(lens)
        #for k in sorted(lens_counts.keys()):
        #    print(k, lens_counts[k]
        return attn_gaps, attn_gaps_idxes, coarse_gaps_in_seq, coarse_time_in_feats, attn_time_in_feats

    attn_train_gaps, attn_train_gaps_idxes, coarse_train_gaps_in_seq, coarse_train_time_in_feats, attn_train_time_in_feats = create_coarse_seq(attn_timeTrainIn)
    attn_dev_gaps, attn_dev_gaps_idxes, coarse_dev_gaps_in_seq, coarse_dev_time_in_feats, attn_dev_time_in_feats = create_coarse_seq(attn_timeDevIn)
    attn_test_gaps, attn_test_gaps_idxes, coarse_test_gaps_in_seq, coarse_test_time_in_feats, attn_test_time_in_feats = create_coarse_seq(attn_timeTestIn)

    ## Testing whether encoder length is subsumed in the attn sequence
    #for i, (attn_feats, gaps) in enumerate(zip(attn_train_time_in_feats, attn_train_gaps)):
    #    print(i)
    #    k=len(gaps)-1
    #    for j in range(len(train_time_in_seq[i])-1, 0, -1):
    #        print(i, j, k, gaps[k], train_time_in_seq[i][j]-train_time_in_seq[i][j-1])
    #        assert gaps[k] == train_time_in_seq[i][j]-train_time_in_seq[i][j-1]
    #        k-=1


    if alg_name in ['rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r',
                    'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                    'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate',]:
        def get_past_attn_feats(attn_time_seq, attn_time_in_feats, attn_gaps):
            attn_feats_past_day = list()
            attn_gaps_past_day = list()
            found_cnt = 0
            for sequence, attn_feats, gaps in zip(attn_time_seq, attn_time_in_feats, attn_gaps):
                last_dt = getDatetime(sequence[-1])
                last_hour = attn_feats[-1]
                assert last_dt.hour == last_hour
                last_day = last_dt.day
                # Get index of nearest hour in the day before dt.date
                ind_found = False
                for ind, ts in zip(range(len(sequence)-1, -1, -1), reversed(sequence)):
                    dt = getDatetime(ts)
                    if dt.day!= last_dt.day and dt.hour == last_dt.hour:
                        print(ind, 'ind found')
                        attn_idx = ind
                        ind_found = True
                        found_cnt += 1
                        break
                if not ind_found:
                    for ind, ts in zip(range(len(sequence)-1, -1, -1), reversed(sequence)):
                        dt = getDatetime(ts)
                        if dt.day!= last_dt.day and abs(dt.hour - last_dt.hour) <=2:
                            print(ind, 'ind found')
                            attn_idx = ind
                            ind_found = True
                            break
                if not ind_found:
                    print('NOT FOUND')
                    attn_idx = len(sequence) - 11

                if attn_idx-10<0:
                    begin_idx = 0
                    end_idx = 20
                elif attn_idx+10>=len(attn_feats):
                    begin_idx = len(attn_feats)-1-20
                    end_idx = len(attn_feats)-1
                else:
                    begin_idx = attn_idx-10
                    end_idx = attn_idx+10

                assert begin_idx>=0 and end_idx>=0

                attn_feats_past_day.append(attn_feats[begin_idx:end_idx])
                attn_gaps_past_day.append(gaps[begin_idx:end_idx])

            print('Fraction of Indices Found:', found_cnt *1.0/len(attn_time_seq))

            return attn_feats_past_day, attn_gaps_past_day

        attn_train_time_in_feats, attn_train_gaps = get_past_attn_feats(attn_timeTrainIn, attn_train_time_in_feats, attn_train_gaps)
        attn_dev_time_in_feats, attn_dev_gaps = get_past_attn_feats(attn_timeDevIn, attn_dev_time_in_feats, attn_dev_gaps)
        attn_test_time_in_feats, attn_test_gaps = get_past_attn_feats(attn_timeTestIn, attn_test_time_in_feats, attn_test_gaps)

    #for sequence in coarse_train_gaps_in_seq[:20]:
    #    for s in sequence:
    #        print(s, end=' ')
    #    print('\n', end='')

    # ----- End: Create coarse sequence by grouping nearby events  ----- #


    return {
        'train_event_in_seq': train_event_in_seq,
        'train_event_out_seq': train_event_out_seq,

        'train_time_in_seq': train_time_in_seq,
        'train_time_out_seq': train_time_out_seq,

        'dev_event_in_seq': dev_event_in_seq,
        'dev_event_out_seq': dev_event_out_seq,

        'dev_time_in_seq': dev_time_in_seq,
        'dev_time_out_seq': dev_time_out_seq,

        'test_event_in_seq': test_event_in_seq,
        'test_event_out_seq': test_event_out_seq,

        'test_time_in_seq': test_time_in_seq,
        'test_time_out_seq': test_time_out_seq,

        'test_actual_time_in_seq': testActualTimeIn,
        'test_actual_time_out_seq': testActualTimeOut,

        'dev_actual_time_in_seq': devActualTimeIn,
        'dev_actual_time_out_seq': devActualTimeOut,

        'train_time_in_feats': train_time_in_feats,
        'dev_time_in_feats': dev_time_in_feats,
        'test_time_in_feats': test_time_in_feats,
        'train_time_out_feats': train_time_out_feats,
        'dev_time_out_feats': dev_time_out_feats,
        'test_time_out_feats': test_time_out_feats,

        'attn_train_gaps': attn_train_gaps,
        'attn_train_gaps_idxes': attn_train_gaps_idxes,
        'coarse_train_gaps_in_seq': coarse_train_gaps_in_seq,
        'coarse_train_time_in_feats': coarse_train_time_in_feats,
        'attn_train_time_in_feats': attn_train_time_in_feats,
        'attn_dev_gaps': attn_dev_gaps,
        'attn_dev_gaps_idxes': attn_dev_gaps_idxes,
        'coarse_dev_gaps_in_seq': coarse_dev_gaps_in_seq,
        'coarse_dev_time_in_feats': coarse_dev_time_in_feats,
        'attn_dev_time_in_feats': attn_dev_time_in_feats,
        'attn_test_gaps': attn_test_gaps,
        'attn_test_gaps_idxes': attn_test_gaps_idxes,
        'coarse_test_gaps_in_seq': coarse_test_gaps_in_seq,
        'coarse_test_time_in_feats': coarse_test_time_in_feats,
        'attn_test_time_in_feats': attn_test_time_in_feats,


        'num_categories': len(unique_samples),
        'encoder_length': len(eventTrainIn[0]),
        'decoder_length': len(eventTrainOut[0]),

        'trainND': trainND,
        'trainNA': trainNA,
        'trainIG': trainIG,
        'devND': devND,
        'devNA': devNA,
        'devIG': devIG,
        'testND': testND,
        'testNA': testNA,
        'testIG': testIG,
    }


def read_data(event_train_file, event_test_file, time_train_file, time_test_file,
              pad=True):
    """Read data from given files and return it as a dictionary."""

    with open(event_train_file, 'r') as in_file:
        eventTrain = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(event_test_file, 'r') as in_file:
        eventTest = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(time_train_file, 'r') as in_file:
        timeTrain = [[float(y) for y in x.strip().split()] for x in in_file]

    with open(time_test_file, 'r') as in_file:
        timeTest = [[float(y) for y in x.strip().split()] for x in in_file]

    assert len(timeTrain) == len(eventTrain)
    assert len(eventTest) == len(timeTest)

    # nb_samples = len(eventTrain)
    # max_seqlen = max(len(x) for x in eventTrain)
    unique_samples = set()

    for x in eventTrain + eventTest:
        unique_samples = unique_samples.union(x)

    maxTime = max(itertools.chain((max(x) for x in timeTrain), (max(x) for x in timeTest)))
    minTime = min(itertools.chain((min(x) for x in timeTrain), (min(x) for x in timeTest)))
    # minTime, maxTime = 0, 1

    eventTrainIn = [x[:-1] for x in eventTrain]
    eventTrainOut = [x[1:] for x in eventTrain]
    timeTrainIn = [[(y - minTime) / (maxTime - minTime) for y in x[:-1]] for x in timeTrain]
    timeTrainOut = [[(y - minTime) / (maxTime - minTime) for y in x[1:]] for x in timeTrain]

    if pad:
        train_event_in_seq = pad_sequences(eventTrainIn, padding='post')
        train_event_out_seq = pad_sequences(eventTrainOut, padding='post')
        train_time_in_seq = pad_sequences(timeTrainIn, dtype=float, padding='post')
        train_time_out_seq = pad_sequences(timeTrainOut, dtype=float, padding='post')
    else:
        train_event_in_seq = eventTrainIn
        train_event_out_seq = eventTrainOut
        train_time_in_seq = timeTrainIn
        train_time_out_seq = timeTrainOut


    eventTestIn = [x[:-1] for x in eventTest]
    eventTestOut = [x[1:] for x in eventTest]
    timeTestIn = [[(y - minTime) / (maxTime - minTime) for y in x[:-1]] for x in timeTest]
    timeTestOut = [[(y - minTime) / (maxTime - minTime) for y in x[1:]] for x in timeTest]

    if pad:
        test_event_in_seq = pad_sequences(eventTestIn, padding='post')
        test_event_out_seq = pad_sequences(eventTestOut, padding='post')
        test_time_in_seq = pad_sequences(timeTestIn, dtype=float, padding='post')
        test_time_out_seq = pad_sequences(timeTestOut, dtype=float, padding='post')
    else:
        test_event_in_seq = eventTestIn
        test_event_out_seq = eventTestOut
        test_time_in_seq = timeTestIn
        test_time_out_seq = timeTestOut

    return {
        'train_event_in_seq': train_event_in_seq,
        'train_event_out_seq': train_event_out_seq,

        'train_time_in_seq': train_time_in_seq,
        'train_time_out_seq': train_time_out_seq,

        'test_event_in_seq': test_event_in_seq,
        'test_event_out_seq': test_event_out_seq,

        'test_time_in_seq': test_time_in_seq,
        'test_time_out_seq': test_time_out_seq,

        'num_categories': len(unique_samples)
    }


def calc_base_rate(data, training=True):
    """Calculates the base-rate for intelligent parameter initialization from the training data."""
    suffix = 'train' if training else 'test'
    in_key = suffix + '_time_in_seq'
    out_key = suffix + '_time_out_seq'
    valid_key = suffix + '_event_in_seq'

    dts = (data[out_key] - data[in_key])[data[valid_key] > 0]
    return 1.0 / np.mean(dts)


def calc_base_event_prob(data, training=True):
    """Calculates the base probability of event types for intelligent parameter initialization from the training data."""
    dict_key = 'train_event_in_seq' if training else 'test_event_in_seq'

    class_count = defaultdict(lambda: 0.0)
    for evts in data[dict_key]:
        for ev in evts:
            class_count[ev] += 1.0

    total_events = 0.0
    probs = []
    for cat in range(1, data['num_categories'] + 1):
        total_events += class_count[cat]

    for cat in range(1, data['num_categories'] + 1):
        probs.append(class_count[cat] / total_events)

    return np.array(probs)


def data_stats(data):
    """Prints basic statistics about the dataset."""
    train_valid = data['train_event_in_seq'] > 0
    test_valid = data['test_event_in_seq'] > 0

    print('Num categories = ', data['num_categories'])
    print('delta-t (training) = ')
    print(pd.Series((data['train_time_out_seq'] - data['train_time_in_seq'])[train_valid]).describe())
    train_base_rate = calc_base_rate(data, training=True)
    print('base-rate = {}, log(base_rate) = {}'.format(train_base_rate, np.log(train_base_rate)))
    print('Class probs = ', calc_base_event_prob(data, training=True))

    print('delta-t (testing) = ')
    print(pd.Series((data['test_time_out_seq'] - data['test_time_in_seq'])[test_valid]).describe())
    test_base_rate = calc_base_rate(data, training=False)
    print('base-rate = {}, log(base_rate) = {}'.format(test_base_rate, np.log(test_base_rate)))
    print('Class probs = ', calc_base_event_prob(data, training=False))

    print('Training sequence lenghts = ')
    print(pd.Series(train_valid.sum(axis=1)).describe())

    print('Testing sequence lenghts = ')
    print(pd.Series(test_valid.sum(axis=1)).describe())


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if name is None:
        name = var.name.split('/')[-1][:-2]

    with tf.name_scope('summaries-' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def MAE(time_preds, time_true, events_out):
    """Calculates the MAE between the provided and the given time, ignoring the inf
    and nans. Returns both the MAE and the number of items considered."""

    # Predictions may not cover the entire time dimension.
    # This clips time_true to the correct size.
    seq_limit = time_preds.shape[1]
    batch_limit = time_preds.shape[0]
    clipped_time_true = time_true[:batch_limit, :seq_limit]
    clipped_events_out = events_out[:batch_limit, :seq_limit]
    #print(clipped_time_true)
    #print(time_preds)

    is_finite = np.isfinite(time_preds) & (clipped_events_out > 0)
    #print(time_preds.shape, clipped_time_true.shape, is_finite.shape)

    return np.mean(np.abs(time_preds - clipped_time_true)[is_finite]), np.sum(is_finite)

def DTW(time_preds, time_true, events_out):

    seq_limit = time_preds.shape[1]
    clipped_time_true = time_true[:, :seq_limit]
    clipped_events_out = events_out[:, :seq_limit]

    euclidean_norm = lambda x, y: np.abs(x - y)
    distance = 0
    for time_preds_, clipped_time_true_ in zip(time_preds, clipped_time_true):
        d, cost_matrix, acc_cost_matrix, path = dtw(time_preds_, clipped_time_true_, dist=euclidean_norm)
        distance += d
    distance = distance / clipped_time_true.shape[0]

    return distance

def RMSE(time_preds, time_true, events_out):
    """Calculates the RMSE between the provided and the given time, ignoring the inf
    and nans. Returns both the MAE and the number of items considered."""

    # Predictions may not cover the entire time dimension.
    # This clips time_true to the correct size.
    seq_limit = time_preds.shape[1]
    clipped_time_true = time_true[:, :seq_limit]
    clipped_events_out = events_out[:, :seq_limit]

    is_finite = np.isfinite(time_preds) & (clipped_events_out > 0)

    return np.sqrt(np.mean(np.square(time_preds - clipped_time_true)[is_finite])), np.sum(is_finite)


def ACC(event_preds, event_true):
    """Returns the accuracy of the event prediction, provided the output probability vector."""
    clipped_event_true = event_true[:event_preds.shape[0], :event_preds.shape[1]]
    is_valid = clipped_event_true > 0

    # The indexes start from 0 whereare event_preds start from 1.
    # highest_prob_ev = event_preds.argmax(axis=-1) + 1
    highest_prob_ev = event_preds
    #print(clipped_event_true)
    #print(highest_prob_ev)
    #print((clipped_event_true==highest_prob_ev).shape)

    return np.sum((highest_prob_ev == clipped_event_true)[is_valid]) / np.sum(is_valid)


def MRR(event_preds_softmax, event_true):
    "Computes Mean Reciprocal Rank of events"

    num_unique_events = event_preds_softmax.shape[-1]
    event_true_flatten = np.reshape(event_true, [-1, 1])
    num_events = event_true_flatten.shape[0]
    event_preds_softmax_flatten = np.reshape(event_preds_softmax, [-1, num_unique_events])

    ranks = np.where(event_true_flatten-1 == (np.argsort(-event_preds_softmax_flatten)))[1] + 1
    #print((np.argsort(-event_preds_softmax_flatten)).shape)
    #for event_tr, softmax, softmax_ranked in zip(event_true_flatten, event_preds_softmax_flatten.tolist(), np.where(event_true_flatten-1 == (np.argsort(-event_preds_softmax_flatten)))[1]):
    #    print(event_tr, softmax, softmax_ranked)
    #print(ranks)

    return np.mean(1.0/ranks)

def PERCENT_ERROR(event_preds, event_true):
    return (1.0 - ACC(event_preds, event_true)) * 100
