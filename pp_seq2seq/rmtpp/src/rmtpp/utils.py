from tensorflow.contrib.keras import preprocessing
from collections import defaultdict
import itertools
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from itertools import chain
from dtw import dtw
from datetime import datetime
import time
from bisect import bisect_right

def getDatetime(epoch):
    date_string=time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime(epoch))
    date= datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
    return date

def timestampTotime(ts):
    #t = time.strftime("%Y %m %d %H %M %S", time.localtime(ts)).split(' ')
    t = time.strftime("%m %d %H %M %S", time.gmtime(ts)).split(' ')
    t = [int(i) for i in t]
    return t

def getHour(epoch):
    return timestampTotime(epoch)[2]

pad_sequences = preprocessing.sequence.pad_sequences
DELTA = 3600.0


def create_dir(dirname):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def change_zero_label(sequences):
    return [[lbl+1 for lbl in sequence] for sequence in sequences]

def generate_norm_seq(timeTrain, timeDev, timeTest, enc_len, normalization, max_offset, check=0):

    def _generate_norm_seq(sequences, enc_len, normalization, check=0):

        def _norm_seq(sequence):
            sequence = np.array(sequence)
            gap_seq = sequence[1:]-sequence[:-1]
            init_gap = np.zeros((1))
            gap_seq = np.hstack((init_gap, gap_seq))

            if normalization in ['minmax']:
                max_gap = np.clip(np.max(gap_seq[:enc_len-1]), 1.0, np.inf)
                min_gap = np.clip(np.min(gap_seq[:enc_len-1]), 1.0, np.inf)
                n_d, n_a = [(max_gap - min_gap)], [(-min_gap/(max_gap - min_gap))]
            elif normalization == 'average_per_seq':
                avg_gap = np.clip(np.mean(gap_seq[:enc_len-1]), 1.0, np.inf)
                n_d, n_a = [avg_gap], [0.0]
            elif normalization == 'max_per_seq':
                max_gap = np.clip(np.max(gap_seq[:enc_len-1]), 1.0, np.inf)
                n_d, n_a = [max_gap], [0.0]
            elif normalization is None or normalization in ['average', 'max_offset']:
                n_d, n_a = [1.0], [0.0]
            else:
                print('Normalization not found')
                assert False

            avg_gap_norm = gap_seq/n_d + n_a
            avg_gap_norm = np.cumsum(avg_gap_norm)

            return avg_gap_norm, n_d, n_a, init_gap

        avg_gaps_norm, normalizer_d, normalizer_a, initial_gaps = list(), list(), list(), list()
        for sequence in sequences:
            avg_gap_norm, n_d, n_a, init_gap = _norm_seq(sequence)
            avg_gaps_norm.append(avg_gap_norm)
            normalizer_d.append(n_d)
            normalizer_a.append(n_a)
            initial_gaps.append(init_gap)

        if normalization == 'average':
            n_d = 0.0
            for gap_seq in avg_gaps_norm:
                n_d += np.sum(gap_seq[1:]-gap_seq[:-1])
            n_d = n_d / (len(sequences) * (enc_len-1))
            avg_gaps_norm = [gap_seq/n_d for gap_seq in avg_gaps_norm]
            normalizer_d = np.ones((len(sequences), 1)) * n_d
        elif normalization == 'max_offset':
            n_d = max_offset
            avg_gaps_norm = [gap_seq/n_d for gap_seq in avg_gaps_norm]
            normalizer_d = np.ones((len(sequences), 1)) * n_d

        return avg_gaps_norm, normalizer_d, normalizer_a, initial_gaps

    timeTrain, trainND, trainNA, trainIG = _generate_norm_seq(timeTrain, enc_len, normalization)
    timeDev, devND, devNA, devIG = _generate_norm_seq(timeDev, enc_len, normalization)
    timeTest, testND, testNA, testIG = _generate_norm_seq(timeTest, enc_len, normalization)

    return timeTrain, trainND, trainNA, trainIG, \
            timeDev, devND, devNA, devIG, \
            timeTest, testND, testNA, testIG

def read_seq2seq_data(dataset_path, alg_name, dec_len,
                      normalization=None, max_offset=0.0, offset=0.0, pad=False):
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

    for x in eventTrain + eventDevIn + eventTestIn:
        unique_samples = unique_samples.union(x)

    enc_len = len(timeTrainIn[0])
    trainActualTimeIn = np.array(timeTrainIn)[:, enc_len-1:enc_len].tolist()
    devActualTimeIn = np.array(timeDevIn)[:, enc_len-1:enc_len].tolist()
    testActualTimeIn = np.array(timeTestIn)[:, enc_len-1:enc_len].tolist()
    trainActualTimeOut, devActualTimeOut, testActualTimeOut = timeTrainOut, timeDevOut, timeTestOut

    # ----- Normalization by gaps ----- #
    timeTrain, trainND, trainNA, trainIG, \
            timeDev, devND, devNA, devIG, \
            timeTest, testND, testNA, testIG \
            = generate_norm_seq(timeTrain, timeDev, timeTest, enc_len, normalization, max_offset)

    timeTrainIn = [seq[:enc_len] for seq in timeTrain]
    timeTrainOut = [seq[enc_len:] for seq in timeTrain]
    timeDevIn = [seq[:enc_len] for seq in timeDev]
    timeDevOut = [seq[enc_len:] for seq in timeDev]
    timeTestIn = [seq[:enc_len] for seq in timeTest]
    timeTestOut = [seq[enc_len:] for seq in timeTest]

    timeTrainFeats = np.array(timeTrainFeats)
    timeDevFeats = np.array(timeDevFeats)
    timeTestFeats = np.array(timeTestFeats)
    timeTrainInFeats = [seq[:enc_len] for seq in timeTrainFeats]
    timeTrainOutFeats = [seq[enc_len:] for seq in timeTrainFeats]
    timeDevInFeats = [seq[:enc_len] for seq in timeDevFeats]
    timeDevOutFeats = [seq[enc_len:] for seq in timeDevFeats]
    timeTestInFeats = [seq[:enc_len] for seq in timeTestFeats]
    timeTestOutFeats = [seq[enc_len:] for seq in timeTestFeats]

    seq_len = enc_len + dec_len - 1
    eventTrainIn = [x[:seq_len] for x in eventTrain]
    eventTrainOut = [x[1:seq_len+1] for x in eventTrain]
    timeTrainIn = [x[:seq_len] for x in timeTrain]
    timeTrainOut = [x[1:seq_len+1] for x in timeTrain]
    timeTrainInFeats = [x[:seq_len] for x in timeTrainFeats]
    timeTrainOutFeats = [x[1:seq_len+1] for x in timeTrainFeats]

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

        'train_actual_time_in_seq': trainActualTimeIn,
        'train_actual_time_out_seq': trainActualTimeOut,

        'dev_actual_time_in_seq': devActualTimeIn,
        'dev_actual_time_out_seq': devActualTimeOut,

        'test_actual_time_in_seq': testActualTimeIn,
        'test_actual_time_out_seq': testActualTimeOut,

        'train_time_in_feats': train_time_in_feats,
        'dev_time_in_feats': dev_time_in_feats,
        'test_time_in_feats': test_time_in_feats,
        'train_time_out_feats': train_time_out_feats,
        'dev_time_out_feats': dev_time_out_feats,
        'test_time_out_feats': test_time_out_feats,

        'attn_train_time_in_seq': attn_timeTrainIn,
        'attn_dev_time_in_seq': attn_timeDevIn,
        'attn_test_time_in_seq': attn_timeTestIn,


        'num_categories': len(unique_samples),
        'encoder_length': len(eventTestIn[0]),
        #'decoder_length': len(eventTrain[0])-len(eventTestIn[0]),
        'decoder_length': dec_len,

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
    time_true = np.array(time_true)
    time_preds = np.array(time_preds)
    events_out = np.array(events_out)
    seq_limit = time_preds.shape[1]
    batch_limit = time_preds.shape[0]
    clipped_time_true = time_true[:batch_limit, :seq_limit]
    clipped_events_out = events_out[:batch_limit, :seq_limit]
    #print(clipped_time_true)
    #print(time_preds)

    is_finite = np.isfinite(time_preds) & (clipped_events_out > 0)

    return np.sum(np.abs(time_preds - clipped_time_true), axis=0), np.sum(is_finite, axis=0)

def DTW(time_preds, time_true, events_out):

    #seq_limit = time_preds.shape[1]
    clipped_time_true = time_true#[:, :seq_limit]
    clipped_events_out = events_out#[:, :seq_limit]

    euclidean_norm = lambda x, y: np.abs(x - y)
    distance = 0
    for time_preds_, clipped_time_true_ in zip(time_preds, clipped_time_true):
        d, cost_matrix, acc_cost_matrix, path = dtw(time_preds_, clipped_time_true_, dist=euclidean_norm)
        distance += d
    distance = distance / len(clipped_time_true)

    return distance

def RMSE(time_preds, time_true, events_out):
    """Calculates the RMSE between the provided and the given time, ignoring the inf
    and nans. Returns both the MAE and the number of items considered."""

    # Predictions may not cover the entire time dimension.
    # This clips time_true to the correct size.
    seq_limit = time_preds.shape[1]
    clipped_time_true = time_true[:, :seq_limit]
    clipped_events_out = events_out[:, :seq_limit]
    #print(clipped_time_true)
    #print(time_preds)

    is_finite = np.isfinite(time_preds) & (clipped_events_out > 0)

    return np.sqrt(np.mean(np.square(time_preds - clipped_time_true)[is_finite])), np.sum(is_finite)


def ACC(event_preds, event_true):
    """Returns the accuracy of the event prediction, provided the output probability vector."""
    event_preds = np.array(event_preds)
    event_true = np.array(event_true)
    clipped_event_true = event_true[:event_preds.shape[0], :event_preds.shape[1]]
    is_valid = clipped_event_true > 0

    # The indexes start from 0 whereare event_preds start from 1.
    # highest_prob_ev = event_preds.argmax(axis=-1) + 1
    highest_prob_ev = event_preds
    #print(clipped_event_true)
    #print(highest_prob_ev)

    return np.sum((highest_prob_ev == clipped_event_true), axis=0), np.sum(is_valid, axis=0)


def MRR(event_preds_softmax, event_true):
    "Computes Mean Reciprocal Rank of events"

    num_unique_events = event_preds_softmax.shape[-1]
    dec_len = event_preds_softmax.shape[1]
    event_true_flatten = np.reshape(event_true, [-1, 1])
    num_events = event_true_flatten.shape[0]
    event_preds_softmax_flatten = np.reshape(event_preds_softmax, [-1, num_unique_events])

    ranks = np.where(event_true_flatten-1 == (np.argsort(-event_preds_softmax_flatten)))[1] + 1
    ranks = np.reshape(ranks, [-1, dec_len])
    #print((np.argsort(-event_preds_softmax_flatten)).shape)
    #for event_tr, softmax, softmax_ranked in zip(event_true_flatten, event_preds_softmax_flatten.tolist(), np.where(event_true_flatten-1 == (np.argsort(-event_preds_softmax_flatten)))[1]):
    #    print(event_tr, softmax, softmax_ranked)
    #print(ranks)

    return np.sum(1.0/ranks, axis=0)

def PERCENT_ERROR(event_preds, event_true):
    return (1.0 - ACC(event_preds, event_true)) * 100

def get_output_indices(input_seqs, horizon_output_seqs, offsets, decoder_length):
    out_begin_indices, out_end_indices = list(), list()

    for input_seq, hor_out_seq, offset in zip(input_seqs, horizon_output_seqs, offsets):
        out_begin_ind = bisect_right(hor_out_seq, input_seq[-1]+offset)
        out_end_ind = out_begin_ind + decoder_length
        out_begin_indices.append(out_begin_ind)
        out_end_indices.append(out_end_ind)

    return out_begin_indices, out_end_indices

def get_attn_seqs_from_offset(input_seqs, attn_seqs, offsets, encoder_length):
    attn_begin_indices, attn_end_indices = list(), list()

    for seq, attn_seq, offset in zip(input_seqs, attn_seqs, offsets):
        key = seq[-1] + offset - 3600.0 * 24.0 # One day in the past
        attn_idx = bisect_right(attn_seq, key)
        #print(key, attn_seq[attn_idx], abs(key-attn_seq[attn_idx]))
        #begin_idx = int(np.clip(attn_idx - encoder_length/2, 0, len(attn_seq)-1))
        #end_idx = int(np.clip(attn_idx + encoder_length/2, 0, len(attn_seq)-1))

        if attn_idx-encoder_length<0:
            begin_idx = 0
            end_idx = encoder_length
        elif attn_idx+encoder_length>=len(attn_seq):
            begin_idx = int(max(0, len(attn_seq)-1-encoder_length))
            end_idx = len(attn_seq)-1
        else:
            begin_idx = attn_idx-encoder_length//2
            end_idx = attn_idx+encoder_length//2

        attn_begin_indices.append(begin_idx)
        attn_end_indices.append(end_idx)

    return attn_begin_indices, attn_end_indices
