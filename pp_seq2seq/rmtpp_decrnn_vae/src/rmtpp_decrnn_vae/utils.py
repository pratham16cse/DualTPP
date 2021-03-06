from tensorflow.contrib.keras import preprocessing
from collections import defaultdict
import itertools
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from itertools import chain


pad_sequences = preprocessing.sequence.pad_sequences


def create_dir(dirname):
    """Creates a directory if it does not already exist."""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def change_zero_label(sequences):
    return [[lbl+1 for lbl in sequence] for sequence in sequences]

def generate_norm_seq(sequences, enc_len, check=0):
    sequences = np.array(sequences)
    gap_seqs = sequences[:, 1:]-sequences[:, :-1]
    avg_gaps = np.average(gap_seqs[:, :enc_len-1], axis=1)
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

def generate_shifted_seq(sequences, enc_len, check=0):
    sequences = np.array(sequences)
    gap_seqs = sequences[:, 1:]-sequences[:, :-1]
    avg_gaps = np.ones_like(np.average(gap_seqs[:, :enc_len-1], axis=1))
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

def read_seq2seq_data(event_train_file, event_dev_file, event_test_file,
                      time_train_file, time_dev_file, time_test_file,
                      normalization=None,
                      pad=True, dataset_path=None):
    """Read data from given files and return it as a dictionary."""

    with open(event_train_file+'.in', 'r') as in_file:
        eventTrainIn = [[int(y) for y in x.strip().split()] for x in in_file]
    with open(event_train_file+'.out', 'r') as in_file:
        eventTrainOut = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(event_dev_file+'.in', 'r') as in_file:
        eventDevIn = [[int(y) for y in x.strip().split()] for x in in_file]
    with open(event_dev_file+'.out', 'r') as in_file:
        eventDevOut = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(event_test_file+'.in', 'r') as in_file:
        eventTestIn = [[int(y) for y in x.strip().split()] for x in in_file]
    with open(event_test_file+'.out', 'r') as in_file:
        eventTestOut = [[int(y) for y in x.strip().split()] for x in in_file]

    with open(time_train_file+'.in', 'r') as in_file:
        timeTrainIn = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(time_train_file+'.out', 'r') as in_file:
        timeTrainOut = [[float(y) for y in x.strip().split()] for x in in_file]

    with open(time_dev_file+'.in', 'r') as in_file:
        timeDevIn = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(time_dev_file+'.out', 'r') as in_file:
        timeDevOut = [[float(y) for y in x.strip().split()] for x in in_file]

    with open(time_test_file+'.in', 'r') as in_file:
        timeTestIn = [[float(y) for y in x.strip().split()] for x in in_file]
    with open(time_test_file+'.out', 'r') as in_file:
        timeTestOut = [[float(y) for y in x.strip().split()] for x in in_file]

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

    # nb_samples = len(eventTrain)
    # max_seqlen = max(len(x) for x in eventTrain)
    unique_samples = set()

    for x in eventTrainIn + eventTrainOut + eventDevIn + eventTestIn:
        unique_samples = unique_samples.union(x)

    enc_len = len(timeTrainIn[0])
    devActualTimeIn = np.array(timeDevIn)[:, enc_len-1:enc_len].tolist()
    testActualTimeIn = np.array(timeTestIn)[:, enc_len-1:enc_len].tolist()
    devActualTimeOut, testActualTimeOut = timeDevOut, timeTestOut

    if normalization is not None:
        if normalization == 'minmax':
            maxTime = max(itertools.chain((max(x) for x in timeTrainIn), (max(x) for x in timeDevIn), (max(x) for x in timeTestIn)))
            minTime = min(itertools.chain((min(x) for x in timeTrainIn), (min(x) for x in timeDevIn), (min(x) for x in timeTestIn)))
        elif normalization == 'average':
            allTimes = [x for x in chain(*timeTrainIn)] + [x for x in chain(*timeDevIn)] + [x for x in chain(*timeTestIn)]
            maxTime = np.mean(allTimes)
            minTime = 0
        elif normalization == 'average_per_seq':
            timeTrain, trainAvgGaps = generate_norm_seq(timeTrain, enc_len)
            timeDev, devAvgGaps = generate_norm_seq(timeDev, enc_len)
            timeTest, testAvgGaps = generate_norm_seq(timeTest, enc_len)
            minTime, maxTime = 0, 1
        else:
            print('Normalization not found')
            assert False
    else:
        timeTrain, trainAvgGaps = generate_shifted_seq(timeTrain, enc_len)
        timeDevIn, devAvgGaps = generate_shifted_seq(timeDevIn, enc_len)
        timeTestIn, testAvgGaps = generate_shifted_seq(timeTestIn, enc_len)
        minTime, maxTime = 0, 1

    timeTrainIn, timeTrainOut = timeTrain[:, :enc_len], timeTrain[:, enc_len:]
    timeDevIn, timeDevOut = timeDev[:, :enc_len], timeDev[:, enc_len:]
    timeTestIn, timeTestOut = timeTest[:, :enc_len], timeTest[:, enc_len:]

    timeTrainIn = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTrainIn]
    timeTrainOut = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTrainOut]

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

    timeDevIn = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeDevIn]
    timeDevOut = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeDevOut]

    if pad:
        dev_event_in_seq = pad_sequences(eventDevIn, padding='post')
        dev_event_out_seq = pad_sequences(eventDevOut, padding='post')
        dev_time_in_seq = pad_sequences(timeDevIn, dtype=float, padding='post')
        dev_time_out_seq = pad_sequences(timeDevOut, dtype=float, padding='post')
    else:
        dev_event_in_seq = eventDevIn
        dev_event_out_seq = eventDevOut
        dev_time_in_seq = timeDevIn
        dev_time_out_seq = timeDevOut

    timeTestIn = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTestIn]
    timeTestOut = [[(y - minTime) / (maxTime - minTime) for y in x] for x in timeTestOut]

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

        'dev_event_in_seq': dev_event_in_seq,
        'dev_event_out_seq': dev_event_out_seq,

        'dev_time_in_seq': dev_time_in_seq,
        'dev_time_out_seq': dev_time_out_seq,

        'test_event_in_seq': test_event_in_seq,
        'test_event_out_seq': test_event_out_seq,

        'test_time_in_seq': test_time_in_seq,
        'test_time_out_seq': test_time_out_seq,

        'dev_avg_gaps': devAvgGaps,
        'test_avg_gaps': testAvgGaps,

        'test_actual_time_in_seq': testActualTimeIn,
        'test_actual_time_out_seq': testActualTimeOut,

        'dev_actual_time_in_seq': devActualTimeIn,
        'dev_actual_time_out_seq': devActualTimeOut,

        'num_categories': len(unique_samples),
        'encoder_length': len(eventTrainIn[0]),
        'decoder_length': len(eventTrainOut[0]),
        'minTime': minTime,
        'maxTime': maxTime,
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

    return np.mean(np.abs(time_preds - clipped_time_true)[is_finite]), np.sum(is_finite)

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

def PERCENT_ERROR(event_preds, event_true):
    return (1.0 - ACC(event_preds, event_true)) * 100
