from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print(times)
    times = pd.Series(times)
    times_grouped = times.groupby(lambda x: pd.Timestamp(times[x], unit='s').floor('H')).agg('count')
    #plt.bar(times_grouped.index, times_grouped.tolist(), width=0.02)
    plt.bar(range(len(times_grouped.index)), times_grouped.values)
    return times_grouped


def get_input_output_seqs(data):
    marks, times = data
    marks = [np.array(x[1:]) for x in marks]
    times = [np.array(x[1:])-np.array(x[:-1]) for x in times]
    marks_in = [x[:-1] for x in marks]
    marks_out = [x[1:] for x in marks]
    times_in = [x[:-1] for x in times]
    times_out = [x[1:] for x in times]

    return marks_in, marks_out, times_in, times_out

def create_train_dev_test_split(data, max_offset):
    marks, times = data
    num_events_per_hour = get_num_events_per_hour(data)
    train_marks, train_times = list(), list()
    dev_marks, dev_times = list(), list()
    test_marks, test_times = list(), list()

    block_begin_idxes = num_events_per_hour.cumsum()
    num_hrs = len(num_events_per_hour)-len(num_events_per_hour)%(4*max_offset)
    for idx in range(0, num_hrs, 4*max_offset):
        print(idx, num_hrs)
        train_start_idx = block_begin_idxes[idx-1]+1 if idx>0 else 0
        train_end_idx = block_begin_idxes[idx+(2*max_offset-1)]
        train_marks.append(marks[train_start_idx:train_end_idx])
        train_times.append(times[train_start_idx:train_end_idx])

        dev_start_idx = block_begin_idxes[idx+(2*max_offset-1)]+1
        dev_end_idx = block_begin_idxes[idx+(3*max_offset-1)]
        dev_marks.append(marks[dev_start_idx:dev_end_idx])
        dev_times.append(times[dev_start_idx:dev_end_idx])

        test_start_idx = block_begin_idxes[idx+(3*max_offset-1)]+1
        test_end_idx = block_begin_idxes[idx+(4*max_offset-1)]
        test_marks.append(marks[test_start_idx:test_end_idx])
        test_times.append(times[test_start_idx:test_end_idx])

    return train_marks, train_times, \
            dev_marks, dev_times, \
            test_marks, test_times

def main():
    for dataset in ['Delhi']:#['barca', 'Delhi', 'jaya', 'Movie', 'Fight', 'Verdict', 'Trump']:
        filename = '../pp_seq2seq/data/DataSetForSeq2SeqPP/'+dataset+'.txt'
        marks, times = read_data(filename)
        num_chops = 1
        #marks, times = split_data((marks, times), num_chops)
        num_events_per_hour = get_num_events_per_hour((marks, times))
        print('Number of hours spanned by '+dataset, len(num_events_per_hour))
        #get_best_num_chops((marks, times))
        #get_best_max_offset((marks, times))
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

