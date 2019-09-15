import sys, os
sys.path.append('../../tf_rmtpp/src/tf_rmtpp/')
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
#from preprocess_rmtpp_data import preprocess

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

def preprocess(raw_dataset_name,
               dataset_name,
               event_train, time_train,
               event_dev, time_dev,
               event_test, time_test,
               encoder_length, decoder_length,
               train_step_length=None, dev_step_length=None, test_step_length=None,
               keep_classes=0):

    sequence_length = encoder_length + decoder_length

    print('Step Lengths:', train_step_length, dev_step_length, test_step_length)

    if train_step_length is None:
        train_step_length = sequence_length
    if dev_step_length is None:
        dev_step_length = decoder_length
    if test_step_length is None:
        test_step_length = decoder_length

    batch_size = 1

    train_event_seq = event_train
    train_time_seq = time_train

    dev_event_seq = event_dev
    dev_time_seq = time_dev

    test_event_seq = event_test
    test_time_seq = time_test

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

    print('Creating training data . . .')
    currItr = train_time_itr.iterFinished
    ctr = 0
    while currItr == train_time_itr.iterFinished:
        ctr += 1
        #print(ctr)
        train_event_batch, _, _, _, _ = train_event_itr.nextBatch(batchSize=batch_size, stepLen=train_step_length)
        train_time_batch, _, _, _, _ = train_time_itr.nextBatch(batchSize=batch_size, stepLen=train_step_length)

        #if ctr == 1: print(train_event_batch.shape)

        pp_train_event_in_seq.extend(train_event_batch[:, :encoder_length].tolist())
        pp_train_time_in_seq.extend(train_time_batch[:, :encoder_length].tolist())
        pp_train_event_out_seq.extend(train_event_batch[:, encoder_length:].tolist())
        pp_train_time_out_seq.extend(train_time_batch[:, encoder_length:].tolist())

    print('Creating dev data . . .')
    currItr = dev_time_itr.iterFinished
    ctr = 0
    while currItr == dev_time_itr.iterFinished:
        ctr += 1
        #print(ctr)
        dev_event_batch, _, _, _, _ = dev_event_itr.nextBatch(batchSize=batch_size, stepLen=dev_step_length)
        dev_time_batch, _, _, _, _ = dev_time_itr.nextBatch(batchSize=batch_size, stepLen=dev_step_length)

        #if ctr == 1: print(dev_event_batch.shape)

        pp_dev_event_in_seq.extend(dev_event_batch[:, :encoder_length].tolist())
        pp_dev_time_in_seq.extend(dev_time_batch[:, :encoder_length].tolist())
        pp_dev_event_out_seq.extend(dev_event_batch[:, encoder_length:].tolist())
        pp_dev_time_out_seq.extend(dev_time_batch[:, encoder_length:].tolist())

    print('Creating test data . . .')
    currItr = test_time_itr.iterFinished
    ctr = 0
    while currItr == test_time_itr.iterFinished:
        ctr += 1
        #print(ctr)

        test_event_batch, _, _, _, _ = test_event_itr.nextBatch(batchSize=batch_size, stepLen=test_step_length)
        test_time_batch, _, _, _, _ = test_time_itr.nextBatch(batchSize=batch_size, stepLen=test_step_length)

        #if ctr == 1: print(test_event_batch.shape)

        pp_test_event_in_seq.extend(test_event_batch[:, :encoder_length].tolist())
        pp_test_time_in_seq.extend(test_time_batch[:, :encoder_length].tolist())
        pp_test_event_out_seq.extend(test_event_batch[:, encoder_length:].tolist())
        pp_test_time_out_seq.extend(test_time_batch[:, encoder_length:].tolist())


    def write_to_file(fptr, sequences):
        for seq in sequences:
            for e in seq:
                f.write(str(e) + ' ')
            f.write('\n')

    dataset_name = dataset_name if dataset_name[-1]!='/' else dataset_name[:-1]
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

    with open(os.path.join(dataset_name, 'labels.in'), 'w') as f:
        for lbl in unique_labels:
            f.write(str(lbl) + '\n')

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
    plt.loglog(train_gaps_freq.keys(), train_gaps_freq.values(), 'ro')
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
    plt.loglog(dev_gaps_freq.keys(), dev_gaps_freq.values(), 'ro')
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
    plt.loglog(test_gaps_freq.keys(), test_gaps_freq.values(), 'ro')
    plt.title(raw_dataset_name+'__test')
    plt.savefig(os.path.join(dataset_name, 'test_gaps_freq.png'))
    plt.close()

    #----------- Gap Statistics End-------------#

def main():
    dataset = sys.argv[1]
    dataset_path = sys.argv[2]
    encoder_length = int(sys.argv[3])
    decoder_length = int(sys.argv[4])
    train_step_length = int(sys.argv[5])
    dev_step_length = int(sys.argv[6])
    test_step_length = int(sys.argv[7])
    keep_classes = int(sys.argv[8])
    sequence_length = encoder_length + decoder_length
    output_path = 'NewDataParsed'
    

    data_path = dataset_path

    data = pd.read_csv(data_path, usecols= [0, 1], delimiter=' ', header=None, names=['ID', 'Time']) 
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

    arr = np.array(lst)
    arr_eve = np.array(eve)

    tr_per = 0.7
    de_per = 0.1
    te_per = 0.2

    time_train = []
    time_dev = []
    time_test = []
    event_train = []
    event_dev = []
    event_test = []

    for x in range(0,int(tr_per*total)):
        time_train.append(arr[x])
        event_train.append(arr_eve[x])

    for x in range(int(tr_per*total),int((tr_per+de_per)*total)):
        time_dev.append(arr[x])
        event_dev.append(arr_eve[x])

    for x in range(int((tr_per+de_per)*total), total):
        time_test.append(arr[x])
        event_test.append(arr_eve[x])

    time_train = [time_train]
    event_train = [event_train]
    time_dev = [time_dev]
    event_dev = [event_dev]
    time_test = [time_test]
    event_test = [event_test]

    # seq_len = 25

    # pad_tr = len(time_train)%seq_len
    # pad_te = len(time_test)%seq_len
    # pad_de = len(time_dev)%seq_len

    # time_train = time_train[:len(time_train)-pad_tr]
    # time_test = time_test[:len(time_test)-pad_te]
    # time_dev = time_dev[:len(time_dev)-pad_de]
    # event_train = event_train[:len(event_train)-pad_tr]
    # event_test = event_test[:len(event_test)-pad_te]
    # event_dev = event_dev[:len(event_dev)-pad_de]

    # time_train = np.reshape(np.array(time_train), (-1, seq_len))
    # time_test = np.reshape(np.array(time_test), (-1, seq_len))
    # time_dev = np.reshape(np.array(time_dev), (-1, seq_len))
    # event_train = np.reshape(np.array(event_train), (-1, seq_len))
    # event_test = np.reshape(np.array(event_test), (-1, seq_len))
    # event_dev = np.reshape(np.array(event_dev), (-1, seq_len))

    # time_train = list(time_train.tolist())
    # time_test = list(time_test.tolist())
    # time_dev = list(time_dev.tolist())
    # event_train = list(event_train.tolist())
    # event_test = list(event_test.tolist())
    # event_dev = list(event_dev.tolist())

    print(np.shape(time_train))
    print(np.shape(time_test))
    print(np.shape(time_dev))
    print(np.shape(event_train))
    print(np.shape(event_test))
    print(np.shape(event_dev))

    preprocess(dataset,
               os.path.join(output_path, dataset),
               event_train, time_train,
               event_dev, time_dev,
               event_test, time_test,
               encoder_length, decoder_length,
               train_step_length, dev_step_length, test_step_length,
               keep_classes)

if __name__ == '__main__':
    main()
