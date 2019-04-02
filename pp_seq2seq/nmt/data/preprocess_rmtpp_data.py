import sys, os
sys.path.append('../../../tf_rmtpp/src/tf_rmtpp/')
from itertools import chain
import numpy as np
import utils
from BatchGenerator import BatchGenerator

def preprocess(dataset_name,
               event_train_file,
               time_train_file,
               event_test_file,
               time_test_file,
               encoder_length,
               decoder_length):
    sequence_length = encoder_length + decoder_length
    batch_size = 100
    data = utils.read_data(event_train_file=event_train_file,
                           event_test_file=event_test_file,
                           time_train_file=time_train_file,
                           time_test_file=time_test_file,
                           feats_train_file=None,
                           feats_test_file=None,
                           pad=False,
                           normalize=False)

    #print(data['train_event_in_seq'][:10])

    train_event_seq = data['train_event_in_seq']
    train_time_seq = data['train_time_in_seq']

    test_event_seq = data['test_event_in_seq']
    test_time_seq = data['test_time_in_seq']

    unique_labels = set([lbl for lbl in chain(*(train_event_seq + test_event_seq))])
    print('Unique Labels:', np.array(sorted(unique_labels)))

    train_event_itr = BatchGenerator(train_event_seq, batchSeqLen=sequence_length, batch_pad=False)
    train_time_itr = BatchGenerator(train_time_seq, batchSeqLen=sequence_length, batch_pad=False)

    test_event_itr = BatchGenerator(test_event_seq, batchSeqLen=sequence_length, batch_pad=False)
    test_time_itr = BatchGenerator(test_time_seq, batchSeqLen=sequence_length, batch_pad=False)

    pp_train_event_in_seq, pp_test_event_in_seq = list(), list()
    pp_train_time_in_seq, pp_test_time_in_seq = list(), list()
    pp_train_event_out_seq, pp_test_event_out_seq = list(), list()
    pp_train_time_out_seq, pp_test_time_out_seq = list(), list()

    print('Creating training data . . .')
    currItr = train_time_itr.iterFinished
    ctr = 0
    while currItr == train_time_itr.iterFinished:
        ctr += 1
        print(ctr)
        train_event_batch, _, _, _, _ = train_event_itr.nextBatch(batchSize=batch_size)
        train_time_batch, _, _, _, _ = train_time_itr.nextBatch(batchSize=batch_size)

        if ctr == 1: print(train_event_batch.shape)

        pp_train_event_in_seq.extend(train_event_batch[:, :encoder_length].tolist())
        pp_train_time_in_seq.extend(train_time_batch[:, :encoder_length].tolist())
        pp_train_event_out_seq.extend(train_event_batch[:, encoder_length:].tolist())
        pp_train_time_out_seq.extend(train_time_batch[:, encoder_length:].tolist())

    print('Creating test data . . .')
    currItr = test_time_itr.iterFinished
    ctr = 0
    while currItr == test_time_itr.iterFinished:
        ctr += 1
        print(ctr)

        test_event_batch, _, _, _, _ = test_event_itr.nextBatch(batchSize=batch_size)
        test_time_batch, _, _, _, _ = test_time_itr.nextBatch(batchSize=batch_size)

        if ctr == 1: print(test_event_batch.shape)

        pp_test_event_in_seq.extend(test_event_batch[:, :encoder_length].tolist())
        pp_test_time_in_seq.extend(test_time_batch[:, :encoder_length].tolist())
        pp_test_event_out_seq.extend(test_event_batch[:, encoder_length:].tolist())
        pp_test_time_out_seq.extend(test_time_batch[:, encoder_length:].tolist())


    def write_to_file(fptr, sequences):
        for seq in sequences:
            for e in seq:
                f.write(str(e) + ' ')
            f.write('\n')

    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)
    with open(os.path.join(dataset_name, 'train_event.in'), 'w') as f:
        write_to_file(f, pp_train_event_in_seq)
    with open(os.path.join(dataset_name, 'train_event.out'), 'w') as f:
        write_to_file(f, pp_train_event_out_seq)
    with open(os.path.join(dataset_name, 'test_event.in'), 'w') as f:
        write_to_file(f, pp_test_event_in_seq)
    with open(os.path.join(dataset_name, 'test_event.out'), 'w') as f:
        write_to_file(f, pp_test_event_out_seq)
    with open(os.path.join(dataset_name, 'train_time.in'), 'w') as f:
        write_to_file(f, pp_train_time_in_seq)
    with open(os.path.join(dataset_name, 'train_time.out'), 'w') as f:
        write_to_file(f, pp_train_time_out_seq)
    with open(os.path.join(dataset_name, 'test_time.in'), 'w') as f:
        write_to_file(f, pp_test_time_in_seq)
    with open(os.path.join(dataset_name, 'test_time.out'), 'w') as f:
        write_to_file(f, pp_test_time_out_seq)

    with open(os.path.join(dataset_name, 'labels.in'), 'w') as f:
        for lbl in unique_labels:
            f.write(str(lbl) + '\n')


def main():

    dataset_name = sys.argv[1]
    event_train_file = sys.argv[2]
    time_train_file = sys.argv[3]
    event_test_file = sys.argv[4]
    time_test_file = sys.argv[5]
    encoder_length = int(sys.argv[6])
    decoder_length = int(sys.argv[7])

    preprocess(dataset_name,
               event_train_file,
               time_train_file,
               event_test_file,
               time_test_file,
               encoder_length,
               decoder_length)

if __name__ == '__main__':
    main()
