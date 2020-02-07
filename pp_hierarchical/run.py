import argparse

import run_rmtpp
import run_hierarchical


parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help='dataset_name')
parser.add_argument('alg_name', type=str, help='alg_name')
parser.add_argument('dataset_path', type=str, help='Path to the raw dataset file')
parser.add_argument('alg_path', type=str, help='Path to the code')

parser.add_argument('--output_dir', type=str,
                    help='Path to store all raw outputs, checkpoints, \
                          summaries, and plots')
parser.add_argument('--seed', type=int,
                    help='Seed for parameter initialization')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of training epochs')
parser.add_argument('--patience', type=int, default=0,
                    help='Number of epochs to wait for \
                          before beginning cross-validation')
parser.add_argument('--bptt', type=int, default=20,
                    help='Truncation length for truncated bptt of rnn')
parser.add_argument('--block_size', type=int, default=1,
                    help='Number of hours to consider for evaluation')
parser.add_argument('--use_marks', action='store_true',
                    help='Consider markers \
                          in the data for training and testing')
parser.add_argument('--use_intensity', action='store_true',
                    help='Use intensity based formulation')
parser.add_argument('--normalization', type=str, default='average',
                    help='gap normalization method')
parser.add_argument('--decoder_length', type=int, default=5,
                    help='Number of events to predict in the prediction range')
parser.add_argument('--compound_event_size', type=int, default=10,
                    help='Number of simple events in a compound event')
parser.add_argument('--learning_rate', type=int, default=1e-2,
                    help='Learning rate for the training algorithm')

parser.add_argument('--generate_plots', action='store_true',
                    help='Generate dev and test plots, both per epochs \
                          and after training')

group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true')
group.add_argument('-q', '--quite', action='store_true')


args = parser.parse_args()
print(args)


if args.alg_name in ['rmtpp']:
    run_rmtpp.run(args)
elif args.alg_name in ['hierarchical']:
    run_hierarchical.run(args)
