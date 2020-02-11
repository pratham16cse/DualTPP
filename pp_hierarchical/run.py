import sys
import argparse
import json
import os
from itertools import product
from argparse import Namespace
import ipdb
import copy
from joblib import Parallel, delayed
from operator import itemgetter

import run_rmtpp
import run_hierarchical

hparams_aliases = json.loads(open('hparams_aliases.json', 'r').read())
hparams = json.loads(open('hparams.json', 'r').read())

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help='dataset_name')
parser.add_argument('alg_name', type=str, help='alg_name')
parser.add_argument('dataset_path', type=str, help='Path to the raw dataset file')
#parser.add_argument('alg_path', type=str, help='Path to the code')

#parser.add_argument('-hls', '--hidden_layer_size', type=int, nargs='+',
#                    help='Number of units in RNN')
#parser.add_argument('--learning_rate', type=float, default=1e-2, nargs='+',
#                    help='Learning rate for the training algorithm')
parser.add_argument('--use_marks', action='store_true',
                    help='Consider markers \
                          in the data for training and testing')
parser.add_argument('--use_intensity', action='store_true',
                    help='Use intensity based formulation')
parser.add_argument('--use_time_embed', action='store_true',
                    help='Discretize and embed hour-of-day features')

#parser.add_argument('--training_mode', action='store_true',
#                    help='Enables training mode. If false, only inference \
#                          is performed')
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
parser.add_argument('--normalization', type=str, default='average',
                    help='gap normalization method')
parser.add_argument('--decoder_length', type=int, default=5,
                    help='Number of events to predict in the prediction range')
parser.add_argument('--compound_event_size', type=int, default=10,
                    help='Number of simple events in a compound event')

parser.add_argument('--generate_plots', action='store_true',
                    help='Generate dev and test plots, both per epochs \
                          and after training')
parser.add_argument('--parallel_hparam', action='store_true',
                    help='Parallel execution of hyperparameters')

group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true')
group.add_argument('-q', '--quite', action='store_true')


args = parser.parse_args()

results = list()
best_result_idx = 0
param_names, param_values = zip(*hparams[args.alg_name].items())

orig_stdout = sys.stdout

def run_config(param_vals, training_mode):
    args_vars_config = copy.deepcopy(vars(args))
    args_vars_config['training_mode'] = training_mode
    for name, val in zip(param_names, param_vals):
        args_vars_config[name] = val

    param_dir = ''
    for name, val in zip(param_names, param_vals):
        param_dir = param_dir + hparams_aliases[name] + '_' + str(val) + '_'

    args_vars_config['output_dir'] \
            = os.path.join(args_vars_config['output_dir'], param_dir)
    if training_mode==1.:
        args_vars_config['output_dir'] \
                = os.path.join(args_vars_config['output_dir'], 'train')
    else:
        args_vars_config['output_dir'] \
                = os.path.join(args_vars_config['output_dir'], 'eval')
    print(args_vars_config['output_dir'])

    os.makedirs(args_vars_config['output_dir'], exist_ok=True)
    f = open(os.path.join(args_vars_config['output_dir'], 'print_dump'), 'w')
    args_vars_config['outfile'] = f

    args_config = Namespace(**args_vars_config)
    print(args_config)
    print(args)

    if args.alg_name in ['rmtpp']:
        result = run_rmtpp.run(args_config)
    elif args.alg_name in ['hierarchical']:
        result = run_hierarchical.run(args_config)

    result['output_dir'] = args_config.output_dir
    for name, val in zip(param_names, param_vals):
        result[name] = val
    print('\n Result:')
    print(result)
    with open(os.path.join(args_config.output_dir, 'result.json'), 'w') as fp:
        result_json = json.dumps(result, indent=4)
        fp.write(result_json)

    f.close()

    return result


param_vals_list = product(*param_values)
param_vals_list = [param_vals for param_vals in param_vals_list]

# Training
print('\n Starting Training. . .')
training_mode_list = [1.] * len(param_vals_list)
if args.parallel_hparam:
    for param_vals, training_mode in zip(param_vals_list, training_mode_list):
        print(param_vals, training_mode)
    results = Parallel(n_jobs=1)(delayed(run_config)(param_vals, training_mode) for param_vals, training_mode in zip(param_vals_list, training_mode_list))
else:
    results = list()
    for param_vals, mode in zip(param_vals_list, training_mode_list):
        result = run_config(param_vals, mode)
        results.append(result)
print('\n Finished Training. . .')


# Inference
print('\n Starting Inference. . .')
training_mode_list = [0.] * len(param_vals_list)
results = Parallel(n_jobs=1)(delayed(run_config)(param_vals, training_mode) for param_vals, training_mode in zip(param_vals_list, training_mode_list))
print('\n Finished Inference. . .')

dev_gap_errors = [result['best_dev_gap_error'] for result in results]
best_result_idx, _ = min(enumerate(dev_gap_errors), key=itemgetter(1))
print(best_result_idx)
best_result = results[best_result_idx]
print('\n best_results')
print(best_result)

with open(os.path.join(args.output_dir, 'best_result.json'), 'w') as fp:
    best_result_json = json.dumps(best_result, indent=4)
    fp.write(best_result_json)
