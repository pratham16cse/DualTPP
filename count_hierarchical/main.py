import argparse
import sys, os, json
import numpy as np
from itertools import product
from argparse import Namespace
import multiprocessing as MP
from operator import itemgetter
import datetime
from generator import generate_dataset, generate_twitter_dataset

import run
import utils

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help='dataset_name')
parser.add_argument('model_name', type=str, help='model_name')

parser.add_argument('--epochs', type=int, default=15,
                    help='number of training epochs')
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs to wait for \
                          before beginning cross-validation')

parser.add_argument('--learning_rate', type=float, default=1e-3, nargs='+',
                   help='Learning rate for the training algorithm')
parser.add_argument('-hls', '--hidden_layer_size', type=int, default=32, nargs='+',
                   help='Number of units in RNN')

parser.add_argument('--output_dir', type=str,
                    help='Path to store all raw outputs, checkpoints, \
                          summaries, and plots', default='Outputs')

parser.add_argument('--seed', type=int,
                    help='Seed for parameter initialization',
                    default=42)

# Bin size T_i - T_(i-1) in seconds
parser.add_argument('--bin_size', type=int, default=0,
                    help='Number of seconds in a bin')

# F(T_(i-1), T_(i-2) ..... , T_(i-r)) -> T(i)
# r_feature_sz = 20
parser.add_argument('--in_bin_sz', type=int,
                    help='Input count of bins r_feature_sz',
                    default=20)

# dec_len = 8   # For All Models
parser.add_argument('--out_bin_sz', type=int,
                    help='Output count of bin',
                    default=1)

# enc_len = 80  # For RMTPP
parser.add_argument('--enc_len', type=int, default=80,
                    help='Input length for rnn of rmtpp')

# wgan_enc_len = 60  # For WGAN
parser.add_argument('--wgan_enc_len', type=int, default=60,
                    help='Input length for rnn of WGAN')
parser.add_argument('--use_wgan_d', action='store_true', default=False,
                    help='Whether to use WGAN discriminator or not')

# interval_size = 360  # For RMTPP
parser.add_argument('--interval_size', type=int, default=360,
                    help='Interval size for threshold query')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Input batch size')
parser.add_argument('--query', type=int, default=1,
                    help='Query number')
parser.add_argument('--normalization', type=str, default='average',
                    help='gap normalization method')

parser.add_argument('--generate_plots', action='store_true', default=False,
                    help='Generate dev and test plots, both per epochs \
                          and after training')
parser.add_argument('--parallel_hparam', action='store_true', default=False,
                    help='Parallel execution of hyperparameters')

args = parser.parse_args()

dataset_names = list()
if args.dataset_name == 'all':
    dataset_names.append('sin')
    dataset_names.append('hawkes')
    dataset_names.append('sin_hawkes_overlay')
    dataset_names.append('taxi')
    dataset_names.append('twitter')
else:
    dataset_names.append(args.dataset_name)

print(dataset_names)

twitter_dataset_names = list()
if 'twitter' in dataset_names:
    dataset_names.remove('twitter')
    twitter_dataset_names.append('Trump')
    #twitter_dataset_names.append('Verdict')
    #twitter_dataset_names.append('Delhi')

for data_name in twitter_dataset_names:
    dataset_names.append(data_name)

args.dataset_name = dataset_names

model_names = list()
if args.model_name == 'all':
    # model_names.append('wgan')
    model_names.append('count_model')
    # model_names.append('hierarchical')
    model_names.append('rmtpp_nll')
    model_names.append('rmtpp_mse')
    model_names.append('rmtpp_count')
else:
    model_names.append(args.model_name)
args.model_name = model_names

run_model_flags = {
    'compute_time_range_pdf': True,

    'run_rmtpp_count_with_optimization': False,
    'run_rmtpp_with_optimization_fixed_cnt': False,
    'run_rmtpp_with_optimization_fixed_cnt_solver_with_nll': True,

    'run_rmtpp_count_cont_rmtpp_with_nll': True,
    'run_rmtpp_count_cont_rmtpp_with_mse': True,
    'run_rmtpp_count_reinit_with_nll': True,
    'run_rmtpp_count_reinit_with_mse': True,

    'run_rmtpp_for_count_with_mse': True,
    'run_rmtpp_for_count_with_nll': True,
    'run_wgan_for_count': False,
}

automate_bin_sz = False
if args.bin_size == 0:
    automate_bin_sz = True

if args.patience >= args.epochs:
    args.patience = 0

id_process = os.getpid()
time_current = datetime.datetime.now().isoformat()

print('args', args)

print("********************************************************************")
print("PID: %s" % str(id_process))
print("Time: %s" % time_current)
print("epochs: %s" % str(args.epochs))
print("learning_rate: %s" % str(args.learning_rate))
print("seed: %s" % str(args.seed))
print("Models: %s" % str(model_names))
print("Datasets: %s" % str(dataset_names))
print("********************************************************************")

print("####################################################################")
np.random.seed(args.seed)
print("Generating Datasets\n")
generate_dataset()
generate_twitter_dataset(twitter_dataset_names)
print("####################################################################")

os.makedirs('Outputs', exist_ok=True)
event_count_result = dict()
for dataset_name in dataset_names:
    print("Processing", dataset_name, "Datasets\n")
    if automate_bin_sz:
        args.current_dataset = dataset_name
        args.bin_size = utils.get_optimal_bin_size(dataset_name)
        print('New bin size is', args.bin_size)
    dataset = utils.get_processed_data(dataset_name, args)

    test_data_out_bin = dataset['test_data_out_bin']
    event_count_preds_true = test_data_out_bin
    count_var = None

    per_model_count = dict()
    per_model_save = {
        'wgan': None,
        'count_model': None,
        'hierarchical': None,
        'rmtpp_mse': None,
        'rmtpp_nll': None,
        'rmtpp_count': None,
    }
    per_model_count['true'] = event_count_preds_true
    for model_name in model_names:
        print("--------------------------------------------------------------------")
        print("Running", model_name, "Model\n")

        model, result = run.run_model(dataset_name, model_name, dataset, args, per_model_save, run_model_flags=run_model_flags)

        if model_name == 'count_model':
            count_var = result['count_var'].numpy()
            result = result['count_preds']

        per_model_count[model_name] = result
        per_model_save[model_name] = model
        print("Finished Running", model_name, "Model\n")

        if model_name != 'rmtpp_count' and per_model_count[model_name] is not None:
            old_stdout = sys.stdout
            sys.stdout=open("Outputs/count_model_"+dataset_name+".txt","a")
            print("____________________________________________________________________")
            print(model_name, 'MAE for Count Prediction:', np.mean(np.abs(per_model_count['true']-per_model_count[model_name])))
            print(model_name, 'MAE for Count Prediction (per bin):', np.mean(np.abs(per_model_count['true']-per_model_count[model_name]), axis=0))
            print("____________________________________________________________________")
            sys.stdout.close()
            sys.stdout = old_stdout

        print('Got result', 'for model', model_name, 'on dataset', dataset_name)

    for idx in range(10):
        utils.generate_plots(args, dataset_name, dataset, per_model_count, test_sample_idx=idx, count_var=count_var)

    event_count_result[dataset_name] = per_model_count
    print("####################################################################")

