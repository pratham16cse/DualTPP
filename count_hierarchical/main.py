import argparse
import sys, os, json
import numpy as np
from itertools import product
from argparse import Namespace
import multiprocessing as MP
from operator import itemgetter
import datetime
from collections import OrderedDict
from generator import generate_dataset, generate_twitter_dataset
import json
import time

import run
import utils

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('dataset_name', type=str, help='dataset_name')
parser.add_argument('model_name', type=str, help='model_name')

parser.add_argument('--use_marks', action='store_true',
                    help='Consider markers in the data for training and testing')

parser.add_argument('--epochs', type=int, default=0,
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

parser.add_argument('--cnt_net_type', type=str,
                    help='Count model network type (ff or rnn)')

# enc_len = 80  # For RMTPP
parser.add_argument('--enc_len', type=int, default=80,
                    help='Input length for rnn of rmtpp')

# comp_enc_len = 40  # For Compound RMTPP
parser.add_argument('--comp_enc_len', type=int, default=40,
                    help='Input length for rnn of compound rmtpp')

# comp_bin_sz = 10  # For Compound RMTPP
parser.add_argument('--comp_bin_sz', type=int, default=10,
                    help='events inside one bin of compound rmtpp')

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
parser.add_argument('--stride_len', type=int, default=1,
                    help='Stride len for RMTPP number')
parser.add_argument('--normalization', type=str, default='average',
                    help='gap normalization method')

parser.add_argument('--generate_plots', action='store_true', default=False,
                    help='Generate dev and test plots, both per epochs \
                          and after training')
parser.add_argument('--parallel_hparam', action='store_true', default=False,
                    help='Parallel execution of hyperparameters')

# Flags for RMTPP calibration
parser.add_argument('--calibrate_rmtpp', action='store_true', default=False,
                    help='Whether to calibrate RMTPP')
parser.add_argument('--extra_var_model', action='store_true', default=False,
                    help='Use a separate model to train the variance of RMTPP')

# Flags for optimizer
parser.add_argument('--opt_num_counts', type=int, default=5,
                    help='Number of counts to try before and after mean for optimizer')
parser.add_argument('--no_rescale_rmtpp_params', action='store_true', default=False,
                    help='Do not rescale RMTPP intensities for optimizer')
parser.add_argument('--use_ratio_constraints', action='store_true', default=False,
                    help='Maintain Ratios of adjacent RMTPP event predictions')

# Parameters for extra_var_model
parser.add_argument('--num_grps', type=int, default=10,
                    help='Number of groups in each bin in forecast horizon')
parser.add_argument('--num_pos', type=int, default=40,
                    help='Number of positions in each group in forecast horizon')

# Time-feature parameters
parser.add_argument('--no_count_model_feats', action='store_true', default=False,
                    help='Do not use time-features for count model')
parser.add_argument('--no_rmtpp_model_feats', action='store_true', default=False,
                    help='Do not use time-features for rmtpp model')


# Trainsformer Paramerters
parser.add_argument('-d_model', type=int, default=32) #64
parser.add_argument('-d_rnn', type=int, default=8) #256
parser.add_argument('-d_inner_hid', type=int, default=32) #128
parser.add_argument('-d_k', type=int, default=8) #16
parser.add_argument('-d_v', type=int, default=8) #16

parser.add_argument('-n_head', type=int, default=2) #4
parser.add_argument('-n_layers', type=int, default=1) #4

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-smooth', type=float, default=0.1)

args = parser.parse_args()

dataset_names = list()
if args.dataset_name == 'all':
    dataset_names.append('sin')
    # dataset_names.append('hawkes')
    # dataset_names.append('sin_hawkes_overlay')
    dataset_names.append('taxi')
    dataset_names.append('911_traffic')
    dataset_names.append('911_ems')
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
    #model_names.append('hawkes_model')
    model_names.append('wgan')
    model_names.append('seq2seq')
    model_names.append('transformer')
    model_names.append('count_model')
    # model_names.append('hierarchical')
    #model_names.append('rmtpp_nll')
    model_names.append('rmtpp_mse')
    model_names.append('rmtpp_mse_var')
    #model_names.append('rmtpp_nll_comp')
    model_names.append('rmtpp_mse_comp')
    model_names.append('rmtpp_mse_var_comp')
    #model_names.append('pure_hierarchical_nll')
    #model_names.append('pure_hierarchical_mse')
    model_names.append('rmtpp_count')
else:
    model_names.append(args.model_name)
args.model_name = model_names

run_model_flags = {
    'compute_time_range_pdf': False,

    'run_rmtpp_count_with_optimization': False,
    'run_rmtpp_with_optimization_fixed_cnt': False,

    'count_only': True,
}
if 'rmtpp_nll' in model_names:
    run_model_flags['rmtpp_nll_opt'] = True
    run_model_flags['rmtpp_nll_cont'] = True
    #run_model_flags['rmtpp_nll_reinit'] = True
    run_model_flags['rmtpp_nll_simu'] = True
if 'rmtpp_mse' in model_names:
    run_model_flags['rmtpp_mse_opt'] = True
    run_model_flags['rmtpp_mse_cont'] = True
    #run_model_flags['rmtpp_mse_reinit'] = True
    run_model_flags['rmtpp_mse_simu'] = True
if 'rmtpp_mse_var' in model_names:
    run_model_flags['rmtpp_mse_var_opt'] = True
    run_model_flags['rmtpp_mse_var_cont'] = True
    #run_model_flags['rmtpp_mse_var_reinit'] = True
    run_model_flags['rmtpp_mse_var_simu'] = True
if 'rmtpp_nll_comp' in model_names:
    #run_model_flags['run_rmtpp_with_joint_optimization_fixed_cnt_solver_nll_comp'] = True
    run_model_flags['rmtpp_nll_opt_comp'] = True
    run_model_flags['rmtpp_nll_cont_comp'] = True
if 'rmtpp_mse_comp' in model_names:
    #run_model_flags['run_rmtpp_with_joint_optimization_fixed_cnt_solver_mse_comp'] = True
    run_model_flags['rmtpp_mse_opt_comp'] = True
    run_model_flags['rmtpp_mse_cont_comp'] = True
if 'rmtpp_mse_var_comp' in model_names:
    #run_model_flags['run_rmtpp_with_joint_optimization_fixed_cnt_solver_mse_var_comp'] = True
    run_model_flags['rmtpp_mse_var_opt_comp'] = True
    run_model_flags['rmtpp_mse_var_cont_comp'] = True
if 'pure_hierarchical_nll' in model_names:
    run_model_flags['run_pure_hierarchical_infer_nll'] = True
if 'pure_hierarchical_mse' in model_names:
    run_model_flags['run_pure_hierarchical_infer_mse'] = True
if 'wgan' in model_names:
    run_model_flags['wgan_simu'] = True
if 'seq2seq' in model_names:
    run_model_flags['seq2seq_simu'] = True
if 'transformer' in model_names:
    run_model_flags['transformer_simu'] = True
if 'hawkes_model' in model_names:
    run_model_flags['hawkes_simu'] = True

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
os.makedirs('Outputs', exist_ok=True)
print("Generating Datasets\n")
generate_dataset()
generate_twitter_dataset(twitter_dataset_names)
print("####################################################################")

event_count_result = OrderedDict()
results = dict()
for dataset_name in dataset_names:
    print("Processing", dataset_name, "Datasets\n")
    args.current_dataset = dataset_name
    if dataset_name == 'Trump':
        args.comp_enc_len = 25
    if automate_bin_sz:
        args.bin_size = utils.get_optimal_bin_size(dataset_name)
        print('New bin size is', args.bin_size, 'sec')
    dataset = utils.get_processed_data(dataset_name, args)

    test_data_out_bin = dataset['test_data_out_bin']
    event_count_preds_true = test_data_out_bin
    count_var = None

    per_model_count = dict()
    per_model_save = {
        'wgan': None,
        'seq2seq': None,
        'transformer': None,
        'count_model': None,
        'hierarchical': None,
        'rmtpp_mse': None,
        'rmtpp_nll': None,
        'rmtpp_mse_var': None,
        'rmtpp_count': None,
    }
    per_model_count['true'] = event_count_preds_true
    for model_name in model_names:
        print("--------------------------------------------------------------------")
        args.current_model = model_name
        print("Running", model_name, "Model\n")

        model, result, rmtpp_var_model, results \
            = run.run_model(dataset_name,
                            model_name,
                            dataset,
                            args,
                            results,
                            prev_models=per_model_save,
                            run_model_flags=run_model_flags)

        if model_name == 'count_model':
            count_var = result['count_var'].numpy()
            result = result['count_preds']

        per_model_count[model_name] = result
        per_model_save[model_name] = model
        if model_name == 'rmtpp_mse' and args.extra_var_model:
            per_model_save['rmtpp_var_model'] = rmtpp_var_model
        print("Finished Running", model_name, "Model\n")

        #if model_name != 'rmtpp_count' and per_model_count[model_name] is not None:
        #    old_stdout = sys.stdout
        #    sys.stdout=open("Outputs/count_model_"+dataset_name+".txt","a")
        #    print("____________________________________________________________________")
        #    print(model_name, 'MAE for Count Prediction:', np.mean(np.abs(per_model_count['true']-per_model_count[model_name])))
        #    print(model_name, 'MAE for Count Prediction (per bin):', np.mean(np.abs(per_model_count['true']-per_model_count[model_name]), axis=0))
        #    print("____________________________________________________________________")
        #    sys.stdout.close()
        #    sys.stdout = old_stdout

        print('Got result', 'for model', model_name, 'on dataset', dataset_name)

    for idx in range(10):
        utils.generate_plots(args, dataset_name, dataset, per_model_count, test_sample_idx=idx, count_var=count_var)

    event_count_result[dataset_name] = per_model_count
    print("####################################################################")


with open('Outputs/results_'+dataset_name+'.txt', 'w') as fp:

    fp.write('\n\nCount MAE and Deep MAE in random interval:')
    fp.write('\nModel Name & Count MAE & Deep MAE & Wass dist & opt_loss & cont_loss & count_loss')
    for model_name, metrics_dict in results.items():
        fp.write(
            '\n & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\'.format(
                model_name,
                metrics_dict['count_mae_rh'],
                metrics_dict['deep_mae_rh'],
                metrics_dict['wass_dist_rh'],
                metrics_dict['opt_loss'],
                metrics_dict['cont_loss'],
                metrics_dict['count_loss'],
            )
        )

    fp.write('\n\nCount MAE and Deep MAE in Forecast Horizon:')
    fp.write('\nModel Name & Count MAE & Deep MAE')
    for model_name, metrics_dict in results.items():
        fp.write(
            '\n & {} & {:.3f} & {:.3f} & {:.3f} \\\\'.format(
                model_name,
                metrics_dict['count_mae_fh'],
                metrics_dict['deep_mae_fh'],
                metrics_dict['wass_dist_fh'],
            )
        )


    fp.write('\n')
    for model_name, metrics_dict in results.items():
        fp.write('\n {}'.format(model_name))
        for metric, val in metrics_dict.items():
            fp.write('\n {}: {:.3f}'.format(metric, val))
        fp.write('\n')


#with open('Outputs/results_'+dataset_name+'.json', 'w') as fp:
#    json.dump(results, fp)
    #results_json = json.dumps(results, indent=4)
    #fp.write(results_json)
