#!/usr/bin/env python
import click
import rmtpp
import tensorflow as tf
import tempfile
from operator import itemgetter
import numpy as np
import os
import json
import multiprocessing as MP
from itertools import product
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#import pathos.pools as pp

hidden_layer_size_list = [16, 32, 64, 128]
hparams = json.loads(open('hparams.json', 'r').read())
hparams_aliases = json.loads(open('hparams_aliases.json', 'r').read())

def_opts = rmtpp.rmtpp_core.def_opts

@click.command()
@click.argument('dataset_name')
@click.argument('alg_name')
@click.argument('event_train_file')
@click.argument('time_train_file')
@click.argument('event_dev_file')
@click.argument('time_dev_file')
@click.argument('event_test_file')
@click.argument('time_test_file')
@click.option('--dataset_path', 'dataset_path', help='Dataset path.', default=None)
@click.option('--summary', 'summary_dir', help='Which folder to save summaries to.', default=None)
@click.option('--save', 'save_dir', help='Which folder to save checkpoints to.', default=None)
@click.option('--epochs', 'num_epochs', help='How many epochs to train for.', default=1)
@click.option('--restart/--no-restart', 'restart', help='Can restart from a saved model from the summary folder, if available.', default=False)
@click.option('--train-eval/--no-train-eval', 'train_eval', help='Should evaluate the model on training data?', default=False)
@click.option('--test-eval/--no-test-eval', 'test_eval', help='Should evaluate the model on test data?', default=True)
@click.option('--scale', 'scale', help='Constant to scale the time fields by.', default=1.0)
@click.option('--batch-size', 'batch_size', help='Batch size.', default=def_opts.batch_size)
@click.option('--bptt', 'bptt', help='Series dependence depth.', default=def_opts.bptt)
@click.option('--decoder-len', 'decoder_length', help='Number of events to predict in future', default=def_opts.decoder_length)
@click.option('--init-learning-rate', 'learning_rate', help='Initial learning rate.', default=def_opts.learning_rate)
@click.option('--cpu-only/--no-cpu-only', 'cpu_only', help='Use only the CPU.', default=def_opts.cpu_only)
@click.option('--normalization', 'normalization', help='The normalization technique', default=def_opts.normalization)
@click.option('--constraints', 'constraints', help='Constraints over wt and D (or any other values), refer to constraints.json', default=def_opts.constraints)
@click.option('--patience', 'patience', help='Number of epochs to wait before applying stop_criteria for training', default=def_opts.patience)
@click.option('--stop-criteria', 'stop_criteria', help='Stopping criteria: per_epoch_val_err or epsilon', default=def_opts.stop_criteria)
@click.option('--epsilon', 'epsilon', help='threshold for epsilon-stopping-criteria', default=def_opts.epsilon)
@click.option('--num-extra-layer', 'num_extra_layer', help='Number of extra layer on top of hidden state ', default=def_opts.num_extra_layer)
@click.option('--mark-loss/--no-mark-loss', 'mark_loss', help='If true, mark_LL is also added to the loss', default=def_opts.mark_loss)
@click.option('--parallel-hparam/--no-parallel-hparam', 'parallel_hparam', help='If true, hparam will run in parallel', default=True)
def cmd(dataset_name, alg_name, dataset_path,
        event_train_file, time_train_file, event_dev_file, time_dev_file, event_test_file, time_test_file,
        save_dir, summary_dir, num_epochs, restart, train_eval, test_eval, scale,
        batch_size, bptt, decoder_length, learning_rate, cpu_only, normalization, constraints,
        patience, stop_criteria, epsilon, num_extra_layer, mark_loss, parallel_hparam):
    """Read data from EVENT_TRAIN_FILE, TIME_TRAIN_FILE and try to predict the values in EVENT_TEST_FILE, TIME_TEST_FILE."""

    data = rmtpp.utils.read_seq2seq_data(
        event_train_file=event_train_file,
        event_dev_file=event_dev_file,
        event_test_file=event_test_file,
        time_train_file=time_train_file,
        time_dev_file=time_dev_file,
        time_test_file=time_test_file,
        normalization=normalization,
        dataset_path=dataset_path
    )

    print('scaling with ', scale)
    data['train_time_out_seq'] /= scale
    data['train_time_in_seq'] /= scale
    data['dev_time_out_seq'] /= scale
    data['dev_time_in_seq'] /= scale
    data['test_time_out_seq'] /= scale
    data['test_time_in_seq'] /= scale

    def model_creator(params):
        tf.reset_default_graph()
        sess = tf.Session()

        #rmtpp.utils.data_stats(data) #TODO(PD) Need to support seq2seq models.

        params_named, restart, num_epochs, save_dir, train_eval = params
        def_opts_local = def_opts
        print('params_named start')
        params_named = [(name, val) for name, val in params_named]
        params_alias_named = [(hparams_aliases[name], val) for name, val in params_named]
        #params_named_print_ = params_named
        #for name, val in params_named_print_:
        #    print(name, val, '-------------')
        #    def_opts_local = def_opts_local.set(name, val)
        #print('params_named end')
        rmtpp_mdl = rmtpp.rmtpp_core.RMTPP(
            sess=sess,
            num_categories=data['num_categories'],
            params_named=params_named,
            params_alias_named=params_alias_named,
            #hidden_layer_size=hidden_layer_size, # A hyperparameter
            alg_name=alg_name,
            save_dir=save_dir,
            summary_dir=summary_dir,
            batch_size=batch_size,
            bptt=data['encoder_length'],
            decoder_length=data['decoder_length'],
            #learning_rate=learning_rate,
            cpu_only=cpu_only,
            constraints=constraints,
            #wt_hparam=wt_hparam,
            patience=patience,
            stop_criteria=stop_criteria,
            epsilon=epsilon,
            #num_extra_layer=num_extra_layer,
            mark_loss=mark_loss,
            _opts=def_opts_local
        )

        return rmtpp_mdl

    def hyperparameter_worker(params, rmtpp_mdl, dec_len, restore_path=None):
        
        params_named, restart, num_epochs, save_dir, train_eval = params
        # TODO: The finalize here has to be false because tf.global_variables()
        # creates a new graph node (why?). Hence, need to be extra careful while
        # saving the model.
        rmtpp_mdl.initialize(finalize=False)
        result = rmtpp_mdl.train(training_data=data, restart=restart,
                                        with_summaries=summary_dir is not None,
                                        num_epochs=num_epochs, eval_train_data=train_eval,
                                        restore_path=restore_path, dec_len_for_eval=dec_len)

        with open('constraints.json', 'r') as fp:
            constraints_json = json.loads(fp.read())
            result['constraints'] = constraints_json[constraints]

        # del rmtpp_mdl
        return result

    decoder_length_run = [0, 1, 2, 3]

    old_save_dir = save_dir
    for dec_len in decoder_length_run:
        save_dir = old_save_dir+'/'+str(dec_len)
        if dec_len != decoder_length_run[0]:
            num_epochs = -1
            stop_criteria = 'epsilon'

        if os.path.isfile(save_dir+'/result.json'):
            print('Model already trained, stored, restoring . . .')
            with open(os.path.join(save_dir)+'/result.json', 'r') as fp:
                result = json.loads(fp.read())
            params = (result[param] for param in hparams[alg_name].keys())
            args = (result['hidden_layer_size'], result['wt_hparam'], True, 0, result['checkpoint_dir'], train_eval)
            rmtpp_mdl = model_creator(args)
            result = hyperparameter_worker(args, rmtpp_mdl, dec_len)
        else:
            # TODO(PD) Run hyperparameter tuning in parallel
            #results  = pp.ProcessPool().map(hyperparameter_worker, hidden_layer_size_list)
            results = []

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            if parallel_hparam:
                global hparam_loop
                def hparam_loop(all_params):
                    loop_idx, (param_names, params) = all_params
                    print('params', params, loop_idx)
                    checkp_dir = old_save_dir+'/'+str(decoder_length_run[0])
                    checkp_dir_hparams = ''
                    params_named = zip(param_names, params)
                    for name, val in params_named:
                        print(name, hparams_aliases[name], val)
                        checkp_dir_hparams = checkp_dir_hparams+hparams_aliases[name]+'_'+str(val)+'_'
                    checkp_dir = checkp_dir + '/' + checkp_dir_hparams
                    state_restart=True
                    if dec_len==decoder_length_run[0]:
                        #checkp_dir=None
                        state_restart = False
                    print("check dir ", checkp_dir)
                    params_named = tuple([zip(param_names, params)])
                    args = (params_named) + (state_restart, num_epochs, save_dir, train_eval)
                    rmtpp_mdl = model_creator(args)
                    result = hyperparameter_worker(args, rmtpp_mdl, dec_len, checkp_dir)
                    for name, val in zip(param_names, params):
                        result[name] = val
                    return result
                    # results.append(result)

                param_names, param_values = zip(*hparams[alg_name].items())

                param_values_lst=list()
                param_names_lst=list()
                for element in product(*param_values):
                    param_values_lst.append(element)
                    param_names_lst.append(param_names)

                with MP.Pool() as pool:
                        results = pool.map(hparam_loop, enumerate(zip(param_names_lst, param_values_lst)))

            else:
                param_names, param_values = zip(*hparams[alg_name].items())
                for params in product(*param_values):
                    checkp_dir = old_save_dir+'/'+str(decoder_length_run[0])
                    checkp_dir_hparams = ''
                    params_named = zip(param_names, params)
                    for name, val in params_named:
                        print(name, hparams_aliases[name], val)
                        checkp_dir_hparams = checkp_dir_hparams+hparams_aliases[name]+'_'+str(val)+'_'
                    checkp_dir = checkp_dir + '/' + checkp_dir_hparams
                    state_restart=True
                    if dec_len==decoder_length_run[0]:
                        #checkp_dir=None
                        state_restart = False
                    print("check dir ", checkp_dir)
                    params_named = tuple([zip(param_names, params)])
                    args = (params_named) + (state_restart, num_epochs, save_dir, train_eval)
                    rmtpp_mdl = model_creator(args)
                    result = hyperparameter_worker(args, rmtpp_mdl, dec_len, checkp_dir)
                    for name, val in zip(param_names, params):
                        result[name] = val
                    results.append(result)
                    # print(result['best_test_mae'], result['best_test_acc'])

            best_result_idx, _ = min(enumerate([result['best_dev_mae'] for result in results]), key=itemgetter(1))
            best_result = results[best_result_idx]
            for param in hparams[alg_name].keys(): assert param in best_result.keys() # Check whether all hyperparameters are stored
            print('best test mae:', best_result['best_test_mae'])
            if save_dir:
                np.savetxt(os.path.join(save_dir)+'/test.pred.events.out.csv', best_result['best_test_event_preds'], delimiter=',')
                np.savetxt(os.path.join(save_dir)+'/test.pred.times.out.csv', best_result['best_test_time_preds'], delimiter=',')
                np.savetxt(os.path.join(save_dir)+'/test.gt.events.out.csv', data['test_event_out_seq'][:,:dec_len], delimiter=',')
                np.savetxt(os.path.join(save_dir)+'/test.gt.times.out.csv', np.array(data['test_actual_time_out_seq'])[:,:dec_len], delimiter=',')

                np.savetxt(os.path.join(save_dir)+'/dev.pred.events.out.csv', best_result['best_dev_event_preds'], delimiter=',')
                np.savetxt(os.path.join(save_dir)+'/dev.pred.times.out.csv', best_result['best_dev_time_preds'], delimiter=',')
                np.savetxt(os.path.join(save_dir)+'/dev.gt.events.out.csv', data['dev_event_out_seq'][:,:dec_len], delimiter=',')
                np.savetxt(os.path.join(save_dir)+'/dev.gt.times.out.csv', np.array(data['dev_actual_time_out_seq'])[:,:dec_len], delimiter=',')

                for result in results:
                    del result['best_train_event_preds'], result['best_train_time_preds'], \
                            result['best_dev_event_preds'], result['best_dev_time_preds'], \
                            result['best_test_event_preds'], result['best_test_time_preds']
                with open(os.path.join(save_dir)+'/result.json', 'w') as fp:
                    best_result_json = json.dumps(best_result, indent=4)
                    fp.write(best_result_json)
                with open(os.path.join(save_dir)+'/all_results.json', 'w') as fp:
                    all_results_json = json.dumps(results, indent=4)
                    fp.write(all_results_json)

                plt.plot(best_result['train_loss_list'])
                plt.ylabel('train_loss')
                plt.xlabel('epoch')
                plt.savefig(os.path.join(save_dir, 'train_loss.png'))
                plt.close()
                plt.plot(best_result['train_time_loss_list'])
                plt.ylabel('train_time_loss')
                plt.xlabel('epoch')
                plt.savefig(os.path.join(save_dir, 'train_time_loss.png'))
                plt.close()
                plt.plot(best_result['train_mark_loss_list'])
                plt.ylabel('train_mark_loss')
                plt.xlabel('epoch')
                plt.savefig(os.path.join(save_dir, 'train_mark_loss.png'))
                plt.close()
                plt.plot(best_result['wt_list'])
                plt.ylabel('wt')
                plt.xlabel('epoch')
                plt.savefig(os.path.join(save_dir, 'wt.png'))
                plt.close()


if __name__ == '__main__':
    cmd()

