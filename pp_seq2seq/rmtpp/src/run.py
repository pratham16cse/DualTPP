#!/usr/bin/env python
import click
import rmtpp
import tensorflow as tf
import tempfile
from operator import itemgetter
import numpy as np
import os
import json
#import pathos.pools as pp

hidden_layer_size_list = [16, 32, 64, 128]

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
def cmd(dataset_name, alg_name,
        event_train_file, time_train_file, event_dev_file, time_dev_file, event_test_file, time_test_file,
        save_dir, summary_dir, num_epochs, restart, train_eval, test_eval, scale,
        batch_size, bptt, decoder_length, learning_rate, cpu_only, normalization):
    """Read data from EVENT_TRAIN_FILE, TIME_TRAIN_FILE and try to predict the values in EVENT_TEST_FILE, TIME_TEST_FILE."""
    data = rmtpp.utils.read_seq2seq_data(
        event_train_file=event_train_file,
        event_dev_file=event_dev_file,
        event_test_file=event_test_file,
        time_train_file=time_train_file,
        time_dev_file=time_dev_file,
        time_test_file=time_test_file,
        normalization=normalization,
    )

    data['train_time_out_seq'] /= scale
    data['train_time_in_seq'] /= scale
    data['dev_time_out_seq'] /= scale
    data['dev_time_in_seq'] /= scale
    data['test_time_out_seq'] /= scale
    data['test_time_in_seq'] /= scale


    def hyperparameter_worker(params):
        tf.reset_default_graph()
        sess = tf.Session()

        #rmtpp.utils.data_stats(data) #TODO(PD) Need to support seq2seq models.

        hidden_layer_size = params
        rmtpp_mdl = rmtpp.rmtpp_core.RMTPP(
            sess=sess,
            num_categories=data['num_categories'],
            hidden_layer_size=hidden_layer_size, # A hyperparameter
            summary_dir=summary_dir if summary_dir is not None else tempfile.mkdtemp(),
            batch_size=batch_size,
            bptt=data['encoder_length'],
            decoder_length=data['decoder_length'],
            learning_rate=learning_rate,
            cpu_only=cpu_only,
            _opts=rmtpp.rmtpp_core.def_opts
        )

        # TODO: The finalize here has to be false because tf.global_variables()
        # creates a new graph node (why?). Hence, need to be extra careful while
        # saving the model.
        rmtpp_mdl.initialize(finalize=False)
        result = rmtpp_mdl.train(training_data=data, restart=restart,
                                 with_summaries=summary_dir is not None,
                                 num_epochs=num_epochs, with_evals=True)
        # del rmtpp_mdl
        return result

    # TODO(PD) Run hyperparameter tuning in parallel
    #results  = pp.ProcessPool().map(hyperparameter_worker, hidden_layer_size_list)
    results = []
    for hidden_layer_size in hidden_layer_size_list:
        result = hyperparameter_worker((hidden_layer_size))
        results.append(result)
        # print(result['best_test_mae'], result['best_test_acc'])

    best_result_idx, _ = min(enumerate([result['best_dev_mae'] for result in results]), key=itemgetter(1))
    best_result = results[best_result_idx]
    print('best test mae:', best_result['best_test_mae'])
    if save_dir:
        np.savetxt(os.path.join(save_dir)+'/pred.events.out.csv', best_result['best_test_event_preds'], delimiter=',')
        np.savetxt(os.path.join(save_dir)+'/pred.times.out.csv', best_result['best_test_time_preds'], delimiter=',')
        np.savetxt(os.path.join(save_dir)+'/gt.events.out.csv', data['test_event_out_seq'], delimiter=',')
        np.savetxt(os.path.join(save_dir)+'/gt.times.out.csv', data['test_time_out_seq'], delimiter=',')

        with open(os.path.join(save_dir)+'/result.json', 'w') as fp:
            json.dump(best_result, fp, indent=4)

if __name__ == '__main__':
    cmd()

