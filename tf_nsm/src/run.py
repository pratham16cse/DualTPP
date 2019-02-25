#!/usr/bin/env python
import click
import tf_rmtpp
import tensorflow as tf
import tempfile

def_opts = tf_rmtpp.rmtpp_core.def_opts

@click.command()
@click.argument('event_train_file')
@click.argument('time_train_file')
@click.argument('event_test_file')
@click.argument('time_test_file')
@click.option('--summary', 'summary_dir', help='Which folder to save summaries to.', default=None)
@click.option('--epochs', 'num_epochs', help='How many epochs to train for.', default=1)
@click.option('--restart/--no-restart', 'restart', help='Can restart from a saved model from the summary folder, if available.', default=False)
@click.option('--train-eval/--no-train-eval', 'train_eval', help='Should evaluate the model on training data?', default=False)
@click.option('--test-eval/--no-test-eval', 'test_eval', help='Should evaluate the model on test data?', default=True)
@click.option('--scale', 'scale', help='Constant to scale the time fields by.', default=1.0)
@click.option('--batch-size', 'batch_size', help='Batch size.', default=def_opts.batch_size)
@click.option('--bptt', 'bptt', help='Series dependence depth.', default=def_opts.bptt)
@click.option('--init-learning-rate', 'learning_rate', help='Initial learning rate.', default=def_opts.learning_rate)
@click.option('--cpu-only/--no-cpu-only', 'cpu_only', help='Use only the CPU.', default=def_opts.cpu_only)
def cmd(event_train_file, time_train_file, event_test_file, time_test_file,
        summary_dir, num_epochs, restart, train_eval, test_eval, scale,
        batch_size, bptt, learning_rate, cpu_only):
    """Read data from EVENT_TRAIN_FILE, TIME_TRAIN_FILE and try to predict the values in EVENT_TEST_FILE, TIME_TEST_FILE."""
    data = tf_rmtpp.utils.read_data(
        event_train_file=event_train_file,
        event_test_file=event_test_file,
        time_train_file=time_train_file,
        time_test_file=time_test_file
    )

    data['train_time_out_seq'] /= scale
    data['train_time_in_seq'] /= scale
    data['test_time_out_seq'] /= scale
    data['test_time_in_seq'] /= scale

    # Set def_opts.bptt = length of largest sequence
    max_seq_len = max(data['train_time_in_seq'].shape[1], data['test_time_in_seq'].shape[1])
    tf_rmtpp.rmtpp_core.def_opts = def_opts.set('bptt', max_seq_len)
    print('data_shape:', data['train_time_in_seq'].shape[1])
    print('BPTT:', tf_rmtpp.rmtpp_core.def_opts.bptt)

    tf.reset_default_graph()
    sess = tf.Session()

    print('data_stats begin')
    tf_rmtpp.utils.data_stats(data)
    print('data_stats end')

    print('NSM begin')
    nsm_model = tf_rmtpp.rmtpp_core.NSM(
        sess=sess,
        num_categories=data['num_categories'],
        summary_dir=summary_dir if summary_dir is not None else tempfile.mkdtemp(),
        batch_size=batch_size,
        bptt=tf_rmtpp.rmtpp_core.def_opts.bptt,
        learning_rate=learning_rate,
        cpu_only=cpu_only,
        _opts=tf_rmtpp.rmtpp_core.def_opts
        )
    print('NSM end')

    return
    
    print('RMTPP begin')
    rmtpp_mdl = tf_rmtpp.rmtpp_core.RMTPP(
        sess=sess,
        num_categories=data['num_categories'],
        summary_dir=summary_dir if summary_dir is not None else tempfile.mkdtemp(),
        batch_size=batch_size,
        bptt=tf_rmtpp.rmtpp_core.def_opts.bptt,
        learning_rate=learning_rate,
        cpu_only=cpu_only,
        _opts=tf_rmtpp.rmtpp_core.def_opts
    )
    print('RMTPP end')

    # TODO: The finalize here has to be false because tf.global_variables()
    # creates a new graph node (why?). Hence, need to be extra careful while
    # saving the model.
    print('begin init')
    rmtpp_mdl.initialize(finalize=False)
    print('end init')
    rmtpp_mdl.train(training_data=data, restart=restart,
                    with_summaries=summary_dir is not None,
                    num_epochs=num_epochs, with_evals=False)

    if train_eval:
        print('\nEvaluation on training data:')
        train_time_preds, train_event_preds = rmtpp_mdl.predict_train(data=data)
        rmtpp_mdl.eval(train_time_preds, data['train_time_out_seq'],
                       train_event_preds, data['train_event_out_seq'])
        print()

    if test_eval:
        print('\nEvaluation on testing data:')
        test_time_preds, test_event_preds = rmtpp_mdl.predict_test(data=data)
        rmtpp_mdl.eval(test_time_preds, data['test_time_out_seq'],
                       test_event_preds, data['test_event_out_seq'])

    print()


if __name__ == '__main__':
    cmd()



'''
#Run Command 

./run.py arithmatic_prog/event-train.txt arithmatic_prog/time-train.txt arithmatic_prog/event-test.txt arithmatic_prog/time-test.txt --summary ./summ/ --epochs 10 --test-eval --init-learning-rate 0.0001 --cpu-only

'''