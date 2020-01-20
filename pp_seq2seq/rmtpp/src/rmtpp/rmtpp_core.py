import tensorflow as tf
import numpy as np
import os
import decorated_options as Deco
from .utils import create_dir, variable_summaries, MAE, RMSE, ACC, MRR, PERCENT_ERROR, DTW
from .utils import get_output_indices
from scipy.integrate import quad
import multiprocessing as MP
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
from collections import OrderedDict, Counter
from operator import itemgetter

trim_seq_dec_len = lambda sequences, dec_len: [seq[:dec_len] for seq in sequences]
ETH = 10.0
clip = lambda D: tf.clip_by_value(D, -10.0, 10.0)
__EMBED_SIZE = 4
__HIDDEN_LAYER_SIZE = 16  # 64, 128, 256, 512, 1024
epsilon = 0.1

def_opts = Deco.Options(
    params_alias_named={},

    hidden_layer_size=16,

    batch_size=64,          # 16, 32, 64

    learning_rate=0.1,      # 0.1, 0.01, 0.001
    momentum=0.9,
    decay_steps=100,
    decay_rate=0.001,

    l2_penalty=0.001,         # Unused

    float_type=tf.float32,

    seed=42,
    scope='RMTPP',
    alg_name='rmtpp',
    save_dir='./save.rmtpp/',
    summary_dir='./summary.rmtpp/',

    device_gpu='/gpu:0',
    device_cpu='/cpu:0',

    bptt=20,
    decoder_length=5,
    cpu_only=False,

    normalization=None,
    constraints="default",

    patience=0,
    stop_criteria='per_epoch_val_err',
    epsilon=0.0001,

    num_extra_layer=0,
    mark_loss=True,
    plot_pred_dev=True,
    plot_pred_test=False,
    num_feats=1,
    use_time_features=False,

    max_offset=0.0,

    wt_hparam=1.0,

    rnn_cell_type='manual',

    embed_size=__EMBED_SIZE,
    Wem=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,
    Wem_feats=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,

    Wt=lambda hidden_layer_size: np.random.normal(size=(1, hidden_layer_size)),
    Wh=lambda hidden_layer_size: np.random.normal(size=(hidden_layer_size)),
    bh=lambda hidden_layer_size: np.random.normal(size=(1, hidden_layer_size)),
    Ws=lambda hidden_layer_size: np.random.normal(size=(hidden_layer_size)),
    bs=lambda hidden_layer_size: np.random.normal(size=(1, hidden_layer_size)),
    wt=1.0,
    Wy=lambda hidden_layer_size: np.random.normal(size=(__EMBED_SIZE, hidden_layer_size)),
    Vy=lambda hidden_layer_size, num_categories: np.random.normal(size=(hidden_layer_size, num_categories)),
    Vt=lambda hidden_layer_size, decoder_length: np.random.normal(size=(decoder_length, hidden_layer_size)),
    Vw=lambda hidden_layer_size, decoder_length: np.random.normal(size=(decoder_length, hidden_layer_size)),
    bt=lambda decoder_length: np.random.normal(size=(decoder_length)), # bt is provided by the base_rate
    bw=lambda decoder_length: np.random.normal(size=(decoder_length)), # bw is provided by the base_rate
    bk=lambda hidden_layer_size, num_categories: np.random.normal(size=(1, num_categories)),
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softplus(x):
    """Numpy counterpart to tf.nn.softplus"""
    return x + np.log1p(np.exp(-x)) if x>=0.0 else np.log1p(np.exp(x))

#def softplus(x):
#    """Numpy counterpart to tf.nn.softplus"""
#    return np.log1p(np.exp(x))

def quad_func(t, c, w):
    """This is the t * f(t) function calculating the mean time to next event,
    given c, w."""
    return c * t * np.exp(w * t - (c / w) * (np.exp(w * t) - 1))

def density_func(t, c, w):
    """This is the t * f(t) function calculating the mean time to next event,
    given c, w."""
    return c * np.exp(w * t - (c / w) * (np.exp(w * t) - 1))

def quad_func_splusintensity(t, D, w):
    return t * (D + w * t) * np.exp(-(D * t) - (w * t * t) / 2)

class RMTPP:
    """Class implementing the Recurrent Marked Temporal Point Process model."""

    @Deco.optioned()
    def __init__(self, sess, num_categories, params_named, params_alias_named, hidden_layer_size, batch_size,
                 learning_rate, momentum, l2_penalty, embed_size,
                 float_type, bptt, decoder_length, seed, scope, alg_name,
                 save_dir, decay_steps, decay_rate,
                 device_gpu, device_cpu, summary_dir, cpu_only, constraints,
                 patience, stop_criteria, epsilon, num_extra_layer, mark_loss,
                 Wt, Wem, Wem_feats, Wh, bh, Ws, bs, wt, Wy, Vy, Vt, Vw, bk, bt, bw, wt_hparam,
                 plot_pred_dev, plot_pred_test, rnn_cell_type, num_feats, use_time_features,
                 max_offset):

        self.seed = seed
        tf.set_random_seed(self.seed)

        self.PARAMS_NAMED = OrderedDict(params_named)
        self.PARAMS_ALIAS_NAMED = params_alias_named

        self.HIDDEN_LAYER_SIZE = self.PARAMS_NAMED['hidden_layer_size'] #hidden_layer_size
        self.NUM_RNN_LAYERS = self.PARAMS_NAMED['num_rnn_layers']
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = self.PARAMS_NAMED['learning_rate'] #learning_rate
        self.MOMENTUM = momentum
        self.L2_PENALTY = l2_penalty
        self.EMBED_SIZE = embed_size
        self.BPTT = bptt
        self.DEC_LEN = decoder_length
        self.ALG_NAME = alg_name
        self.SAVE_DIR = save_dir
        self.SUMMARY_DIR = summary_dir
        self.CONSTRAINTS = constraints

        self.NUM_CATEGORIES = num_categories
        self.FLOAT_TYPE = float_type

        self.DEVICE_CPU = device_cpu
        self.DEVICE_GPU = device_gpu

        self.wt_hparam = self.PARAMS_NAMED['wt_hparam'] #wt_hparam

        self.RNN_CELL_TYPE = rnn_cell_type

        self.RNN_REG_PARAM = self.PARAMS_NAMED['rnn_reg_param'] #rnn_reg_param
        self.EXTLYR_REG_PARAM = self.PARAMS_NAMED['extlyr_reg_param'] #extlyr_reg_param
        print('RNN_REG_PARAM:', self.RNN_REG_PARAM)
        print('EXTLYR_REG_PARAM:', self.EXTLYR_REG_PARAM)
        self.RNN_REGULARIZER = tf.contrib.layers.l2_regularizer(scale=self.RNN_REG_PARAM)
        self.EXTLYR_REGULARIZER = tf.contrib.layers.l2_regularizer(scale=self.EXTLYR_REG_PARAM)

        self.PATIENCE = patience
        self.STOP_CRITERIA = stop_criteria
        self.EPSILON = epsilon
        if self.STOP_CRITERIA == 'epsilon':
            assert self.EPSILON > 0.0

        self.NUM_EXTRA_LAYER = self.PARAMS_NAMED['num_extra_layer'] #num_extra_layer
        self.MARK_LOSS = mark_loss
        self.PLOT_PRED_DEV = plot_pred_dev
        self.PLOT_PRED_TEST = plot_pred_test

        self.NUM_FEATS = num_feats
        self.USE_TIME_FEATS = use_time_features

        self.MAX_OFFSET = max_offset

        if True:
            self.DEC_STATE_SIZE = 2 * self.HIDDEN_LAYER_SIZE

        self.sess = sess
        self.last_epoch = 0

        self.rs = np.random.RandomState(self.seed)
        np.random.seed(self.seed)

        def get_wt_constraint():
            if self.CONSTRAINTS == 'default':
                return lambda x: tf.clip_by_value(x, 1e-5, 20.0)
                #return lambda x: tf.clip_by_value(x, 1e-5, np.inf)
            elif self.CONSTRAINTS == 'c1':
                return lambda x: tf.clip_by_value(x, 1.0, np.inf)
            elif self.CONSTRAINTS == 'c2':
                return lambda x: tf.clip_by_value(x, 1e-5, np.inf)
            elif self.CONSTRAINTS == 'unconstrained':
                return lambda x: x
            else:
                print('Constraint on wt not found.')
                assert False

        def get_D_constraint():
            if self.CONSTRAINTS == 'default':
                return lambda x: x
            elif self.CONSTRAINTS in ['c1', 'c2']:
                return lambda x: -tf.nn.softplus(-x)
            elif self.CONSTRAINTS == 'unconstrained':
                return lambda x: x
            else:
                print('Constraint on wt not found.')
                assert False

        def get_WT_constraint():
            if self.CONSTRAINTS == 'default':
                return lambda x: tf.clip_by_value(x, 1e-5, np.inf)
            elif self.CONSTRAINTS in ['c1', 'c2']:
                return lambda x: tf.nn.softplus(x)
            else:
                print('Constraint on wt not found.')
                assert False


        with tf.variable_scope(scope):
            with tf.device(device_gpu if not cpu_only else device_cpu):
                # Make input variables
                self.events_in = tf.placeholder(tf.int32, [None, self.BPTT], name='events_in')
                self.times_in = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_in')

                self.events_out = tf.placeholder(tf.int32, [None, self.BPTT], name='events_out')
                self.times_out = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_out')

                #TODO make use of self.NUM_FEATS flag to add multiple features
                self.times_in_feats = tf.placeholder(tf.int32, [None, self.BPTT], name='times_in_feats')

                self.mode = tf.placeholder(tf.float32, name='mode')

                self.batch_num_events = tf.placeholder(self.FLOAT_TYPE, [], name='bptt_events')

                self.inf_batch_size = tf.shape(self.events_in)[0]

                # Make variables
                with tf.variable_scope('hidden_state'):
                    self.Wt = tf.get_variable(name='Wt',
                                              shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(Wt(self.HIDDEN_LAYER_SIZE)))

                    self.Wem = tf.get_variable(name='Wem', shape=(self.NUM_CATEGORIES, self.EMBED_SIZE),
                                               dtype=self.FLOAT_TYPE,
                                               regularizer=self.RNN_REGULARIZER,
                                               initializer=tf.constant_initializer(Wem(self.NUM_CATEGORIES)))
                    self.Wem_feats = tf.get_variable(name='Wem_feats', shape=(24, self.EMBED_SIZE),
                                                     dtype=self.FLOAT_TYPE,
                                                     regularizer=self.RNN_REGULARIZER,
                                                     initializer=tf.constant_initializer(Wem_feats(24)))
                    self.Wh = tf.get_variable(name='Wh', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(Wh(self.HIDDEN_LAYER_SIZE)))
                    self.bh = tf.get_variable(name='bh', shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(bh(self.HIDDEN_LAYER_SIZE)))

                with tf.variable_scope('decoder_state'):
                    self.Ws = tf.get_variable(name='Ws', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                                  dtype=self.FLOAT_TYPE,
                                                  regularizer=self.RNN_REGULARIZER,
                                                  initializer=tf.constant_initializer(Ws(self.HIDDEN_LAYER_SIZE)))
                    self.bs = tf.get_variable(name='bs', shape=(1, self.HIDDEN_LAYER_SIZE),
                                                  dtype=self.FLOAT_TYPE,
                                                  regularizer=self.RNN_REGULARIZER,
                                                  initializer=tf.constant_initializer(bs(self.HIDDEN_LAYER_SIZE)))


                with tf.variable_scope('output'):
                    self.wt = tf.get_variable(name='wt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(wt),
                                              constraint=get_wt_constraint())

                    self.Wy = tf.get_variable(name='Wy', shape=(self.EMBED_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(Wy(self.HIDDEN_LAYER_SIZE)))

                    # The first column of Vy is merely a placeholder (will not be trained).
                    self.Vy = tf.get_variable(name='Vy', shape=(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(Vy(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES)))
                    self.Vt = tf.get_variable(name='Vt', shape=(self.DEC_LEN, self.DEC_STATE_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(Vt(self.DEC_STATE_SIZE, self.DEC_LEN)))
                    self.bt = tf.get_variable(name='bt', shape=(1, self.DEC_LEN),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(bt(self.DEC_LEN)))
                    self.bk = tf.get_variable(name='bk', shape=(1, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(bk(self.DEC_STATE_SIZE, num_categories)))
                    self.Vw = tf.get_variable(name='Vw', shape=(self.DEC_LEN, self.DEC_STATE_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(Vw(self.DEC_STATE_SIZE, self.DEC_LEN)))
                    self.bw = tf.get_variable(name='bw', shape=(1, self.DEC_LEN),
                                              dtype=self.FLOAT_TYPE,
                                              regularizer=self.RNN_REGULARIZER,
                                              initializer=tf.constant_initializer(bw(self.DEC_LEN)))

                    if True:
                        print('Always Sharing pseudo-Decoder Parameters')
                        self.Vt = tf.transpose(self.Vt[0:1, :self.HIDDEN_LAYER_SIZE], [1, 0])
                        self.bt = self.bt[:, 0:1]
                        self.Vw = tf.transpose(self.Vw[0:1, :self.HIDDEN_LAYER_SIZE], [1, 0])
                        self.bw = self.bw[:, 0:1]

                self.all_vars = [self.Wt, self.Wem, self.Wem_feats, self.Wh, self.bh, self.Ws, self.bs,
                                 self.wt, self.Wy, self.Vy, self.Vt, self.bt, self.bk, self.Vw, self.bw]

                # Add summaries for all (trainable) variables
                with tf.device(device_cpu):
                    for v in self.all_vars:
                        variable_summaries(v)

                # Make graph
                # RNNcell = RNN_CELL_TYPE(HIDDEN_LAYER_SIZE)
                if self.RNN_CELL_TYPE == 'lstm':
                    cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE, forget_bias=1.0)
                    self.cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.NUM_RNN_LAYERS)],
                                                            state_is_tuple=True)
                    internal_state = self.cell.zero_state(self.inf_batch_size, dtype = tf.float32)

                # Initial state for GRU cells
                self.initial_state = state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE],
                                                      dtype=self.FLOAT_TYPE,
                                                      name='initial_state')
                self.initial_time = last_time = tf.zeros((self.inf_batch_size,),
                                                         dtype=self.FLOAT_TYPE,
                                                         name='initial_time')

                self.loss, self.time_loss, self.mark_loss = 0.0, 0.0, 0.0
                ones_2d = tf.ones((self.inf_batch_size, 1), dtype=self.FLOAT_TYPE)
                # ones_1d = tf.ones((self.inf_batch_size,), dtype=self.FLOAT_TYPE)

                self.hidden_states = []
                self.D_list = []
                self.event_preds = []

                self.time_LLs = []
                self.mark_LLs = []
                self.log_lambdas = []
                # self.delta_ts = []
                self.times = []

                with tf.name_scope('BPTT'):
                    for i in range(self.BPTT):

                        events_embedded = tf.nn.embedding_lookup(self.Wem,
                                                                 tf.mod(self.events_in[:, i] - 1, self.NUM_CATEGORIES))
                        time = self.times_in[:, i]
                        time_next = self.times_out[:, i]
                        time_feat_embd = self.embedFeatures(self.times_in_feats[:, i])

                        delta_t_prev = tf.expand_dims(time - last_time, axis=-1)
                        delta_t_next = tf.expand_dims(time_next - time, axis=-1)

                        last_time = time

                        time_2d = tf.expand_dims(time, axis=-1)

                        # output, state = RNNcell(events_embedded, state)

                        # TODO Does TF automatically broadcast? Then we'll not
                        # need multiplication with tf.ones
                        type_delta_t = True

                        with tf.variable_scope('state_recursion', reuse=tf.AUTO_REUSE):

                            if self.RNN_CELL_TYPE == 'manual':
                                raise NotImplemented('Manual RNN disabled')
                                new_state = tf.tanh(
                                    tf.matmul(state, self.Wh) +
                                    tf.matmul(events_embedded, self.Wy) +
                                    # Two ways of interpretting this term
                                    (tf.matmul(delta_t_prev, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                                    tf.matmul(ones_2d, self.bh),
                                    name='h_t'
                                )
                            elif self.RNN_CELL_TYPE == 'lstm':
                                inputs = tf.concat([state, events_embedded, delta_t_prev], axis=-1)
                                if self.USE_TIME_FEATS:
                                    inputs = tf.concat([inputs, time_feat_embd], axis=-1)
                                new_state, internal_state = self.cell(inputs,  internal_state)

                            if self.NUM_EXTRA_LAYER:
                                names = ['hidden_layer_'+str(hl_id) for hl_id in range(1, self.NUM_EXTRA_LAYER+1)]
                                for name in names:
                                    new_state = tf.layers.dense(new_state,
                                                                self.HIDDEN_LAYER_SIZE,
                                                                name=name,
                                                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                                                                activation=tf.nn.relu,
                                                                kernel_regularizer=self.EXTLYR_REGULARIZER,
                                                                bias_regularizer=self.EXTLYR_REGULARIZER)

                            state = tf.where(self.events_in[:, i] > 0, new_state, state)

                        with tf.name_scope('loss_calc'):
                            base_intensity_bt = tf.matmul(ones_2d, self.bt)
                            base_intensity_bw = tf.matmul(ones_2d, self.bw)
                            # wt_non_zero = tf.sign(self.wt) * tf.maximum(1e-9, tf.abs(self.wt))
                            #base_intensity_bt = tf.Print(base_intensity_bt, [base_intensity_bt, self.Vt], message='Printing Vt and bt')
                            self.D = tf.matmul(state, self.Vt) + base_intensity_bt
                            self.D = get_D_constraint()(self.D)
                            #self.D = tf.Print(self.D, [self.D], message='Printing D Before')
                            if self.ALG_NAME in ['rmtpp_splusintensity']:
                                self.D = tf.nn.softplus(self.D)
                            #self.D = tf.Print(self.D, [self.D], message='Printing D After')

                            if self.ALG_NAME in ['rmtpp_wcmpt', 'rmtpp_mode_wcmpt']:
                                self.WT = tf.matmul(state, self.Vw) + base_intensity_bw
                                self.WT = get_WT_constraint()(self.WT)
                                self.WT = tf.clip_by_value(self.WT, 0.0, 10.0)
                            elif self.ALG_NAME in ['rmtpp', 'rmtpp_mode', 'rmtpp_splusintensity',
                                                   'zero_pred', 'average_gap_pred', 'average_tr_gap_pred']:
                                self.WT = self.wt
                            elif self.ALG_NAME in ['rmtpp_negw', 'rmtpp_splusintensity_negw']:
                                self.WT = -1.0 * self.wt
                            elif self.ALG_NAME in ['rmtpp_whparam', 'rmtpp_mode_whparam']:
                                self.WT = self.wt_hparam

                            if self.ALG_NAME in ['rmtpp_splusintensity', 'rmtpp_splusintensity_negw']:
                                lambda_ = self.D + (delta_t_next * self.WT)
                                log_lambda_ = tf.log(lambda_)
                                log_f_star = (log_lambda_
                                              - self.D * delta_t_next
                                              - (self.WT/2.0) * tf.square(delta_t_next))
                            elif self.ALG_NAME in ['rmtpp', 'rmtpp_negw']:
                                log_lambda_ = (self.D + (delta_t_next * self.WT))
                                lambda_ = tf.exp(tf.minimum(ETH, log_lambda_), name='lambda_')
                                log_f_star = (log_lambda_
                                              + (1.0 / self.WT) * tf.exp(tf.minimum(ETH, self.D))
                                              - (1.0 / self.WT) * lambda_)

                            events_pred = tf.nn.softmax(
                                tf.minimum(ETH,
                                           tf.matmul(state, self.Vy) + ones_2d * self.bk),
                                name='Pr_events'
                            )

                            time_LL = log_f_star
                            mark_LL = tf.expand_dims(
                                tf.log(
                                    tf.maximum(
                                        1e-6,
                                        tf.gather_nd(
                                            events_pred,
                                            tf.concat([
                                                tf.expand_dims(tf.range(self.inf_batch_size), -1),
                                                tf.expand_dims(tf.mod(self.events_out[:, i] - 1, self.NUM_CATEGORIES), -1)
                                            ], axis=1, name='Pr_next_event'
                                            )
                                        )
                                    )
                                ), axis=-1, name='log_Pr_next_event'
                            )
                            step_LL = time_LL + mark_LL

                            # In the batch some of the sequences may have ended before we get to the
                            # end of the seq. In such cases, the events will be zero.
                            # TODO Figure out how to do this with RNNCell, LSTM, etc.
                            num_events = tf.reduce_sum(tf.where(self.events_in[:, i] > 0,
                                                       tf.ones(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE),
                                                       tf.zeros(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE)),
                                                       name='num_events')

                            if i + self.DEC_LEN > self.BPTT - 1:
                                self.loss -= tf.reduce_sum(
                                    tf.where(self.events_in[:, i] > 0,
                                             tf.squeeze(step_LL),
                                             tf.zeros(shape=(self.inf_batch_size,)))
                                )
                                self.time_loss -= tf.reduce_sum(
                                    tf.where(self.events_in[:, i] > 0,
                                             tf.squeeze(time_LL),
                                             tf.zeros(shape=(self.inf_batch_size,)))
                                )
                                self.mark_loss -= tf.reduce_sum(
                                    tf.where(self.events_in[:, i] > 0,
                                             tf.squeeze(mark_LL),
                                             tf.zeros(shape=(self.inf_batch_size,)))
                                )

                            # self.loss -= tf.cond(num_events > 0,
                            #                      lambda: tf.reduce_sum(
                            #                          tf.where(self.events_in[:, i] > 0,
                            #                                   tf.squeeze(step_LL),
                            #                                   tf.zeros(shape=(self.inf_batch_size,))),
                            #                          name='batch_bptt_loss'),
                            #                      lambda: 0.0)

                        self.time_LLs.append(time_LL)
                        self.mark_LLs.append(mark_LL)
                        self.log_lambdas.append(log_lambda_)

                        self.hidden_states.append(state)
                        self.D_list.append(self.D)
                        self.event_preds.append(events_pred)

                        # self.delta_ts.append(tf.clip_by_value(delta_t, 0.0, np.inf))
                        self.times.append(time)

                    self.D_list = tf.stack(self.D_list, axis=1)

                    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    print('REGULARIZATION VARIABLES:', reg_variables)
                    if self.RNN_REG_PARAM:
                        reg_term = tf.contrib.layers.apply_regularization(self.RNN_REGULARIZER, reg_variables)
                    if self.EXTLYR_REG_PARAM:
                        reg_term_dense_layers = tf.losses.get_regularization_loss()

                    if self.RNN_REG_PARAM:
                        self.loss = self.loss + reg_term
                    if self.EXTLYR_REG_PARAM:
                        self.loss = self.loss + reg_term_dense_layers


                self.final_state = self.hidden_states[-1]

                # ----- Prediction using Inverse Transform Sampling ----- #
                #u = tf.random.uniform((self.inf_batch_size, 5000), minval=0.0, maxval=0.1, seed=self.seed)
                if self.ALG_NAME in ['rmtpp', 'rmtpp_splusintensity']:
                    u = tf.ones((self.inf_batch_size, self.BPTT, 1)) * tf.range(0.0, 1.0, 1.0/5000)
                elif self.ALG_NAME in ['rmtpp_negw', 'rmtpp_splusintensity_negw']:
                    lim = 1 - tf.exp((tf.exp(clip(self.D_list)))/self.WT)
                    u = tf.ones((self.inf_batch_size, self.BPTT, 1)) * tf.range(0.0, 0.99, 0.99/5000)
                    u  = u * lim

                if self.ALG_NAME in ['rmtpp', 'rmtpp_negw']:
                    c = -tf.exp(clip(self.D_list))
                    self.val = (1.0/self.WT) * tf.log((self.WT/c) * tf.log(1.0 - u) + 1)
                    #self.val = tf.Print(self.val, [tf.shape(self.val), tf.log((self.WT/c) * tf.log(1.0 - u) + 1), 1.0/self.WT], message='Printing log and 1/w')
                    #self.val = tf.Print(self.val, [tf.reduce_sum(tf.cast(tf.is_finite(((self.WT/c) * tf.log(1.0 - u))), tf.int32)), 1.0/self.WT], message='Printing log and 1/w')
                    #self.val = tf.Print(self.val, [tf.reduce_sum(tf.cast(tf.is_finite((self.WT/c)), tf.int32)), 1.0/self.WT], message='Printing log and 1/w')
                    #self.val = tf.Print(self.val, [tf.reduce_sum(tf.cast(tf.is_finite(c), tf.int32)), 1.0/self.WT], message='Printing c')
                    #self.val = tf.Print(self.val, [tf.reduce_sum(tf.cast(tf.is_finite(1.0/c), tf.int32)), 1.0/self.WT], message='Printing 1/c')
                    #self.val = tf.Print(self.val, [tf.reduce_sum(tf.cast(tf.is_finite(tf.log(1.0 - u)), tf.int32)), 1.0/self.WT], message='Printing log and 1/w')
                    self.val = tf.reduce_mean(self.val, axis=-1)
                elif self.ALG_NAME in ['rmtpp_splusintensity', 'rmtpp_splusintensity_negw']:
                    self.val = (1.0/self.WT) * (-self.D_list + tf.sqrt(tf.square(self.D_list) - 2*self.WT*tf.log(1.0-u)))
                    self.val = tf.reduce_mean(self.val, axis=-1)

                #self.val = tf.Print(self.val, [self.val], message='Printing val')

                with tf.device(device_cpu):
                    # Global step needs to be on the CPU (Why?)
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.learning_rate = tf.train.inverse_time_decay(self.LEARNING_RATE,
                                                                 global_step=self.global_step,
                                                                 decay_steps=decay_steps,
                                                                 decay_rate=decay_rate)
                # self.global_step is incremented automatically by the
                # optimizer.

                # self.increment_global_step = tf.assign(
                #     self.global_step,
                #     self.global_step + 1,
                #     name='update_global_step'
                # )

                # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        beta1=self.MOMENTUM)

                # Capping the gradient before minimizing.
                # update = optimizer.minimize(loss)

                # Performing manual gradient clipping.
                self.gvs = self.optimizer.compute_gradients(self.loss)
                # update = optimizer.apply_gradients(gvs)

                # capped_gvs = [(tf.clip_by_norm(grad, 100.0), var) for grad, var in gvs]
                grads, vars_ = list(zip(*self.gvs))

                self.norm_grads, self.global_norm = tf.clip_by_global_norm(grads, 10.0)
                capped_gvs = list(zip(self.norm_grads, vars_))

                with tf.device(device_cpu):
                    tf.contrib.training.add_gradients_summaries(self.gvs)
                    # for g, v in zip(grads, vars_):
                    #     variable_summaries(g, name='grad-' + v.name.split('/')[-1][:-2])

                    variable_summaries(self.loss, name='loss')
                    variable_summaries(self.hidden_states, name='agg-hidden-states')
                    variable_summaries(self.event_preds, name='agg-event-preds-softmax')
                    variable_summaries(self.time_LLs, name='agg-time-LL')
                    variable_summaries(self.mark_LLs, name='agg-mark-LL')
                    variable_summaries(self.time_LLs + self.mark_LLs, name='agg-total-LL')
                    # variable_summaries(self.delta_ts, name='agg-delta-ts')
                    variable_summaries(self.times, name='agg-times')
                    variable_summaries(self.log_lambdas, name='agg-log-lambdas')
                    variable_summaries(tf.nn.softplus(self.wt), name='wt-soft-plus')

                    self.tf_merged_summaries = tf.summary.merge_all()

                self.update = self.optimizer.apply_gradients(capped_gvs,
                                                             global_step=self.global_step)

                self.tf_init = tf.global_variables_initializer()
                # self.check_nan = tf.add_check_numerics_ops()

    def embedFeatures(self, inputs):
        return tf.nn.embedding_lookup(self.Wem_feats, inputs)

    def initialize(self, finalize=False):
        """Initialize the global trainable variables."""
        self.sess.run(self.tf_init)

        if finalize:
            # This prevents memory leaks by disallowing changes to the graph
            # after initialization.
            self.sess.graph.finalize()


    def make_feed_dict(self, training_data, batch_idxes, bptt_idx,
                       init_hidden_state=None):
        """Creates a batch for the given batch_idxes starting from bptt_idx.
        The hidden state will be initialized with all zeros if no such state is
        provided.
        """

        if init_hidden_state is None:
            cur_state = np.zeros((len(batch_idxes), self.HIDDEN_LAYER_SIZE))
        else:
            cur_state = init_hidden_state

        train_event_in_seq = training_data['train_event_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']
        train_event_out_seq = training_data['train_event_out_seq']
        train_time_out_seq = training_data['train_time_out_seq']

        batch_event_train_in = train_event_in_seq[batch_idxes, :]
        batch_event_train_out = train_event_out_seq[batch_idxes, :]
        batch_time_train_in = train_time_in_seq[batch_idxes, :]
        batch_time_train_out = train_time_out_seq[batch_idxes, :]

        bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
        bptt_event_in = batch_event_train_in[:, bptt_range]
        bptt_event_out = batch_event_train_out[:, bptt_range]
        bptt_time_in = batch_time_train_in[:, bptt_range]
        bptt_time_out = batch_time_train_out[:, bptt_range]

        if bptt_idx > 0:
            initial_time = batch_time_train_in[:, bptt_idx - 1]
        else:
            initial_time = np.zeros(batch_time_train_in.shape[0])

        feed_dict = {
            self.initial_state: cur_state,
            self.initial_time: initial_time,
            self.events_in: bptt_event_in,
            self.events_out: bptt_event_out,
            self.times_in: bptt_time_in,
            self.times_out: bptt_time_out,
            self.batch_num_events: np.sum(batch_event_train_in > 0)
        }

        return feed_dict

    def train(self, training_data, num_epochs=1,
              restart=False, check_nans=False, one_batch=False,
              with_summaries=False, eval_train_data=False, stop_criteria=None, 
              restore_path=None, dec_len_for_eval=0):

        if dec_len_for_eval==0:
            dec_len_for_eval=training_data['decoder_length']
        if restore_path is None:
            restore_path = self.SAVE_DIR
        """Train the model given the training data.

        If with_evals is an integer, then that many elements from the test set
        will be tested.
        """
        # create_dir(self.SAVE_DIR)
        ckpt = tf.train.get_checkpoint_state(restore_path)

        # TODO: Why does this create new nodes in the graph? Possibly memory leak?
        saver = tf.train.Saver(tf.global_variables())

        if with_summaries:
            train_writer = tf.summary.FileWriter(self.SUMMARY_DIR + '/train',
                                                 self.sess.graph)

        if ckpt and restart:
            print('Restoring from {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)

        if stop_criteria is not None:
            self.STOP_CRITERIA = stop_criteria

        train_event_in_seq = training_data['train_event_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']
        train_event_out_seq = training_data['train_event_out_seq']
        train_time_out_seq = training_data['train_time_out_seq']
        train_time_in_feats = training_data['train_time_in_feats']
        train_time_out_feats = training_data['train_time_out_feats']
        train_actual_time_in_seq = np.array(training_data['train_actual_time_in_seq'])

        best_dev_mae, best_test_mae = np.inf, np.inf
        best_dev_gap_mae, best_test_gap_mae = np.inf, np.inf
        best_dev_gap_dtw, best_test_gap_dtw = np.inf, np.inf
        best_dev_time_preds, best_dev_event_preds = [], []
        best_test_time_preds, best_test_event_preds = [], []
        best_epoch = 0
        total_loss = 0.0
        train_loss_list, dev_loss_list, test_loss_list = list(), list(), list()
        train_time_loss_list, dev_time_loss_list, test_time_loss_list = list(), list(), list()
        train_mark_loss_list, dev_mark_loss_list, test_mark_loss_list = list(), list(), list()
        wt_list = list()
        train_epoch_times, train_inference_times = list(), list()
        dev_inference_times = list()
        test_inference_times = list()

        idxes = list(range(len(train_event_in_seq)))
        n_batches = len(idxes) // self.BATCH_SIZE

        print('Training with ', self.STOP_CRITERIA, 'stop_criteria and ', num_epochs, 'num_epochs')
        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            self.rs.shuffle(idxes)

            print("Starting epoch...", epoch)
            total_loss_prev = total_loss
            total_loss = 0.0
            time_loss, mark_loss = 0.0, 0.0
            epoch_start_time = time.time()

            for batch_idx in range(n_batches):
                batch_idxes = idxes[batch_idx * self.BATCH_SIZE:(batch_idx + 1) * self.BATCH_SIZE]

                batch_event_train_in = [train_event_in_seq[batch_idx][:self.BPTT] for batch_idx in batch_idxes]
                batch_time_train_in = [train_time_in_seq[batch_idx][:self.BPTT] for batch_idx in batch_idxes]
                batch_time_train_feats = [train_time_in_feats[batch_idx][:self.BPTT] for batch_idx in batch_idxes]

                batch_time_train_out = [train_time_out_seq[batch_idx][:self.BPTT] for batch_idx in batch_idxes]
                batch_event_train_out = [train_event_out_seq[batch_idx][:self.BPTT] for batch_idx in batch_idxes]
                #offsets = np.zeros((self.BATCH_SIZE))
                #out_begin_indices, out_end_indices \
                #        = get_output_indices(batch_time_train_in, batch_time_train_out, offsets, self.DEC_LEN)
                ##print(offsets)
                #for beg_ind, end_ind, seq in zip(out_begin_indices, out_end_indices, batch_event_train_out):
                #    print(beg_ind, end_ind, len(seq))
                #    assert end_ind < len(seq)
                #batch_event_train_out = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                #                         zip(batch_event_train_out, out_begin_indices, out_end_indices)]
                #batch_time_train_out = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                #                        zip(batch_time_train_out, out_begin_indices, out_end_indices)]
                batch_train_actual_time_in_seq = [train_actual_time_in_seq[batch_idx] for batch_idx in batch_idxes]


                cur_state = np.zeros((self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE))
                batch_loss, batch_time_loss, batch_mark_loss = 0.0, 0.0, 0.0

                batch_num_events = np.sum(np.array(batch_event_train_in) > 0)

                initial_time = np.zeros(len(batch_time_train_in))

                feed_dict = {
                    self.initial_state: cur_state,
                    self.initial_time: initial_time,
                    self.events_in: batch_event_train_in,
                    self.events_out: batch_event_train_out,
                    self.times_in: batch_time_train_in,
                    self.times_out: batch_time_train_out,
                    self.times_in_feats: batch_time_train_feats,
                    self.batch_num_events: batch_num_events
                }

                if check_nans:
                    raise NotImplemented('tf.add_check_numerics_ops is '
                                         'incompatible with tf.cond and '
                                         'tf.while_loop.')
                    # _, _, cur_state, loss_ = \
                    #     self.sess.run([self.check_nan, self.update,
                    #                    self.final_state, self.loss],
                    #                   feed_dict=feed_dict)
                else:
                    if with_summaries:
                        _, summaries, cur_state, loss_, step = \
                            self.sess.run([self.update,
                                           self.tf_merged_summaries,
                                           self.final_state,
                                           self.loss,
                                           self.global_step],
                                          feed_dict=feed_dict)
                        train_writer.add_summary(summaries, step)
                    else:
                        _, cur_state, loss_, time_loss_, mark_loss_ = \
                            self.sess.run([self.update,
                                           self.final_state, self.loss,
                                           self.time_loss, self.mark_loss],
                                          feed_dict=feed_dict)
                batch_loss = loss_
                total_loss += batch_loss
                time_loss += time_loss_
                mark_loss += mark_loss_
                if batch_idx % 10 == 0:
                    print('Loss during batch {} last BPTT = {:.3f}, lr = {:.5f}'
                          .format(batch_idx, batch_loss, self.sess.run(self.learning_rate)))
            epoch_end_time = time.time()
            train_epoch_times.append(epoch_end_time - epoch_start_time)

            if self.STOP_CRITERIA == 'per_epoch_val_err' and epoch >= self.PATIENCE:

                if eval_train_data:
                    print('Running evaluation on train, dev, test: ...')
                else:
                    print('Running evaluation on dev, test: ...')

                if eval_train_data:
                    raise NotImplementedError
                else:
                    train_mae, train_acc, train_mrr, train_time_preds, train_event_preds = None, None, None, np.array([]), np.array([])

                dev_offsets = np.zeros((len(training_data['dev_time_in_seq']))) * self.MAX_OFFSET
                dev_offsets_normalized = dev_offsets / np.squeeze(np.array(training_data['devND']), axis=-1)
                dev_time_preds, dev_gaps_preds, dev_event_preds, \
                        dev_event_preds_softmax, inference_time, _ \
                        = self.predict(training_data['dev_event_in_seq'],
                                       training_data['dev_time_in_seq'],
                                       training_data['dev_time_in_feats'],
                                       dec_len_for_eval,
                                       training_data['train_time_in_seq'],
                                       training_data['devND'],
                                       dev_offsets,
                                       dev_offsets_normalized,
                                       single_threaded=True,
                                       is_nowcast=True)
                dev_loss, dev_time_loss, dev_mark_loss \
                        = self.evaluate_likelihood(training_data['dev_event_in_seq'],
                                                   training_data['dev_time_in_seq'],
                                                   training_data['dev_event_out_seq'],
                                                   training_data['dev_time_out_seq'],
                                                   training_data['dev_time_in_feats'],
                                                   training_data['dev_time_out_feats'],
                                                   dec_len_for_eval)
                dev_loss_list.append(dev_loss)
                dev_time_loss_list.append(dev_time_loss)
                dev_mark_loss_list.append(dev_mark_loss)
                dev_inference_times.append(inference_time)
                dev_time_in_seq = training_data['dev_time_in_seq']
                dev_actual_time_in_seq = training_data['dev_actual_time_in_seq']
                dev_unnorm_time_in_seq = training_data['dev_unnorm_time_in_seq']
                dev_time_out_seq = training_data['dev_nowcast_time_out_seq']
                dev_event_out_seq = training_data['dev_nowcast_event_out_seq']
                dev_offsets = dev_offsets_normalized * np.squeeze(training_data['devND'], axis=-1)

                #out_begin_indices, out_end_indices \
                #        = get_output_indices(dev_actual_time_in_seq, dev_time_out_seq, dev_offsets, self.DEC_LEN)
                #for beg_ind, end_ind, seq in zip(out_begin_indices, out_end_indices, dev_event_out_seq):
                #    #print(beg_ind, end_ind, len(seq))
                #    assert end_ind < len(seq)
                #dev_event_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                #                        zip(dev_event_out_seq, out_begin_indices, out_end_indices)]
                #dev_time_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                #                        zip(dev_time_out_seq, out_begin_indices, out_end_indices)]

                gaps = dev_gaps_preds
                unnorm_gaps = [seq * devND for seq, devND in zip(gaps, training_data['devND'])]
                #dev_time_preds = [np.cumsum(gap_seq) + seq + off - devIG for gap_seq, seq, off, devIG in
                #                    zip(unnorm_gaps, dev_actual_time_in_seq, dev_offsets, training_data['devIG'])]

                dev_time_preds = [gap_seq + np.array(unnorm_time_in[:-1]) for gap_seq, unnorm_time_in in \
                                    zip(unnorm_gaps, dev_unnorm_time_in_seq)]

                dev_mae, dev_total_valid, dev_acc, dev_gap_mae, dev_mrr, dev_gap_dtw, \
                        cum_dev_gap_mae, cum_dev_acc, cum_dev_mrr \
                        = self.eval(dev_time_preds, dev_time_out_seq,
                                    dev_event_preds, dev_event_out_seq,
                                    training_data['dev_actual_time_in_seq'],
                                    dev_event_preds_softmax,
                                    dev_offsets,
                                    is_nowcast=True,
                                    time_inputs=dev_unnorm_time_in_seq)
                #print('DEV: MAE = {:.5f}; valid = {}, ACC = {:.5f}, MAGE = {:.5f}, DTW = {:.5f}'.format(
                #    dev_mae, dev_total_valid, dev_acc, dev_gap_mae, dev_gap_dtw))
                print('DEV: MAE =', dev_mae, '; valid =', dev_total_valid, 'ACC =', dev_acc, 'MAGE =', dev_gap_mae, 'DTW =', dev_gap_dtw)

                if self.PLOT_PRED_DEV:
                    idx = 4
                    #true_gaps_plot = dev_time_out_seq[idx] - np.concatenate([dev_actual_time_in_seq[idx]+dev_offsets[idx], dev_time_out_seq[idx][:-1]])
                    true_gaps_plot = np.array(dev_time_out_seq[idx])-np.array(dev_unnorm_time_in_seq[idx][:-1])
                    pred_gaps_plot = unnorm_gaps[idx]
                    #inp_tru_gaps = training_data['dev_time_in_seq'][idx][1:] \
                    #               - training_data['dev_time_in_seq'][idx][:-1]
                    #inp_tru_gaps = inp_tru_gaps * training_data['devND'][idx]
                    #true_gaps_plot = list(inp_tru_gaps) + list(true_gaps_plot)
                    #pred_gaps_plot = list(inp_tru_gaps) + list(pred_gaps_plot)

                    plot_dir = os.path.join(self.SAVE_DIR,'dev_plots')
                    #if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
                    plot_hparam_dir = 'pred_plot_'
                    for name, val in self.PARAMS_ALIAS_NAMED:
                        plot_hparam_dir += str(name) + '_' + str(val) + '_'
                    plot_dir = os.path.join(plot_dir, plot_hparam_dir)
                    if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
                    name_plot = os.path.join(plot_dir, 'epoch_' + str(epoch))

                    assert len(true_gaps_plot) == len(pred_gaps_plot)

                    fig_pred_gaps = plt.figure()
                    ax1 = fig_pred_gaps.add_subplot(111)
                    ax1.scatter(list(range(1, len(pred_gaps_plot)+1)), pred_gaps_plot, c='r', label='Pred gaps')
                    ax1.scatter(list(range(1, len(true_gaps_plot)+1)), true_gaps_plot, c='b', label='True gaps')
                    #ax1.plot([self.BPTT-self.DEC_LEN+0.5, self.BPTT-self.DEC_LEN+0.5],
                    #         [0, max(np.concatenate([true_gaps_plot, pred_gaps_plot]))],
                    #         'g-')
                    ax1.set_xlabel('Index')
                    ax1.set_ylabel('Gaps')
                    plt.grid()

                    plt.savefig(name_plot+'.png')
                    plt.close()
    
                test_offsets = np.zeros((len(training_data['test_time_in_seq']))) * self.MAX_OFFSET
                test_offsets_normalized = test_offsets / np.squeeze(np.array(training_data['testND']), axis=-1)
                test_time_preds, test_gaps_preds, test_event_preds, \
                        test_event_preds_softmax, inference_time, _ \
                        = self.predict(training_data['test_event_in_seq'],
                                       training_data['test_time_in_seq'],
                                       training_data['test_time_in_feats'],
                                       dec_len_for_eval,
                                       training_data['train_time_in_seq'],
                                       training_data['testND'],
                                       test_offsets,
                                       test_offsets_normalized,
                                       single_threaded=True,
                                       is_nowcast=True)
                test_loss, test_time_loss, test_mark_loss \
                        = self.evaluate_likelihood(training_data['test_event_in_seq'],
                                                   training_data['test_time_in_seq'],
                                                   training_data['test_event_out_seq'],
                                                   training_data['test_time_out_seq'],
                                                   training_data['test_time_in_feats'],
                                                   training_data['test_time_out_feats'],
                                                   dec_len_for_eval)
                test_loss_list.append(test_loss)
                test_time_loss_list.append(test_time_loss)
                test_mark_loss_list.append(test_mark_loss)
                test_inference_times.append(inference_time)
                test_time_in_seq = training_data['test_time_in_seq']
                test_actual_time_in_seq = training_data['test_actual_time_in_seq']
                test_unnorm_time_in_seq = training_data['test_unnorm_time_in_seq']
                test_time_out_seq = training_data['test_nowcast_time_out_seq']
                test_event_out_seq = training_data['test_nowcast_event_out_seq']
                test_offsets = test_offsets_normalized * np.squeeze(training_data['testND'], axis=-1)

                #out_begin_indices, out_end_indices \
                #        = get_output_indices(test_actual_time_in_seq, test_time_out_seq, test_offsets, self.DEC_LEN)
                #for beg_ind, end_ind, seq in zip(out_begin_indices, out_end_indices, test_event_out_seq):
                #    #print(beg_ind, end_ind, len(seq))
                #    assert end_ind < len(seq)
                #test_event_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                #                        zip(test_event_out_seq, out_begin_indices, out_end_indices)]
                #test_time_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                #                        zip(test_time_out_seq, out_begin_indices, out_end_indices)]

                gaps = test_gaps_preds
                unnorm_gaps = [seq * testND for seq, testND in zip(gaps, training_data['testND'])]
                #test_time_preds = [np.cumsum(gap_seq) + seq + off - testIG for gap_seq, seq, off, testIG in
                #                    zip(unnorm_gaps, test_actual_time_in_seq, test_offsets, training_data['testIG'])]

                test_time_preds = [gap_seq + np.array(unnorm_time_in[:-1]) for gap_seq, unnorm_time_in in \
                                    zip(unnorm_gaps, test_unnorm_time_in_seq)]

                test_mae, test_total_valid, test_acc, test_gap_mae, test_mrr, test_gap_dtw, \
                        cum_test_gap_mae, cum_test_acc, cum_test_mrr \
                        = self.eval(test_time_preds, test_time_out_seq,
                                    test_event_preds, test_event_out_seq,
                                    training_data['test_actual_time_in_seq'],
                                    test_event_preds_softmax,
                                    test_offsets,
                                    is_nowcast=True,
                                    time_inputs=test_unnorm_time_in_seq)
                #print('TEST: MAE = {:.5f}; valid = {}, ACC = {:.5f}, MAGE = {:.5f}, DTW = {:.5f}'.format(
                #    test_mae, test_total_valid, test_acc, test_gap_mae, test_gap_dtw))
                print('TEST: MAE =', test_mae, '; valid =', test_total_valid, 'ACC =', test_acc, 'MAGE =', test_gap_mae, 'DTW =', test_gap_dtw)

                if self.PLOT_PRED_TEST:
                    random_plot_number = 4
                    true_gaps_plot = tru_gaps[random_plot_number,:]
                    pred_gaps_plot = unnorm_gaps[random_plot_number,:]
                    inp_tru_gaps = training_data['test_time_in_seq'][random_plot_number, 1:] \
                                   - training_data['test_time_in_seq'][random_plot_number, :-1]
                    inp_tru_gaps = inp_tru_gaps * training_data['testND'][random_plot_number]
                    true_gaps_plot = list(inp_tru_gaps) + list(true_gaps_plot)
                    pred_gaps_plot = list(inp_tru_gaps) + list(pred_gaps_plot)

                    plot_dir = os.path.join(self.SAVE_DIR,'test_plots')
                    if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
                    name_plot = os.path.join(plot_dir, "pred_plot_"+str(self.HIDDEN_LAYER_SIZE)+"_"+str(epoch))

                    plot_dir = os.path.join(self.SAVE_DIR,'test_plots')
                    if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
                    plot_hparam_dir = 'pred_plot_'
                    for name, val in self.PARAMS_ALIAS_NAMED:
                        plot_hparam_dir += str(name) + '_' + str(val) + '_'
                    plot_dir = os.path.join(plot_dir, plot_hparam_dir)
                    if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
                    name_plot = os.path.join(plot_dir, 'epoch_' + str(epoch))

                    assert len(true_gaps_plot) == len(pred_gaps_plot)

                    fig_pred_gaps = plt.figure()
                    ax1 = fig_pred_gaps.add_subplot(111)
                    ax1.scatter(list(range(len(pred_gaps_plot))), pred_gaps_plot, c='r', label='Pred gaps')
                    ax1.scatter(list(range(len(true_gaps_plot))), true_gaps_plot, c='b', label='True gaps')

                    plt.savefig(name_plot)
                    plt.close()

                if dev_gap_dtw < best_dev_gap_dtw:
                    best_epoch = epoch
                    best_train_mae, best_dev_mae, best_test_mae, best_dev_gap_mae, best_test_gap_mae = train_mae, dev_mae, test_mae, dev_gap_mae, test_gap_mae
                    best_dev_gap_dtw, best_test_gap_dtw = dev_gap_dtw, test_gap_dtw
                    best_cum_dev_gap_mae, best_cum_dev_acc, best_cum_dev_mrr = cum_dev_gap_mae, cum_dev_acc, cum_dev_mrr
                    best_cum_test_gap_mae, best_cum_test_acc, best_cum_test_mrr = cum_test_gap_mae, cum_test_acc, cum_test_mrr
                    best_train_acc, best_dev_acc, best_test_acc = train_acc, dev_acc, test_acc
                    best_train_mrr, best_dev_mrr, best_test_mrr = train_mrr, dev_mrr, test_mrr
                    best_train_event_preds, best_train_time_preds  = train_event_preds, train_time_preds
                    best_dev_event_preds, best_dev_time_preds  = dev_event_preds, dev_time_preds
                    best_test_event_preds, best_test_time_preds  = test_event_preds, test_time_preds
                    best_dev_loss, best_test_loss = dev_loss, test_loss
                    best_dev_time_loss, best_test_time_loss = dev_time_loss, test_time_loss
                    best_dev_mark_loss, best_test_mark_loss = dev_mark_loss, test_mark_loss
                    best_w = self.sess.run(self.wt).tolist()
    
                    checkpoint_dir = restore_path
                    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path)# , global_step=step)
                    print('Model saved at {}'.format(checkpoint_path))


            # self.sess.run(self.increment_global_step)
            train_loss_list.append(total_loss)
            train_time_loss_list.append(np.float64(time_loss))
            train_mark_loss_list.append(np.float64(mark_loss))
            wt_list.append(self.sess.run(self.wt).tolist()[0][0])
            print('Loss on last epoch = {:.4f}, train_loss = {:.4f}, mark_loss = {:.4f}, new lr = {:.5f}, global_step = {}'
                  .format(total_loss, np.float64(time_loss), np.float64(mark_loss),
                          self.sess.run(self.learning_rate),
                          self.sess.run(self.global_step)))

            if self.STOP_CRITERIA == 'epsilon' and epoch >= self.PATIENCE and abs(total_loss-total_loss_prev) < self.EPSILON:
                break

            if one_batch:
                print('Breaking after just one batch.')
                break

        # Remember how many epochs we have trained.
        if num_epochs>0:
            self.last_epoch += num_epochs

        if self.STOP_CRITERIA == 'epsilon':

            if eval_train_data:
                print('Running evaluation on train, dev, test: ...')
            else:
                print('Running evaluation on dev, test: ...')

            if eval_train_data: 
                raise NotImplementedError
            else:
                train_mae, train_acc, train_mrr, train_time_preds, train_event_preds = None, None, None, np.array([]), np.array([])

            dev_offsets = np.ones((len(training_data['dev_time_in_seq']))) * self.MAX_OFFSET
            dev_offsets_normalized = dev_offsets / np.squeeze(np.array(training_data['devND']), axis=-1)
            dev_time_preds, dev_gaps_preds, dev_event_preds, \
                    dev_event_preds_softmax, inference_time, _ \
                    = self.predict(training_data['dev_event_in_seq'],
                                   training_data['dev_time_in_seq'],
                                   training_data['dev_time_in_feats'],
                                   dec_len_for_eval,
                                   training_data['train_time_in_seq'],
                                   training_data['devND'],
                                   dev_offsets,
                                   dev_offsets_normalized,
                                   single_threaded=True)
            dev_loss, dev_time_loss, dev_mark_loss \
                    = self.evaluate_likelihood(training_data['dev_event_in_seq'],
                                               training_data['dev_time_in_seq'],
                                               training_data['dev_event_out_seq'],
                                               training_data['dev_time_out_seq'],
                                               training_data['dev_time_in_feats'],
                                               training_data['dev_time_out_feats'],
                                               dec_len_for_eval)
            dev_loss_list.append(dev_loss)
            dev_time_loss_list.append(dev_time_loss)
            dev_mark_loss_list.append(dev_mark_loss)
            dev_inference_times.append(inference_time)
            dev_time_in_seq = training_data['dev_time_in_seq']
            dev_actual_time_in_seq = training_data['dev_actual_time_in_seq']
            dev_time_out_seq = training_data['dev_actual_time_out_seq']
            dev_event_out_seq = training_data['dev_event_out_seq']
            dev_offsets = dev_offsets_normalized * np.squeeze(training_data['devND'], axis=-1)

            out_begin_indices, out_end_indices \
                    = get_output_indices(dev_actual_time_in_seq, dev_time_out_seq, dev_offsets, self.DEC_LEN)
            for beg_ind, end_ind, seq in zip(out_begin_indices, out_end_indices, dev_event_out_seq):
                #print(beg_ind, end_ind, len(seq))
                assert end_ind < len(seq)
            dev_event_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                                    zip(dev_event_out_seq, out_begin_indices, out_end_indices)]
            dev_time_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                                    zip(dev_time_out_seq, out_begin_indices, out_end_indices)]

            gaps = dev_gaps_preds
            unnorm_gaps = [seq * devND for seq, devND in zip(gaps, training_data['devND'])]
            dev_time_preds = [np.cumsum(gap_seq) + seq + off - devIG for gap_seq, seq, off, devIG in
                                zip(unnorm_gaps, dev_actual_time_in_seq, dev_offsets, training_data['devIG'])]

            dev_time_out_seq = trim_seq_dec_len(dev_time_out_seq, dec_len_for_eval)
            dev_time_preds = trim_seq_dec_len(dev_time_preds, dec_len_for_eval)
            dev_event_out_seq = trim_seq_dec_len(dev_event_out_seq, dec_len_for_eval)
            dev_event_preds = trim_seq_dec_len(dev_event_preds, dec_len_for_eval)

            dev_mae, dev_total_valid, dev_acc, dev_gap_mae, dev_mrr, dev_gap_dtw, \
                    cum_dev_gap_mae, cum_dev_acc, cum_dev_mrr \
                    = self.eval(dev_time_preds, dev_time_out_seq,
                                dev_event_preds, dev_event_out_seq,
                                training_data['dev_actual_time_in_seq'],
                                dev_event_preds_softmax,
                                dev_offsets)
            #print('DEV: MAE = {:.5f}; valid = {}, ACC = {:.5f}, MAGE = {:.5f}, DTW = {:.5f}'.format(
            #    dev_mae, dev_total_valid, dev_acc, dev_gap_mae, dev_gap_dtw))
            print('DEV: MAE =', dev_mae, '; valid =', dev_total_valid, 'ACC =', dev_acc, 'MAGE =', dev_gap_mae, 'DTW =', dev_gap_dtw)

            test_offsets = np.ones((len(training_data['test_time_in_seq']))) * self.MAX_OFFSET
            test_offsets_normalized = test_offsets / np.squeeze(np.array(training_data['testND']), axis=-1)
            test_time_preds, test_gaps_preds, test_event_preds, \
                    test_event_preds_softmax, inference_time, _ \
                    = self.predict(training_data['test_event_in_seq'],
                                   training_data['test_time_in_seq'],
                                   training_data['test_time_in_feats'],
                                   dec_len_for_eval,
                                   training_data['train_time_in_seq'],
                                   training_data['testND'],
                                   test_offsets,
                                   test_offsets_normalized,
                                   single_threaded=True)
            test_loss, test_time_loss, test_mark_loss \
                    = self.evaluate_likelihood(training_data['test_event_in_seq'],
                                               training_data['test_time_in_seq'],
                                               training_data['test_event_out_seq'],
                                               training_data['test_time_out_seq'],
                                               training_data['test_time_in_feats'],
                                               training_data['test_time_out_feats'],
                                               dec_len_for_eval)
            test_loss_list.append(test_loss)
            test_time_loss_list.append(test_time_loss)
            test_mark_loss_list.append(test_mark_loss)
            test_inference_times.append(inference_time)
            test_time_in_seq = training_data['test_time_in_seq']
            test_actual_time_in_seq = training_data['test_actual_time_in_seq']
            test_time_out_seq = training_data['test_actual_time_out_seq']
            test_event_out_seq = training_data['test_event_out_seq']
            test_offsets = test_offsets_normalized * np.squeeze(training_data['testND'], axis=-1)

            out_begin_indices, out_end_indices \
                    = get_output_indices(test_actual_time_in_seq, test_time_out_seq, test_offsets, self.DEC_LEN)
            for beg_ind, end_ind, seq in zip(out_begin_indices, out_end_indices, test_event_out_seq):
                #print(beg_ind, end_ind, len(seq))
                assert end_ind < len(seq)
            test_event_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                                    zip(test_event_out_seq, out_begin_indices, out_end_indices)]
            test_time_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                                    zip(test_time_out_seq, out_begin_indices, out_end_indices)]

            gaps = test_gaps_preds
            unnorm_gaps = [seq * testND for seq, testND in zip(gaps, training_data['testND'])]
            test_time_preds = [np.cumsum(gap_seq) + seq + off - testIG for gap_seq, seq, off, testIG in
                                zip(unnorm_gaps, test_actual_time_in_seq, test_offsets, training_data['testIG'])]

            test_time_out_seq = trim_seq_dec_len(test_time_out_seq, dec_len_for_eval)
            test_time_preds = trim_seq_dec_len(test_time_preds, dec_len_for_eval)
            test_event_out_seq = trim_seq_dec_len(test_event_out_seq, dec_len_for_eval)
            test_event_preds = trim_seq_dec_len(test_event_preds, dec_len_for_eval)

            test_mae, test_total_valid, test_acc, test_gap_mae, test_mrr, test_gap_dtw, \
                    cum_test_gap_mae, cum_test_acc, cum_test_mrr \
                    = self.eval(test_time_preds, test_time_out_seq,
                                test_event_preds, test_event_out_seq,
                                training_data['test_actual_time_in_seq'],
                                test_event_preds_softmax,
                                test_offsets)
            #print('TEST: MAE = {:.5f}; valid = {}, ACC = {:.5f}, MAGE = {:.5f}, DTW = {:.5f}'.format(
            #    test_mae, test_total_valid, test_acc, test_gap_mae, test_gap_dtw))
            print('TEST: MAE =', test_mae, '; valid =', test_total_valid, 'ACC =', test_acc, 'MAGE =', test_gap_mae, 'DTW =', test_gap_dtw)

            if dev_gap_dtw < best_dev_gap_dtw:
                best_epoch = num_epochs
                best_train_mae, best_dev_mae, best_test_mae, best_dev_gap_mae, best_test_gap_mae = train_mae, dev_mae, test_mae, dev_gap_mae, test_gap_mae
                best_dev_gap_dtw, best_test_gap_dtw = dev_gap_dtw, test_gap_dtw
                best_cum_dev_gap_mae, best_cum_dev_acc, best_cum_dev_mrr = cum_dev_gap_mae, cum_dev_acc, cum_dev_mrr
                best_cum_test_gap_mae, best_cum_test_acc, best_cum_test_mrr = cum_test_gap_mae, cum_test_acc, cum_test_mrr
                best_train_acc, best_dev_acc, best_test_acc = train_acc, dev_acc, test_acc
                best_train_mrr, best_dev_mrr, best_test_mrr = train_mrr, dev_mrr, test_mrr
                best_train_event_preds, best_train_time_preds  = train_event_preds, train_time_preds
                best_dev_event_preds, best_dev_time_preds  = dev_event_preds, dev_time_preds
                best_test_event_preds, best_test_time_preds  = test_event_preds, test_time_preds
                best_dev_loss, best_test_loss = dev_loss, test_loss
                best_dev_time_loss, best_test_time_loss = dev_time_loss, test_time_loss
                best_dev_mark_loss, best_test_mark_loss = dev_mark_loss, test_mark_loss
                best_w = self.sess.run(self.wt).tolist()

                checkpoint_dir = restore_path
                checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
                saver.save(self.sess, checkpoint_path)# , global_step=step)
                print('Model saved at {}'.format(checkpoint_path))


        avg_train_epoch_time = np.average(train_epoch_times)
        avg_dev_inference_time = np.average(dev_inference_times)
        avg_train_inference_time = np.average(train_inference_times)
        avg_test_inference_time = np.average(test_inference_times)

        return {
                'best_epoch': best_epoch,
                'best_dev_gap_mae': best_dev_gap_mae,
                'best_test_gap_mae': best_test_gap_mae,
                'best_dev_gap_dtw': best_dev_gap_dtw,
                'best_test_gap_dtw': best_test_gap_dtw,
                'best_cum_dev_gap_mae': best_cum_dev_gap_mae,
                'best_cum_dev_acc': best_cum_dev_acc,
                'best_cum_dev_mrr':best_cum_dev_mrr,
                'best_cum_test_gap_mae': best_cum_test_gap_mae,
                'best_cum_test_acc': best_cum_test_acc,
                'best_cum_test_mrr':best_cum_test_mrr,
                'best_dev_mae': best_dev_mae,
                'best_dev_acc': best_dev_acc,
                'best_dev_mrr': best_dev_mrr,
                'best_test_mae': best_test_mae,
                'best_test_acc': best_test_acc,
                'best_test_mrr': best_test_mrr,
                'best_dev_loss': best_dev_loss,
                'best_test_loss': best_test_loss,
                'best_train_mae': best_train_mae,
                'best_train_acc': best_train_acc,
                'best_train_event_preds': best_train_event_preds,
                'best_train_time_preds': best_train_time_preds,
                'best_dev_event_preds': best_dev_event_preds,
                'best_dev_time_preds': best_dev_time_preds,
                'best_test_event_preds': best_test_event_preds,
                'best_test_time_preds': best_test_time_preds,
                'dev_event_out_seq': dev_event_out_seq,
                'dev_time_out_seq': dev_time_out_seq,
                'test_event_out_seq': test_event_out_seq,
                'test_time_out_seq': test_time_out_seq,
                'best_w': best_w,
                'wt_hparam': self.wt_hparam,
                'checkpoint_dir': checkpoint_dir,
                'train_loss_list': train_loss_list,
                'train_time_loss_list': train_time_loss_list,
                'train_mark_loss_list': train_mark_loss_list,
                'dev_loss_list': dev_loss_list,
                'dev_time_loss_list': dev_time_loss_list,
                'dev_mark_loss_list': dev_mark_loss_list,
                'test_loss_list': test_loss_list,
                'test_time_loss_list': test_time_loss_list,
                'test_mark_loss_list': test_mark_loss_list,
                'wt_list': wt_list,
                'avg_train_inference_time':avg_train_inference_time,
                'avg_dev_inference_time':avg_dev_inference_time,
                'avg_test_inference_time':avg_test_inference_time,
                'avg_train_epoch_time': avg_train_epoch_time,
               }


    def restore(self):
        """Restore the model from saved state."""
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)
        print('Loading the model from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def evaluate_likelihood(self, event_in_seq, time_in_seq, event_out_seq, time_out_seq, time_in_feats, time_out_feats, decoder_length):

        num_events = np.sum(np.array(event_in_seq) > 0)
        cur_state = np.zeros((len(event_in_seq), self.HIDDEN_LAYER_SIZE))
        initial_time = np.zeros(len(event_in_seq))

        event_seq = [(in_seq + out_seq)[:self.BPTT+1] for in_seq, out_seq in zip(event_in_seq, event_out_seq)]
        time_seq = [(in_seq.tolist() + out_seq.tolist())[:self.BPTT+1] for in_seq, out_seq in zip(time_in_seq, time_out_seq)]
        time_feats = [(in_seq + out_seq)[:self.BPTT+1] for in_seq, out_seq in zip(time_in_feats, time_out_feats)]

        events_in = [seq[:-1] for seq in event_seq]
        events_out = [seq[1:] for seq in event_seq]
        times_in = [seq[:-1] for seq in time_seq]
        times_out = [seq[1:] for seq in time_seq]
        times_in_feats = [seq[:-1] for seq in time_feats]

        feed_dict = {
            self.initial_state: cur_state,
            self.initial_time: initial_time,
            self.events_in: events_in,
            self.events_out: events_out,
            self.times_in: times_in,
            self.times_out: times_out,
            self.times_in_feats: times_in_feats,
            self.batch_num_events: num_events
        }

        loss_, time_loss_, mark_loss_ = self.sess.run([self.loss,
                                                       self.time_loss, self.mark_loss],
                                                       feed_dict=feed_dict)

        print("Log Likelihood:", loss_)
        return float(loss_), float(time_loss_), float(mark_loss_)


    def predict(self, event_in_seq, time_in_seq, time_in_feats, dec_len_for_eval,
                train_in_seq, ND, offsets, offsets_normalized, single_threaded=False,
                is_nowcast=False):
        """Treats the entire dataset as a single batch and processes it."""


        if self.ALG_NAME in ['zero_pred']:
            start_time = time.time()
            event_out_seq = np.tile(event_in_seq[:, -1:], [1, dec_len_for_eval])
            time_out_seq = np.tile(time_in_seq[:, -1:], [1, dec_len_for_eval])
            all_event_preds_softmax = np.zeros((len(event_in_seq), dec_len_for_eval, self.NUM_CATEGORIES))
            for i, row in enumerate(event_in_seq):
                last_event = row[-1]
                events_id2freq = Counter(row)
                max_freq = max(events_id2freq.items(), key=itemgetter(1))[1]
                events_id2freq[last_event] = max_freq * 1.1
                event_ids, event_freqs = events_id2freq.keys(), events_id2freq.values()
                event_ids = np.array(list(event_ids))-1
                event_freqs = list(event_freqs)
                event_preds_softmax = np.zeros((1, 1, self.NUM_CATEGORIES))
                event_preds_softmax[0, 0, event_ids] = event_freqs
                all_event_preds_softmax[i] = np.repeat(event_preds_softmax, dec_len_for_eval, axis=1)
            end_time = time.time()
            inference_time = end_time - start_time
            return time_out_seq, event_out_seq, all_event_preds_softmax, inference_time

        #if self.ALG_NAME in ['average_gap_pred']:
        if 'average' in self.ALG_NAME:
            start_time = time.time()
            event_out_seq = [max(Counter(row).items(), key=itemgetter(1))[0] for row in event_in_seq]
            event_out_seq = np.expand_dims(np.array(event_out_seq), axis=1)
            event_out_seq = np.tile(event_out_seq, [1, dec_len_for_eval])
            if self.ALG_NAME in ['average_gap_pred']:
                input_gaps = time_in_seq[:, 1:] - time_in_seq[:, :-1]
                output_gaps = np.mean(input_gaps, axis=1)
            elif self.ALG_NAME in ['average_tr_gap_pred']:
                train_in_gaps = train_in_seq[:, 1:] - train_in_seq[:, :-1]
                train_in_gaps = np.reshape(train_in_gaps, [-1])
                train_in_gaps_sorted = np.sort(train_in_gaps)
                train_in_gaps_sorted = train_in_gaps_sorted[:int(0.95*train_in_gaps_sorted.shape[0])]
                output_gaps = np.mean(train_in_gaps_sorted)
                output_gaps = np.ones((time_in_seq.shape[0])) * output_gaps
            output_gaps = np.expand_dims(np.array(output_gaps), axis=1)
            output_gaps = np.tile(output_gaps, [1, dec_len_for_eval])
            output_gaps = np.cumsum(output_gaps, axis=1)
            time_out_seq = time_in_seq[:, -1:] + output_gaps
            all_event_preds_softmax = np.zeros((len(event_in_seq), dec_len_for_eval, self.NUM_CATEGORIES))
            for i, row in enumerate(event_in_seq):
                event_ids, event_freqs = Counter(row).keys(), Counter(row).values()
                event_ids = np.array(list(event_ids))-1
                event_freqs = list(event_freqs)
                event_preds_softmax = np.zeros((1, 1, self.NUM_CATEGORIES))
                event_preds_softmax[0, 0, event_ids] = event_freqs
                all_event_preds_softmax[i] = np.repeat(event_preds_softmax, dec_len_for_eval, axis=1)
            end_time = time.time()
            inference_time = end_time - start_time
            return time_out_seq, event_out_seq, all_event_preds_softmax, inference_time

        start_time = time.time()

        N = len(time_in_seq)

        # Default offsets = self.MAX_OFFSET
        #offsets = np.random.uniform(low=0.0, high=self.MAX_OFFSET, size=(self.BATCH_SIZE))
        #offsets = np.ones((N)) * self.MAX_OFFSET
        #offsets_normalized = offsets / np.squeeze(np.array(ND), axis=-1)

        all_hidden_states = []
        simul_event_preds_softmax = []
        simul_event_preds = []
        simul_time_preds = []

        cur_state = np.zeros((len(event_in_seq), self.HIDDEN_LAYER_SIZE))

        total_vals = np.zeros(N)
        pred_idxes = -1.0 * np.ones(N)
        begin_idxes, end_idxes = np.zeros(N, dtype=int), np.zeros(N, dtype=int)
        simul_idx = 0
        #for pred_idx in range(0, dec_len_for_eval):
        while any(total_vals<offsets_normalized) or any(pred_idxes<dec_len_for_eval):
            if simul_idx % 1000 == 0:
                print('simul_idx', simul_idx, dec_len_for_eval)
            if simul_idx == 0:
                #bptt_range = range(simul_idx, (simul_idx + self.BPTT))
                #bptt_event_in = event_in_seq[:, bptt_range]
                #bptt_time_in = time_in_seq[:, bptt_range]
                #bptt_event_in = np.concatenate([event_in_seq, np.zeros((N, self.DEC_LEN-1))], axis=1)
                #bptt_time_in = np.concatenate([time_in_seq, np.zeros((N, self.DEC_LEN-1))], axis=1)
                #bptt_time_in_feats = np.concatenate([time_in_feats, np.zeros((N, self.DEC_LEN-1))], axis=1)
                bptt_event_in = np.array(event_in_seq)
                bptt_time_in = np.array(time_in_seq)
                bptt_time_in_feats = np.array(time_in_feats)
            else:
                #bptt_event_in = event_in_seq[:, self.BPTT-1+simul_idx]
                bptt_event_in = np.asarray(simul_event_preds[-1])
                bptt_event_in = np.concatenate([np.expand_dims(bptt_event_in, axis=-1),
                                                np.zeros((bptt_event_in.shape[0], self.BPTT-1))],
                                                axis=-1)
                #bptt_time_in = time_in_seq[:, self.BPTT-1+simul_idx]
                bptt_time_in = np.asarray(simul_time_preds[-1])
                bptt_time_in = np.concatenate([np.expand_dims(bptt_time_in, axis=-1),
                                               np.zeros((bptt_time_in.shape[0], self.BPTT-1))],
                                               axis=-1)

                bptt_time_in_feats = np.asarray(simul_time_preds[-1]) // 3600 % 24
                bptt_time_in_feats = np.concatenate([np.expand_dims(bptt_time_in_feats, axis=-1),
                                                     np.zeros((bptt_time_in_feats.shape[0], self.BPTT-1))],
                                                     axis=-1)

            if simul_idx == 0:
                initial_time = np.zeros(bptt_time_in.shape[0])
            elif simul_idx == 1:
                initial_time = [seq[-1] for seq in time_in_seq]
            elif simul_idx > 1:
                initial_time = simul_time_preds[-2]

            feed_dict = {
                self.initial_state: cur_state,
                self.initial_time: initial_time,
                self.events_in: bptt_event_in,
                self.times_in: bptt_time_in,
                self.times_in_feats: bptt_time_in_feats,
            }

            bptt_hidden_states, bptt_events_pred, cur_state, D, WT = self.sess.run(
                [self.hidden_states, self.event_preds, self.final_state, self.D, self.WT],
                feed_dict=feed_dict
            )

            # TODO: This calculation is completely ignoring the clipping which
            # happens during the inference step.
            [Vt, Vw, bt, bw, wt]  = self.sess.run([self.Vt, self.Vw, self.bt, self.bw, self.wt])

            val = self.sess.run(self.val, feed_dict=feed_dict)
            #print(val)
            #print(val.shape[0], np.sum(np.isfinite(val)), val)
            if not is_nowcast:
                #print(D, WT)
                if self.ALG_NAME in ['rmtpp', 'rmtpp_mode', 'rmtpp_splusintensity', 'rmtpp_negw', 'rmtpp_splusintensity_negw']:
                    WT = np.ones((len(event_in_seq), 1)) * WT
                elif self.ALG_NAME in ['rmtpp_whparam', 'rmtpp_mode_whparam']:
                    raise NotImplemented('For whparam methods')
                    WT = self.wt_hparam

                all_hidden_states.extend(bptt_hidden_states)
                simul_event_preds_softmax.append(bptt_events_pred[-1])
                #print(bptt_events_pred[-1], np.array(bptt_events_pred[-1]).shape)
                simul_event_preds.extend([np.argmax(bptt_events_pred[-1], axis=-1)+1])

                if simul_idx==0:
                    time_pred_last = [seq[-1] for seq in time_in_seq]
                else:
                    time_pred_last = simul_time_preds[-1]

                step_time_preds = time_pred_last + val[:, 0]

                total_vals += val[:, 0]
                for idx, (t_val, offset) in enumerate(zip(total_vals, offsets_normalized)):
                    if t_val>offset:
                        pred_idxes[idx] += 1
                        if pred_idxes[idx] == 0:
                            begin_idxes[idx] = simul_idx
                        if pred_idxes[idx] == dec_len_for_eval:
                            end_idxes[idx] = simul_idx


                simul_time_preds.append(step_time_preds)
                simul_idx += 1

            else:
                all_gaps_preds = val[:, :-1]
                bptt_events_pred = np.stack(bptt_events_pred, axis=1)[:, :-1]
                print(bptt_events_pred.shape, val.shape)
                all_event_preds_softmax = bptt_events_pred
                all_event_preds = np.argmax(bptt_events_pred, axis=-1) + 1
                #all_gaps_preds = [time_preds-time_in \
                #                    for time_preds, time_in in \
                #                    zip(all_time_preds, time_in_seq)]
                #all_time_preds = [np.cumsum(gaps_preds) for gaps_preds in all_gaps_preds]
                all_time_preds = [gaps_preds + time_in[:-1] \
                                    for gaps_preds, time_in in \
                                    zip(all_gaps_preds, time_in_seq)]
                break


        if not is_nowcast:
            simul_time_preds = np.array(simul_time_preds).T
            all_time_preds = [sml_pred[b_idx:e_idx] for sml_pred, b_idx, e_idx in
                                zip(simul_time_preds, begin_idxes, end_idxes)]
            all_time_preds = np.array(all_time_preds)
            all_gaps_preds = [time_preds-np.concatenate([[time_in[-1]+off], time_preds[:-1]]) \
                                for off, time_preds, time_in in \
                                zip(offsets_normalized, all_time_preds, time_in_seq)]
            assert np.isfinite(all_time_preds).sum() == all_time_preds.size

            simul_event_preds_softmax = np.transpose(np.array(simul_event_preds_softmax),
                                                     axes=[1, 0, 2])
            simul_event_preds = np.array(simul_event_preds).T

            all_event_preds = [sml_pred[b_idx:e_idx] for sml_pred, b_idx, e_idx in
                                zip(simul_event_preds, begin_idxes, end_idxes)]
            all_event_preds = np.array(all_event_preds)

            all_event_preds_softmax = [sml_pred[b_idx:e_idx] for sml_pred, b_idx, e_idx in
                                zip(simul_event_preds_softmax, begin_idxes, end_idxes)]
            all_event_preds_softmax = np.array(all_event_preds_softmax)

            all_time_preds = all_time_preds[:, :dec_len_for_eval]
            all_event_preds = all_event_preds[:, :dec_len_for_eval]
            all_event_preds_softmax = all_event_preds_softmax[:, :dec_len_for_eval]

        end_time = time.time()
        inference_time = end_time - start_time

        return all_time_preds, all_gaps_preds, all_event_preds, all_event_preds_softmax, inference_time, offsets_normalized

    def eval(self, time_preds, time_true, event_preds, event_true,
             time_input_last, event_preds_softmax, offsets,
             is_nowcast=False, time_inputs=None):
        """Prints evaluation of the model on the given dataset."""
        # Print test error once every epoch:
        mae, total_valid = MAE(time_preds, time_true, event_true)
        acc, _ = ACC(event_preds, event_true)
        #print('** MAE = {:.3f}; valid = {}, ACC = {:.3f}'.format(
        #    mae, total_valid, acc))
        if not is_nowcast:
            if time_input_last is not None:
                #gap_true = time_true - np.concatenate([time_input_last, time_true[:, :-1]], axis=1)
                #gap_preds = time_preds - np.concatenate([time_input_last, time_preds[:, :-1]], axis=1)
                gap_true = [seq - np.concatenate([last+off, seq[:-1]]) for seq, last, off in zip(time_true, time_input_last, offsets)]
                gap_preds = [seq - np.concatenate([last+off, seq[:-1]]) for seq, last, off in zip(time_preds, time_input_last, offsets)]
                gap_mae, gap_total_valid = MAE(gap_true, gap_preds, event_true)
                gap_dtw = DTW(gap_true, gap_preds, event_true)
            else:
                gap_mae = None
                gap_dtw = None
        else:
            gap_true = [np.array(true_out)-np.array(true_in[:-1]) for true_out, true_in\
                            in zip(time_true, time_inputs)]
            gap_preds = [np.array(preds)-np.array(true[:-1]) for preds, true \
                            in zip(time_preds, time_inputs)]
            gap_mae, gap_total_valid = MAE(gap_true, gap_preds, event_true)
            gap_dtw = DTW(gap_true, gap_preds, event_true)


        if event_preds_softmax is not None:
            mrr = MRR(event_preds_softmax, event_true)
        else:
            mrr = None

        step_mae = mae/total_valid
        step_acc = acc/total_valid
        cum_acc = np.cumsum(acc)/np.cumsum(total_valid)
        step_gap_mae = gap_mae/total_valid
        cum_gap_mae = np.cumsum(gap_mae)/np.cumsum(total_valid)
        step_mrr = mrr/total_valid
        cum_mrr = np.cumsum(mrr)/np.cumsum(total_valid)

        return step_mae.tolist(), total_valid.tolist(), step_acc.tolist(), step_gap_mae.tolist(), step_mrr.tolist(), gap_dtw.tolist(), cum_gap_mae.tolist(), cum_acc.tolist(), cum_mrr.tolist()

    def predict_test(self, data, single_threaded=False):
        """Make (time, event) predictions on the test data."""
        return self.predict(event_in_seq=data['test_event_in_seq'],
                            time_in_seq=data['test_time_in_seq'],
                            decoder_length=data['decoder_length'],
                            single_threaded=single_threaded)

    def predict_train(self, data, single_threaded=False, batch_size=None):
        """Make (time, event) predictions on the training data."""
        if batch_size == None:
            batch_size = data['train_event_in_seq'].shape[0]

        return self.predict(event_in_seq=data['train_event_in_seq'][0:batch_size, :],
                            time_in_seq=data['train_time_in_seq'][0:batch_size, :],
                            single_threaded=single_threaded)
