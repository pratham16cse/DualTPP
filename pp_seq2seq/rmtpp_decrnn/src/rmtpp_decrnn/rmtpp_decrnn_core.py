import tensorflow as tf
import numpy as np
import os
import decorated_options as Deco
from .utils import create_dir, variable_summaries, MAE, RMSE, ACC, PERCENT_ERROR
from scipy.integrate import quad
import multiprocessing as MP
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

ETH = 10.0
__EMBED_SIZE = 4
__HIDDEN_LAYER_SIZE = 16  # 64, 128, 256, 512, 1024
epsilon = 0.1

def_opts = Deco.Options(
    batch_size=64,          # 16, 32, 64

    learning_rate=0.1,      # 0.1, 0.01, 0.001
    momentum=0.9,
    decay_steps=100,
    decay_rate=0.001,

    l2_penalty=0.001,         # Unused

    float_type=tf.float32,

    seed=42,
    scope='RMTPP_DECRNN',
    alg_name='rmtpp_decrnn',
    save_dir='./save.rmtpp_decrnn/',
    summary_dir='./summary.rmtpp_decrnn/',

    device_gpu='/gpu:0',
    device_cpu='/cpu:0',

    bptt=20,
    decoder_length=10,
    cpu_only=False,

    normalization=None,
    constraints="default",

    patience=0,
    stop_criteria='per_epoch_val_err',
    epsilon=0.0,

    share_dec_params=True,
    init_zero_dec_state=True,
    concat_final_enc_state=True,
    num_extra_dec_layer=0,
    concat_before_dec_update=False,
    mark_triggers_time=True,

    wt_hparam=1.0,

    embed_size=__EMBED_SIZE,
    Wem=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,

    Wt=lambda hidden_layer_size: 0.001*np.ones((1, hidden_layer_size)),
    Wh=lambda hidden_layer_size: 0.001*np.ones((hidden_layer_size)),
    bh=lambda hidden_layer_size: 0.001*np.ones((1, hidden_layer_size)),
    Ws=lambda hidden_layer_size: 0.001*np.ones((hidden_layer_size)),
    bs=lambda hidden_layer_size: 0.001*np.ones((1, hidden_layer_size)),
    wt=1.0,
    Wy=lambda hidden_layer_size: 0.001*np.ones((__EMBED_SIZE, hidden_layer_size)),
    Vy=lambda hidden_layer_size, num_categories: 0.001*np.ones((hidden_layer_size, num_categories)),
    Vt=lambda hidden_layer_size, decoder_length: 0.001*np.ones((decoder_length, hidden_layer_size)),
    Vw=lambda hidden_layer_size, decoder_length: 0.001*np.ones((decoder_length, hidden_layer_size)),
    bt=lambda decoder_length: 0.001*np.ones((decoder_length)), # bt is provided by the base_rate
    bw=lambda decoder_length: 0.001*np.ones((decoder_length)), # bw is provided by the base_rate
    bk=lambda hidden_layer_size, num_categories: 0.001*np.ones((1, num_categories)),
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


class RMTPP_DECRNN:
    """Class implementing the Recurrent Marked Temporal Point Process model."""

    @Deco.optioned()
    def __init__(self, sess, num_categories, hidden_layer_size, batch_size,
                 learning_rate, momentum, l2_penalty, embed_size,
                 float_type, bptt, decoder_length, seed, scope, alg_name,
                 save_dir, decay_steps, decay_rate,
                 device_gpu, device_cpu, summary_dir, cpu_only, constraints,
                 patience, stop_criteria, epsilon, share_dec_params,
                 init_zero_dec_state, concat_final_enc_state, num_extra_dec_layer, concat_before_dec_update,
                 mark_triggers_time,
                 Wt, Wem, Wh, bh, Ws, bs, wt, Wy, Vy, Vt, Vw, bk, bt, bw, wt_hparam):
        self.HIDDEN_LAYER_SIZE = hidden_layer_size
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
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

        self.wt_hparam = wt_hparam

        self.PATIENCE = patience
        self.STOP_CRITERIA = stop_criteria
        self.EPSILON = epsilon
        if self.STOP_CRITERIA == 'epsilon':
            assert self.EPSILON > 0.0

        self.SHARE_DEC_PARAMS = share_dec_params
        self.INIT_ZERO_DEC_STATE = init_zero_dec_state
        self.CONCAT_FINAL_ENC_STATE = concat_final_enc_state
        self.NUM_EXTRA_DEC_LAYER = num_extra_dec_layer
        self.CONCAT_BEFORE_DEC_UPDATE = concat_before_dec_update
        self.MARK_TRIGGERS_TIME = mark_triggers_time

        if self.CONCAT_FINAL_ENC_STATE:
            self.DEC_STATE_SIZE = 2 * self.HIDDEN_LAYER_SIZE
        else:
            self.DEC_STATE_SIZE = self.HIDDEN_LAYER_SIZE

        self.sess = sess
        self.seed = seed
        self.last_epoch = 0

        self.rs = np.random.RandomState(seed + 42)
        np.random.seed(42)

        def get_wt_constraint():
            if self.CONSTRAINTS == 'default':
                return lambda x: tf.clip_by_value(x, 1e-5, 20.0)
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

                self.events_out = tf.placeholder(tf.int32, [None, self.DEC_LEN], name='events_out')
                self.times_out = tf.placeholder(self.FLOAT_TYPE, [None, self.DEC_LEN], name='times_out')

                self.mode = tf.placeholder(tf.float32, name='mode')

                self.batch_num_events = tf.placeholder(self.FLOAT_TYPE, [], name='bptt_events')

                self.inf_batch_size = tf.shape(self.events_in)[0]

                # Make variables
                with tf.variable_scope('encoder_state'):
                    self.Wt = tf.get_variable(name='Wt',
                                              shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wt(self.HIDDEN_LAYER_SIZE)))

                    self.Wem = tf.get_variable(name='Wem', shape=(self.NUM_CATEGORIES, self.EMBED_SIZE),
                                               dtype=self.FLOAT_TYPE,
                                               initializer=tf.constant_initializer(Wem(self.NUM_CATEGORIES)))
                    self.Wh = tf.get_variable(name='Wh', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wh(self.HIDDEN_LAYER_SIZE)))
                    self.bh = tf.get_variable(name='bh', shape=(1, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bh(self.HIDDEN_LAYER_SIZE)))

                with tf.variable_scope('decoder_state'):
                    self.Ws = tf.get_variable(name='Ws', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE),
                                                  dtype=self.FLOAT_TYPE,
                                                  initializer=tf.constant_initializer(Ws(self.HIDDEN_LAYER_SIZE)))
                    self.bs = tf.get_variable(name='bs', shape=(1, self.HIDDEN_LAYER_SIZE),
                                                  dtype=self.FLOAT_TYPE,
                                                  initializer=tf.constant_initializer(bs(self.HIDDEN_LAYER_SIZE)))


                with tf.variable_scope('output'):
                    self.wt = tf.get_variable(name='wt', shape=(1, 1),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(wt),
                                              constraint=get_wt_constraint())

                    self.Wy = tf.get_variable(name='Wy', shape=(self.EMBED_SIZE, self.HIDDEN_LAYER_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Wy(self.HIDDEN_LAYER_SIZE)))

                    # The first column of Vy is merely a placeholder (will not be trained).
                    self.Vy = tf.get_variable(name='Vy', shape=(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vy(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES)))
                    self.Vt = tf.get_variable(name='Vt', shape=(self.DEC_LEN, self.DEC_STATE_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vt(self.DEC_STATE_SIZE, self.DEC_LEN)))
                    self.bt = tf.get_variable(name='bt', shape=(1, self.DEC_LEN),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bt(self.DEC_LEN)))
                    self.bk = tf.get_variable(name='bk', shape=(1, self.NUM_CATEGORIES),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bk(self.DEC_STATE_SIZE, num_categories)))
                    self.Vw = tf.get_variable(name='Vw', shape=(self.DEC_LEN, self.DEC_STATE_SIZE),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(Vw(self.DEC_STATE_SIZE, self.DEC_LEN)))
                    self.bw = tf.get_variable(name='bw', shape=(1, self.DEC_LEN),
                                              dtype=self.FLOAT_TYPE,
                                              initializer=tf.constant_initializer(bw(self.DEC_LEN)))

                    if self.SHARE_DEC_PARAMS:
                        print('Sharing Decoder Parameters')
                        self.Vt = tf.tile(self.Vt[0:1, :], [self.DEC_LEN, 1])
                        self.bt = tf.tile(self.bt[:, 0:1], [1, self.DEC_LEN])
                        self.Vw = tf.tile(self.Vw[0:1, :], [self.DEC_LEN, 1])
                        self.bw = tf.tile(self.bw[:, 0:1], [1, self.DEC_LEN])
                    else:
                        print('NOT Sharing Decoder Parameters')

                self.all_vars = [self.Wt, self.Wem, self.Wh, self.bh, self.Ws, self.bs,
                                 self.wt, self.Wy, self.Vy, self.Vt, self.bt, self.bk, self.Vw, self.bw]

                # Add summaries for all (trainable) variables
                with tf.device(device_cpu):
                    for v in self.all_vars:
                        variable_summaries(v)

                # Make graph
                # RNNcell = RNN_CELL_TYPE(HIDDEN_LAYER_SIZE)

                # Initial state for GRU cells
                self.initial_state = state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE],
                                                      dtype=self.FLOAT_TYPE,
                                                      name='initial_state')
                s_state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE],
                                                      dtype=self.FLOAT_TYPE,
                                                      name='initial_state')
                self.initial_time = last_time = tf.zeros((self.inf_batch_size,),
                                                         dtype=self.FLOAT_TYPE,
                                                         name='initial_time')

                self.loss = 0.0
                ones_2d = tf.ones((self.inf_batch_size, 1), dtype=self.FLOAT_TYPE)
                # ones_1d = tf.ones((self.inf_batch_size,), dtype=self.FLOAT_TYPE)

                self.hidden_states = []
                self.event_preds = []

                self.time_LLs = []
                self.mark_LLs = []
                self.log_lambdas = []
                # self.delta_ts = []
                self.times = []

                #----------- Encoder Begin ----------#
                with tf.name_scope('BPTT'):
                    for i in range(self.BPTT):

                        events_embedded = tf.nn.embedding_lookup(self.Wem,
                                                                 tf.mod(self.events_in[:, i] - 1, self.NUM_CATEGORIES))
                        time = self.times_in[:, i]

                        delta_t_prev = tf.expand_dims(time - last_time, axis=-1)

                        last_time = time

                        time_2d = tf.expand_dims(time, axis=-1)

                        # output, state = RNNcell(events_embedded, state)

                        # TODO Does TF automatically broadcast? Then we'll not
                        # need multiplication with tf.ones
                        type_delta_t = True

                        with tf.name_scope('state_recursion'):
                            new_state = tf.tanh(
                                tf.matmul(state, self.Wh) +
                                tf.matmul(events_embedded, self.Wy) +
                                # Two ways of interpretting this term
                                (tf.matmul(delta_t_prev, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                                tf.matmul(ones_2d, self.bh),
                                name='h_t'
                            )
                            state = tf.where(self.events_in[:, i] > 0, new_state, state)
                        self.hidden_states.append(state)
                    self.final_state = self.hidden_states[-1]
                #----------- Encoder End ----------#


                #----------- Decoder Begin ----------#
                # TODO Does affine transformations (Wy) need to be different? Wt is not \
                  # required in the decoder

                if not self.INIT_ZERO_DEC_STATE:
                    s_state = self.final_state
                #s_state = tf.Print(s_state, [self.mode, tf.equal(self.mode, 1.0)], message='mode ')
                self.decoder_states = []
                with tf.name_scope('Decoder'):
                    for i in range(self.DEC_LEN):

                        if not mark_triggers_time:
                            self.decoder_states.append(s_state)

                        events_pred = tf.nn.softmax(
                            tf.minimum(ETH,
                                       tf.matmul(s_state, self.Vy) + ones_2d * self.bk),
                            name='Pr_events'
                        )
                        # events_pred = tf.Print(events_pred, [self.mode], message='mode')
                        self.event_preds.append(events_pred)
                        events = tf.cond(tf.equal(self.mode, 1.0),
                                         lambda: self.events_out[:, i],
                                         #lambda: tf.argmax(events_pred, axis=-1, output_type=tf.int32),
                                         lambda: tf.argmax(events_pred, axis=-1, output_type=tf.int32) + 1)
                        #events = tf.Print(events, [events], message='events')
                        mark_LL = tf.expand_dims(
                            tf.log(
                                tf.maximum(
                                    1e-6,
                                    tf.gather_nd(
                                        events_pred,
                                        tf.concat([
                                            tf.expand_dims(tf.range(self.inf_batch_size), -1),
                                            tf.expand_dims(tf.mod(self.events_out[:,i] - 1, self.NUM_CATEGORIES), -1)
                                        ], axis=1, name='Pr_next_event'
                                        )
                                    )
                                )
                            ), axis=-1, name='log_Pr_next_event'
                        )
                        self.mark_LLs.append(mark_LL)

                        events_embedded = tf.nn.embedding_lookup(self.Wem,
                                                                 tf.mod(events - 1, self.NUM_CATEGORIES))

                        with tf.variable_scope('state_recursion', reuse=tf.AUTO_REUSE):
                            new_state = tf.tanh(
                                tf.matmul(s_state, self.Ws) +
                                tf.matmul(events_embedded, self.Wy) +
                                tf.matmul(ones_2d, self.bs),
                                name='s_t'
                            )
                            if self.CONCAT_BEFORE_DEC_UPDATE:
                                new_state = tf.concat([new_state, self.final_state], axis=-1)
                            if self.NUM_EXTRA_DEC_LAYER:
                                names = ['hidden_layer_'+str(hl_id) for hl_id in range(1, self.NUM_EXTRA_DEC_LAYER+1)]
                                for name in names:
                                    new_state = tf.layers.dense(new_state,
                                                                self.HIDDEN_LAYER_SIZE,
                                                                name=name,
                                                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                                                                activation=tf.nn.relu)
                            # if self.CONCAT_FINAL_ENC_STATE:
                            #     new_state = tf.concat([new_state_, self.final_state], axis=-1)
                            # else:
                            #     new_state = new_state_

                            s_state = new_state

                        if mark_triggers_time:
                            self.decoder_states.append(s_state)

                    self.event_preds = tf.stack(self.event_preds, axis=1)

                    # ------ Begin time-prediction ------ #
                    self.decoder_states = tf.stack(self.decoder_states, axis=1)
                    if self.CONCAT_FINAL_ENC_STATE:
                        decoder_states_concat = tf.concat([self.decoder_states,
                                                           tf.tile(tf.expand_dims(self.final_state, axis=1),
                                                                   [1, self.DEC_LEN, 1])],
                                                           axis=-1)
                    else:
                        decoder_states_concat = self.decoder_states
                    times_out_prev = tf.concat([self.times_in[:, -1:], self.times_out[:, :-1]], axis=1)

                    gaps = self.times_out-times_out_prev

                    times_prev = tf.cumsum(tf.concat([self.times_in[:, -1:], gaps[:, :-1]], axis=1), axis=1)

                    base_intensity_bt = self.bt
                    base_intensity_bw = self.bw

                    newVt = tf.tile(tf.expand_dims(self.Vt, axis=0), [tf.shape(self.decoder_states)[0], 1, 1]) 
                    newVw = tf.tile(tf.expand_dims(self.Vw, axis=0), [tf.shape(self.decoder_states)[0], 1, 1]) 

                    D = tf.reduce_sum(decoder_states_concat * newVt, axis=2) + base_intensity_bt

                    # D = tf.squeeze(tf.tensordot(self.decoder_states, self.Vt, axes=[[2],[0]]), axis=-1) + base_intensity_bt
                    D = get_D_constraint()(D)
                    if self.ALG_NAME in ['rmtpp_decrnn_wcmpt', 'rmtpp_decrnn_mode_wcmpt']:
                        WT = tf.reduce_sum(decoder_states_concat * newVw, axis=2) + base_intensity_bw
                        # WT = tf.squeeze(tf.tensordot(self.decoder_states, self.Vw, axes=[[2],[0]]), axis=-1) + base_intensity_bw
                        WT = get_WT_constraint()(WT)
                        WT = tf.clip_by_value(WT, 0.0, 10.0)
                    elif self.ALG_NAME in ['rmtpp_decrnn', 'rmtpp_decrnn_mode']:
                        WT = self.wt
                    elif self.ALG_NAME in ['rmtpp_decrnn_whparam', 'rmtpp_decrnn_mode_whparam']:
                        WT = self.wt_hparam
                    log_lambda_ = (D + gaps * WT)
                    lambda_ = tf.exp(tf.minimum(ETH, log_lambda_), name='lambda_')
                    log_f_star = (log_lambda_
                                  + (1.0 / WT) * tf.exp(tf.minimum(ETH, D))
                                  - (1.0 / WT) * lambda_)


                with tf.name_scope('loss_calc'):

                    self.mark_LLs = tf.squeeze(tf.stack(self.mark_LLs, axis=1), axis=-1)
                    self.time_LLs = log_f_star
                    step_LLs = self.time_LLs + self.mark_LLs
                    #step_LLs = self.mark_LLs
                    #step_LLs = self.time_LLs

                    # In the batch some of the sequences may have ended before we get to the
                    # end of the seq. In such cases, the events will be zero.
                    # TODO Figure out how to do this with RNNCell, LSTM, etc.
                    num_events = tf.reduce_sum(tf.where(self.events_out > 0,
                                               tf.ones(shape=(self.inf_batch_size, self.DEC_LEN), dtype=self.FLOAT_TYPE),
                                               tf.zeros(shape=(self.inf_batch_size, self.DEC_LEN), dtype=self.FLOAT_TYPE)),
                                               name='num_events')

                    # TODO(PD) Is the sign of loss correct?
                    self.loss = (-1) * tf.reduce_sum(
                        tf.where(self.events_out > 0,
                                 step_LLs / self.batch_num_events,
                                 tf.zeros(shape=(self.inf_batch_size, self.DEC_LEN)))
                    )

                    # self.loss -= tf.cond(num_events > 0,
                    #                      lambda: tf.reduce_sum(
                    #                          tf.where(self.events_out[:, i] > 0,
                    #                                   tf.squeeze(step_LL),
                    #                                   tf.zeros(shape=(self.inf_batch_size,))),
                    #                          name='batch_bptt_loss'),
                    #                      lambda: 0.0)

                self.log_lambdas = log_lambda_

                # self.delta_ts.append(tf.clip_by_value(delta_t, 0.0, np.inf))
                self.times = self.times_in


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
                    variable_summaries(self.decoder_states, name='agg-decoder-states')
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

        # if dec_len_for_eval==0:
        #     dec_len_for_eval=training_data['decoder_length']
        if restore_path is None:
            restore_path = self.SAVE_DIR
        """Train the model given the training data.

        If with_evals is an integer, then that many elements from the test set
        will be tested.
        """
        create_dir(self.SAVE_DIR)
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

        best_dev_mae, best_test_mae = np.inf, np.inf
        best_dev_time_preds, best_dev_event_preds = [], []
        best_test_time_preds, best_test_event_preds = [], []
        best_epoch = 0
        total_loss = 0.0
        train_loss_list = list()

        idxes = list(range(len(train_event_in_seq)))
        n_batches = len(idxes) // self.BATCH_SIZE

        print('Training with ', self.STOP_CRITERIA, 'stop_criteria and ', num_epochs, 'num_epochs')
        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            self.rs.shuffle(idxes)

            print("Starting epoch...", epoch)
            total_loss_prev = total_loss
            total_loss = 0.0

            for batch_idx in range(n_batches):
                batch_idxes = idxes[batch_idx * self.BATCH_SIZE:(batch_idx + 1) * self.BATCH_SIZE]
                batch_event_train_in = train_event_in_seq[batch_idxes, :]
                batch_event_train_out = train_event_out_seq[batch_idxes, :]
                batch_time_train_in = train_time_in_seq[batch_idxes, :]
                batch_time_train_out = train_time_out_seq[batch_idxes, :]

                cur_state = np.zeros((self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE))
                batch_loss = 0.0

                batch_num_events = np.sum(batch_event_train_in > 0)

                initial_time = np.zeros(batch_time_train_in.shape[0])

                feed_dict = {
                    self.initial_state: cur_state,
                    self.initial_time: initial_time,
                    self.events_in: batch_event_train_in,
                    self.events_out: batch_event_train_out,
                    self.times_in: batch_time_train_in,
                    self.times_out: batch_time_train_out,
                    self.batch_num_events: batch_num_events,
                    self.mode: 1.0, # Train Mode
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
                        _, cur_state, loss_ = \
                            self.sess.run([self.update,
                                           self.final_state, self.loss],
                                          feed_dict=feed_dict)
                batch_loss = loss_
                total_loss += batch_loss
                if batch_idx % 10 == 0:
                    print('Loss during batch {} last BPTT = {:.3f}, lr = {:.5f}'
                          .format(batch_idx, batch_loss, self.sess.run(self.learning_rate)))

            if self.STOP_CRITERIA == 'per_epoch_val_err' and epoch >= self.PATIENCE:

                minTime, maxTime = training_data['minTime'], training_data['maxTime']

                if eval_train_data:
                    print('Running evaluation on train, dev, test: ...')
                else:
                    print('Running evaluation on dev, test: ...')

                if eval_train_data:
                    plt_time_out_seq = training_data['train_time_out_seq']
                    plt_tru_gaps = plt_time_out_seq - np.concatenate([training_data['train_time_in_seq'][:, -1:], plt_time_out_seq[:, :-1]], axis=1)
                    train_time_preds, train_event_preds = self.predict(training_data['train_event_in_seq'],
                                                                   training_data['train_time_in_seq'],
                                                                   training_data['decoder_length'],
                                                                   plt_tru_gaps,
                                                                   single_threaded=True)
                    train_time_preds = train_time_preds * (maxTime - minTime) + minTime
                    train_time_out_seq = training_data['train_time_out_seq'] * (maxTime - minTime) + minTime
                    train_mae, train_total_valid, train_acc = self.eval(train_time_preds, train_time_out_seq,
                                                                  train_event_preds, training_data['train_event_out_seq'])
                    print('TRAIN: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                        train_mae, train_total_valid, train_acc))
                else:
                    train_mae, train_acc, train_time_preds, train_event_preds = None, None, np.array([]), np.array([])

                plt_time_out_seq = training_data['dev_time_out_seq']
                plt_tru_gaps = plt_time_out_seq - np.concatenate([training_data['dev_time_in_seq'][:, -1:], plt_time_out_seq[:, :-1]], axis=1)
                dev_time_preds, dev_event_preds = self.predict(training_data['dev_event_in_seq'],
                                                               training_data['dev_time_in_seq'],
                                                               training_data['decoder_length'],
                                                               plt_tru_gaps,
                                                               single_threaded=True)
                # dev_time_out_seq = training_data['dev_time_out_seq'] * (maxTime - minTime) + minTime
                dev_time_preds = dev_time_preds * (maxTime - minTime) + minTime
                dev_time_out_seq = np.array(training_data['dev_actual_time_out_seq'])
                dev_time_in_seq = training_data['dev_time_in_seq'] * (maxTime - minTime) + minTime
                gaps = dev_time_preds - np.concatenate([dev_time_in_seq[:, -1:], dev_time_preds[:, :-1]], axis=-1)
                unnorm_gaps = gaps * training_data['dev_avg_gaps']
                unnorm_gaps = np.cumsum(unnorm_gaps, axis=1)
                dev_time_preds = unnorm_gaps + training_data['dev_actual_time_in_seq']
                
                dev_mae, dev_total_valid, dev_acc = self.eval(dev_time_preds, dev_time_out_seq,
                                                              dev_event_preds, training_data['dev_event_out_seq'])
                print('DEV: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                    dev_mae, dev_total_valid, dev_acc))
    
                plt_time_out_seq = training_data['test_time_out_seq']
                plt_tru_gaps = plt_time_out_seq - np.concatenate([training_data['test_time_in_seq'][:, -1:], plt_time_out_seq[:, :-1]], axis=1)
                test_time_preds, test_event_preds = self.predict(training_data['test_event_in_seq'],
                                                                 training_data['test_time_in_seq'],
                                                                 training_data['decoder_length'],
                                                                 plt_tru_gaps,
                                                                 single_threaded=True)
                test_time_preds = test_time_preds * (maxTime - minTime) + minTime
                # test_time_out_seq = training_data['test_time_out_seq'] * (maxTime - minTime) + minTime
                test_time_out_seq = np.array(training_data['test_actual_time_out_seq'])
                test_time_in_seq = training_data['test_time_in_seq'] * (maxTime - minTime) + minTime
                gaps = test_time_preds - np.concatenate([test_time_in_seq[:, -1:], test_time_preds[:, :-1]], axis=-1)
                # gaps = test_time_preds - np.concatenate([training_data['test_actual_time_in_seq'], test_time_preds[:, :-1]], axis=-1)
                # tru_gaps = test_time_out_seq - np.concatenate([training_data['test_actual_time_in_seq'], test_time_out_seq[:, :-1]], axis=1)
                tru_gaps = test_time_out_seq - np.concatenate([training_data['test_actual_time_in_seq'], test_time_out_seq[:, :-1]], axis=1)
                # unnorm_true_gaps = tru_gaps * training_data['test_avg_gaps']
                unnorm_gaps = gaps * training_data['test_avg_gaps']
                unnorm_gaps = np.cumsum(unnorm_gaps, axis=1)
                test_time_preds = unnorm_gaps + training_data['test_actual_time_in_seq']

                # print('Predicted gaps')
                # print(gaps)

                # print('UnNormed Predicted gaps')
                # print(unnorm_gaps)
                # print('True gaps')
                # print(tru_gaps)

                # print('UnNormed True gaps')
                # print(unnorm_true_gaps)
                test_mae, test_total_valid, test_acc = self.eval(test_time_preds, test_time_out_seq,
                                                                 test_event_preds, training_data['test_event_out_seq'])
                print('TEST: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                    test_mae, test_total_valid, test_acc))
    
                if dev_mae < best_dev_mae:
                    best_epoch = epoch
                    best_train_mae, best_dev_mae, best_test_mae = train_mae, dev_mae, test_mae
                    best_train_acc, best_dev_acc, best_test_acc = train_acc, dev_acc, test_acc
                    best_train_event_preds, best_train_time_preds  = train_event_preds, train_time_preds
                    best_dev_event_preds, best_dev_time_preds  = dev_event_preds, dev_time_preds
                    best_test_event_preds, best_test_time_preds  = test_event_preds, test_time_preds
                    best_w = self.sess.run(self.wt).tolist()
    
                    checkpoint_dir = os.path.join(self.SAVE_DIR, 'hls_'+str(self.HIDDEN_LAYER_SIZE))
                    checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
                    saver.save(self.sess, checkpoint_path)# , global_step=step)
                    print('Model saved at {}'.format(checkpoint_path))


            # self.sess.run(self.increment_global_step)
            train_loss_list.append(total_loss)
            print('Loss on last epoch = {:.4f}, new lr = {:.5f}, global_step = {}'
                  .format(total_loss,
                          self.sess.run(self.learning_rate),
                          self.sess.run(self.global_step)))

            print(total_loss, total_loss_prev, abs(total_loss-total_loss_prev))
            if self.STOP_CRITERIA == 'epsilon' and epoch >= self.PATIENCE and abs(total_loss-total_loss_prev) < self.EPSILON:
                break

            if one_batch:
                print('Breaking after just one batch.')
                break

        # Remember how many epochs we have trained.
        if num_epochs>0:
            self.last_epoch += num_epochs

        if self.STOP_CRITERIA == 'epsilon':
            print('w:', self.sess.run(self.wt).tolist())

            minTime, maxTime = training_data['minTime'], training_data['maxTime']
            if eval_train_data:
                print('Running evaluation on train, dev, test: ...')
            else:
                print('Running evaluation on dev, test: ...')

            if eval_train_data: 
                assert 1==0
                #Not yet Implemented
                plt_time_out_seq = training_data['train_time_out_seq']
                plt_tru_gaps = plt_time_out_seq[:,:dec_len_for_eval] - np.concatenate([training_data['train_time_in_seq'][:, -1:], plt_time_out_seq[:, :dec_len_for_eval-1]], axis=1)
                train_time_preds, train_event_preds = self.predict(training_data['train_event_in_seq'],
                                                               training_data['train_time_in_seq'],
                                                               training_data['decoder_length'],
                                                               plt_tru_gaps,
                                                               single_threaded=True)
                train_time_preds = train_time_preds[:,:dec_len_for_eval] * (maxTime - minTime) + minTime
                train_time_out_seq = training_data['train_time_out_seq'] * (maxTime - minTime) + minTime
                train_mae, train_total_valid, train_acc = self.eval(train_time_preds, train_time_out_seq,
                                                              train_event_preds, training_data['train_event_out_seq'])
                print('TRAIN: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                    train_mae, train_total_valid, train_acc))
            else:
                train_mae, train_acc, train_time_preds, train_event_preds = None, None, np.array([]), np.array([])


            plt_time_out_seq = training_data['dev_time_out_seq']
            plt_tru_gaps = plt_time_out_seq[:,:dec_len_for_eval] - np.concatenate([training_data['dev_time_in_seq'][:, -1:], plt_time_out_seq[:, :dec_len_for_eval-1]], axis=1)
            dev_time_preds, dev_event_preds = self.predict(training_data['dev_event_in_seq'],
                                                           training_data['dev_time_in_seq'],
                                                           training_data['decoder_length'],
                                                           plt_tru_gaps,
                                                           single_threaded=True)
            dev_time_preds = dev_time_preds[:,:dec_len_for_eval] * (maxTime - minTime) + minTime
            dev_time_out_seq = np.array(training_data['dev_actual_time_out_seq'])[:,:dec_len_for_eval]
            dev_time_in_seq = training_data['dev_time_in_seq'] * (maxTime - minTime) + minTime
            gaps = dev_time_preds - np.concatenate([dev_time_in_seq[:, -1:], dev_time_preds[:, :-1]], axis=-1)
            unnorm_gaps = gaps * training_data['dev_avg_gaps']
            unnorm_gaps = np.cumsum(unnorm_gaps, axis=1)
            dev_time_preds = unnorm_gaps + training_data['dev_actual_time_in_seq']
            
            dev_mae, dev_total_valid, dev_acc = self.eval(dev_time_preds, dev_time_out_seq,
                                                          dev_event_preds, training_data['dev_event_out_seq'])
            print('DEV: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                dev_mae, dev_total_valid, dev_acc))

            plt_time_out_seq = training_data['test_time_out_seq']
            plt_tru_gaps = plt_time_out_seq[:,:dec_len_for_eval] - np.concatenate([training_data['test_time_in_seq'][:, -1:], plt_time_out_seq[:, :dec_len_for_eval-1]], axis=1)
            test_time_preds, test_event_preds = self.predict(training_data['test_event_in_seq'],
                                                             training_data['test_time_in_seq'],
                                                             training_data['decoder_length'],
                                                             plt_tru_gaps,
                                                             single_threaded=True)
            test_time_preds = test_time_preds[:,:dec_len_for_eval] * (maxTime - minTime) + minTime
            test_time_out_seq = np.array(training_data['test_actual_time_out_seq'])[:,:dec_len_for_eval]
            test_time_in_seq = training_data['test_time_in_seq'] * (maxTime - minTime) + minTime
            gaps = test_time_preds - np.concatenate([test_time_in_seq[:, -1:], test_time_preds[:, :-1]], axis=-1)
            unnorm_gaps = gaps * training_data['test_avg_gaps']
            unnorm_gaps = np.cumsum(unnorm_gaps, axis=1)
            tru_gaps = test_time_out_seq - np.concatenate([training_data['test_actual_time_in_seq'], test_time_out_seq[:, :-1]], axis=1)
            test_time_preds = unnorm_gaps + training_data['test_actual_time_in_seq']

            # print('UnNormed Predicted gaps')
            # print(unnorm_gaps)
            # print('True gaps')
            # print(tru_gaps)
            
            test_mae, test_total_valid, test_acc = self.eval(test_time_preds, test_time_out_seq,
                                                             test_event_preds, training_data['test_event_out_seq'])
            print('TEST: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                test_mae, test_total_valid, test_acc))

            if dev_mae < best_dev_mae:
                best_epoch = -1
                best_train_mae, best_dev_mae, best_test_mae = train_mae, dev_mae, test_mae
                best_train_acc, best_dev_acc, best_test_acc = train_acc, dev_acc, test_acc
                best_train_event_preds, best_train_time_preds  = train_event_preds, train_time_preds
                best_dev_event_preds, best_dev_time_preds  = dev_event_preds, dev_time_preds
                best_test_event_preds, best_test_time_preds  = test_event_preds, test_time_preds
                best_w = self.sess.run(self.wt).tolist()

                checkpoint_dir = os.path.join(self.SAVE_DIR, 'hls_'+str(self.HIDDEN_LAYER_SIZE))
                checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
                saver.save(self.sess, checkpoint_path)# , global_step=step)
                print('Model saved at {}'.format(checkpoint_path))


        if ckpt and num_epochs==0:
            self.restore()
            minTime, maxTime = training_data['minTime'], training_data['maxTime']

            plt_time_out_seq = training_data['train_time_out_seq']
            plt_tru_gaps = plt_time_out_seq - np.concatenate([training_data['train_time_in_seq'][:, -1:], plt_time_out_seq[:, :-1]], axis=1) 
            plot_dir = os.path.join(self.SAVE_DIR,'train')
            if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
            best_train_time_preds, best_train_event_preds = self.predict(training_data['train_event_in_seq'],
                                                           training_data['train_time_in_seq'],
                                                           training_data['decoder_length'],
                                                           plt_tru_gaps, plot_dir=plot_dir, single_threaded=True)
            best_train_time_preds = best_train_time_preds * (maxTime - minTime) + minTime
            train_time_out_seq = training_data['train_time_out_seq'] * (maxTime - minTime) + minTime
            best_train_mae, train_total_valid, best_train_acc = self.eval(best_train_time_preds, train_time_out_seq,
                                                          best_train_event_preds, training_data['train_event_out_seq'])
            print('TRAIN: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                best_train_mae, train_total_valid, best_train_acc))

            plt_time_out_seq = training_data['dev_time_out_seq']
            plt_tru_gaps = plt_time_out_seq - np.concatenate([training_data['dev_time_in_seq'][:, -1:], plt_time_out_seq[:, :-1]], axis=1) 
            plot_dir = os.path.join(self.SAVE_DIR,'dev')
            if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
            best_dev_time_preds, best_dev_event_preds = self.predict(training_data['dev_event_in_seq'],
                                                           training_data['dev_time_in_seq'],
                                                           training_data['decoder_length'],
                                                           plt_tru_gaps, plot_dir=plot_dir, single_threaded=True)
            best_dev_time_preds = best_dev_time_preds * (maxTime - minTime) + minTime
            dev_time_out_seq = training_data['dev_time_out_seq'] * (maxTime - minTime) + minTime
            best_dev_mae, dev_total_valid, best_dev_acc = self.eval(best_dev_time_preds, dev_time_out_seq,
                                                          best_dev_event_preds, training_data['dev_event_out_seq'])
            print('DEV: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                best_dev_mae, dev_total_valid, best_dev_acc))
    
            plt_time_out_seq = training_data['test_time_out_seq']
            plt_tru_gaps = plt_time_out_seq - np.concatenate([training_data['test_time_in_seq'][:, -1:], plt_time_out_seq[:, :-1]], axis=1)
            plot_dir = os.path.join(self.SAVE_DIR,'test')
            if not os.path.isdir(plot_dir): os.mkdir(plot_dir)
            best_test_time_preds, best_test_event_preds = self.predict(training_data['test_event_in_seq'],
                                                             training_data['test_time_in_seq'],
                                                             training_data['decoder_length'],
                                                             plt_tru_gaps, plot_dir=plot_dir, single_threaded=True)
            best_test_time_preds = best_test_time_preds * (maxTime - minTime) + minTime
            test_time_out_seq = training_data['test_time_out_seq'] * (maxTime - minTime) + minTime
            test_time_in_seq = training_data['test_time_in_seq'] * (maxTime - minTime) + minTime
            gaps = best_test_time_preds - np.concatenate([test_time_in_seq[:, -1:], best_test_time_preds[:, :-1]], axis=-1)
            tru_gaps = test_time_out_seq - np.concatenate([test_time_in_seq[:, -1:], test_time_out_seq[:, :-1]], axis=1)
            print('Predicted gaps')
            print(gaps)
            print(plt_tru_gaps)
            best_test_mae, test_total_valid, best_test_acc = self.eval(best_test_time_preds, test_time_out_seq,
                                                             best_test_event_preds, training_data['test_event_out_seq'])
            print('TEST: MAE = {:.5f}; valid = {}, ACC = {:.5f}'.format(
                best_test_mae, test_total_valid, best_test_acc))
    
            print('Best Epoch:{}, Best Dev MAE:{:.5f}, Best Test MAE:{:.5f}'.format(
                best_epoch, best_dev_mae, best_test_mae))
    
            best_w = self.sess.run(self.wt).tolist()

            return None

        return {
                'best_epoch': best_epoch,
                'best_train_mae': best_train_mae,
                'best_train_acc': best_train_acc,
                'best_dev_mae': best_dev_mae,
                'best_dev_acc': best_dev_acc,
                'best_test_mae': best_test_mae,
                'best_test_acc': best_test_acc,
                'best_train_event_preds': best_train_event_preds.tolist(),
                'best_train_time_preds': best_train_time_preds.tolist(),
                'best_dev_event_preds': best_dev_event_preds.tolist(),
                'best_dev_time_preds': best_dev_time_preds.tolist(),
                'best_test_event_preds': best_test_event_preds.tolist(),
                'best_test_time_preds': best_test_time_preds.tolist(),
                'best_w': best_w,
                'hidden_layer_size': self.HIDDEN_LAYER_SIZE,
                'wt_hparam': self.wt_hparam,
                'checkpoint_dir': checkpoint_dir,
                'train_loss_list': train_loss_list,
               }


    def restore(self):
        """Restore the model from saved state."""
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.SAVE_DIR)
        print('Loading the model from {}'.format(ckpt.model_checkpoint_path))
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, event_in_seq, time_in_seq, decoder_length, plt_tru_gaps, single_threaded=False, plot_dir=False):
        """Treats the entire dataset as a single batch and processes it."""

        def get_wt_constraint():
            if self.CONSTRAINTS == 'default':
                return lambda x: tf.clip_by_value(x, 1e-5, 20.0)
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
                return lambda x: -softplus(-x)
            elif self.CONSTRAINTS == 'unconstrained':
                return lambda x: x
            else:
                print('Constraint on wt not found.')
                assert False

        def get_WT_constraint():
            if self.CONSTRAINTS == 'default':
                return lambda x: np.clip(x, 1e-5, np.inf)
            elif self.CONSTRAINTS in ['c1', 'c2']:
                return lambda x: softplus(x)
            else:
                print('Constraint on wt not found.')
                assert False

        cur_state = np.zeros((len(event_in_seq), self.HIDDEN_LAYER_SIZE))
        initial_time = np.zeros(time_in_seq.shape[0])
        # Feeding dummy values to self.<events/tims>_out placeholders
        event_out_seq = time_out_seq = np.zeros((event_in_seq.shape[0], self.DEC_LEN))

        feed_dict = {
            self.initial_state: cur_state,
            self.initial_time: initial_time,
            self.events_in: event_in_seq,
            self.times_in: time_in_seq,
            self.events_out: event_out_seq,
            self.times_out: time_out_seq,
            self.mode: 0.0 #Test Mode
        }

        all_encoder_states, all_decoder_states, all_event_preds, cur_state = self.sess.run(
            [self.hidden_states, self.decoder_states, self.event_preds, self.final_state],
            feed_dict=feed_dict
        )
        all_event_preds = np.argmax(all_event_preds, axis=-1) + 1
        all_event_preds = np.transpose(all_event_preds)

        # TODO: This calculation is completely ignoring the clipping which
        # happens during the inference step.
        [Vt, Vw, bt, bw, wt]  = self.sess.run([self.Vt, self.Vw, self.bt, self.bw, self.wt])

        global _quad_worker
        def _quad_worker(params):
            batch_idx, (decoder_states, time_pred_last, tru_gap) = params
            preds_i = []
            #print(np.matmul(decoder_states, Vt) + bt)
            for pred_idx, s_i in enumerate(decoder_states):
                t_last = time_pred_last if pred_idx==0 else preds_i[-1]
                D = (np.dot(s_i, Vt[pred_idx,:]) + bt[:,pred_idx]).reshape(-1)
                D = np.clip(D, np.ones_like(D)*-50.0, np.ones_like(D)*50.0)
                D = get_D_constraint()(D)
                c_ = np.exp(np.maximum(D, np.ones_like(D)*-87.0))
                if self.ALG_NAME in ['rmtpp_decrnn_wcmpt', 'rmtpp_decrnn_mode_wcmpt']:
                    WT = (np.dot(s_i, Vw[pred_idx,:]) + bw[:,pred_idx]).reshape(-1)
                    WT = get_WT_constraint()(WT)
                    WT = np.clip(WT, 0.0, 10.0)
                elif self.ALG_NAME in ['rmtpp_decrnn', 'rmtpp_decrnn_mode']:
                    WT = wt
                elif self.ALG_NAME in ['rmtpp_decrnn_whparam', 'rmtpp_decrnn_mode_whparam']:
                    WT = self.wt_hparam

                if self.ALG_NAME in ['rmtpp_decrnn', 'rmtpp_decrnn_wcmpt', 'rmtpp_decrnn_whparam']:
                    args = (c_, WT)
                    val, _err = quad(quad_func, 0, np.inf, args=args)
                    #print(batch_idx, D, c_, WT, val)
                elif self.ALG_NAME in ['rmtpp_decrnn_mode', 'rmtpp_decrnn_mode_wcmpt', 'rmtpp_decrnn_mode_whparam']:
                    val_raw = (np.log(WT) - D)/WT
                    val = np.where(val_raw<0.0, 0.0, val_raw)
                    val = val.reshape(-1)[0]
                    #print(batch_idx, D, c_, WT, val, val_raw)

                assert np.isfinite(val)
                preds_i.append(t_last + val)

                if plot_dir:
                    if self.ALG_NAME in ['rmtpp', 'rmtpp_wcmpt']:
                        mean = val
                        mode = (np.log(WT) - D)/WT
                        mode = np.where(mode<0.0, 0.0, mode)
                        mode = mode.reshape(-1)[0]
                    elif self.ALG_NAME in ['rmtpp_mode', 'rmtpp_mode_wcmpt']:
                        args = (c_, WT)
                        mode = val
                        mean, _ = quad(quad_func, 0, np.inf, args=args)

                    plt_x = np.arange(val-2.0, val+2.0, 0.05)
                    plt_y = density_func(plt_x, c_, WT)
                    plt.plot(plt_x, plt_y.reshape(-1), label='Density')
                    plt.plot(mean, 0.0, 'go', label='mean')
                    plt.plot(mode, 0.0, 'r*', label='mode')
                    plt.plot(tru_gap, 0.0, 'b^', label='True gap')
                    plt.xlabel('Gap')
                    plt.ylabel('Density')
                    plt.legend(loc='best')
                    plt.savefig(os.path.join(plot_dir,'instance_'+str(batch_idx)+'.png'))
                    plt.close()
    
                    #print(batch_idx, D, wt, mode, mean, density_func(mode, D, wt), density_func(mean, D, wt))
                    print(batch_idx, D, c_, WT, mean, density_func(mean, c_, WT))

            return preds_i

        time_pred_last = time_in_seq[:, -1]
        print(all_decoder_states.shape)
        if self.CONCAT_FINAL_ENC_STATE:
            all_decoder_states = np.concatenate([all_decoder_states, np.tile(np.expand_dims(cur_state, axis=1), [1, self.DEC_LEN, 1])], axis=-1)

        if single_threaded:
            all_time_preds = [_quad_worker((idx, (state, t_last, tru_gap))) for idx, (state, t_last, tru_gap) in enumerate(zip(all_decoder_states, time_pred_last, plt_tru_gaps))]
        else:
            with MP.Pool() as pool:
                all_time_preds = pool.map(_quad_worker, enumerate(zip(all_decoder_states, time_pred_last, plt_tru_gaps)))

        all_time_preds = np.asarray(all_time_preds).T
        assert np.isfinite(all_time_preds).sum() == all_time_preds.size

        print('all_time_preds shape:', all_time_preds.shape)

        return np.asarray(all_time_preds).T, np.asarray(all_event_preds).swapaxes(0, 1)

    def eval(self, time_preds, time_true, event_preds, event_true):
        """Prints evaluation of the model on the given dataset."""
        # Print test error once every epoch:
        mae, total_valid = MAE(time_preds, time_true, event_true)
        acc = ACC(event_preds, event_true)
        #print('** MAE = {:.3f}; valid = {}, ACC = {:.3f}'.format(
        #    mae, total_valid, acc))
        return mae, total_valid, acc

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
