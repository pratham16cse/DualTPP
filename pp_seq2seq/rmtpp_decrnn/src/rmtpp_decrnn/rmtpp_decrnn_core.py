from tensorflow.contrib.keras import preprocessing
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import os
import decorated_options as Deco
from .utils import create_dir, variable_summaries, MAE, RMSE, ACC, MRR, PERCENT_ERROR, DTW
from .utils import get_output_indices, get_attn_seqs_from_offset, timestampTotime
from scipy.integrate import quad
import multiprocessing as MP
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time, sys
from collections import OrderedDict, Counter
from operator import itemgetter

pad_sequences = preprocessing.sequence.pad_sequences
def getHour(t):
    ts = timestampTotime(t)
    return ts[2] + ts[3]*1.0/60 + ts[4]*1.0/3600

trim_seq_dec_len = lambda sequences, dec_len: [seq[:dec_len] for seq in sequences]
ETH = 10.0
clip = lambda D: tf.clip_by_value(D, -10.0, 10.0)
ONE_DAY_SECS = 3600.0*24.0

__EMBED_SIZE = 4
__HIDDEN_LAYER_SIZE = 16  # 64, 128, 256, 512, 1024
epsilon = 0.1

def_opts = Deco.Options(
    params_alias_named={},

    hidden_layer_size=16,

    batch_size=64,          # 16, 32, 64

    learning_rate=0.1,      # 0.1, 0.01, 0.001
    temperature=5.0,
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
    decoder_length=5,
    cpu_only=False,

    normalization=None,
    constraints="default",

    patience=0,
    stop_criteria='per_epoch_val_err',
    epsilon=0.0001,

    share_dec_params=True,
    init_zero_dec_state=True,
    concat_final_enc_state=True,
    num_extra_dec_layer=0,
    concat_before_dec_update=False,
    mark_triggers_time=True,
    mark_loss=True,
    plot_pred_dev=True,
    plot_pred_test=False,
    position_encode=False,
    use_intensity=True,

    num_feats=1,
    use_time_features=False,
    use_avg_gaps=False,

    offset=0.0,
    max_offset=0.0,
    discrete_offset_feats=False,
    sample_trn_offsets=True,

    wt_hparam=1.0,

    enc_cell_type='manual',
    dec_cell_type='manual',

    num_discrete_states = 5,

    embed_size=__EMBED_SIZE,
    embed_gaps=None,
    gap_embed_size=8,
    Wem=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,
    Wem_feats=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,

    Wt=lambda hidden_layer_size: np.random.normal(size=(1, hidden_layer_size)),
    Wh=lambda hidden_layer_size: np.random.normal(size=(hidden_layer_size)),
    bh=lambda hidden_layer_size: np.random.normal(size=(1, hidden_layer_size)),
    Ws=lambda hidden_layer_size: np.random.normal(size=(hidden_layer_size)),
    bs=lambda hidden_layer_size: np.random.normal(size=(1, hidden_layer_size)),
    wt=1.0,
    wt_attn=1.0,
    Wy=lambda hidden_layer_size: np.random.normal(size=(__EMBED_SIZE, hidden_layer_size)),
    Vy=lambda hidden_layer_size, num_categories: np.random.normal(size=(hidden_layer_size, num_categories)),
    Vt=lambda hidden_layer_size, decoder_length: np.random.normal(size=(decoder_length, hidden_layer_size)),
    Vw=lambda hidden_layer_size, decoder_length: np.random.normal(size=(decoder_length, hidden_layer_size)),
    bt=lambda decoder_length: np.random.normal(size=(decoder_length)), # bt is provided by the base_rate
    bw=lambda decoder_length: np.random.normal(size=(decoder_length)), # bw is provided by the base_rate
    bk=lambda hidden_layer_size, num_categories: np.random.normal(size=(1, num_categories)),
    Wem_position=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,
    enc_Wem_position=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,
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

class RMTPP_DECRNN:
    """Class implementing the Recurrent Marked Temporal Point Process model."""

    @Deco.optioned()
    def __init__(self, sess, num_categories, params_named, params_alias_named, hidden_layer_size, batch_size,
                 learning_rate, temperature, momentum, l2_penalty, embed_size,
                 float_type, bptt, decoder_length, seed, scope, alg_name,
                 save_dir, decay_steps, decay_rate,
                 device_gpu, device_cpu, summary_dir, cpu_only, constraints,
                 patience, stop_criteria, epsilon, share_dec_params,
                 init_zero_dec_state, concat_final_enc_state, num_extra_dec_layer, concat_before_dec_update,
                 mark_triggers_time, mark_loss,
                 Wt, Wem, Wh, bh, Ws, bs, wt, wt_attn, Wy, Vy, Vt, Vw, bk, bt, bw, wt_hparam, Wem_position, enc_Wem_position,
                 plot_pred_dev, plot_pred_test, enc_cell_type, dec_cell_type, num_discrete_states,
                 position_encode, use_intensity, num_feats, use_time_features, use_avg_gaps, offset, max_offset,
                 discrete_offset_feats, sample_trn_offsets, embed_gaps, gap_embed_size):

        self.seed = seed
        tf.set_random_seed(self.seed)

        self.PARAMS_NAMED = OrderedDict(params_named)
        self.PARAMS_ALIAS_NAMED = params_alias_named

        self.HIDDEN_LAYER_SIZE = self.PARAMS_NAMED['hidden_layer_size'] #hidden_layer_size
        self.NUM_ENC_LAYERS = self.PARAMS_NAMED['num_enc_layers']
        self.NUM_DEC_LAYERS = self.PARAMS_NAMED['num_dec_layers']
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = self.PARAMS_NAMED['learning_rate'] #learning_rate
        self.TEMPERATURE = temperature
        self.MOMENTUM = momentum
        self.L2_PENALTY = l2_penalty
        self.EMBED_SIZE = embed_size
        self.EMBED_GAPS = embed_gaps
        self.GAP_EMBED_SIZE = gap_embed_size
        self.BPTT = bptt
        self.DEC_LEN = decoder_length
        self.ALG_NAME = alg_name
        self.SAVE_DIR = save_dir
        self.SUMMARY_DIR = summary_dir
        self.CONSTRAINTS = constraints

        self.NUM_CATEGORIES = num_categories
        self.FLOAT_TYPE = float_type
        self.NUM_SAMP_INV = 1000

        self.DEVICE_CPU = device_cpu
        self.DEVICE_GPU = device_gpu

        self.wt_hparam = self.PARAMS_NAMED['wt_hparam'] #wt_hparam

        self.ENC_CELL_TYPE = enc_cell_type
        self.DEC_CELL_TYPE = dec_cell_type

        self.NUM_DISCRETE_STATES = num_discrete_states
        #self.NUM_DISCRETE_STATES = self.BPTT - self.DEC_LEN

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

        self.SHARE_DEC_PARAMS = share_dec_params
        self.INIT_ZERO_DEC_STATE = init_zero_dec_state
        self.CONCAT_FINAL_ENC_STATE = concat_final_enc_state
        self.NUM_EXTRA_DEC_LAYER = self.PARAMS_NAMED['num_extra_dec_layer'] #num_extra_dec_layer
        self.CONCAT_BEFORE_DEC_UPDATE = concat_before_dec_update
        self.MARK_TRIGGERS_TIME = mark_triggers_time
        self.MARK_LOSS = mark_loss
        self.PLOT_PRED_DEV = plot_pred_dev
        self.PLOT_PRED_TEST = plot_pred_test
        self.POSITION_ENCODE = position_encode
        self.USE_INTENSITY = use_intensity

        self.NUM_FEATS = num_feats
        self.USE_TIME_FEATS = use_time_features
        self.USE_AVG_GAPS = use_avg_gaps

        self.OFFSET = offset
        self.MAX_OFFSET = max_offset
        self.DISCRETE_OFFSET_FEATS = discrete_offset_feats
        self.SAMPLE_TRN_OFFSETS = sample_trn_offsets

        if self.CONCAT_FINAL_ENC_STATE:
            self.DEC_STATE_SIZE = 2 * self.HIDDEN_LAYER_SIZE
        else:
            self.DEC_STATE_SIZE = self.HIDDEN_LAYER_SIZE

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

                self.events_out = tf.placeholder(tf.int32, [None, self.DEC_LEN], name='events_out')
                self.times_out = tf.placeholder(self.FLOAT_TYPE, [None, self.DEC_LEN], name='times_out')

                #TODO make use of self.NUM_FEATS flag to add multiple features
                self.times_in_feats = tf.placeholder(tf.int32, [None, self.BPTT], name='times_in_feats')
                self.times_out_feats = tf.placeholder(tf.int32, [None, self.DEC_LEN], name='times_out_feats')

                self.last_input_timestamps = tf.placeholder(self.FLOAT_TYPE, [None, 1], name='last_input_timestamps')
                self.ts_indices = tf.placeholder(tf.int32, [None], name='ts_indices')
                self.seq_lens = tf.placeholder(tf.int32, [None], name='seq_lens')

                self.offset_begin_max_norm = tf.placeholder(self.FLOAT_TYPE, [None], name='offset_begin_max_norm')
                self.offset_begin = tf.placeholder(self.FLOAT_TYPE, [None], name='offset_begin')
                self.offset_begin_feats = tf.placeholder(self.FLOAT_TYPE, [None, 1], name='offset_begin_feats')

                if 'pastattn' in self.ALG_NAME:
                    self.attn_timestamps = tf.placeholder(self.FLOAT_TYPE, [None, None], name='attn_timestamps')
                    self.attn_gaps = tf.placeholder(self.FLOAT_TYPE, [None, None], name='attn_gaps')
                    self.attn_times_in_feats = tf.placeholder(tf.int32, [None, None], name='attn_times_in_feats')

                self.mode = tf.placeholder(tf.float32, name='mode')

                self.batch_num_events = tf.placeholder(self.FLOAT_TYPE, [], name='bptt_events')

                self.inf_batch_size = tf.shape(self.events_in)[0]

                # Make variables
                with tf.variable_scope('encoder_state'):
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
                                                     initializer=tf.constant_initializer(Wem(24)))
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

                    self.wt_attn = tf.get_variable(name='wt_attn', shape=(1, 1),
                                                   dtype=self.FLOAT_TYPE,
                                                   regularizer=self.RNN_REGULARIZER,
                                                   initializer=tf.constant_initializer(wt_attn),
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
                    self.Wem_position = tf.get_variable(name='Wem_position', shape=(self.DEC_LEN, self.EMBED_SIZE),
                                               dtype=self.FLOAT_TYPE,
                                               regularizer=self.RNN_REGULARIZER,
                                               initializer=tf.constant_initializer(Wem_position(self.DEC_LEN)))
                    self.enc_Wem_position = tf.get_variable(name='enc_Wem_position', shape=(self.BPTT, self.EMBED_SIZE),
                                               dtype=self.FLOAT_TYPE,
                                               regularizer=self.RNN_REGULARIZER,
                                               initializer=tf.constant_initializer(enc_Wem_position(self.BPTT)))

                    if self.SHARE_DEC_PARAMS:
                        print('Sharing Decoder Parameters')
                        self.Vt = tf.tile(self.Vt[0:1, :], [self.DEC_LEN, 1])
                        self.bt = tf.tile(self.bt[:, 0:1], [1, self.DEC_LEN])
                        self.Vw = tf.tile(self.Vw[0:1, :], [self.DEC_LEN, 1])
                        self.bw = tf.tile(self.bw[:, 0:1], [1, self.DEC_LEN])
                    else:
                        print('NOT Sharing Decoder Parameters')

                self.all_vars = [self.Wt, self.Wem, self.Wem_feats, self.Wh, self.bh, self.Ws, self.bs,
                                 self.wt, self.wt_attn, self.Wy, self.Vy, self.Vt, self.bt, self.bk, self.Vw, self.bw, Wem_position, enc_Wem_position]

                with tf.device(device_cpu):
                    # Global step needs to be on the CPU (Why?)
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.temperature = tf.train.inverse_time_decay(self.TEMPERATURE,
                                                               global_step=self.global_step,
                                                               decay_steps=decay_steps,
                                                               decay_rate=decay_rate)

                # Add summaries for all (trainable) variables
                # with tf.device(device_cpu):
                #     for v in self.all_vars:
                #         variable_summaries(v)

                # Make graph
                # RNNcell = RNN_CELL_TYPE(HIDDEN_LAYER_SIZE)

                # ----- Start: Create input for pastattn_rnn ----- #
                if 'pastattn' in self.ALG_NAME:
                    batch_attn_timestamps = self.attn_timestamps
                    batch_attn_gaps_in = self.attn_gaps
                    batch_attn_times_in_feats = self.attn_times_in_feats
                    match_idxes = tf.searchsorted(batch_attn_timestamps,
                                                  self.last_input_timestamps - ONE_DAY_SECS,
                                                  side='left')
                    begin_idxes = match_idxes - self.BPTT//2
                    seq_lens = tf.expand_dims(self.seq_lens, axis=-1)
                    begin_idxes = tf.where(begin_idxes < 0,
                                           tf.zeros_like(begin_idxes),
                                           begin_idxes)
                    begin_idxes = tf.where(begin_idxes+self.BPTT >= seq_lens,
                                           tf.ones_like(begin_idxes) * seq_lens-self.BPTT-1,
                                           begin_idxes)
                    end_idxes = begin_idxes + self.BPTT
                    attn_idxes = begin_idxes + tf.cumsum(tf.ones((1, self.BPTT), dtype=tf.int32), axis=-1)
                    attn_idxes = tf.expand_dims(attn_idxes, axis=-1)

                    input_batch_attn_gaps_in = tf.gather_nd(batch_attn_gaps_in, attn_idxes, batch_dims=1)
                    input_batch_attn_times_in_feats = tf.gather_nd(batch_attn_times_in_feats, attn_idxes, batch_dims=1)
                # ----- End: Create input for pastattn_rnn ----- #

                if self.ENC_CELL_TYPE == 'lstm':
                    enc_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE, forget_bias=1.0)
                    self.enc_cell = tf.contrib.rnn.MultiRNNCell([enc_cell for _ in range(self.NUM_ENC_LAYERS)],
                                                            state_is_tuple=True)
                    enc_internal_state = self.enc_cell.zero_state(self.inf_batch_size, dtype = tf.float32)

                if self.DEC_CELL_TYPE == 'lstm':
                    dec_cell = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE, forget_bias=1.0)
                    self.dec_cell = tf.contrib.rnn.MultiRNNCell([dec_cell for _ in range(self.NUM_DEC_LAYERS)],
                                                            state_is_tuple=True)
                    dec_internal_state = self.dec_cell.zero_state(self.inf_batch_size, dtype = tf.float32)

                if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                     'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                     'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                                     'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                    enc_cell_t = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE, forget_bias=1.0, name='attn_rnn')
                    self.enc_cell_t = tf.contrib.rnn.MultiRNNCell([enc_cell_t for _ in range(self.NUM_ENC_LAYERS)],
                                                            state_is_tuple=True)
                    enc_internal_state_t = self.enc_cell_t.zero_state(self.inf_batch_size, dtype = tf.float32)

                if self.ALG_NAME in ['rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                                     'rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r',
                                     'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                    enc_cell_c = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE, forget_bias=1.0, name='pastattn_rnn')
                    self.enc_cell_c = tf.contrib.rnn.MultiRNNCell([enc_cell_c for _ in range(self.NUM_ENC_LAYERS)],
                                                                  state_is_tuple=True)
                    enc_internal_state_c = self.enc_cell_c.zero_state(self.inf_batch_size, dtype = tf.float32)


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

                if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                     'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                     'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                                     'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                    state_t = tf.zeros([self.inf_batch_size, int(self.HIDDEN_LAYER_SIZE)],
                                       dtype=self.FLOAT_TYPE,
                                       name='attn_rnn_init_state')
                    self.hidden_states_t = []

                if self.ALG_NAME in ['rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                                     'rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r',
                                     'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                    state_c = self.enc_cell_c.zero_state(self.inf_batch_size, dtype=tf.float32)
                    self.hidden_states_c = []


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
                with tf.variable_scope('BPTT', reuse=tf.AUTO_REUSE):
                    for i in range(self.BPTT):

                        events_embedded = tf.nn.embedding_lookup(self.Wem,
                                                                 tf.mod(self.events_in[:, i] - 1, self.NUM_CATEGORIES))
                        time = self.times_in[:, i]

                        delta_t_prev = tf.expand_dims(time - last_time, axis=-1)
                        delta_t_prev = self.embedGaps(delta_t_prev)

                        time_feat_embd = self.embedFeatures(self.times_in_feats[:, i])

                        last_time = time

                        time_2d = tf.expand_dims(time, axis=-1)

                        # output, state = RNNcell(events_embedded, state)

                        # TODO Does TF automatically broadcast? Then we'll not
                        # need multiplication with tf.ones
                        type_delta_t = True

                        #with tf.name_scope('state_recursion'):
                        with tf.variable_scope('enc_state_recursion', reuse=tf.AUTO_REUSE):

                            if self.ENC_CELL_TYPE == 'manual':
                                new_state = tf.tanh(
                                    tf.matmul(state, self.Wh) +
                                    tf.matmul(events_embedded, self.Wy) +
                                    # Two ways of interpretting this term
                                    (tf.matmul(delta_t_prev, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                                    tf.matmul(ones_2d, self.bh),
                                    name='h_t'
                                )
                            elif self.ENC_CELL_TYPE == 'lstm':
                                inputs = tf.concat([events_embedded, delta_t_prev], axis=-1)
                                if self.USE_TIME_FEATS:
                                    inputs = tf.concat([inputs, time_feat_embd], axis=-1)
                                new_state, enc_internal_state = self.enc_cell(inputs,  enc_internal_state)

                                if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                                     'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                                     'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                                                     'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                                    p_embedded = tf.nn.embedding_lookup(self.enc_Wem_position,
                                                                        i * tf.ones((self.inf_batch_size), dtype=tf.int32))
                                    inputs_t = tf.concat([delta_t_prev, time_feat_embd], axis=-1)
                                    new_state_t, enc_internal_state_t = self.enc_cell_t(inputs_t,  enc_internal_state_t)

                            state = tf.where(self.events_in[:, i] > 0, new_state, state)
                            if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                                 'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                                 'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                                                 'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                                state_t = tf.where(self.events_in[:, i] > 0, new_state_t, state_t)

                        self.hidden_states.append(state)
                        if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                             'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                             'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                                             'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                            self.hidden_states_t.append(state_t)

                    if self.ALG_NAME in ['rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                                         'rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r',
                                         'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                        for i in range(self.BPTT):
                            input_batch_attn_times_in_feats_embd = self.embedFeatures(input_batch_attn_times_in_feats[:, i])
                            input_batch_attn_gaps_in_ = self.embedGaps(tf.expand_dims(input_batch_attn_gaps_in[:, i], axis=-1))
                            inputs = tf.concat([input_batch_attn_times_in_feats_embd, input_batch_attn_gaps_in_], axis=-1)
                            new_state_c, enc_internal_state_c = self.enc_cell_c(inputs, enc_internal_state_c)
                            self.hidden_states_c.append(new_state_c)
                        self.hidden_states_c = tf.stack(self.hidden_states_c, axis=1)


                    self.final_state = self.hidden_states[-1]
                    self.hidden_states = tf.stack(self.hidden_states, axis=1)
                    if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                         'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                         'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                                         'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                        self.hidden_states_t = tf.stack(self.hidden_states_t, axis=1)

                    # Computing z-attention values
                    if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                         'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                                         'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                        keys = self.hidden_states[:, :self.BPTT-self.DEC_LEN]
                        values = self.hidden_states_t[:, :self.BPTT-self.DEC_LEN]
                        self.z = tf.reduce_sum(values * tf.expand_dims(keys[:, -1], axis=1), axis=2)
                        #gaps_in_ = tf.concat([tf.zeros((self.inf_batch_size, 1)), self.times_in[:, 1:] - self.times_in[:, :-1]], axis=1)
                        #self.z = -tf.abs(gaps_in_ - gaps_in_[:, -1:])[:, :self.BPTT-self.DEC_LEN]
                        self.z = tf.nn.softmax(self.z)
                    elif self.ALG_NAME in ['rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                                           'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                        keys = self.hidden_states
                        values = self.hidden_states_c[:, :-self.DEC_LEN]
                        self.z = tf.reduce_sum(values * tf.expand_dims(keys[:, -1], axis=1), axis=2)
                        self.z = tf.nn.softmax(self.z)
                #----------- Encoder End ----------#

                with tf.variable_scope('offset', reuse=tf.AUTO_REUSE):
                    if self.MAX_OFFSET > 0.0:
                        self.offset_state = self.offset_network(self.final_state)
                    else:
                        self.offset_state = self.final_state

                #----------- Decoder Begin ----------#
                # TODO Does affine transformations (Wy) need to be different? Wt is not \
                  # required in the decoder

                if not self.INIT_ZERO_DEC_STATE:
                    s_state = self.offset_state
                #s_state = tf.Print(s_state, [self.mode, tf.equal(self.mode, 1.0)], message='mode ')
                average_gaps = tf.reduce_mean(self.times_in - tf.concat([tf.zeros_like(self.times_in[:, 0:1]), self.times_in[:, :-1]], axis=1), axis=1, keep_dims=True)
                self.decoder_states = []
                if self.ALG_NAME in ['rmtpp_decrnn_latentz']:
                    z_current = self.sample_z(s_state)
                    self.z_list = []
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
                                         lambda: tf.argmax(events_pred, axis=-1, output_type=tf.int32) + 1)
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

                            if self.DEC_CELL_TYPE == 'manual':
                                new_state = tf.tanh(
                                    tf.matmul(s_state, self.Ws) +
                                    tf.matmul(events_embedded, self.Wy) +
                                    tf.matmul(ones_2d, self.bs),
                                    name='s_t'
                                )
                            elif self.DEC_CELL_TYPE == 'lstm':
                                if self.ALG_NAME in ['rmtpp_decrnn_latentz']:
                                    inputs = tf.concat([s_state, events_embedded, z_current], axis=-1)
                                else:
                                    inputs = tf.concat([s_state, events_embedded], axis=-1)

                                if self.POSITION_ENCODE:
                                    p_embedded = tf.nn.embedding_lookup(self.Wem_position,
                                                                        i * tf.ones((self.inf_batch_size), dtype=tf.int32))
                                    inputs = tf.concat([inputs, p_embedded], axis=-1)

                                if self.USE_AVG_GAPS:
                                    inputs = tf.concat([inputs, average_gaps], axis=1)

                                new_state, dec_internal_state = self.dec_cell(inputs,  dec_internal_state)

                            if self.CONCAT_BEFORE_DEC_UPDATE:
                                new_state = tf.concat([new_state, self.offset_state], axis=-1)
                            if self.NUM_EXTRA_DEC_LAYER:
                                names = ['hidden_layer_'+str(hl_id) for hl_id in range(1, self.NUM_EXTRA_DEC_LAYER+1)]
                                for name in names:
                                    new_state = tf.layers.dense(new_state,
                                                                self.HIDDEN_LAYER_SIZE,
                                                                name=name,
                                                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                                                                activation=tf.nn.relu,
                                                                kernel_regularizer=self.EXTLYR_REGULARIZER,
                                                                bias_regularizer=self.EXTLYR_REGULARIZER)

                            s_state = new_state

                            if self.ALG_NAME in ['rmtpp_decrnn_latentz']:
                                z_current = self.sample_z(s_state)
                                self.z_list.append(z_current)

                        if mark_triggers_time:
                            self.decoder_states.append(s_state)

                    self.event_preds = tf.stack(self.event_preds, axis=1)


                    self.decoder_states = tf.stack(self.decoder_states, axis=1)

                    if self.ALG_NAME in ['rmtpp_decrnn_latentz']:
                        self.z_list = tf.stack(self.z_list, axis=1)
                    if self.CONCAT_FINAL_ENC_STATE:
                        decoder_states_concat = tf.concat([self.decoder_states,
                                                           tf.tile(tf.expand_dims(self.offset_state, axis=1),
                                                                   [1, self.DEC_LEN, 1])],
                                                           axis=-1)
                    else:
                        decoder_states_concat = self.decoder_states

                    offset = tf.expand_dims(self.offset_begin, axis=-1)
                    times_out_prev = tf.concat([self.times_in[:, -1:]+offset, self.times_out[:, :-1]], axis=1)

                    gaps = self.times_out-times_out_prev

                    times_prev = tf.cumsum(tf.concat([self.times_in[:, -1:], gaps[:, :-1]], axis=1), axis=1)

                    base_intensity_bt = self.bt
                    base_intensity_bw = self.bw

                    newVt = tf.tile(tf.expand_dims(self.Vt, axis=0), [tf.shape(self.decoder_states)[0], 1, 1]) 
                    newVw = tf.tile(tf.expand_dims(self.Vw, axis=0), [tf.shape(self.decoder_states)[0], 1, 1]) 

                    #self.D = tf.reduce_sum(decoder_states_concat * newVt, axis=2) + base_intensity_bt
                    if self.ALG_NAME in ['rmtpp_decrnn_latentz']:
                        self.D += tf.squeeze(tf.layers.dense(self.z_list, 1, name='z_to_lambda_layer',
                                                             kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed)),
                                             axis=-1)
                    if self.MAX_OFFSET > 0.0:
                        #self.D += self.offset_feats_network()
                        #self.D += self.offset_diff_network()
                        self.D, WT = self.DW_network(decoder_states_concat)

                    # self.D = tf.squeeze(tf.tensordot(self.decoder_states, self.Vt, axes=[[2],[0]]), axis=-1) + base_intensity_bt
                    self.D = get_D_constraint()(self.D)
                    if self.ALG_NAME in ['rmtpp_decrnn_splusintensity']:
                        self.D = tf.nn.softplus(self.D)

                    if self.ALG_NAME in ['rmtpp_decrnn_wcmpt']:
                        #self.WT = tf.reduce_sum(decoder_states_concat * newVw, axis=2) + base_intensity_bw
                        # self.WT = tf.squeeze(tf.tensordot(self.decoder_states, self.Vw, axes=[[2],[0]]), axis=-1) + base_intensity_bw
                        if self.MAX_OFFSET > 0.0:
                            self.WT = WT
                        self.WT = get_WT_constraint()(self.WT)
                        #self.WT = tf.clip_by_value(self.WT, 0.0, 10.0)
                    elif self.ALG_NAME in ['rmtpp_decrnn', 'rmtpp_decrnn_inv', 'rmtpp_decrnn_splusintensity',
                                           'rmtpp_decrnn_latentz',
                                           'rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn', 'rmtpp_decrnn_truemarks',
                                           'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                           'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn', 
                                           'rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r',
                                           'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                                           'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                        self.WT = self.wt
                    elif self.ALG_NAME in ['rmtpp_decrnn_negw', 'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                        self.WT = -1.0 * self.wt
                    elif self.ALG_NAME in ['rmtpp_decrnn_whparam']:
                        self.WT = self.wt_hparam

                    if self.ALG_NAME in ['rmtpp_decrnn_splusintensity']:
                        lambda_ = (self.D + gaps * self.WT)
                        log_lambda_ = tf.log(lambda_)
                        if self.USE_INTENSITY:
                            log_f_star = (log_lambda_
                                          - self.D * gaps
                                          - (self.WT/2.0) * tf.square(gaps))
                        else:
                            log_f_star = -tf.square(self.D - gaps)
                    elif 'attn' in self.ALG_NAME:
                        if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn', 'rmtpp_decrnn_attn_negw']:
                            z_indices_sorted = tf.argsort(self.z, axis=-1, direction='DESCENDING', stable=True, name='argsort_z')
                            z_indices_topk = z_indices_sorted[:, :self.NUM_DISCRETE_STATES]
                            self.z_indices_topk = z_indices_topk
                            self.z_topk = tf.tile(tf.expand_dims(tf.gather(self.z, z_indices_topk, batch_dims=1), axis=1), [1, self.DEC_LEN, 1])
                            #self.z_topk = tf.nn.softmax(self.z_topk)

                            gap_append_indices = tf.expand_dims(z_indices_topk, axis=1) + tf.expand_dims(tf.expand_dims(tf.range(0, self.DEC_LEN), axis=0), axis=-1)
                            self.gap_append_indices = gap_append_indices
                            gaps_in = tf.concat([tf.zeros((self.inf_batch_size, 1)), self.times_in[:, 1:] - self.times_in[:, :-1]], axis=1)
                            lookup_gaps = tf.gather(gaps_in, gap_append_indices, batch_dims=1)
                        elif self.ALG_NAME in ['rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r']:
                            self.z = tf.reduce_sum(tf.expand_dims(decoder_states_concat, axis=2) * tf.expand_dims(self.hidden_states_t, axis=1), axis=-1)
                            self.z = tf.nn.softmax(self.z)
                            z_indices_sorted = tf.argsort(self.z, axis=-1, direction='DESCENDING', stable=True, name='argsort_z')
                            z_indices_topk = z_indices_sorted[:, :, 0:1]
                            gap_append_indices = z_indices_topk
                            self.z_topk = tf.gather(self.z, z_indices_topk, batch_dims=2)
                            gaps_in = tf.concat([tf.zeros((self.inf_batch_size, 1)), self.times_in[:, 1:] - self.times_in[:, :-1]], axis=1)
                            lookup_gaps = tf.gather(gaps_in, gap_append_indices, batch_dims=1)
                        elif self.ALG_NAME in ['rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn']:
                            z_indices_sorted = tf.argsort(self.z, axis=-1, direction='DESCENDING', stable=True, name='argsort_z')
                            z_indices_topk = z_indices_sorted[:, :self.NUM_DISCRETE_STATES]
                            self.z_indices_topk = z_indices_topk
                            self.z_topk = tf.tile(tf.expand_dims(tf.gather(self.z, z_indices_topk, batch_dims=1), axis=1), [1, self.DEC_LEN, 1])
                            self.z_topk = tf.nn.softmax(self.z_topk)

                            #gap_append_indices = tf.expand_dims(z_indices_topk, axis=1) + tf.expand_dims(tf.expand_dims(tf.range(0, self.DEC_LEN), axis=0), axis=-1)
                            offset_unnormalized = tf.expand_dims(tf.expand_dims(self.offset_begin_max_norm * self.MAX_OFFSET * 1.0, axis=-1), axis=-1)
                            #lookup_indices = begin_idxes + self.z_indices_topk
                            lookup_indices = tf.expand_dims(begin_idxes, axis=-1) + tf.expand_dims(z_indices_topk, axis=1) + tf.expand_dims(tf.expand_dims(tf.range(0, self.DEC_LEN), axis=0), axis=-1)
                            lookup_indices = tf.expand_dims(lookup_indices, axis=-1)
                            lookup_timestamps = tf.gather_nd(batch_attn_timestamps,
                                                             lookup_indices,
                                                             batch_dims=1)
                            lookup_timestamps = lookup_timestamps + offset_unnormalized
                            batch_attn_timestamps_tiled = tf.tile(tf.expand_dims(batch_attn_timestamps, axis=1), [1, self.DEC_LEN, 1])
                            gap_append_indices \
                                    = tf.searchsorted(batch_attn_timestamps_tiled,
                                                      lookup_timestamps,
                                                      side='right')
                            self.gap_append_indices = gap_append_indices
                            seq_lens_expand = tf.expand_dims(seq_lens, axis=-1)
                            gap_append_indices = tf.where(gap_append_indices<seq_lens_expand,
                                                          gap_append_indices,
                                                          tf.ones_like(gap_append_indices)*(seq_lens_expand-1))
                            lookup_gaps = tf.gather_nd(batch_attn_gaps_in,
                                                       tf.expand_dims(gap_append_indices, axis=-1),
                                                       batch_dims=1)
                        elif self.ALG_NAME in ['rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r']:
                            self.z = tf.reduce_sum(tf.expand_dims(decoder_states_concat, axis=2) * tf.expand_dims(self.hidden_states_c, axis=1), axis=-1)
                            self.z = tf.nn.softmax(self.z)
                            z_indices_sorted = tf.argsort(self.z, axis=-1, direction='DESCENDING', stable=True, name='argsort_z') 
                            z_indices_topk = z_indices_sorted[:, :, 0:1]
                            self.z_topk = tf.gather(self.z, z_indices_topk, batch_dims=2)
                            gap_append_indices = z_indices_topk
                            lookup_gaps = tf.gather(attn_gaps_in, gap_append_indices, batch_dims=1)
                        elif self.ALG_NAME in ['rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate', 'rmtpp_decrnn_attnstate_negw']:
                            z_indices_sorted = tf.argsort(self.z, axis=-1, direction='DESCENDING', stable=True, name='argsort_z')
                            z_indices_topk = z_indices_sorted[:, :self.NUM_DISCRETE_STATES]
                            self.z_indices_topk = z_indices_topk
                            self.z_topk = tf.tile(tf.expand_dims(tf.gather(self.z, z_indices_topk, batch_dims=1), axis=1), [1, self.DEC_LEN, 1])
                            #self.z_topk = tf.nn.softmax(self.z_topk)

                            gap_append_indices = tf.expand_dims(z_indices_topk, axis=1) + tf.expand_dims(tf.expand_dims(tf.range(0, self.DEC_LEN), axis=0), axis=-1)
                            self.gap_append_indices = gap_append_indices
                            gaps_in = tf.concat([tf.zeros((self.inf_batch_size, 1)), self.times_in[:, 1:] - self.times_in[:, :-1]], axis=1)
                            lookup_gaps = tf.gather(gaps_in, gap_append_indices, batch_dims=1)
                            append_states = tf.gather(self.hidden_states_t, gap_append_indices, batch_dims=1)
                        elif self.ALG_NAME in ['rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                            z_indices_sorted = tf.argsort(self.z, axis=-1, direction='DESCENDING', stable=True, name='argsort_z')
                            z_indices_topk = z_indices_sorted[:, :self.NUM_DISCRETE_STATES]
                            self.z_indices_topk = z_indices_topk
                            self.z_topk = tf.tile(tf.expand_dims(tf.gather(self.z, z_indices_topk, batch_dims=1), axis=1), [1, self.DEC_LEN, 1])
                            #self.z_topk = tf.nn.softmax(self.z_topk)

                            gap_append_indices = tf.expand_dims(z_indices_topk, axis=1) + tf.expand_dims(tf.expand_dims(tf.range(0, self.DEC_LEN), axis=0), axis=-1)
                            self.gap_append_indices = gap_append_indices
                            lookup_gaps = tf.gather(attn_gaps_in, gap_append_indices, batch_dims=1)
                            append_states = tf.gather(self.hidden_states_c, gap_append_indices, batch_dims=1)

                        self.gap_append_indices = gap_append_indices
                        self.lookup_gaps = lookup_gaps


                        #self.D = tf.reduce_sum(decoder_states_concat * newVt, axis=-1) + base_intensity_bt
                        if 'attnstate' not in self.ALG_NAME:
                            if '_r' not in self.ALG_NAME:
                                decoder_states_concat = tf.tile(tf.expand_dims(decoder_states_concat, axis=2), [1, 1, self.NUM_DISCRETE_STATES, 1])
                            else:
                                decoder_states_concat = tf.expand_dims(decoder_states_concat, axis=2)
                            lookup_gaps = tf.expand_dims(lookup_gaps, axis=-1)
                            lookup_gaps = self.embedGaps(lookup_gaps)
                            decoder_states_concat = tf.concat([decoder_states_concat, lookup_gaps], axis=-1)
                            self.D = tf.squeeze(tf.layers.dense(decoder_states_concat, 1, name='attn_D_layer',
                                                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                                                                kernel_regularizer=self.EXTLYR_REGULARIZER,
                                                                bias_regularizer=self.EXTLYR_REGULARIZER), axis=-1)
                            if 'splusintensity' in self.ALG_NAME:
                                self.D = tf.nn.softplus(self.D)
                        elif 'attnstate' in self.ALG_NAME:
                            decoder_states_concat = tf.tile(tf.expand_dims(decoder_states_concat, axis=2), [1, 1, self.NUM_DISCRETE_STATES, 1])
                            append_states = tf.tile(append_states, [1, 1, 1, decoder_states_concat.get_shape().as_list()[-1]])
                            decoder_states_concat = tf.concat([decoder_states_concat, append_states], axis=-1)
                            self.D = tf.squeeze(tf.layers.dense(decoder_states_concat, 1, name='attnstate_D_layer',
                                                                kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                                                                kernel_regularizer=self.EXTLYR_REGULARIZER,
                                                                bias_regularizer=self.EXTLYR_REGULARIZER), axis=-1)
                            if 'splusintensity' in self.ALG_NAME:
                                self.D = tf.nn.softplus(self.D)


                        if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_attn_r',
                                             'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_pastattn_r',
                                             'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_pastattnstate',
                                             'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                            log_lambda_ = (self.D + tf.expand_dims(gaps, axis=-1) * self.WT)
                            lambda_ = tf.exp(tf.minimum(ETH, log_lambda_), name='lambda_')
                            if self.USE_INTENSITY:
                                log_f_star = (log_lambda_
                                              + (1.0 / self.WT) * tf.exp(tf.minimum(ETH, self.D))
                                              - (1.0 / self.WT) * lambda_)
                                log_f_star = tf.reduce_sum(log_f_star * self.z_topk, axis=-1)
                            else:
                                log_f_star = -tf.reduce_sum(tf.square(self.D - tf.expand_dims(gaps, axis=-1)) * self.z_topk, axis=-1)
                        elif self.ALG_NAME in ['rmtpp_decrnn_splusintensity_attn', 'rmtpp_decrnn_splusintensity_attn_r',
                                               'rmtpp_decrnn_splusintensity_pastattn', 'rmtpp_decrnn_splusintensity_pastattn_r',
                                               'rmtpp_decrnn_splusintensity_attnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                            lambda_ = (self.D + tf.expand_dims(gaps, axis=-1) * self.WT)
                            log_lambda_ = tf.log(lambda_)
                            if self.USE_INTENSITY:
                                log_f_star = (log_lambda_
                                              - self.D * tf.expand_dims(gaps, axis=-1)
                                              - (self.WT/2.0) * tf.square(tf.expand_dims(gaps, axis=-1)))
                                log_f_star = tf.reduce_sum(log_f_star * self.z_topk, axis=-1)
                            else:
                                log_f_star = -tf.reduce_sum(tf.square(self.D - tf.expand_dims(gaps, axis=-1)) * self.z_topk, axis=-1)

                    else:
                        log_lambda_ = (self.D + gaps * self.WT)
                        lambda_ = tf.exp(tf.minimum(ETH, log_lambda_), name='lambda_')
                        if self.USE_INTENSITY:
                            log_f_star = (log_lambda_
                                          + (1.0 / self.WT) * tf.exp(tf.minimum(ETH, self.D))
                                          - (1.0 / self.WT) * lambda_)
                        else:
                            log_f_star = -tf.square(self.D - gaps)

                with tf.name_scope('loss_calc'):

                    self.mark_LLs = tf.squeeze(tf.stack(self.mark_LLs, axis=1), axis=-1)
                    self.time_LLs = log_f_star

                    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    print('REGULARIZATION VARIABLES:', reg_variables)
                    if self.RNN_REG_PARAM:
                        reg_term = tf.contrib.layers.apply_regularization(self.RNN_REGULARIZER, reg_variables)
                    if self.EXTLYR_REG_PARAM:
                        reg_term_dense_layers = tf.losses.get_regularization_loss()

                    step_LLs = self.time_LLs
                    if self.MARK_LOSS:
                        step_LLs += self.mark_LLs

                    attn_loss = 0.0
                    if 'attn' in self.ALG_NAME:
                        with tf.variable_scope('attn_pred', reuse=tf.AUTO_REUSE):
                            first_output_gap = self.times_out[:, 0:1] - self.times_in[:, -1:]
                            if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn', 'rmtpp_decrnn_attn_negw']:
                                closest_input_gaps_indices = tf.argmin(tf.abs(first_output_gap - gaps_in[:, :self.BPTT-self.DEC_LEN]), axis=-1)
                            elif self.ALG_NAME in ['rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r']:
                                output_gaps = tf.concat([first_output_gap, self.times_out[:, 1:]-self.times_out[:, :-1]], axis=-1)
                                closest_input_gaps_indices = tf.argmin(tf.abs(tf.expand_dims(output_gaps, axis=2)-tf.expand_dims(gaps_in, axis=1)), axis=-1)
                            elif self.ALG_NAME in ['rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn']:
                                closest_input_gaps_indices = tf.argmin(tf.abs(first_output_gap - input_batch_attn_gaps_in[:, :-self.DEC_LEN]), axis=-1)
                            elif self.ALG_NAME in ['rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r']:
                                output_gaps = tf.concat([first_output_gap, self.times_out[:, 1:]-self.times_out[:, :-1]], axis=-1)
                                closest_input_gaps_indices = tf.argmin(tf.abs(tf.expand_dims(output_gaps, axis=2)-tf.expand_dims(attn_gaps_in, axis=1)), axis=-1)
                            elif self.ALG_NAME in ['rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate', 'rmtpp_decrnn_attnstate_negw']:
                                closest_input_gaps_indices = tf.argmin(tf.abs(first_output_gap - gaps_in[:, :self.BPTT-self.DEC_LEN]), axis=-1)
                            elif self.ALG_NAME in ['rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                                closest_input_gaps_indices = tf.argmin(tf.abs(first_output_gap - attn_gaps_in[:, :-self.DEC_LEN]), axis=-1)

                            self.closest_input_gaps_indices = closest_input_gaps_indices

                            attn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=closest_input_gaps_indices, logits=self.z)
                            attn_loss = tf.reduce_sum(attn_loss)


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
                                 step_LLs,
                                 tf.zeros(shape=(self.inf_batch_size, self.DEC_LEN)))
                    )
                    if self.RNN_REG_PARAM:
                        self.loss = self.loss + reg_term
                    if self.EXTLYR_REG_PARAM:
                        self.loss = self.loss + reg_term_dense_layers

                    if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                         'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                         'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                                         'rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r',
                                         'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                        self.loss += attn_loss

                    self.time_loss = (-1) * tf.reduce_sum(
                        tf.where(self.events_out > 0,
                                 self.time_LLs,
                                 tf.zeros(shape=(self.inf_batch_size, self.DEC_LEN)))
                    )
                    self.mark_loss = (-1) * tf.reduce_sum(
                        tf.where(self.events_out > 0,
                                 self.mark_LLs,
                                 tf.zeros(shape=(self.inf_batch_size, self.DEC_LEN)))
                    )
                    self.attn_loss = tf.reduce_sum(attn_loss)

                    # self.loss -= tf.cond(num_events > 0,
                    #                      lambda: tf.reduce_sum(
                    #                          tf.where(self.events_out[:, i] > 0,
                    #                                   tf.squeeze(step_LL),
                    #                                   tf.zeros(shape=(self.inf_batch_size,))),
                    #                          name='batch_bptt_loss'),
                    #                      lambda: 0.0)

                self.log_lambdas = log_lambda_

                # ----- Prediction using Inverse Transform Sampling ----- #
                #u = tf.random.uniform((self.inf_batch_size, self.DEC_LEN, 5000), minval=0.0, maxval=0.1, seed=self.seed)
                if self.ALG_NAME in ['rmtpp_decrnn_negw']:
                    lim = 1 - tf.exp((tf.exp(clip(self.D)))/self.WT)
                    lim = tf.clip_by_value(lim, 0.0, 1.0)
                    u = tf.ones((self.inf_batch_size, self.DEC_LEN, 1)) * tf.range(0.0, 0.99, 0.99/5000)
                    u  = u * tf.expand_dims(lim, axis=-1)
                elif self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_attn_r',
                                       'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_pastattn_r',
                                       'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_pastattnstate']:
                    u = tf.ones((self.inf_batch_size, self.DEC_LEN, self.NUM_DISCRETE_STATES, 1)) * tf.range(0.0, 1.0, 1.0/5000)
                elif self.ALG_NAME in ['rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                    lim = 1 - tf.exp(clip((tf.exp(clip(self.D)))/self.WT))
                    self.lim = lim
                    u = tf.ones((self.inf_batch_size, self.DEC_LEN, self.NUM_DISCRETE_STATES, 1)) * tf.range(0.0, 0.99, 0.99/5000)
                    u  = u * tf.expand_dims(lim, axis=-1)
                else:
                    u = tf.ones((self.inf_batch_size, self.DEC_LEN, 1)) * tf.range(0.0, 1.0, 1.0/5000)

                if self.ALG_NAME in ['rmtpp_decrnn', 'rmtpp_decrnn_negw']:
                    if self.USE_INTENSITY:
                        c = -tf.exp(clip(self.D))
                        c = tf.expand_dims(c, axis=-1)
                        self.val = (1.0/self.WT) * tf.log((self.WT/c) * tf.log(1.0 - u) + 1)
                    else:
                        self.val = self.D
                elif self.ALG_NAME in ['rmtpp_decrnn_splusintensity']:
                    if self.USE_INTENSITY:
                        D = tf.expand_dims(self.D, axis=-1)
                        self.val = (1.0/self.WT) * (-D + tf.sqrt(tf.square(D) - 2*self.WT*tf.log(1.0-u)))
                    else:
                        self.val = self.D
                elif self.ALG_NAME in ['rmtpp_decrnn_wcmpt']:
                    if self.USE_INTENSITY:
                        c = -tf.exp(clip(self.D))
                        c = tf.expand_dims(c, axis=-1)
                        WT = tf.expand_dims(self.WT, axis=-1)
                        self.val = (1.0/WT) * tf.log((WT/c) * tf.log(1.0 - u) + 1)
                    else:
                        self.val = self.D
                elif self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_attn_r',
                                       'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_pastattn_r',
                                       'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_pastattnstate',
                                       'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                    if self.USE_INTENSITY:
                        c = -tf.exp(clip(self.D))
                        c = tf.expand_dims(c, axis=-1)
                        self.val = (1.0/self.WT) * tf.log(tf.maximum((self.WT/c) * tf.log(1.0 - u) + 1, 1e-10))
                        self.val = tf.clip_by_value(self.val, 0.0, np.inf)
                        self.log_part = (self.WT/c) * tf.log(1.0 - u) + 1
                        self.log_1mu = tf.log(1.0 - u)
                        self.c = c
                    else:
                        self.val = tf.expand_dims(self.D, axis=-1)
                elif self.ALG_NAME in ['rmtpp_decrnn_splusintensity_attn', 'rmtpp_decrnn_splusintensity_attn_r',
                                       'rmtpp_decrnn_splusintensity_pastattn', 'rmtpp_decrnn_splusintensity_pastattn_r',
                                       'rmtpp_decrnn_splusintensity_attnstate', 'rmtpp_decrnn_splusintensity_pastattnstate']:
                    if self.USE_INTENSITY:
                        u = tf.ones((self.inf_batch_size, self.DEC_LEN, self.NUM_DISCRETE_STATES, 1)) * tf.range(0.0, 1.0, 1.0/5000)
                        D = tf.expand_dims(self.D, axis=-1)
                        self.val = (1.0/self.WT) * (-D + tf.sqrt(tf.square(D) - 2*self.WT*tf.log(1.0-u)))
                    else:
                        self.val = tf.expand_dims(self.D, axis=-1)

                if self.USE_INTENSITY:
                    self.val = tf.reduce_mean(self.val, axis=-1)

                if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                                     'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                                     'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                                     'rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r',
                                     'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate', 
                                     'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate',
                                     'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
                    self.val = tf.reduce_sum(self.val * self.z_topk, axis=-1)

                self.times = self.times_in


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

    def embedFeatures(self, inputs):
        return tf.nn.embedding_lookup(self.Wem_feats, inputs)

    def time_encoding(self, inputs, num_units, scope='time_kernal', reuse=None, return_weight=False):
        '''Shift-invariant time encoding kernal
        
        Args:
          inputs: A 2d float32 tensor with shate of [N, max_len]
          num_units: integer

        Returns:
          A 3d float tensor which embeds the input or 
          A tuple with one 3d float tensor (embeddings) and 2d float tensor (attention weight)
        '''
        assert(num_units % 2 == 0)
        effe_numits = num_units // 2
        
        with tf.variable_scope(scope, reuse=reuse):        
            init_freq_base = np.linspace(0, 8, effe_numits)
            init_freq_base = init_freq_base.astype(np.float32)
            #init_freq = 1 / 10 ** init_freq_base


            cos_freq_var = tf.get_variable('time_cos_freq', dtype=tf.float32, 
                                                initializer = tf.constant(init_freq_base))
            cos_freq_var = 1 / 10.0 ** cos_freq_var
            sin_freq_var = tf.get_variable('time_sin_freq', dtype=tf.float32, 
                                                initializer = tf.constant(init_freq_base))
            sin_freq_var = 1 / 10.0 ** sin_freq_var
            
            ones = np.ones(num_units).astype(np.float32)
            beta_var = tf.get_variable('time_beta', dtype=tf.float32, initializer = tf.constant(ones))
            
            # print("inputs shape ", inputs)
            sys.stdout.flush()
            expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))
            # print("expand_input shape ", expand_input)
            
            cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(cos_freq_var, [1, 1, effe_numits])))
            sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sin_freq_var, [1, 1, effe_numits])))
            
            freq_feat = tf.concat([cos_feat, sin_feat], axis=2) # [N, max_len, num_units]
            
            # print("freq_feat shape ", freq_feat)
            
            output = tf.multiply(freq_feat, tf.reshape(beta_var, [1, 1, num_units]))
            
            if return_weight:
                return output, tf.concat([cos_freq_var, sin_freq_var], axis=-1)
            return output

    def rand_time_encode(self, inputs, num_units, scope='rand_time_kernal', reuse=None, min_w=0, max_w=8):
        '''Bochner time encoding with uniformly random sampled frequencies

        Args:
          inputs: A 2d float32 tensor with shate of [N, max_len]
          num_units: An integer for the number of dimensions
          scope: string, scope for tensorflow variables
          reuse: bool, if true the layer could be reused
          min_w: float, min(log10(period))
          max_w: float, max(log10(period))
        
        Returns:
          A 3d float tensor which embeds the input
        '''
        assert(num_units % 2 == 0)
        effe_numits = num_units // 2
        with tf.variable_scope(scope, reuse=reuse):
            sampled_freq = tf.random_uniform(
                                            [effe_numits],
                                            minval=10 ** min_w,
                                            maxval=10 ** max_w,
                                            dtype=tf.float32,
                                            seed=None,
                                            name=None
                                        )
            sampled_freq = tf.ones_like(sampled_freq) / sampled_freq
            sampled_freq = tf.contrib.framework.sort(sampled_freq)
            expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))
            cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
            sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
            
            output = tf.concat([cos_feat, sin_feat], axis=2) # [N, max_len, num_units]
            return output
        
    def gaussian_time_encode(self, inputs, num_units, scope='gaussian_time_kernal', reuse=None):
        '''Bochner time encoding with frequencies sampled from Gaussian family

        Args:
          inputs: A 2d float32 tensor with shate of [N, max_len]
          num_units: An integer for the number of dimensions
          scope: string, scope for tensorflow variables
          reuse: bool, if true the layer could be reused
        
        Returns:
          A 3d float tensor which embeds the input
        '''
        assert(num_units % 2 == 0)
        effe_numits = num_units // 2
        with tf.variable_scope(scope, reuse=reuse):
            sampled_freq = tf.random_normal(
                                            [effe_numits],
                                            dtype=tf.float32,
                                            seed=None,
                                            name=None
                                        )
            
            expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))
            #sampled_freq = tf.ones_like(sampled_freq) / 10 ** sampled_freq
            
            init_freq_base = np.linspace(0, 8, effe_numits) / np.pi / 2
            init_freq_base = init_freq_base.astype(np.float32)
            init_freq = 1 / 10 ** init_freq_base
            mean_vec = tf.get_variable('mean_vec', dtype=tf.float32, initializer=tf.constant(init_freq))
            std_vec = tf.ones_like(mean_vec)
            sampled_freq = std_vec * sampled_freq + mean_vec
            
            #sampled_freq = tf.contrib.framework.sort(sampled_freq)
            #print('no sort')
                            
            #print(expand_input.shape)
            
            cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
            sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
            
            output = tf.concat([cos_feat, sin_feat], axis=2) # [N, max_len, num_units]
            return output
        
        
    def inverse_cdf_time_encode(self, inputs, num_units, method='maf', scope='inverse_cdf_time_kernal', reuse=None):
        '''Bochner time encoding with different inverse CDF methods

        Args:
          inputs: A 2d float32 tensor with shate of [N, max_len]
          num_units: An integer for the number of dimensions
          method: str, method for the inverse CDF
          scope: string, scope for tensorflow variables
          reuse: bool, if true the layer could be reused
        
        Returns:
          A 3d float tensor which embeds the input
        '''
        assert(num_units % 2 == 0)
        effe_numits = num_units // 2

        tfd = tfp.distributions
        tfb = tfp.bijectors
        with tf.variable_scope(scope, reuse=reuse):
            
            expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))

            if method == 'mlp_res':
                print('inv cdf method: mlp_res')
                sampled_freq = tf.random_uniform(
                                            [1, effe_numits],
                                            minval=0.0,
                                            maxval=1.0,
                                            dtype=tf.float32,
                                            seed=None,
                                            name=None
                                        )
                sampled_freq = tf.ones_like(sampled_freq) / 10 ** sampled_freq
                sampled_freq1 = tf.keras.layers.Dense(units=effe_numits, activation=tf.nn.relu, use_bias=True, bias_initializer='zeros')(sampled_freq)
                sampled_freq2 = tf.keras.layers.Dense(units=effe_numits, activation=None, use_bias=True, bias_initializer='zeros')(sampled_freq1)
                sampled_freq = tf.keras.layers.Dense(units=effe_numits, activation=None, use_bias=True, bias_initializer='zeros')(tf.add(sampled_freq2, sampled_freq))

            elif method == 'maf':
                print('inv cdf method: maf')
                maf = tfd.TransformedDistribution(
                    distribution=tfd.Normal(loc=0., scale=1.),
                    bijector=tfb.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                            hidden_layers=[256, 256])),
                            event_shape=[1])
                sampled_freq = maf.sample([1,effe_numits])

            elif method == 'iaf':
                print('inv cdf method: iaf')
                iaf = tfd.TransformedDistribution(
                    distribution=tfd.Normal(loc=0., scale=1.),
                    bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                            hidden_layers=[256, 256]))),
                    event_shape=[1])
                sampled_freq = iaf.sample([1,effe_numits])

            elif method == 'NVP':
                print('inv cdf method: NVP')
                nvp = tfd.TransformedDistribution(
                    distribution=tfd.Normal(loc=0., scale=1.),
                    bijector=tfb.RealNVP(
                        num_masked=2,
                        shift_and_log_scale_fn=tfb.real_nvp_default_template(
                            hidden_layers=[256, 256])))
                sampled_freq = nvp.sample([1,effe_numits])
                
            else:
                raise ValueError('method not found')

            sampled_freq = tf.exp(sampled_freq)
            
            cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
            sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
            
            output = tf.concat([cos_feat, sin_feat], axis=2) # [N, max_len, num_units]
            return output

    def embedGaps(self, inputs):
        if self.EMBED_GAPS is None:
            out = inputs
        elif self.EMBED_GAPS=='soft_onehot':
            num_units = self.GAP_EMBED_SIZE
            h_1 = tf.layers.dense(inputs, num_units, name='embed_gaps_softmax',
                                  kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                                  activation=tf.nn.softmax)
            out = tf.layers.dense(h_1, num_units, name='embed_gaps_linear',
                                  kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
        elif self.EMBED_GAPS=='time_kernal':
            num_units = self.GAP_EMBED_SIZE
            out = self.time_encoding(inputs, num_units)
            out = tf.squeeze(out, axis=1)
        elif self.EMBED_GAPS=='rand_time_kernal':
            num_units = self.GAP_EMBED_SIZE
            out = self.rand_time_encode(inputs, num_units)
            out = tf.squeeze(out, axis=1)
        elif self.EMBED_GAPS=='gaussian_time_kernal':
            num_units = self.GAP_EMBED_SIZE
            out = self.gaussian_time_encode(inputs, num_units)
            out = tf.squeeze(out, axis=1)
        elif self.EMBED_GAPS=='inverse_cdf_time_kernal':
            num_units = self.GAP_EMBED_SIZE
            out = self.inverse_cdf_time_encode(inputs, num_units)
            out = tf.squeeze(out, axis=1)
        else:
            raise ValueError('encoding type not found')

        return out

    def DW_network(self, inputs):
        h_1 = tf.layers.dense(inputs, 2, name='dw_nw_1',
                              kernel_initializer=tf.random_normal_initializer(),
                              bias_initializer=tf.random_normal_initializer(),
                              activation=None,
                              kernel_regularizer=self.EXTLYR_REGULARIZER,
                              bias_regularizer=self.EXTLYR_REGULARIZER)
        #h_2 = tf.layers.dense(h_1, self.HIDDEN_LAYER_SIZE, name='offset_nw_1',
        #                      kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
        #                      activation=tf.nn.sigmoid)

        out = h_1
        D, WT = out[:, :, 0], out[:, :, 1]
        return D, WT

    def offset_network(self, inputs):
        offset_begin_feats_discrete = tf.cast(self.offset_begin_feats, tf.int32)
        self.offset_begin_feats_embd = self.embedFeatures(offset_begin_feats_discrete)
        if self.DISCRETE_OFFSET_FEATS:
            offset_feats_inputs = self.offset_begin_feats_embd
        else:
            offset_feats_inputs = self.offset_begin_feats
        offset_begin_max_norm = tf.expand_dims(self.offset_begin_max_norm, axis=-1)
        inputs = tf.concat([inputs,
                            offset_feats_inputs,
                            offset_begin_max_norm],
                           axis=-1)
        #inputs = tf.Print(inputs, [inputs[:, :, -3:]])

        h_1 = tf.layers.dense(inputs, self.HIDDEN_LAYER_SIZE, name='offset_nw_1',
                              kernel_initializer=tf.random_normal_initializer(seed=self.seed),
                              bias_initializer=tf.random_normal_initializer(seed=self.seed),
                              activation=None,
                              kernel_regularizer=self.EXTLYR_REGULARIZER,
                              bias_regularizer=self.EXTLYR_REGULARIZER)
        #offset_state = tf.layers.dense(h_1, self.HIDDEN_LAYER_SIZE, name='offset_nw_2',
        #                               kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
        #                               activation=tf.nn.sigmoid)
        return h_1

    #def offset_network(self, inputs):
    #    offset_begin_feats_discrete = tf.cast(self.offset_begin_feats, tf.int32)
    #    self.offset_begin_feats_embd = self.embedFeatures(offset_begin_feats_discrete)
    #    if self.DISCRETE_OFFSET_FEATS:
    #        offset_feats_inputs = self.offset_begin_feats_embd
    #    else:
    #        offset_feats_inputs = tf.expand_dims(self.offset_begin_feats, axis=-1)
    #    offset_begin_max_norm = tf.expand_dims(tf.expand_dims(self.offset_begin_max_norm, axis=-1), axis=-1)
    #    inputs = tf.concat([inputs,
    #                        tf.tile(offset_feats_inputs, [1, self.DEC_LEN, 1]),
    #                        tf.tile(offset_begin_max_norm, [1, self.DEC_LEN, 1])], axis=-1)
    #    #inputs = tf.Print(inputs, [inputs[:, :, -3:]])

    #    h_1 = tf.layers.dense(inputs, self.HIDDEN_LAYER_SIZE/2, name='offset_nw_1',
    #                          kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
    #                          activation=tf.nn.sigmoid)
    #    h_2 = tf.layers.dense(h_1, self.HIDDEN_LAYER_SIZE/2, name='offset_nw_2',
    #                          kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
    #                          activation=tf.nn.sigmoid)
    #    out = tf.layers.dense(h_2, 2, name='offset_nw_out',
    #                          kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
    #    #out = tf.squeeze(out, axis=-1)
    #    D, WT = out[:, :, 0], out[:, :, 1]

    #    return D, WT

    def offset_feats_network(self):
        offset_begin_feats_discrete = tf.cast(self.offset_begin_feats, tf.int32)
        self.offset_begin_feats_embd = self.embedFeatures(offset_begin_feats_discrete)
        if self.DISCRETE_OFFSET_FEATS:
            inputs = self.offset_begin_feats_embd
        else:
            inputs = tf.expand_dims(self.offset_begin_feats, axis=-1)
            
        num_units = self.offset_begin_feats_embd.get_shape().as_list()[-1]
        h_1 = tf.layers.dense(inputs, num_units, name='offset_feats_nw_1',
                              kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                              activation=tf.nn.relu)
        h_2 = tf.layers.dense(h_1, h_1.get_shape().as_list()[-1], name='offset_feats_nw_2',
                              kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                              activation=tf.nn.relu)
        out = tf.layers.dense(h_2, 1, name='offset_feats_nw_out',
                              kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
        out = tf.squeeze(out, axis=-1)

        return out

    def offset_diff_network(self):
        inputs = tf.expand_dims(self.offset_begin_max_norm, axis=-1)
        h_1 = tf.layers.dense(inputs, self.HIDDEN_LAYER_SIZE/2, name='offset_diff_nw_1',
                              kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                              activation=tf.nn.relu)
        h_2 = tf.layers.dense(h_1, self.HIDDEN_LAYER_SIZE/2, name='offset_diff_nw_2',
                              kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed),
                              activation=tf.nn.relu)
        out = tf.layers.dense(h_2, 1, name='offset_diff_nw_out',
                              kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
        #out = tf.squeeze(out, axis=-1)

        return out


    def sample_z(self, inputs):
        with tf.variable_scope('latent_discrete_state', reuse=tf.AUTO_REUSE):

            z_logits = tf.layers.dense(inputs, self.NUM_DISCRETE_STATES, name='z_layer',
                                       kernel_initializer=tf.glorot_uniform_initializer(seed=self.seed))
            z_distribution = tf.contrib.distributions.RelaxedOneHotCategorical(self.temperature, logits=z_logits)
            z_sampled = z_distribution.sample(seed=self.seed)
            return z_sampled

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
        """Train the model given the training data.

        If with_evals is an integer, then that many elements from the test set
        will be tested.
        """

        if dec_len_for_eval==0:
            dec_len_for_eval=training_data['decoder_length']
        if restore_path is None:
            restore_path = self.SAVE_DIR
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
        train_actual_time_out_seq = training_data['train_actual_time_out_seq']
        train_time_in_feats = training_data['train_time_in_feats']
        train_time_out_feats = training_data['train_time_out_feats']
        train_actual_time_in_seq = np.array(training_data['train_actual_time_in_seq'])
        attn_train_time_in_feats = training_data['attn_train_time_in_feats']
        attn_train_gaps = training_data['attn_train_gaps']
        attn_train_time_in_seq = training_data['attn_train_time_in_seq']
        trainND = training_data['trainND']
        train_time_indices = training_data['train_time_indices']

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
        num_sample_offsets = 5

        print('Training with ', self.STOP_CRITERIA, 'stop_criteria and ', num_epochs, 'num_epochs')
        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            self.rs.shuffle(idxes)

            print("Starting epoch...", epoch)
            total_loss_prev = total_loss
            total_loss = 0.0
            time_loss, mark_loss, attn_loss = 0.0, 0.0, 0.0
            epoch_start_time = time.time()

            for batch_idx in range(n_batches):
                batch_idxes = idxes[batch_idx * self.BATCH_SIZE:(batch_idx + 1) * self.BATCH_SIZE]

                batch_event_train_in = [train_event_in_seq[batch_idx] for batch_idx in batch_idxes]
                batch_time_train_in = [train_time_in_seq[batch_idx] for batch_idx in batch_idxes]
                batch_time_train_feats = [train_time_in_feats[batch_idx] for batch_idx in batch_idxes]

                batch_time_train_out_full = [train_time_out_seq[batch_idx] for batch_idx in batch_idxes]
                batch_event_train_out_full = [train_event_out_seq[batch_idx] for batch_idx in batch_idxes]
                batch_trainND = [trainND[batch_idx] for batch_idx in batch_idxes]
                batch_time_train_out_feats_full = [train_time_out_feats[batch_idx] for batch_idx in batch_idxes]
                for _ in range(num_sample_offsets):
                    if self.SAMPLE_TRN_OFFSETS:
                        offsets = np.random.uniform(low=0.0, high=self.MAX_OFFSET, size=(self.BATCH_SIZE))
                    else:
                        offsets = np.zeros((self.BATCH_SIZE), dtype=float)
                    offsets_normalized = offsets / np.squeeze(np.array(batch_trainND), axis=-1)
                    offsets_feed = offsets * 1.0 / self.MAX_OFFSET
                    batch_train_actual_time_in_seq = [train_actual_time_in_seq[batch_idx] for batch_idx in batch_idxes]
                    batch_time_train_actual_out = [train_actual_time_out_seq[batch_idx] for batch_idx in batch_idxes]
                    out_begin_indices, out_end_indices \
                            = get_output_indices(batch_train_actual_time_in_seq, batch_time_train_actual_out, offsets, self.DEC_LEN)
                    #print(offsets)
                    for beg_ind, end_ind, seq in zip(out_begin_indices, out_end_indices, batch_event_train_out_full):
                        #print(beg_ind, end_ind, len(seq))
                        assert end_ind < len(seq)
                    batch_event_train_out = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                                             zip(batch_event_train_out_full, out_begin_indices, out_end_indices)]
                    batch_time_train_out = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                                            zip(batch_time_train_out_full, out_begin_indices, out_end_indices)]
                    batch_time_train_out_feats = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                                                  zip(batch_time_train_out_feats_full, out_begin_indices, out_end_indices)]

                    offset_feats = [[getHour(s+offset)/24.0 for s in seq] for seq, offset in zip(batch_train_actual_time_in_seq, offsets)]

                    batch_train_time_indices = [train_time_indices[batch_idx] for batch_idx in batch_idxes]
                    batch_attn_train_time_in_seq = [attn_train_time_in_seq[idx] for idx in batch_train_time_indices]
                    batch_attn_train_gaps = [attn_train_gaps[idx] for idx in batch_train_time_indices]
                    batch_attn_train_time_in_feats = [attn_train_time_in_feats[idx] for idx in batch_train_time_indices]
                    batch_train_seq_lens = [len(attn_gaps_seq) for attn_gaps_seq in batch_attn_train_gaps]

                    num_elems = np.sum(np.ones_like(batch_time_train_feats))

                    num_elems = np.sum(np.ones_like(batch_time_train_in))

                    cur_state = np.zeros((self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE))
                    batch_loss = 0.0

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
                        self.times_out_feats: batch_time_train_out_feats,
                        self.batch_num_events: batch_num_events,
                        self.last_input_timestamps: batch_train_actual_time_in_seq,
                        self.ts_indices: batch_train_time_indices,
                        self.seq_lens: batch_train_seq_lens,
                        self.offset_begin_max_norm: offsets_feed,
                        self.offset_begin: offsets_normalized,
                        self.offset_begin_feats: offset_feats,
                        self.mode: 1.0, # Train Mode
                    }
                    if 'pastattn' in self.ALG_NAME:
                        feed_dict[self.attn_timestamps] = batch_attn_train_time_in_seq
                        feed_dict[self.attn_times_in_feats] = batch_attn_train_time_in_feats
                        feed_dict[self.attn_gaps] = batch_attn_train_gaps

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
                        _, cur_state, loss_, time_loss_, mark_loss_, attn_loss_ = \
                            self.sess.run([self.update,
                                           self.final_state, self.loss,
                                           self.time_loss, self.mark_loss, self.attn_loss],
                                          feed_dict=feed_dict)

                    batch_loss = loss_
                    total_loss += batch_loss
                    time_loss += time_loss_
                    mark_loss += mark_loss_
                    attn_loss += attn_loss_
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

                dev_time_preds, dev_gaps_preds, dev_event_preds, \
                        dev_event_preds_softmax, inference_time, dev_offsets_normalized \
                        = self.predict(training_data['dev_event_in_seq'],
                                       training_data['dev_time_in_seq'],
                                       training_data['dev_time_in_feats'],
                                       dec_len_for_eval,
                                       training_data['attn_dev_time_in_seq'],
                                       training_data['attn_dev_gaps'],
                                       training_data['dev_time_out_feats'],
                                       training_data['attn_dev_time_in_feats'],
                                       np.array(training_data['dev_actual_time_in_seq']),
                                       training_data['attn_dev_time_in_seq'],
                                       training_data['dev_time_indices'],
                                       training_data['devND'],
                                       single_threaded=True,
                                       event_out_seq=training_data['dev_event_out_seq'] if self.ALG_NAME=='rmtpp_decrnn_truemarks' else None)
                dev_loss, dev_time_loss, dev_mark_loss \
                        = self.evaluate_likelihood(training_data['dev_event_in_seq'],
                                                   training_data['dev_time_in_seq'],
                                                   training_data['dev_event_out_seq'],
                                                   training_data['dev_time_out_seq'],
                                                   training_data['dev_time_in_feats'],
                                                   training_data['dev_time_out_feats'],
                                                   dec_len_for_eval,
                                                   training_data['attn_dev_time_in_seq'],
                                                   training_data['attn_dev_gaps'],
                                                   training_data['dev_time_out_feats'],
                                                   training_data['attn_dev_time_in_feats'],
                                                   np.array(training_data['dev_actual_time_in_seq']),
                                                   training_data['attn_dev_time_in_seq'],
                                                   training_data['dev_time_indices'],
                                                   training_data['devND'])

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

                if self.PLOT_PRED_DEV:
                    idx = 4
                    true_gaps_plot = dev_time_out_seq[idx] - np.concatenate([dev_actual_time_in_seq[idx]+dev_offsets[idx], dev_time_out_seq[idx][:-1]])
                    pred_gaps_plot = unnorm_gaps[idx]
                    inp_tru_gaps = training_data['dev_time_in_seq'][idx][1:] \
                                   - training_data['dev_time_in_seq'][idx][:-1]
                    inp_tru_gaps = inp_tru_gaps * training_data['devND'][idx]
                    true_gaps_plot = list(inp_tru_gaps) + list(true_gaps_plot)
                    pred_gaps_plot = list(inp_tru_gaps) + list(pred_gaps_plot)

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
                    ax1.plot([self.BPTT-0.5, self.BPTT-0.5],
                             [0, max(np.concatenate([true_gaps_plot, pred_gaps_plot]))],
                             'g-')
                    ax1.set_xlabel('Index')
                    ax1.set_ylabel('Gaps')
                    plt.grid()

                    plt.savefig(name_plot+'.png')
                    plt.close()
    
                test_time_preds, test_gaps_preds, test_event_preds, \
                        test_event_preds_softmax, inference_time, test_offsets_normalized \
                        = self.predict(training_data['test_event_in_seq'],
                                       training_data['test_time_in_seq'],
                                       training_data['test_time_in_feats'],
                                       dec_len_for_eval,
                                       training_data['attn_test_time_in_seq'],
                                       training_data['attn_test_gaps'],
                                       training_data['test_time_out_feats'],
                                       training_data['attn_test_time_in_feats'],
                                       np.array(training_data['test_actual_time_in_seq']),
                                       training_data['attn_test_time_in_seq'],
                                       training_data['test_time_indices'],
                                       training_data['testND'],
                                       single_threaded=True,
                                       event_out_seq=training_data['test_event_out_seq'] if self.ALG_NAME=='rmtpp_decrnn_truemarks' else None)
                test_loss, test_time_loss, test_mark_loss \
                        = self.evaluate_likelihood(training_data['test_event_in_seq'],
                                                   training_data['test_time_in_seq'],
                                                   training_data['test_event_out_seq'],
                                                   training_data['test_time_out_seq'],
                                                   training_data['test_time_in_feats'],
                                                   training_data['test_time_out_feats'],
                                                   dec_len_for_eval,
                                                   training_data['attn_test_time_in_seq'],
                                                   training_data['attn_test_gaps'],
                                                   training_data['test_time_out_feats'],
                                                   training_data['attn_test_time_in_feats'],
                                                   np.array(training_data['test_actual_time_in_seq']),
                                                   training_data['attn_test_time_in_seq'],
                                                   training_data['test_time_indices'],
                                                   training_data['testND'])
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
            print('Loss on last epoch = {:.4f}, train_loss = {:.4f}, mark_loss = {:.4f}, attn_loss = {:.4f}, new lr = {:.5f}, global_step = {}'
                  .format(total_loss, np.float64(time_loss), np.float64(mark_loss), np.float64(attn_loss),
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

            dev_time_preds, dev_gaps_preds, dev_event_preds, \
                    dev_event_preds_softmax, inference_time, dev_offsets_normalized \
                    = self.predict(training_data['dev_event_in_seq'],
                                   training_data['dev_time_in_seq'],
                                   training_data['dev_time_in_feats'],
                                   dec_len_for_eval,
                                   training_data['attn_dev_time_in_seq'],
                                   training_data['attn_dev_gaps'],
                                   training_data['dev_time_out_feats'],
                                   training_data['attn_dev_time_in_feats'],
                                   np.array(training_data['dev_actual_time_in_seq']),
                                   training_data['attn_dev_time_in_seq'],
                                   training_data['dev_time_indices'],
                                   training_data['devND'],
                                   single_threaded=True,
                                   event_out_seq=training_data['dev_event_out_seq'] if self.ALG_NAME=='rmtpp_decrnn_truemarks' else None)
            dev_loss, dev_time_loss, dev_mark_loss \
                    = self.evaluate_likelihood(training_data['dev_event_in_seq'],
                                               training_data['dev_time_in_seq'],
                                               training_data['dev_event_out_seq'],
                                               training_data['dev_time_out_seq'],
                                               training_data['dev_time_in_feats'],
                                               training_data['dev_time_out_feats'],
                                               dec_len_for_eval,
                                               training_data['attn_dev_time_in_seq'],
                                               training_data['attn_dev_gaps'],
                                               training_data['dev_time_out_feats'],
                                               training_data['attn_dev_time_in_feats'],
                                               np.array(training_data['dev_actual_time_in_seq']),
                                               training_data['attn_dev_time_in_seq'],
                                               training_data['dev_time_indices'],
                                               training_data['devND'])
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

            test_time_preds, test_gaps_preds, test_event_preds, \
                    test_event_preds_softmax, inference_time, test_offsets_normalized \
                    = self.predict(training_data['test_event_in_seq'],
                                   training_data['test_time_in_seq'],
                                   training_data['test_time_in_feats'],
                                   dec_len_for_eval,
                                   training_data['attn_test_time_in_seq'],
                                   training_data['attn_test_gaps'],
                                   training_data['test_time_out_feats'],
                                   training_data['attn_test_time_in_feats'],
                                   np.array(training_data['test_actual_time_in_seq']),
                                   training_data['attn_test_time_in_seq'],
                                   training_data['test_time_indices'],
                                   training_data['testND'],
                                   single_threaded=True,
                                   event_out_seq=training_data['test_event_out_seq'] if self.ALG_NAME=='rmtpp_decrnn_truemarks' else None)
            test_loss, test_time_loss, test_mark_loss \
                    = self.evaluate_likelihood(training_data['test_event_in_seq'],
                                               training_data['test_time_in_seq'],
                                               training_data['test_event_out_seq'],
                                               training_data['test_time_out_seq'],
                                               training_data['test_time_in_feats'],
                                               training_data['test_time_out_feats'],
                                               dec_len_for_eval,
                                               training_data['attn_test_time_in_seq'],
                                               training_data['attn_test_gaps'],
                                               training_data['test_time_out_feats'],
                                               training_data['attn_test_time_in_feats'],
                                               np.array(training_data['test_actual_time_in_seq']),
                                               training_data['attn_test_time_in_seq'],
                                               training_data['test_time_indices'],
                                               training_data['testND'])

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

    def evaluate_likelihood(self, event_in_seq, time_in_seq, event_out_seq, time_out_seq,
                            time_in_feats, time_out_feats, decoder_length,
                            attn_time_in_seq, attn_gaps,
                            times_out_feats, attn_times_in_feats, actual_time_in_seq,
                            attn_timestamps, ts_indices, ND):

        offsets = np.ones((len(time_in_seq))) * self.MAX_OFFSET
        offsets_normalized = offsets / np.squeeze(np.array(ND), axis=-1)
        offsets_feed = offsets * 1.0 / self.MAX_OFFSET

        offset_feats = [[getHour(s+offset)/24.0 for s in seq] for seq, offset in zip(actual_time_in_seq, offsets)]

        out_begin_indices, out_end_indices \
                = get_output_indices(time_in_seq, time_out_seq, offsets_normalized, self.DEC_LEN)
        #print(offsets)
        for beg_ind, end_ind, seq in zip(out_begin_indices, out_end_indices, event_out_seq):
            #print(beg_ind, end_ind, len(seq))
            assert end_ind < len(seq)
        event_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                            zip(event_out_seq, out_begin_indices, out_end_indices)]
        time_out_seq = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                            zip(time_out_seq, out_begin_indices, out_end_indices)]
        time_out_feats = [seq[beg_ind:end_ind] for seq, beg_ind, end_ind in
                            zip(time_out_feats, out_begin_indices, out_end_indices)]

        attn_timestamps = [attn_timestamps[idx] for idx in ts_indices]
        attn_gaps = [attn_gaps[idx] for idx in ts_indices]
        attn_times_in_feats = [attn_times_in_feats[idx] for idx in ts_indices]
        seq_lens = [len(seq) for seq in attn_gaps]

        num_events = sum([e>0 for seq in event_in_seq for e in seq])
        cur_state = np.zeros((len(event_in_seq), self.HIDDEN_LAYER_SIZE))
        initial_time = np.zeros(len(event_in_seq))

        feed_dict = {
            self.initial_state: cur_state,
            self.initial_time: initial_time,
            self.events_in: event_in_seq,
            self.events_out: event_out_seq,
            self.times_in: time_in_seq,
            self.times_out: time_out_seq,
            self.times_in_feats: time_in_feats,
            self.times_out_feats: time_out_feats,
            self.batch_num_events: num_events,
            self.last_input_timestamps: actual_time_in_seq,
            self.ts_indices: ts_indices,
            self.seq_lens: seq_lens,
            self.offset_begin_max_norm: offsets_feed,
            self.offset_begin: offsets_normalized,
            self.offset_begin_feats: offset_feats,
            self.mode: 1.0,
        }
        if 'pastattn' in self.ALG_NAME:
            feed_dict[self.attn_timestamps] = attn_timestamps
            feed_dict[self.attn_times_in_feats] = attn_times_in_feats
            feed_dict[self.attn_gaps] = attn_gaps

        loss_, time_loss_, mark_loss_ = self.sess.run([self.loss,
                                                       self.time_loss, self.mark_loss],
                                                       feed_dict=feed_dict)

        print("Log Likelihood:", loss_)
        return float(loss_), float(time_loss_), float(mark_loss_)


    def predict(self, event_in_seq, time_in_seq, time_in_feats, dec_len_for_eval,
                attn_time_in_seq, attn_gaps,
                times_out_feats, attn_times_in_feats, actual_time_in_seq,
                attn_timestamps, ts_indices,
                ND, single_threaded=False, plot_dir=False, event_out_seq=None):
        """Treats the entire dataset as a single batch and processes it."""

        start_time = time.time()

        # Default offsets = self.MAX_OFFSET
        #offsets = np.random.uniform(low=0.0, high=self.MAX_OFFSET, size=(self.BATCH_SIZE))
        offsets = np.ones((len(time_in_seq))) * self.MAX_OFFSET
        offsets_normalized = offsets / np.squeeze(np.array(ND), axis=-1)
        offsets_feed = offsets * 1.0 / self.MAX_OFFSET
        offset_feats = [[getHour(s+offset)/24.0 for s in seq] for seq, offset in zip(actual_time_in_seq, offsets)]

        attn_timestamps = [attn_timestamps[idx] for idx in ts_indices]
        attn_gaps = [attn_gaps[idx] for idx in ts_indices]
        attn_times_in_feats = [attn_times_in_feats[idx] for idx in ts_indices]
        seq_lens = [len(seq) for seq in attn_gaps]

        cur_state = np.zeros((len(event_in_seq), self.HIDDEN_LAYER_SIZE))
        initial_time = np.zeros(len(time_in_seq))
        # Feeding dummy values to self.<events/times>_out placeholders
        time_out_seq = np.zeros((len(event_in_seq), self.DEC_LEN))
        if self.ALG_NAME in ['rmtpp_decrnn_truemarks']:
            assert event_out_seq is not None
            mode = 1.0
        else:
            assert event_out_seq is None
            event_out_seq = np.zeros_like(time_out_seq)
            mode = 0.0

        print(np.shape(attn_timestamps), np.shape(attn_gaps), np.shape(attn_times_in_feats))

        feed_dict = {
            self.initial_state: cur_state,
            self.initial_time: initial_time,
            self.events_in: event_in_seq,
            self.times_in: time_in_seq,
            self.events_out: event_out_seq,
            self.times_out: time_out_seq,
            self.times_in_feats: time_in_feats,
            self.last_input_timestamps: actual_time_in_seq,
            self.ts_indices: ts_indices,
            self.seq_lens: seq_lens,
            self.offset_begin_max_norm: offsets_feed,
            self.offset_begin: offsets_normalized,
            self.offset_begin_feats: offset_feats,
            self.mode: mode #Test Mode
        }
        if 'pastattn' in self.ALG_NAME:
            feed_dict[self.attn_timestamps] = attn_timestamps
            feed_dict[self.attn_times_in_feats] = attn_times_in_feats
            feed_dict[self.attn_gaps] = attn_gaps


        all_encoder_states, all_decoder_states, all_event_preds, cur_state, D, WT = self.sess.run(
            [self.hidden_states, self.decoder_states, self.event_preds, self.offset_state, self.D, self.WT],
            feed_dict=feed_dict
        )
        #for d in D:
        #    print(d, WT)

        if self.ALG_NAME in ['rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn',
                             'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                             'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                             'rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r', 
                             'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate', 
                             'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate',
                             'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
            z_topk = self.sess.run(self.z_topk, feed_dict=feed_dict)
            lookup_gaps = self.sess.run(self.lookup_gaps, feed_dict=feed_dict)
            gap_append_indices = self.sess.run(self.gap_append_indices, feed_dict=feed_dict)
            closest_input_gaps_indices = self.sess.run(self.closest_input_gaps_indices, feed_dict=feed_dict)
        else:
            z_topk = np.zeros_like(event_out_seq)

        if self.ALG_NAME in ['rmtpp_decrnn', 'rmtpp_decrnn_inv', 'rmtpp_decrnn_splusintensity', 'rmtpp_decrnn_latentz',
                             'rmtpp_decrnn_attn', 'rmtpp_decrnn_splusintensity_attn', 'rmtpp_decrnn_truemarks',
                             'rmtpp_decrnn_attn_r', 'rmtpp_decrnn_splusintensity_attn_r',
                             'rmtpp_decrnn_pastattn', 'rmtpp_decrnn_splusintensity_pastattn',
                             'rmtpp_decrnn_pastattn_r', 'rmtpp_decrnn_splusintensity_pastattn_r',
                             'rmtpp_decrnn_attnstate', 'rmtpp_decrnn_splusintensity_attnstate',
                             'rmtpp_decrnn_pastattnstate', 'rmtpp_decrnn_splusintensity_pastattnstate', 
                             'rmtpp_decrnn_negw', 'rmtpp_decrnn_attn_negw', 'rmtpp_decrnn_attnstate_negw']:
            WT = np.ones((len(event_in_seq), self.DEC_LEN, 1)) * WT
        elif self.ALG_NAME in ['rmtpp_decrnn_whparam']:
            raise NotImplemented('For whparam methods')
            WT = self.wt_hparam

        all_event_preds_softmax = all_event_preds
        if self.ALG_NAME in ['rmtpp_decrnn_truemarks']:
            all_event_preds = np.transpose(event_out_seq)
        else:
            all_event_preds = np.argmax(all_event_preds, axis=-1) + 1
            all_event_preds = np.transpose(all_event_preds)

        # TODO: This calculation is completely ignoring the clipping which
        # happens during the inference step.
        [Vt, Vw, bt, bw, wt]  = self.sess.run([self.Vt, self.Vw, self.bt, self.bw, self.wt])

        time_pred_last = [seq[-1:] for seq in time_in_seq]
        val = self.sess.run(self.val, feed_dict=feed_dict)
        all_time_preds = np.cumsum(val, axis=1) + time_pred_last + self.OFFSET
        if self.MAX_OFFSET > 0.0:
            all_time_preds = np.cumsum(val, axis=1) + time_pred_last + np.expand_dims(offsets_normalized, axis=-1)

        all_time_preds = np.asarray(all_time_preds).T
        all_gaps_preds = val
        if not np.isfinite(all_time_preds).sum() == all_time_preds.size:
            print(all_time_preds)
        assert np.isfinite(all_time_preds).sum() == all_time_preds.size

        end_time = time.time()
        inference_time = end_time - start_time

        all_time_preds = np.asarray(all_time_preds).T[:, :dec_len_for_eval]
        all_event_preds = np.asarray(all_event_preds).swapaxes(0, 1)[:, :dec_len_for_eval]
        all_event_preds_softmax = all_event_preds_softmax[:, :dec_len_for_eval]


        return all_time_preds, all_gaps_preds, all_event_preds, all_event_preds_softmax, inference_time, offsets_normalized

    def eval(self, time_preds, time_true, event_preds, event_true, time_input_last, event_preds_softmax, offsets):
        """Prints evaluation of the model on the given dataset."""
        mae, total_valid = MAE(time_preds, time_true, event_true)
        acc, _ = ACC(event_preds, event_true)
        if time_input_last is not None:
            gap_true = [seq - np.concatenate([last+off, seq[:-1]]) for seq, last, off in zip(time_true, time_input_last, offsets)]
            gap_preds = [seq - np.concatenate([last+off, seq[:-1]]) for seq, last, off in zip(time_preds, time_input_last, offsets)]
            gap_mae, gap_total_valid = MAE(gap_true, gap_preds, event_true)
            gap_dtw = DTW(gap_true, gap_preds, event_true)
        else:
            gap_mae = None
            gap_dtw = None

        mrr = MRR(event_preds_softmax, event_true)

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
