import math
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

from utils import get_time_features

import transformer_helpers.Constants as Constants
from transformer_helpers.Layers import EncoderLayer

ETH = 10.0
one_by = tf.math.reciprocal_no_nan


class RMTPP_VAR(tf.keras.Model):
    def __init__(self, hidden_layer_size,
                 num_bins, bin_embd_dim,
                 num_grps, grp_embd_dim,
                 num_pos, pos_embd_dims,
                 name='RMTPP_VAR', **kwargs):
        super(RMTPP_VAR, self).__init__(name=name, **kwargs)

        self.hidden_layer_size = hidden_layer_size
        self.layer_bin_embd = layers.Embedding(input_dim=num_bins+1,
                                               output_dim=bin_embd_dim,
                                               mask_zero=True)
        self.layer_grp_embd = layers.Embedding(input_dim=num_grps+1,
                                               output_dim=grp_embd_dim,
                                               mask_zero=True)
        self.layer_pos_embd = layers.Embedding(input_dim=num_pos+1,
                                               output_dim=pos_embd_dims,
                                               mask_zero=True)

        self.l_1 = tf.keras.layers.Dense(hidden_layer_size,
                                         activation=tf.nn.relu,
                                         name='bl_1')
        self.l_2 = tf.keras.layers.Dense(hidden_layer_size,
                                         activation=tf.nn.relu,
                                         name='bl_2')
        self.l_out = tf.keras.layers.Dense(1, activation=tf.nn.softplus,
                                           name='bl_out')

    def call(self, inputs, bin_id, grp_id, pos_id, debug=False):
        bin_embed = self.layer_bin_embd(bin_id)
        grp_embed = self.layer_grp_embd(grp_id)
        pos_embed = self.layer_pos_embd(pos_id)


        inputs = tf.concat([bin_embed, grp_embed, pos_embed], axis=-1)
        l_1_out = self.l_1(inputs)
        l_2_out = self.l_2(l_1_out)
        l_out = self.l_out(l_2_out)
        output = l_out

        output = output + tf.ones_like(output) * 0.00000001

        return output

def calibration_model(args):
    learning_rate = args.learning_rate

    model = keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=[1]),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    # optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

class InverseTransformSampling(layers.Layer):
    """Uses (D, WT) to sample E[f*(g)], expected gap before next event."""
    def call(self, inputs):
        D, WT = inputs
        # u = tf.ones_like(D) * tf.range(0.0, 1.0, 1.0/500.0)
        u = tf.ones_like(D) * tf.random.uniform([500], minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
        c = -tf.exp(D)
        val = one_by(WT) * tf.math.log(WT * one_by(c) * tf.math.log(1.0 - u) + 1.0)
        val = tf.reduce_mean(val, axis=-1, keepdims=True)
        return val

class RMTPP(tf.keras.Model):
    def __init__(self,
                 hidden_layer_size,
                 embed_size,
                 name='RMTPP',
                 num_types=False,
                 use_intensity=True,
                 use_count_model=False,
                 use_var_model=False,
                 use_time_feats=True,
                 **kwargs):
        super(RMTPP, self).__init__(name=name, **kwargs)
        self.use_intensity = use_intensity
        self.use_count_model = use_count_model
        self.use_var_model = use_var_model
        self.use_time_feats = use_time_feats
        self.num_types = num_types
        self.embed_size = embed_size
        
        self.rnn_layer = layers.GRU(hidden_layer_size, return_sequences=True,
                                    return_state=True, stateful=False,
                                    name='GRU_Layer')

        if self.num_types>1:
            self.embedding_layer = layers.Embedding(num_types+1, embed_size,
                                                    mask_zero=True,
                                                    name='marks_embedding')
        if not self.use_intensity:
            self.D_layer = layers.Dense(1, activation=tf.nn.softplus, name='D_layer')
        else:
            self.D_layer = layers.Dense(1, activation=tf.nn.softplus, name='D_layer')

        if self.use_count_model:
            self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')

        if self.use_var_model:
            self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')
            
        if self.use_intensity:
            self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')
            self.gaps_output_layer = InverseTransformSampling()
        if self.num_types>1:
            self.marks_output_layer = layers.Dense(num_types,
                                                   activation='softmax',
                                                   name='marks_output_layer')

    def call(self, gaps, feats, types, initial_state=None, next_state_sno=1):
        ''' Forward pass of the RMTPP model'''

        self.gaps = gaps
        self.types = types
        self.initial_state = initial_state
        
        # Gather input for the rnn
        if self.num_types>1:
            self.types_embd = self.embedding_layer(self.types)
        if self.use_time_feats:
            feats = feats/24.
            rnn_inputs = tf.concat([self.gaps, feats], axis=-1)
        if self.num_types>1:
            rnn_inputs = tf.concat([rnn_inputs, self.types_embd], axis=-1)
        else:
            rnn_inputs = self.gaps

        self.hidden_states, self.final_state \
                = self.rnn_layer(rnn_inputs,
                                 mask=gaps>0.,
                                 initial_state=self.initial_state)

        # Generate D, WT, and gaps_pred
        self.D = self.D_layer(self.hidden_states)
        if self.use_intensity:
            self.D = -self.D
        
        if self.use_intensity:
            self.WT = self.WT_layer(self.hidden_states)
            self.gaps_pred = self.gaps_output_layer((self.D, self.WT))
        elif self.use_count_model:
            self.WT = self.WT_layer(self.hidden_states)
            self.gaps_pred = self.WT
        elif self.use_var_model:
            self.WT = self.WT_layer(self.hidden_states) # Mean of sistribution
            out_mean = self.D
            out_stddev = self.WT
            gaussian_distribution = tfp.distributions.Normal(
                out_mean, out_stddev, validate_args=False, allow_nan_stats=True, 
                name='Normal'
            )
            # output_samples = gaussian_distribution.sample(1000)
            # self.gaps_pred = tf.reduce_mean(output_samples, axis=0)
            self.gaps_pred = out_mean
        else:
            self.gaps_pred = self.D
            self.WT = tf.zeros_like(self.D)

        if self.num_types>1:
            self.types_logits = self.marks_output_layer(self.hidden_states)
        else:
            # Dummy logits
            self.types_logits = tf.concat(
                [tf.ones_like(self.gaps_pred),
                 tf.zeros_like(self.gaps_pred)],
                axis=-1,
            )
        
        next_initial_state = self.hidden_states[:,next_state_sno-1]
        final_state = self.hidden_states[:,-1]
        return self.gaps_pred, self.types_logits, self.D, self.WT, next_initial_state, final_state

def build_rmtpp_model(args, use_intensity, use_var_model, num_types):
    hidden_layer_size = args.hidden_layer_size
    batch_size = args.batch_size
    enc_len = args.enc_len
    learning_rate = args.learning_rate
    model = RMTPP(
        hidden_layer_size,
        args.embed_size,
        use_intensity=use_intensity,
        num_types=num_types,
        use_var_model=use_var_model,
        use_time_feats=(not args.no_rmtpp_model_feats)
    )
    #model.build(input_shape=(batch_size, enc_len, 1))
    optimizer = keras.optimizers.Adam(learning_rate)
    return model, optimizer

class PureHierarchical(tf.keras.Model):
    def __init__(self,
                 hidden_layer_size,
                 name='PureHierarchical',
                 use_intensity=True,
                 use_count_model=False,
                 comp_bin_sz=False,
                 **kwargs):
        super(PureHierarchical, self).__init__(name=name, **kwargs)
        self.use_intensity = use_intensity
        self.use_count_model = use_count_model
        self.comp_bin_sz = comp_bin_sz

        self.rnn_layer1 = RMTPP(hidden_layer_size, name="RMTPP_layer1", use_intensity=use_intensity, use_var_model=False)
        self.rnn_layer2 = RMTPP(hidden_layer_size, name="RMTPP_layer2", use_intensity=use_intensity, use_var_model=False)

        self.state_transform_nw = layers.Dense(hidden_layer_size, name='state_transform_nw')

        if not self.use_intensity:
            self.D_layer = layers.Dense(1, name='D_layer')
        else:
            self.D_layer = layers.Dense(1, activation=tf.nn.softplus, name='D_layer')
            self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')
            self.gaps_output_layer = InverseTransformSampling()

        if self.use_count_model:
            self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')


    def call(self, gaps, initial_state=None, gaps_out=None, next_state_sno=1):
        ''' Forward pass of the PureHierarchical model'''

        self.gaps = gaps
        self.gaps_out = gaps_out
        self.initial_state = initial_state
        
        rnn_inputs_l2 = tf.reduce_sum(self.gaps, axis=2)

        self.gaps_pred_l2, self.D_l2, self.WT_l2, next_initial_state, final_state \
                = self.rnn_layer2(rnn_inputs_l2,
                                 initial_state=self.initial_state)

        self.hidden_state_l2 = self.rnn_layer2.hidden_states
        self.hidden_state_l2_transformed = self.state_transform_nw(self.hidden_state_l2)

        prev_gaps = rnn_inputs_l2/self.comp_bin_sz

        if self.gaps_out is not None:
            rnn_inputs_l1 = tf.concat([tf.expand_dims(prev_gaps, axis=-1), self.gaps_out[:,:,:-1]], axis=2)
            gaps_pred_l1_lst = list()
            D_lst = list()
            WT_lst = list()
            for idx in range(self.gaps.shape[1]):
                gaps_pred_tmp, D_pred, WT_pred, _, _ = \
                    self.rnn_layer1(rnn_inputs_l1[:,idx], 
                        initial_state = self.hidden_state_l2_transformed[:,idx])
                gaps_pred_l1_lst.append(gaps_pred_tmp)
                D_lst.append(D_pred)
                WT_lst.append(WT_pred)
            self.gaps_pred_l1 = tf.stack(gaps_pred_l1_lst, axis=1)
            self.D = tf.stack(D_lst, axis=1)
            self.WT = tf.stack(WT_lst, axis=1)
        else:
            gaps_pred_l1_lst = list()
            init_prev_state = self.hidden_state_l2_transformed
            seq_idx = self.gaps.shape[1]-1
            prev_state = init_prev_state[:,seq_idx]
            prev_gaps_inp = prev_gaps[:,seq_idx]
            rnn_inputs_l1 = tf.expand_dims(prev_gaps_inp, axis=-1)

            for idx in range(self.gaps.shape[2]):
                rnn_inputs_l1, _, _, prev_state, _ = \
                    self.rnn_layer1(rnn_inputs_l1, 
                        initial_state = prev_state)
                gaps_pred_l1_lst.append(rnn_inputs_l1)

            self.gaps_pred_l1 = tf.stack(gaps_pred_l1_lst, axis=2)
            self.gaps_pred_l1 = tf.concat([self.gaps[:,:-1], self.gaps_pred_l1], axis=1)
            self.D = None
            self.WT = None

        return [self.gaps_pred_l2, self.D_l2, self.WT_l2,
               self.gaps_pred_l1, self.D, self.WT, 
               next_initial_state, final_state]

def build_pure_hierarchical_model(args, use_intensity):
    hidden_layer_size = args.hidden_layer_size
    batch_size = args.batch_size
    enc_len = args.comp_enc_len
    comp_bin_sz = args.comp_bin_sz
    learning_rate = args.learning_rate
    model = PureHierarchical(hidden_layer_size, use_intensity=use_intensity, comp_bin_sz=comp_bin_sz)
    model.build(input_shape=(batch_size, enc_len, comp_bin_sz, 1))
    optimizer = keras.optimizers.Adam(learning_rate)
    return model, optimizer

def hierarchical_model(args):
    hidden_layer_size = args.hidden_layer_size
    in_bin_sz = args.in_bin_sz
    out_bin_sz = args.out_bin_sz
    learning_rate = args.learning_rate

    model = keras.Sequential([
        layers.Dense(hidden_layer_size, activation='relu', input_shape=[in_bin_sz]),
        layers.Dense(hidden_layer_size, activation='relu'),
        layers.Dense(hidden_layer_size, activation='relu'),
        layers.Dense(out_bin_sz)
    ])

    # optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

class NegativeLogLikelihood_CountModel(tf.keras.losses.Loss):
    def __init__(self, distribution_params, distribution_name,
                 reduction=keras.losses.Reduction.AUTO,
                 name='NegativeLogLikelihood_CountModel'):
        super(NegativeLogLikelihood_CountModel, self).__init__(reduction=reduction, name=name)
        self.distribution_name = distribution_name
        if self.distribution_name == 'NegativeBinomial':
            self.total_count = distribution_params[0]
            self.probs = distribution_params[1]
        elif self.distribution_name == 'Gaussian':
            self.mu = distribution_params[0]
            self.stddev = distribution_params[1]
        elif self.distribution_name == 'var_model':
            self.output_variance = distribution_params[1]

    def call(self, y_true, y_pred):
        count_distribution = None
        loss = None

        if self.distribution_name == 'NegativeBinomial':
            nb_distribution = tfp.distributions.NegativeBinomial(
                self.total_count, logits=None, probs=self.probs, validate_args=False, allow_nan_stats=True,
                name='NegativeBinomial'
            )
            count_distribution = nb_distribution
            loss = -tf.reduce_mean(count_distribution.log_prob(y_true))

        elif self.distribution_name == 'Gaussian':
            gaussian_distribution = tfp.distributions.Normal(
                self.mu, self.stddev, validate_args=False, allow_nan_stats=True, 
                name='Normal'
            )
            count_distribution = gaussian_distribution
            loss = -tf.reduce_mean(count_distribution.log_prob(y_true))

        elif self.distribution_name == 'var_model':
            count_loss = tf.reduce_mean((y_true - y_pred) * (y_true - y_pred))
            assumed_var = ((y_true - y_pred)*(y_true - y_pred))
            var_loss = tf.reduce_mean((self.output_variance - assumed_var) * (self.output_variance - assumed_var))
            loss = count_loss + var_loss

        return loss

class COUNT_MODEL(tf.keras.Model):
    def __init__(self,
                hidden_layer_size,
                out_bin_sz,
                distribution_name,
                bin_size,
                use_time_feats=True,
                network_type='ff', # ff or rnn or deepar
                name='count_model',
                **kwargs):
        super(COUNT_MODEL, self).__init__(name=name, **kwargs)
        self.distribution_name = distribution_name
        self.out_bin_sz = out_bin_sz
        self.use_time_feats = use_time_feats
        self.network_type = network_type
        self.bin_size = bin_size
        if self.network_type in ['ff', 'rnn']:
            out_layer_size = out_bin_sz
        elif self.network_type == 'deepar':
            out_layer_size = 1

        if self.network_type == 'ff':
            self.dense1 = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu, name="count_dense1")
        elif self.network_type == 'rnn':
            self.rnn_layer = layers.GRU(hidden_layer_size, return_sequences=True,
                                        return_state=True, stateful=False,
                                        name='GRU_Layer')
        elif self.network_type == 'deepar':
            self.enc_layer = layers.GRU(hidden_layer_size, return_sequences=True,
                                        return_state=True, stateful=False,
                                        name='enc_layer')
            self.dec_layer = layers.GRU(hidden_layer_size, return_sequences=True,
                                        return_state=True, stateful=False,
                                        name='dec_layer')

        self.dense2 = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu, name="count_dense2")
        self.out_layer = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu, name="out_layer")

        if self.distribution_name in ['NegativeBinomial', 'Gaussian']:
            self.out_alpha_layer = tf.keras.layers.Dense(out_layer_size, name="out_alpha_layer")
            self.out_mu_layer = tf.keras.layers.Dense(out_layer_size, name="out_mu_layer")

        elif self.distribution_name == 'var_model':
            self.count_out_layer = tf.keras.layers.Dense(out_layer_size, name="count_out_layer")
            self.var_dense1 = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu, name="var_count_dense1")
            self.var_out_layer = tf.keras.layers.Dense(out_layer_size, activation=tf.keras.activations.softplus, name="var_out_layer")

    def call(self, inputs, feats_in, true_outputs=None, debug=False):

        if self.use_time_feats:
            if self.network_type == 'deepar':
                feats_out = feats_in[:, -1:] + tf.cumsum(tf.ones([1, self.out_bin_sz, 1]), axis=1) * self.bin_size/3600.
                feats_out = get_time_features(feats_out * 3600.)
                feats_out = feats_out / 24.

            feats_in = feats_in/24.
            inputs = tf.concat([inputs, feats_in], axis=-1)


        if self.network_type == 'ff':
            inputs = tf.reshape(inputs, [inputs.shape[0], -1])
            hidden_state_1 = self.dense1(inputs)
        elif self.network_type == 'rnn':
            hidden_states, final_state \
                = self.rnn_layer(inputs)
            hidden_state_1 = final_state
        elif self.network_type == 'deepar':
            hidden_states, final_enc_state \
                = self.enc_layer(inputs)


        def step(hidden_state):

            hidden_state_2 = self.dense2(hidden_state)
            output_state = self.out_layer(hidden_state_2)

            bin_count_output = None
    
            if self.distribution_name == 'NegativeBinomial':
                out_alpha = self.out_alpha_layer(output_state)
                out_mu = self.out_mu_layer(output_state)
    
                out_alpha = (tf.math.softplus(out_alpha))
                out_mu = (tf.math.softplus(out_mu))
    
                out_alpha = tf.clip_by_value(out_alpha, 1e-3, 50.0)
                out_mu = tf.clip_by_value(out_mu, 1e-3, 50.0)
    
                total_count = 1.0 / out_alpha
    
                alpha_mu = (out_alpha * out_mu)
                probs = (alpha_mu / (1.0+alpha_mu))
    
                nb_distribution = tfp.distributions.NegativeBinomial(
                    total_count, logits=None, probs=probs, validate_args=False, allow_nan_stats=True,
                    name='NegativeBinomial'
                )
                # output_samples = nb_distribution.sample(1000)
                # bin_count_output = tf.reduce_mean(output_samples, axis=0)
                #dist_params = [total_count, probs]
                dist_params_mu = total_count
                dist_params_stddev = probs
                bin_count_output = total_count
    
            elif self.distribution_name == 'Gaussian':
                out_alpha = self.out_alpha_layer(output_state)
                out_mu = self.out_mu_layer(output_state)
    
                #TODO: Can this layer outputs stddev of distributions
                out_stddev = (tf.math.softplus(out_alpha))
    
                gaussian_distribution = tfp.distributions.Normal(
                    out_mu, out_stddev, validate_args=False, allow_nan_stats=True, 
                    name='Normal'
                )
                # output_samples = gaussian_distribution.sample(1000)
                # bin_count_output = tf.reduce_mean(output_samples, axis=0)
                #dist_params = [out_mu, out_stddev]
                dist_params_mu = out_mu
                dist_params_stddev = out_stddev
                bin_count_output = out_mu
    
            elif self.distribution_name == 'var_model':
                bin_count_output = self.count_out_layer(output_state)
    
                var_hidden_state_1 = self.var_dense1(output_state)
                output_variance = self.var_out_layer(var_hidden_state_1)
                #dist_params = [None, output_variance]
                dist_params_mu = None
                dist_params_stddev = output_variance

            return bin_count_output, dist_params_mu, dist_params_stddev

        if self.network_type in ['ff', 'rnn']:
            (
                bin_count_output,
                dist_params_mu,
                dist_params_stddev
            ) = step(hidden_state_1)
        elif self.network_type == 'deepar':
            all_bin_count_output = list()
            all_dist_params_mu = list()
            all_dist_params_stddev = list()

            curr_dec_state = final_enc_state
            if true_outputs is not None:

                inputs = tf.expand_dims(true_outputs, axis=-1)
                if self.use_time_feats:
                    inputs = tf.concat([inputs, feats_out], axis=-1)
                dec_states, _ = self.dec_layer(inputs, initial_state=curr_dec_state)
                (
                    bin_count_output,
                    dist_params_mu,
                    dist_params_stddev
                ) = step(dec_states)
                bin_count_output = tf.squeeze(bin_count_output, axis=-1)
                dist_params_mu = tf.squeeze(dist_params_mu, axis=-1)
                dist_params_stddev = tf.squeeze(dist_params_stddev, axis=-1)
            else:
                for j in range(self.out_bin_sz):
                    (
                        bin_count_output,
                        dist_params_mu,
                        dist_params_stddev
                    ) = step(curr_dec_state)
    
                    all_bin_count_output.append(bin_count_output)
                    all_dist_params_mu.append(dist_params_mu)
                    all_dist_params_stddev.append(dist_params_stddev)
    
                    curr_feats_out = feats_in[:, -1] + (j+1.) * self.bin_size/3600.
                    curr_feats_out = get_time_features(curr_feats_out * 3600.)
                    curr_feats_out = curr_feats_out / 24.
                    if true_outputs is not None: # Training Mode
                        inputs = true_outputs[:, j:j+1]
                    else: # Inference Mode
                        inputs = dist_params_mu

                    if self.use_time_feats:
                        inputs = tf.stack([inputs, curr_feats_out], axis=-1)
    
                    _, curr_dec_state = self.dec_layer(inputs, initial_state=curr_dec_state)
    
                bin_count_output = tf.concat(all_bin_count_output, axis=1)
                dist_params_mu = tf.concat(all_dist_params_mu, axis=1)
                dist_params_stddev = tf.concat(all_dist_params_stddev, axis=1)

        dist_params = [dist_params_mu, dist_params_stddev]

        return bin_count_output, dist_params


def build_count_model(args, distribution_name):
    hidden_layer_size = args.hidden_layer_size
    batch_size = args.batch_size
    in_bin_sz = args.in_bin_sz
    out_bin_sz = args.out_bin_sz
    learning_rate = args.learning_rate

    model = COUNT_MODEL(
        hidden_layer_size,
        out_bin_sz, 
        distribution_name,
        args.bin_size,
        use_time_feats=(not args.no_count_model_feats),
        network_type=args.cnt_net_type,
    )
    #model.build(input_shape=(batch_size, in_bin_sz))
    optimizer = keras.optimizers.Adam(learning_rate)
    return model, optimizer


# ----- Baseline WGAN Model ----- #
class WGAN(tf.keras.Model):
    def __init__(self,
                 g_cell_type='LSTM',
                 g_num_layers=1,
                 g_state_size=64,
                 d_cell_type='LSTM',
                 d_num_layers=1,
                 d_state_size=64,
                 use_time_feats=True,
                 name='WGAN',
                 **kwargs):
        super(WGAN, self).__init__(name=name, **kwargs)

        '''
        TODO:
        Add multi-layer extension

        '''

        self.keep_prob = tf.constant(0.9)
        self.use_time_feats = use_time_feats

        self.enc_rnn_layer = layers.LSTM(g_state_size, return_sequences=True,
                                         return_state=True, stateful=False,
                                         name='enc_rnn_layer')

        if g_cell_type=='Basic':
            pass
        elif g_cell_type=='LSTM':
            self.g_rnn_layer = layers.LSTM(g_state_size, return_sequences=True,
                                         return_state=True, stateful=False,
                                         name='g_lstm_layer')

        if d_cell_type=='Basic':
            pass
        elif d_cell_type=='LSTM':
            self.d_rnn_layer = layers.LSTM(g_state_size, return_sequences=True,
                                         return_state=True, stateful=False,
                                         name='d_lstm_layer')


        self.g_full_connect = layers.Dense(1, activation=tf.nn.softplus, name='g_full_connect',
                                           bias_initializer=tf.keras.initializers.Zeros())


        self.softmax_layer = layers.Dense(1, activation=None, name='softmax_layer',
                                           bias_initializer=tf.keras.initializers.Zeros())

    def run_encoder(self, enc_input):
        '''
        Encode the input in a hidden_state
        '''
        rnn_outputs, h_state, c_state = self.enc_rnn_layer(enc_input)
        final_state = [h_state, c_state]
        return rnn_outputs, final_state

    def generator(self,
                  g_inputs, #dims batch_size x num_steps x input_size
                  enc_inputs=None,
                  enc_feats=None,
                  g_init_state=None):
        '''
        Generates output sequence from on the noise sequence conditioned
        on the input.
        rnn_inputs: Input-conditioned noise sequence
        generator_initial_state -- needs to be conditioned on the input
        TODO:
        Make sure LSTM states are properly handled
        What is G_DIFF and D_DIFF?
            - G_DIFF and D_DIFF are true when point process is represented
                by gaps instead of times
        '''

        if enc_inputs is None:
            assert g_init_state is not None

        if enc_inputs is not None:
            if self.use_time_feats:
                enc_feats = enc_feats/24.
                enc_inputs = tf.concat([enc_inputs, enc_feats], axis=-1)
            _, g_init_state = self.run_encoder(enc_inputs)

        # rnn_outputs, self.g_h_state, self.g_c_state \
        #         = self.g_rnn_layer(g_inputs)
        rnn_outputs, self.g_h_state, self.g_c_state \
                = self.g_rnn_layer(g_inputs,
                                   initial_state=g_init_state)

        # Add dropout
        # rnn_outputs = tf.nn.dropout(rnn_outputs, self.keep_prob)

        # Softmax layer
        logits_t = self.g_full_connect(rnn_outputs)# +1 #abs, exp, or nothing is better
        #if not D_DIFF and G_DIFF: # depend on D_DIFF
        #    logits_t = tf.cumsum(logits_t,axis=1)

        self.g_state = [self.g_h_state, self.g_c_state]

        return logits_t

    def discriminator(self,
                      enc_inputs, #dims batch_size x num_steps x input_size
                      rnn_inputs):
        '''
        TODO
        What is COST_ALL?
            - Ignore COST_ALL for now
        what is lower_triangular_ones?
            - lower_triangular_ones is for sequence length masking
            - Make sure sequence lenght masking is done properly
        '''

        _, g_init_state = self.run_encoder(enc_inputs)

        rnn_outputs, h_state, c_state \
                = self.d_rnn_layer(rnn_inputs,
                                   initial_state=g_init_state)

        # Add dropout
        rnn_outputs = tf.nn.dropout(rnn_outputs, self.keep_prob)

        # Softmax layer
        logits = self.softmax_layer(rnn_outputs)

        fval = tf.reduce_mean(logits, axis=1)
        # TODO Incorporate sequence_length while calculating fval


        return fval

    def call(self, enc_inputs, rnn_inputs):
        '''
        WGAN forward pass:
            Encode the input through encoder_rnn
            Use final state as the initial_state for the generator
            Generate the output sequence using rnn_inputs
        '''
        return self.generator(enc_inputs, rnn_inputs)


# ----- Start: Baseline Seq2Seq Model ----- #
class Seq2Seq(tf.keras.Model):
    '''
        Implementation of the paper:
        Learning Conditional Generative Models for Temporal Point Processes
        https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16163/16203
    '''
    def __init__(self,
                 g_cell_type='LSTM',
                 g_num_layers=1,
                 g_state_size=64,
                 d_cell_type='LSTM',
                 d_num_layers=1,
                 d_state_size=64,
                 use_time_feats=True,
                 name='Seq2Seq',
                 **kwargs):
        super(Seq2Seq, self).__init__(name=name, **kwargs)

        '''
        TODO:
        Add multi-layer extension

        '''

        self.keep_prob = tf.constant(0.9)
        self.use_time_feats = use_time_feats

        self.enc_rnn_layer = layers.LSTM(g_state_size, return_sequences=True,
                                         return_state=True, stateful=False,
                                         name='enc_rnn_layer')

        self.dec_rnn_layer = layers.LSTM(g_state_size, return_sequences=True,
                                         return_state=True, stateful=False,
                                         name='dec_rnn_layer')

        if g_cell_type=='Basic':
            pass
        elif g_cell_type=='LSTM':
            self.g_rnn_layer = layers.LSTM(g_state_size, return_sequences=True,
                                         return_state=True, stateful=False,
                                         name='g_lstm_layer')

        self.d_layer_1 = tf.keras.layers.Conv1D(3, 10, activation='relu')
        self.d_layer_2 = tf.keras.layers.Conv1D(5, 20, activation='relu')
        #TODO How to add skip-connection?


        self.g_full_connect = layers.Dense(1, activation=tf.nn.softplus, name='g_full_connect',
                                           bias_initializer=tf.keras.initializers.Zeros())


        self.softmax_layer = layers.Dense(1, activation=None, name='softmax_layer',
                                           bias_initializer=tf.keras.initializers.Zeros())

    def run_encoder(self, enc_input):
        '''
        Encode the input in a hidden_state
        '''
        rnn_outputs, h_state, c_state = self.enc_rnn_layer(enc_input)
        final_state = [h_state, c_state]
        logits_t = self.g_full_connect(rnn_outputs)# +1 #abs, exp, or nothing is better
        return logits_t, rnn_outputs, final_state

    def run_decoder(self, dec_input, init_state):
        '''
        Encode the input in a hidden_state
        '''
        rnn_outputs, h_state, c_state = self.dec_rnn_layer(dec_input, initial_state=init_state)
        final_state = [h_state, c_state]
        logits_t = self.g_full_connect(rnn_outputs)# +1 #abs, exp, or nothing is better
        return logits_t, rnn_outputs, final_state

    def generator(self,
                  dec_inputs, #dims batch_size x num_steps x input_size
                  dec_feats,
                  enc_inputs=None,
                  enc_feats=None,
                  dec_init_state=None):
        '''
        Generates output sequence from on the noise sequence conditioned
        on the input.
        rnn_inputs: Input-conditioned noise sequence
        generator_initial_state -- needs to be conditioned on the input
        TODO:
        Make sure LSTM states are properly handled
        What is G_DIFF and D_DIFF?
            - G_DIFF and D_DIFF are true when point process is represented
                by gaps instead of times
        '''

        if enc_inputs is None:
            assert dec_init_state is not None

        if enc_inputs is not None:
            if self.use_time_feats:
                enc_feats = enc_feats/24.
                enc_inputs = tf.concat([enc_inputs, enc_feats], axis=-1)
            _, _, dec_init_state = self.run_encoder(enc_inputs)

        # rnn_outputs, self.g_h_state, self.g_c_state \
        #         = self.g_rnn_layer(g_inputs)
        dec_inputs = tf.concat([dec_inputs, dec_feats/24.], axis=-1)
        rnn_outputs, self.g_h_state, self.g_c_state \
                = self.enc_rnn_layer(dec_inputs,
                                     initial_state=dec_init_state)

        # Add dropout
        # rnn_outputs = tf.nn.dropout(rnn_outputs, self.keep_prob)

        # Softmax layer
        logits_t = self.g_full_connect(rnn_outputs)# +1 #abs, exp, or nothing is better
        #if not D_DIFF and G_DIFF: # depend on D_DIFF
        #    logits_t = tf.cumsum(logits_t,axis=1)

        self.g_state = [self.g_h_state, self.g_c_state]

        return logits_t

    def discriminator(self, zeta, rho):
        '''
        Inputs:
            zeta: Input sequence, (batch_size x num_steps x input_size=1)
            rho: Output sequence (batch_size x _ x input_size=1)
        TODO:
            - Add RCNN as given in the paper
            - 
        '''

        self.rcnn_input = tf.concat([zeta, rho], axis=1)

        layer_1_out = self.d_layer_1(self.rcnn_input)
        layer_2_out = self.d_layer_2(layer_1_out)


        # Softmax layer
        logits = self.softmax_layer(layer_2_out)

        #fval = tf.reduce_mean(logits, axis=1)

        return logits

    def call(self, enc_inputs, rnn_inputs):
        '''
        Seq2Seq forward pass:
            Encode the input through encoder_rnn
            Use final state as the initial_state for the generator
            Generate the output sequence using rnn_inputs
        '''
        return self.generator(enc_inputs, rnn_inputs)

# ----- End: Baseline Seq2Seq Model ----- #


# ----- Start: Implementation of Transformer ----- #

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    #assert seq.dim() == 2
    assert len(seq.shape) == 2
    #return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
    non_pad_mask = seq != tf.expand_dims(tf.cast(Constants.PAD, tf.float32), axis=-1)
    non_pad_mask = tf.cast(non_pad_mask, tf.float32)
    return non_pad_mask


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    #len_q = seq_q.size(1)
    len_q = seq_q.shape[1]
    #padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = seq_k == Constants.PAD
    #padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    padding_mask = tf.tile(tf.expand_dims(padding_mask, axis=1), (1, len_q, 1))  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.shape
    #subsequent_mask = torch.triu(
    #    torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = tf.linalg.band_part(
        tf.ones((len_s, len_s)), 0, -1)
    #subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    subsequent_mask = tf.tile(tf.expand_dims(subsequent_mask, axis=0), [sz_b, 1, 1])  # b x ls x ls
    return subsequent_mask

class Encoder(tf.keras.Model):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,
            name='Encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = tf.constant(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            dtype=tf.float32,
        )

        # event type embedding
        #self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)
        self.event_emb = layers.Embedding(num_types+1, d_model, mask_zero=True)
        self.feature_enc_layer = layers.Dense(d_model)

        #TODO: What to do with nn.ModuleList?
        #self.layer_stack = nn.ModuleList([
        #    EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        #    for _ in range(n_layers)])
        self.layer_stack = [
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)]

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time / self.position_vec
        result = result.numpy()
        result[:, :, 0::2] = np.sin(result[:, :, 0::2])
        result[:, :, 1::2] = np.cos(result[:, :, 1::2])
        result = tf.constant(result)
        non_pad_mask = tf.cast(tf.expand_dims(non_pad_mask, axis=-1), tf.float32)
        #return result * non_pad_mask
        return result

    def call(self, event_type, event_time, event_feats, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        #slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask_keypad = tf.cast(slf_attn_mask_keypad, dtype=slf_attn_mask_subseq.dtype)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq)>(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        feats_enc = self.feature_enc_layer(event_feats)
        enc_output = self.event_emb(event_type)

        for enc_layer in self.layer_stack:
            enc_output += (tem_enc + feats_enc)
            enc_output, _ = enc_layer(
                enc_output,
                event_feats,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output

class TypePredictor(tf.keras.Model):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types, name='TypePredictor', **kwargs):
        super(TypePredictor, self).__init__(name=name, **kwargs)

        #self.linear = nn.Linear(dim, num_types, bias=False)
        #nn.init.xavier_normal_(self.linear.weight)
        self.linear = layers.Dense(
            num_types, use_bias=False,
            kernel_initializer=tf.initializers.GlorotNormal())

    def call(self, data, non_pad_mask):
        out = self.linear(data)
        non_pad_mask = tf.cast(tf.expand_dims(non_pad_mask, axis=-1), tf.float32)
        #out = out * non_pad_mask
        return out

class TimePredictor(tf.keras.Model):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types, name='TimePredictor', **kwargs):
        super(TimePredictor, self).__init__(name=name, **kwargs)

        #self.linear = nn.Linear(dim, num_types, bias=False)
        #nn.init.xavier_normal_(self.linear.weight)
        self.linear = layers.Dense(
            1, activation=tf.nn.softplus, use_bias=False,
            kernel_initializer=tf.initializers.GlorotNormal()
        )

    def call(self, data, non_pad_mask):
        out = self.linear(data)
        non_pad_mask = tf.cast(tf.expand_dims(non_pad_mask, axis=-1), tf.float32)
        #out = out * non_pad_mask
        return out


class RNN_layers(tf.keras.Model):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn, name='RNN_layers', **kwargs):
        super(RNN_layers, self).__init__(name=name, **kwargs)

        # TODO: Add multi-layer LSTM for tensorflow
        #self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.rnn = layers.LSTM(
            d_rnn, return_sequences=True,
            return_state=True, stateful=False)

        #self.rnn = tf.keras.layers.StackedRNNCells(
        #    [tf.keras.layers.LSTMCell(d_rnn) for _ in range(num_layers)]
        #)

        #self.projection = nn.Linear(d_rnn, d_model)
        self.projection = layers.Dense(d_model)

    def call(self, data, non_pad_mask):
        #TODO: Resolve packedsequence part

        #lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        #lengths = tf.reduce_sum(tf.squeeze(non_pad_mask, axis=2), axis=1)
        #import ipdb
        #ipdb.set_trace()
        #lengths = tf.reduce_sum(tf.cast(non_pad_mask, tf.float32), axis=1)
        #pack_enc_output = nn.utils.rnn.pack_padded_sequence(
        #    data, lengths, batch_first=True, enforce_sorted=False)
        #temp = self.rnn(pack_enc_output)[0]
        #out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out, _, _ = self.rnn(data)

        out = self.projection(out)
        return out

class Transformer(tf.keras.Model):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
            use_time_feats=True,
            use_marks=False,
            name='Transformer', **kwargs):
        super(Transformer, self).__init__(name=name, **kwargs)

        self.use_time_feats = use_time_feats
        self.use_marks = use_marks

        self.encoder = Encoder(
            num_types=num_types, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

        self.num_types = num_types

        # convert hidden vectors into a scalar
        #self.linear = nn.Linear(d_model, num_types)
        self.linear = layers.Dense(num_types)

        # parameter for the weight of time difference
        self.alpha = -0.1

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = TimePredictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = TypePredictor(d_model, num_types)

    def call(self, event_time, event_feats, event_type):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        event_feats = event_feats/24.

        event_type = tf.cast(event_type, dtype=tf.float32)

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, event_feats, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction)

# ----- End: Implementation of Transformer ----- #
