import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

ETH = 10.0
one_by = tf.math.reciprocal_no_nan

class InverseTransformSampling(layers.Layer):
    """Uses (D, WT) to sample E[f*(g)], expected gap before next event."""
    def call(self, inputs):
        D, WT = inputs
        u = tf.ones_like(D) * tf.range(0.0, 1.0, 1.0/500.0)
        c = -tf.exp(D)
        val = one_by(WT) * tf.math.log(WT * one_by(c) * tf.math.log(1.0 - u) + 1.0)
        val = tf.reduce_mean(val, axis=-1, keepdims=True)
        return val

class RMTPP(tf.keras.Model):
    def __init__(self,
                 hidden_layer_size,
                 name='RMTPP',
                 use_intensity=True,
                 use_count_model=False,
                 **kwargs):
        super(RMTPP, self).__init__(name=name, **kwargs)
        self.use_intensity = use_intensity
        self.use_count_model = use_count_model
        
        self.rnn_layer = layers.GRU(hidden_layer_size, return_sequences=True,
                                    return_state=True, stateful=False,
                                    name='GRU_Layer')
        
        self.D_layer = layers.Dense(1, name='D_layer')

        if self.use_count_model:
            self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')
            
        if self.use_intensity:
            self.WT_layer = layers.Dense(1, activation=tf.nn.softplus, name='WT_layer')
            self.gaps_output_layer = InverseTransformSampling()

    def call(self, gaps, initial_state=None):
        ''' Forward pass of the RMTPP model'''

        self.gaps = gaps
        self.initial_state = initial_state
        
        # Gather input for the rnn
        rnn_inputs = self.gaps

        self.hidden_states, self.final_state \
                = self.rnn_layer(rnn_inputs,
                                 initial_state=self.initial_state)

        # Generate D, WT, and gaps_pred
        self.D = self.D_layer(self.hidden_states)
        
        if self.use_intensity:
            self.WT = self.WT_layer(self.hidden_states)
            self.gaps_pred = self.gaps_output_layer((self.D, self.WT))
        elif self.use_count_model:
            self.WT = self.WT_layer(self.hidden_states)
            self.gaps_pred = self.WT
        else:
            self.gaps_pred = tf.nn.softplus(self.D)
            self.WT = tf.zeros_like(self.D)
        
        next_initial_state = self.hidden_states[:,0]
        final_state = self.hidden_states[:,-1]
        return self.gaps_pred, self.D, self.WT, next_initial_state, final_state

def build_rmtpp_model(args):
    hidden_layer_size = args.hidden_layer_size
    batch_size = args.batch_size
    enc_len = args.enc_len
    learning_rate = args.learning_rate
    model = RMTPP(hidden_layer_size)
    model.build(input_shape=(batch_size, enc_len, 1))
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
            self.var = distribution_params[1]

    def call(self, y_true, y_pred):
        count_distribution = None

        if self.distribution_name == 'NegativeBinomial':
            nb_distribution = tfp.distributions.NegativeBinomial(
                self.total_count, logits=None, probs=self.probs, validate_args=False, allow_nan_stats=True,
                name='NegativeBinomial'
            )
            count_distribution = nb_distribution

        elif self.distribution_name == 'Gaussian':
            gaussian_distribution = tfp.distributions.Normal(
                self.mu, self.var, validate_args=False, allow_nan_stats=True, 
                name='Normal'
            )
            count_distribution = gaussian_distribution

        return -tf.reduce_mean(count_distribution.log_prob(y_true))

class COUNT_MODEL(tf.keras.Model):
    def __init__(self,
                hidden_layer_size,
                out_bin_sz,
                distribution_name,
                name='count_model',
                **kwargs):
        super(COUNT_MODEL, self).__init__(name=name, **kwargs)
        self.dense1 = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu, name="count_dense1")
        self.dense2 = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu, name="count_dense2")
        self.out_layer = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu, name="out_layer")

        self.out_alpha_layer = tf.keras.layers.Dense(out_bin_sz, name="out_alpha_layer")
        self.out_mu_layer = tf.keras.layers.Dense(out_bin_sz, name="out_mu_layer")
        self.distribution_name = distribution_name

    def call(self, inputs, debug=False):
        hidden_state_1 = self.dense1(inputs)
        hidden_state_2 = self.dense2(hidden_state_1)
        output_state = self.out_layer(hidden_state_2)

        out_alpha = self.out_alpha_layer(output_state)
        out_mu = self.out_mu_layer(output_state)

        if self.distribution_name == 'NegativeBinomial':
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
            output_samples = nb_distribution.sample(1000)
            distribution_params = [total_count, probs]

        elif self.distribution_name == 'Gaussian':
            out_var = (tf.math.softplus(out_alpha))

            gaussian_distribution = tfp.distributions.Normal(
                out_mu, out_var, validate_args=False, allow_nan_stats=True, 
                name='Normal'
            )
            output_samples = gaussian_distribution.sample(1000)
            distribution_params = [out_mu, out_var]

        bin_count_output = tf.reduce_mean(output_samples, axis=0)
        return bin_count_output, distribution_params


def build_count_model(args, distribution_name):
    hidden_layer_size = args.hidden_layer_size
    batch_size = args.batch_size
    in_bin_sz = args.in_bin_sz
    out_bin_sz = args.out_bin_sz
    learning_rate = args.learning_rate

    model = COUNT_MODEL(hidden_layer_size, out_bin_sz, distribution_name)
    model.build(input_shape=(batch_size, in_bin_sz))
    optimizer = keras.optimizers.Adam(learning_rate)
    return model, optimizer
