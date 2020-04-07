import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
    batch_size = args.batch_size
    enc_len = args.enc_len
    learning_rate = args.learning_rate
    model = RMTPP(batch_size)
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
        layers.Dense(out_bin_sz)
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model
