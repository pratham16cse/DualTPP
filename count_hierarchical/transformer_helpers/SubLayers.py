import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_helpers.Constants as Constants
from transformer_helpers.Modules import ScaledDotProductAttention

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow_addons as tfa


class MultiHeadAttention(tf.keras.Model):
    """ Multi-Head Attention module """

    def __init__(
        self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True,
        name='MultiHeadAttention', **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = layers.Dense(n_head * d_k, use_bias=False)
        self.w_ks = layers.Dense(n_head * d_k, use_bias=False)
        self.w_vs = layers.Dense(n_head * d_v, use_bias=False)
        # Note: layers.Dense uses xavier_uniform_ initialization by default
        #nn.init.xavier_uniform_(self.w_qs.weight)
        #nn.init.xavier_uniform_(self.w_ks.weight)
        #nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = layers.Dense(d_model)
        #nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.dropout = nn.Dropout(dropout)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = tf.reshape(self.w_qs(q), [sz_b, len_q, n_head, d_k])
        k = tf.reshape(self.w_ks(k), [sz_b, len_k, n_head, d_k])
        v = tf.reshape(self.w_vs(v), [sz_b, len_v, n_head, d_v])

        # Transpose for attention dot product: b x n x lq x dv
        #q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        #output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = tf.reshape(tf.transpose(output, perm=[0, 2, 1, 3]), [sz_b, len_q, -1])
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(tf.keras.Model):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(
        self, d_in, d_hid, dropout=0.1, normalize_before=True,
        name='PositionwiseFeedForward', **kwargs):
        super(PositionwiseFeedForward, self).__init__(name=name, **kwargs)

        self.normalize_before = normalize_before

        self.w_1 = layers.Dense(d_hid)  # position-wise
        self.w_2 = layers.Dense(d_in)  # position-wise

        #self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #self.dropout = nn.Dropout(dropout)
        self.dropout = tf.keras.layers.Dropout(dropout)


    def call(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = tfa.activations.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
