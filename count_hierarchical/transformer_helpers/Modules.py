import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf


class ScaledDotProductAttention(tf.keras.Model):
    """ Scaled Dot-Product Attention """

    def __init__(
        self, temperature, attn_dropout=0.2,
        name='ScaledDotProductAttention', **kwargs):
        super(ScaledDotProductAttention, self).__init__(name=name, **kwargs)

        self.temperature = temperature
        #self.dropout = nn.Dropout(attn_dropout)
        self.dropout = tf.keras.layers.Dropout(attn_dropout)

    def call(self, q, k, v, mask=None):
        #attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = tf.matmul(q / self.temperature, tf.transpose(k, perm=[0, 1, 3, 2]))

        if mask is not None:
            #attn = attn.masked_fill(mask, -1e9)
            attn = tf.where(mask, attn, -1e9)

        attn = self.dropout(tf.nn.softmax(attn, axis=-1))
        #output = torch.matmul(attn, v)
        output = tf.matmul(attn, v)

        return output, attn
