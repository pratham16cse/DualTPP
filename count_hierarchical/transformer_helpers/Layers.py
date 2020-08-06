#import torch.nn as nn
import tensorflow as tf
from transformer_helpers.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(tf.keras.Model):
    """ Compose with two layers """

    def __init__(
        self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True,
        name='EncoderLayer', **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def call(self, enc_input, feats, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        non_pad_mask = tf.cast(tf.expand_dims(non_pad_mask, axis=-1), tf.float32)

        #enc_output = tf.concat([enc_output, feats], axis=-1)
        #enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        #enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
