import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow import keras

#from transformer_helpers.Models import get_non_pad_mask
from models import get_non_pad_mask
import transformer_helpers.Constants as Constants


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    non_pad_mask_not = tf.cast(non_pad_mask==Constants.PAD, tf.float32)
    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    #event.masked_fill_(~non_pad_mask.bool(), 1.0)
    event = tf.where(event==non_pad_mask_not, 1., event)

    result = tf.math.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    #temp_time = diff_time.unsqueeze(2) * \
    #            torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time = tf.expand_dims(diff_time, axis=2) * \
                tf.random.uniform([*(diff_time.shape.as_list()), num_samples])
    #temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    temp_time /= tf.expand_dims((time[:, :-1] + 1), axis=2)

    temp_hid = model.linear(data)[:, 1:, :]
    #temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)
    temp_hid = tf.reduce_sum(temp_hid * type_mask[:, 1:, :], axis=2, keepdims=True)

    #all_lambda = F.softplus(temp_hid + model.alpha * temp_time, threshold=10)
    # No threshold parameter for tf.nn.softplus is available, (or not required).
    all_lambda = tf.nn.softplus(temp_hid + model.alpha * temp_time)
    #all_lambda = torch.sum(all_lambda, dim=2) / num_samples
    all_lambda = tf.reduce_sum(all_lambda, axis=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types):
    """ Log-likelihood of sequence. """

    #non_pad_mask = get_non_pad_mask(types).squeeze(2)
    non_pad_mask = get_non_pad_mask(types)

    #type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)
    #type_mask = tf.zeros([*(types.shape.as_list()), model.num_types])
    #for i in range(model.num_types):
    #    #type_mask[:, :, i] = (types == i + 1).bool().to(data.device)
    #    type_mask[:, :, i] = tf.cast((types == i + 1), tf.bool)
    type_ids = tf.expand_dims(tf.expand_dims(tf.range(1, model.num_types+1, dtype=tf.float32), axis=0), axis=1)
    type_mask = tf.cast((tf.expand_dims(types, axis=-1) == type_ids), tf.float32)

    all_hid = model.linear(data)
    #all_lambda = F.softplus(all_hid, threshold=10)
    all_lambda = tf.nn.softplus(all_hid)
    #type_lambda = torch.sum(all_lambda * type_mask, dim=2)
    type_lambda = tf.reduce_sum(all_lambda * type_mask, axis=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    #event_ll = torch.sum(event_ll, dim=-1)
    event_ll = tf.reduce_sum(event_ll, axis=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    #non_event_ll = torch.sum(non_event_ll, dim=-1)
    non_event_ll = tf.reduce_sum(non_event_ll, axis=-1)

    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    #pred_type = torch.max(prediction, dim=-1)[1]
    pred_type = tf.cast(tf.argmax(prediction, axis=-1), tf.float32)
    #correct_num = torch.sum(pred_type == truth)
    correct_num = tf.reduce_sum(tf.cast(pred_type == truth, tf.float32))

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        #loss = loss_func(prediction.transpose(1, 2), truth)
        loss = loss_func(truth, prediction)

    #loss = torch.sum(loss)
    loss = tf.reduce_sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    #prediction.squeeze_(-1)
    #tf.squeeze(prediction, axis=-1)

    #true = event_time[:, 1:] - event_time[:, :-1]
    true = event_time
    #prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    #se = torch.sum(diff * diff)
    se = tf.reduce_sum(diff * diff)
    return se


class LabelSmoothingLoss(tf.keras.losses.Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(
        self, label_smoothing, tgt_vocab_size, ignore_index=-100,
        reduction=keras.losses.Reduction.AUTO,
        name='LabelSmoothingLoss'):

        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__(reduction=reduction, name=name)

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def call(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = tf.cast((target != tf.cast((self.ignore_index), tf.float32)), tf.float32)

        #target[target.eq(self.ignore_index)] = 0
        #target[target == (self.ignore_index)] = 0
        target = tf.where(target==self.ignore_index, 0, target)
        #one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = tf.cast(tf.one_hot(tf.cast(target, tf.int64), depth=self.num_classes), tf.float32)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        #log_prb = F.log_softmax(output, dim=-1)
        log_prb = tf.nn.log_softmax(output, axis=-1)
        #loss = -(one_hot * log_prb).sum(dim=-1)
        loss = tf.reduce_sum(-(one_hot * log_prb), axis=-1)
        loss = loss * non_pad_mask
        return loss
