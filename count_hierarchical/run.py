import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras.preprocessing.sequence import pad_sequences

import cvxpy as cp
import numpy as np
import pandas as pd
from itertools import chain
from bisect import bisect_right, bisect_left
from multiprocessing import Pool
import matplotlib.pyplot as plt
import properscoring as ps
from operator import itemgetter

import models
import utils
import os
import sys
import time

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from utils import IntensityHomogenuosPoisson, generate_sample
from utils import get_time_features
from utils import add_metrics_to_dict
from utils import write_arr_to_file
from utils import write_pe_metrics_to_file
from utils import write_opt_losses_to_file

#from transformer_helpers import Utils
import transformer_utils

train_gap_metric_mae = tf.keras.metrics.MeanAbsoluteError()
train_gap_metric_mse = tf.keras.metrics.MeanSquaredError()
dev_gap_metric_mae = tf.keras.metrics.MeanAbsoluteError()
dev_gap_metric_mse = tf.keras.metrics.MeanSquaredError()
test_gap_metric_mae = tf.keras.metrics.MeanAbsoluteError()
test_gap_metric_mse = tf.keras.metrics.MeanSquaredError()


types_metric = tf.keras.metrics.SparseCategoricalAccuracy()
#dev_types_metric = tf.keras.metrics.SparseCategoricalAccuracy()
#test_types_metric = tf.keras.metrics.SparseCategoricalAccuracy()

ETH = 10.0
one_by = tf.math.reciprocal_no_nan
flatten = lambda l: [item for sublist in l for item in sublist]


#####################################################
# 				Loss Functions						#
#####################################################


class NegativeLogLikelihood(tf.keras.losses.Loss):
	def __init__(self, D, WT,
				 reduction=keras.losses.Reduction.AUTO,
				 name='negative_log_likelihood'):
		super(NegativeLogLikelihood, self).__init__(reduction=reduction,
													name=name)
		self.D = D
		self.WT = WT

	def call(self, gaps_true, gaps_pred):
		log_lambda_ = (self.D + (gaps_true * self.WT))
		lambda_ = tf.exp(tf.minimum(ETH, log_lambda_), name='lambda_')
		log_f_star = (log_lambda_
					  + one_by(self.WT) * tf.exp(tf.minimum(ETH, self.D))
					  - one_by(self.WT) * lambda_)
		return -log_f_star

class MeanSquareLoss(tf.keras.losses.Loss):
	def __init__(self,
				 reduction=keras.losses.Reduction.AUTO,
				 name='mean_square_likelihood'):
		super(MeanSquareLoss, self).__init__(reduction=reduction,
													name=name)
	
	def call(self, gaps_true, gaps_pred):
		error = gaps_true - gaps_pred
		return tf.reduce_mean(error * error)

class Gaussian_MSE(tf.keras.losses.Loss):
	def __init__(self, D, WT,
				 reduction=keras.losses.Reduction.AUTO,
				 name='mean_square_guassian'):
		super(Gaussian_MSE, self).__init__(reduction=reduction,
													name=name)
		self.out_mean = D
		self.out_stddev = WT
	
	def call(self, gaps_true, gaps_pred):
		gaussian_distribution = tfp.distributions.Normal(
			self.out_mean, self.out_stddev, validate_args=False, allow_nan_stats=True,
			name='Normal'
		)
		loss = -tf.reduce_mean(gaussian_distribution.log_prob(gaps_true))
		return loss

#####################################################
# 				Run Models Function					#
#####################################################

# RMTPP_VAR model
# Trains a separate model to predict the variance of the RMTPP predictions.
# Predicted variance is function of distance between predicted last encoder input
# and predicted timestamp. The model is trained to maximize the likelihood of 
# the observed gaps.
def run_rmtpp_var(args, data, test_data, trained_rmtpp_model):
	model = models.RMTPP_VAR(args.hidden_layer_size)

	[train_dataset_gaps, nc_event_dev_in_gaps, nc_event_dev_out_gaps, train_norm_gaps] = data
	[event_train_norma, event_train_normd] = train_norm_gaps
	model_name = args.current_model
	optimizer = keras.optimizers.Adam(args.learning_rate)

	enc_len = args.enc_len

	dev_data_gaps = nc_event_dev_in_gaps

	os.makedirs('saved_models/training_'+model_name+'_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_"+model_name+"_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mse = np.inf
	best_dev_epoch = 0

	train_losses = list()
	for epoch in range(args.epochs):
		print('Starting epoch', epoch)
		step_train_loss = 0.0
		step_cnt = 0
		next_initial_state = None
		for sm_step, (gaps_batch, _) in enumerate(train_dataset_gaps):
			gaps_batch_in = gaps_batch[:, :int(enc_len/2)]
			gaps_batch_out = gaps_batch[:, int(enc_len/2):]
			with tf.GradientTape() as tape:

				# TODO: Pass the correct initial state based on 
				# batch_size and stride etc..
				gaps_pred = simulate_fixed_cnt(trained_rmtpp_model,
											   gaps_batch_in,
											   int(enc_len/2),
											   prev_hidden_state=next_initial_state)
				model_inputs = tf.cumsum(gaps_pred, axis=1)
				var_pred = model(model_inputs)

				# Compute the loss for this minibatch.
				import ipdb
				ipdb.set_trace()
				gap_loss_fn = Gaussian_MSE(gaps_pred, var_pred)
				
				gap_loss = gap_loss_fn(gaps_batch_out, None)
				loss = gap_loss
				step_train_loss+=loss.numpy()
				
			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			train_gap_metric_mae(gaps_batch_out, gaps_pred)
			train_gap_metric_mse(gaps_batch_out, gaps_pred)
			train_gap_mae = train_gap_metric_mae.result()
			train_gap_mse = train_gap_metric_mse.result()
			train_gap_metric_mae.reset_states()
			train_gap_metric_mse.reset_states()

			# print(float(train_gap_mae), float(train_gap_mse))
			print('Training loss (for one batch) at step %s: %s' %(sm_step, float(loss)))
			step_cnt += 1
		
		# Dev calculations
		nc_event_dev_in_gaps = dev_data_gaps[:, :int(enc_len/2)]
		nc_event_dev_out_gaps = dev_data_gaps[:, int(enc_len/2):]
		dev_gaps_pred = simulate_fixed_cnt(trained_rmtpp_model,
										   nc_event_dev_in_gaps,
										   int(enc_len)/2,
										   prev_hidden_state=next)
		model_dev_inputs = tf.cumsum(dev_gaps_pred, axis=1)
		dev_var_pred = model(model_dev_inputs)
		dev_loss_fn = Gaussian_MSE(dev_gaps_pred, dev_var_pred)
		dev_loss = dev_loss_fn(nc_event_dev_out_gaps, None)
		dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred,
										event_train_norma, event_train_normd)
		
		dev_gap_metric_mae(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_metric_mse(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_mae = dev_gap_metric_mae.result()
		dev_gap_mse = dev_gap_metric_mse.result()
		dev_gap_metric_mae.reset_states()
		dev_gap_metric_mse.reset_states()
		if best_dev_gap_mse > dev_gap_mse:
			best_dev_gap_mse = dev_gap_mse
			best_dev_epoch = epoch
			print('Saving model at epoch', epoch)
			model.save_weights(checkpoint_path)

		step_train_loss /= step_cnt
		print('Training loss after epoch %s: %s' %(epoch, float(step_train_loss)))
		print('MAE and MSE of Dev data %s: %s' \
			%(float(dev_gap_mae), float(dev_gap_mse)))
		train_losses.append(step_train_loss)

	plt.plot(range(len(train_losses)), train_losses)
	plt.savefig(os.path.join(args.output_dir, 'train_'+args.current_model+'_'+args.current_dataset+'_loss.png'))
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)

	dev_gaps_pred, _, _, _, _ = model(nc_event_dev_in_gaps)
	dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
									event_train_norma, event_train_normd)
	
	dev_gap_metric_mae(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_metric_mse(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	print('Best MAE and MSE of Dev data %s: %s' \
		%(float(dev_gap_mae), float(dev_gap_mse)))
		
	return model, None
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# RMTPP model
def run_rmtpp(args, model, optimizer, data, var_data, NLL_loss,
			  rmtpp_epochs=10, use_var_model=False, comp_model=False,
			  rmtpp_type=None):
	[train_dataset_gaps,
	 nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types,
	 nc_event_dev_out_gaps, nc_event_dev_out_types,
	 train_norm_gaps] = data
	[event_train_norma, event_train_normd] = train_norm_gaps
	#[var_dataset_gaps, train_end_hr_bins_relative,
	# train_data_in_time_end_bin,
	# train_gap_in_bin_norm_a, train_gap_in_bin_norm_d] = var_data
	model_name = args.current_model

	os.makedirs('saved_models/training_'+model_name+'_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_"+model_name+"_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mse = np.inf
	best_dev_epoch = 0
	enc_len = args.enc_len
	comp_enc_len = args.comp_enc_len
	batch_size = args.batch_size
	stride_move = batch_size
	dataset_name = args.current_dataset

	if not comp_model:
		if dataset_name in ['taxi', '911_traffic', '911_ems']:
			stride_move = batch_size * args.stride_len
			if stride_move > enc_len:
				print("Training considering independent sequence")
				stride_move = 0
				
	if comp_model:
		if stride_move > comp_enc_len:
			print("Training considering independent sequence")
			stride_move = 0

	if args.extra_var_model and rmtpp_type=='mse':
		hls = args.hidden_layer_size
		num_grps = args.num_grps #TODO make it command line argument later
		num_pos = args.num_pos
		rmtpp_var_model = models.RMTPP_VAR(hls,
										   args.out_bin_sz, hls,
										   num_grps, hls,
										   num_pos, hls)
		var_optimizer = keras.optimizers.Adam(args.learning_rate)

		os.makedirs('saved_models/training_rmtpp_var_model_'+args.current_dataset+'/', exist_ok=True)
		var_checkpoint_path = "saved_models/training_rmtpp_var_model_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"

		best_dev_var_loss = np.inf
		best_dev_var_epoch = 0
		best_var_loss = np.inf
		best_var_epoch = 0
	else:
		rmtpp_var_model = None

	train_losses = list()
	for epoch in range(args.epochs):
		print('Starting epoch', epoch)
		var_epoch_loss = 0.
		step_train_loss = 0.0
		step_cnt = 0
		next_initial_state = None
		st = time.time()
		for sm_step, (gaps_batch_in, feats_batch_in, types_batch_in,
					  gaps_batch_out, feats_batch_out, types_batch_out) \
						in enumerate(train_dataset_gaps):
			with tf.GradientTape() as tape:
				gaps_pred, types_logits_pred, D, WT, _ = model(
					gaps_batch_in,
					feats_batch_in,
					types_batch_in)

				# Compute the loss for this minibatch.
				if use_var_model:
					gap_loss_fn = Gaussian_MSE(D, WT)
				elif NLL_loss:
					gap_loss_fn = NegativeLogLikelihood(D, WT)
				else:
					gap_loss_fn = MeanSquareLoss()

				type_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
				
				gap_loss = gap_loss_fn(gaps_batch_out, gaps_pred)
				type_loss = type_loss_fn(types_batch_out-1, types_logits_pred)
				loss = gap_loss + type_loss
				step_train_loss+=loss.numpy()

			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			train_gap_metric_mae(gaps_batch_out, gaps_pred)
			train_gap_metric_mse(gaps_batch_out, gaps_pred)
			train_gap_mae = train_gap_metric_mae.result()
			train_gap_mse = train_gap_metric_mse.result()
			train_gap_metric_mae.reset_states()
			train_gap_metric_mse.reset_states()

			# print(float(train_gap_mae), float(train_gap_mse))
			print('Training loss (for one batch) at step %s: %s, %s, %s' %(sm_step, float(loss), float(gap_loss), float(type_loss)))
			step_cnt += 1
		et = time.time()
		print(model_name, 'time_reqd:', et-st)
		
		# Dev calculations
		dev_gaps_pred, dev_logits_pred, _, _, _ = model(nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types)
		dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
													 event_train_norma,
													 event_train_normd)
		
		dev_gap_metric_mae(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_metric_mse(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_mae = dev_gap_metric_mae.result()
		dev_gap_mse = dev_gap_metric_mse.result()
		dev_gap_metric_mae.reset_states()
		dev_gap_metric_mse.reset_states()

		dev_types_acc = types_metric(nc_event_dev_out_types-1, dev_logits_pred)
		types_metric.reset_states()

		if best_dev_gap_mse > dev_gap_mse:
			best_dev_gap_mse = dev_gap_mse
			best_dev_epoch = epoch
			print('Saving model at epoch', epoch)
			model.save_weights(checkpoint_path)

		step_train_loss /= step_cnt
		print('Training loss after epoch %s: %s' %(epoch, float(step_train_loss)))
		print('MAE and MSE of Dev data %s: %s' \
			%(float(dev_gap_mae), float(dev_gap_mse)))
		train_losses.append(step_train_loss)

	if args.extra_var_model and rmtpp_type=='mse':# and epoch%5==0:
		var_gaps_pred_lst, bin_ids_lst, grp_ids_lst, pos_ids_lst = [], [], [], []
		for epoch in range(args.epochs+5):
			var_epoch_loss = 0.
			for sm_step, (var_gaps_batch_in, var_gaps_batch_out,
					var_bch_in_time_end_bin, var_bch_end_hr_bins) \
				in enumerate(var_dataset_gaps):
				with tf.GradientTape() as var_tape:
					if epoch==0:
						var_gaps_pred, _, bin_ids, grp_ids, pos_ids \
							= simulate_v2(model,
										var_bch_in_time_end_bin,
										var_gaps_batch_in,
										var_bch_end_hr_bins.numpy(),
										(train_gap_in_bin_norm_a,
										train_gap_in_bin_norm_d),
										(args.out_bin_sz, num_grps, num_pos),
										var_gaps_batch_out.numpy(),
										prev_hidden_state=None)
						var_gaps_pred_lst.append(var_gaps_pred)
						bin_ids_lst.append(bin_ids)
						grp_ids_lst.append(grp_ids)
						pos_ids_lst.append(pos_ids)
					else:
						var_gaps_pred = var_gaps_pred_lst[sm_step]
						bin_ids = bin_ids_lst[sm_step]
						grp_ids = grp_ids_lst[sm_step]
						pos_ids = pos_ids_lst[sm_step]
					var_model_inputs = tf.cumsum(var_gaps_pred, axis=1)
					var_pred = rmtpp_var_model(var_model_inputs,
											   bin_ids, grp_ids, pos_ids)
					var_pred = tf.squeeze(var_pred, axis=-1)
					var_loss_fn = Gaussian_MSE(var_gaps_pred, var_pred)
					var_loss = var_loss_fn(var_gaps_batch_out, None)
					var_epoch_loss+=var_loss.numpy()
	
				var_grads = var_tape.gradient(var_loss, rmtpp_var_model.trainable_weights)
				var_optimizer.apply_gradients(zip(var_grads, rmtpp_var_model.trainable_weights))
				print('Var model Training loss (for one batch) at step %s: %s' \
					%(sm_step, float(var_loss)))

			if best_var_loss > var_epoch_loss:
				best_var_loss = var_epoch_loss
				best_var_epoch = epoch
				print('Saving rmtpp_var_model at epoch', epoch)
				rmtpp_var_model.save_weights(var_checkpoint_path)


	plt.plot(range(len(train_losses)), train_losses)
	plt.savefig(os.path.join(args.output_dir, 'train_'+model_name+'_'+args.current_dataset+'_loss.png'))
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	dev_gaps_pred, dev_logits_pred, _, _, _ = model(nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types)
	dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
												 event_train_norma,
												 event_train_normd)

	dev_gap_metric_mae(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_metric_mse(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	dev_types_acc = types_metric(nc_event_dev_out_types-1, dev_logits_pred)
	types_metric.reset_states()
	print('Best MAE, MSE, type_acc of Dev data: %s, %s, %s' \
		%(float(dev_gap_mae), float(dev_gap_mse), float(dev_types_acc)))

	if args.extra_var_model and rmtpp_type=='mse':	
		print("Loading best rmtpp_var_model from epoch", best_var_epoch)
		rmtpp_var_model.load_weights(var_checkpoint_path)

	return train_losses, rmtpp_var_model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# RMTPP run initialize with loss function for run_rmtpp
def run_rmtpp_init(args, data, test_data, var_data,
				   NLL_loss=False, use_var_model=False, rmtpp_type='mse'):
	[event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_out_binend, event_test_in_lasttime, 
	 event_test_norma, event_test_normd] = test_data
	rmtpp_epochs = args.epochs
	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	num_types = args.num_types

	use_intensity = True
	if not NLL_loss:
		use_intensity = False
	model, optimizer = models.build_rmtpp_model(args, use_intensity, use_var_model, num_types)
	#model.summary()
	if use_var_model:
		print('\nTraining Model with MSE Loss with Variance')
	elif NLL_loss:
		print('\nTraining Model with NLL Loss')
	else:
		print('\nTraining Model with Mean Square Loss')
	train_loss, rmtpp_var_model = run_rmtpp(args, model, optimizer,
											data, var_data, NLL_loss=NLL_loss, 
											rmtpp_epochs=rmtpp_epochs,
											use_var_model=use_var_model,
											rmtpp_type=rmtpp_type)

	return model, rmtpp_var_model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Pure Hierarchical model
def run_pure_hierarchical(args, model, optimizer, data, NLL_loss, rmtpp_epochs=10):
	[train_dataset_gaps, nc_event_dev_in_gaps, nc_event_dev_out_gaps, train_norm_gaps] = data
	[event_train_norma, event_train_normd] = train_norm_gaps
	model_name = args.current_model

	os.makedirs('saved_models/training_'+model_name+'_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_"+model_name+"_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mse = np.inf
	best_dev_epoch = 0
	enc_len = args.enc_len
	comp_enc_len = args.comp_enc_len
	batch_size = args.batch_size
	dataset_name = args.current_dataset

	train_losses = list()
	for epoch in range(args.epochs):
		print('Starting epoch', epoch)
		step_train_loss = 0.0
		step_cnt = 0
		next_initial_state = None
		for sm_step, (gaps_batch_in, gaps_batch_out) in enumerate(train_dataset_gaps):
			with tf.GradientTape() as tape:
				# TODO: Make sure to pass correct next_stat
				gaps_batch_out = tf.cast(gaps_batch_out, tf.float32)
				gaps_pred_l2, D_l2, WT_l2, gaps_pred, D_l1, WT_l1, next_initial_state, _ = model(gaps_batch_in, 
						initial_state=None, 
						gaps_out=gaps_batch_out,
						next_state_sno=1)

				# print(gaps_pred[0,0,:5])
				# print(gaps_pred_l2[0,:5])

				# Compute the loss for this minibatch.
				if NLL_loss:
					gap_loss_fn = NegativeLogLikelihood(D_l1, WT_l1)
					gap_loss_fn_l2 = NegativeLogLikelihood(D_l2, WT_l2)
				else:
					gap_loss_fn = MeanSquareLoss()
					gap_loss_fn_l2 = MeanSquareLoss()

				gaps_batch_out_l2 = tf.reduce_sum(gaps_batch_out, axis=2)
				gap_loss = gap_loss_fn(gaps_batch_out, gaps_pred)
				gap_loss_l2 = gap_loss_fn_l2(gaps_batch_out_l2, gaps_pred_l2)
				loss = gap_loss + gap_loss_l2
				step_train_loss+=loss.numpy()

				#TODO: Pred for l2 is same as l1?


			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			train_gap_metric_mae(gaps_batch_out, gaps_pred)
			train_gap_metric_mse(gaps_batch_out, gaps_pred)
			train_gap_mae = train_gap_metric_mae.result()
			train_gap_mse = train_gap_metric_mse.result()
			train_gap_metric_mae.reset_states()
			train_gap_metric_mse.reset_states()

			# print(float(train_gap_mae), float(train_gap_mse))
			print('Training loss (for one batch) at step %s: %s' %(sm_step, float(loss)))
			step_cnt += 1
		
		# Dev calculations
		dev_gaps_pred_l2, _,_, dev_gaps_pred, _,_,_,_ = model(nc_event_dev_in_gaps)
		dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
										event_train_norma, event_train_normd)
		
		dev_gap_metric_mae(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_metric_mse(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_mae = dev_gap_metric_mae.result()
		dev_gap_mse = dev_gap_metric_mse.result()
		dev_gap_metric_mae.reset_states()
		dev_gap_metric_mse.reset_states()

		if best_dev_gap_mse > dev_gap_mse:
			best_dev_gap_mse = dev_gap_mse
			best_dev_epoch = epoch
			print('Saving model at epoch', epoch)
			model.save_weights(checkpoint_path)

		step_train_loss /= step_cnt
		print('Training loss after epoch %s: %s' %(epoch, float(step_train_loss)))
		print('MAE and MSE of Dev data %s: %s' \
			%(float(dev_gap_mae), float(dev_gap_mse)))
		train_losses.append(step_train_loss)

	plt.plot(range(len(train_losses)), train_losses)
	plt.savefig(os.path.join(args.output_dir, 'train_'+model_name+'_'+args.current_dataset+'_loss.png'))
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	dev_gaps_pred_l2, _,_, dev_gaps_pred, _,_,_,_ = model(nc_event_dev_in_gaps)
	dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
									event_train_norma, event_train_normd)
	
	dev_gap_metric_mae(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_metric_mse(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	print('Best MAE and MSE of Dev data %s: %s' \
		%(float(dev_gap_mae), float(dev_gap_mse)))

	return train_losses
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Pure RMTPP Hierarchical run initialize with loss function for run_rmtpp
def run_pure_hierarchical_init(args, data, test_data,
							   NLL_loss=False):

	#TODO:
	#	1. Add rmtpp_type
	#	2. use_var_model?
	#

	[event_test_in_gaps, event_test_in_feats,
	 count_test_out_binend, event_test_in_lasttime, 
	 event_test_norma, event_test_normd] =  test_data
	rmtpp_epochs = args.epochs
	
	use_intensity = True
	if not NLL_loss:
		use_intensity = False
	model, optimizer = models.build_pure_hierarchical_model(args, use_intensity)
	# model.summary()
	if NLL_loss:
		print('\nTraining Model with NLL Loss')
	else:
		print('\nTraining Model with Mean Square Loss')
	train_loss = run_pure_hierarchical(args, model, optimizer,
									   data, NLL_loss=NLL_loss, 
									   rmtpp_epochs=rmtpp_epochs)
	return model, None
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# RMTPP Comp run initialize with loss function for run_rmtpp with compound layer
def run_rmtpp_comp_init(args, data, test_data, var_data,
						NLL_loss=False, use_var_model=False, rmtpp_type='mse'):
	[event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_out_binend, event_test_in_lasttime, 
	 event_test_norma, event_test_normd] =  test_data
	rmtpp_epochs = args.epochs
	num_types = 1 # num_types is always 1 for compound model
	
	use_intensity = True
	if not NLL_loss:
		use_intensity = False
	model, optimizer = models.build_rmtpp_model(args, use_intensity, use_var_model, num_types)
	#model.summary()
	if use_var_model:
		print('\nTraining Model with MSE Loss with Variance')
	elif NLL_loss:
		print('\nTraining Model with NLL Loss')
	else:
		print('\nTraining Model with Mean Square Loss')
	train_loss, rmtpp_var_model = run_rmtpp(args, model, optimizer,
											data, var_data, NLL_loss=NLL_loss, 
											rmtpp_epochs=rmtpp_epochs,
											use_var_model=use_var_model,
											rmtpp_type=rmtpp_type,
											comp_model=True)
	return model, rmtpp_var_model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Count Model with bin count from plain FF network
def run_hierarchical(args, data, test_data):
	validation_split = 0.2
	num_epochs = args.epochs * 100

	count_train_in_counts, count_train_out_counts = data
	(count_test_in_counts, count_test_in_feats, count_test_out_counts,
	 count_test_normm, count_test_norms) = test_data
	batch_size = args.batch_size

	model_name = args.current_model
	os.makedirs('saved_models/training_'+model_name+'_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_"+model_name+"_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"

	model_cnt = models.hierarchical_model(args)
	model_cnt.summary()

	if num_epochs > 0:
		history_cnt = model_cnt.fit(count_train_in_counts, count_train_out_counts, batch_size=batch_size,
						epochs=num_epochs, validation_split=validation_split, verbose=0)
		model_cnt.save_weights(checkpoint_path)
		hist = pd.DataFrame(history_cnt.history)
		hist['epoch'] = history_cnt.epoch
		print(hist)
	else:
		model_cnt.load_weights(checkpoint_path)

	test_data_out_norm = utils.normalize_data_given_param(count_test_out_counts, count_test_normm, count_test_norms)
	loss, mae, mse = model_cnt.evaluate(count_test_in_counts, test_data_out_norm, verbose=0)
	print('Normalized loss, mae, mse', loss, mae, mse)

	test_predictions_norm_cnt = model_cnt.predict(count_test_in_counts)
	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, 
											count_test_normm, count_test_norms)
	event_count_preds_cnt = test_predictions_cnt
	return model_cnt, event_count_preds_cnt
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Count model with NB or Gaussian distribution
def run_count_model(args, data, test_data):
	validation_split = 0.2
	num_epochs = args.epochs * 100
	patience = args.patience * 0
	distribution_name = 'Gaussian'
	#distribution_name = 'var_model'

	(count_train_in_counts, count_train_in_feats, count_train_out_counts,
	 count_dev_in_counts, count_dev_in_feats, count_dev_out_counts) = data
	(count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_normm, count_test_norms) = test_data

	dataset_size = len(count_train_in_counts)
	train_data_size = dataset_size - round(validation_split*dataset_size)

	count_train_in_counts = count_train_in_counts.astype(np.float32)
	count_train_out_counts = count_train_out_counts.astype(np.float32)
	count_test_in_counts = count_test_in_counts.astype(np.float32)
	count_test_out_counts = count_test_out_counts.astype(np.float32)

	#dev_data_in_bin = count_train_in_counts[train_data_size:]
	#dev_data_in_bin_feats = count_train_in_feats[train_data_size:]
	#dev_data_out_bin = count_train_out_counts[train_data_size:]
	#count_train_in_counts = count_train_in_counts[:train_data_size]
	#count_train_out_counts = count_train_out_counts[:train_data_size]
	#count_train_in_feats = count_train_in_feats[:train_data_size]

	batch_size = args.batch_size
	train_dataset = tf.data.Dataset.from_tensor_slices(
		(
			count_train_in_counts,
			count_train_in_feats,
			count_train_out_counts
		)
	).batch(batch_size, drop_remainder=True)

	model, optimizer = models.build_count_model(args, distribution_name)
	#model.summary()

	os.makedirs('saved_models/training_count_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_count_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mae = np.inf
	best_dev_epoch = 0

	train_losses = list()
	stddev_sample = list()
	dev_mae_list = list()
	for epoch in range(num_epochs):
		#print('Starting epoch', epoch)
		step_train_loss = 0.0
		step_cnt = 0
		for sm_step, (bin_count_batch_in, bin_count_batch_in_feats,
						bin_count_batch_out) in enumerate(train_dataset):
			with tf.GradientTape() as tape:
				#import ipdb
				#ipdb.set_trace()
				bin_counts_pred, distribution_params = model(
					bin_count_batch_in,
					bin_count_batch_in_feats,
					bin_count_batch_out
				)

				loss_fn = models.NegativeLogLikelihood_CountModel(distribution_params, distribution_name)
				loss = loss_fn(bin_count_batch_out, bin_counts_pred)

				step_train_loss+=loss.numpy()
				
				grads = tape.gradient(loss, model.trainable_weights)
				optimizer.apply_gradients(zip(grads, model.trainable_weights))

			# print('Training loss (for one batch) at step %s: %s' %(sm_step, float(loss)))
			step_cnt += 1
		
		# Dev calculations
		count_dev_pred, _ = model(count_dev_in_counts, count_dev_in_feats)
		count_dev_pred_unnorm = utils.denormalize_data(count_dev_pred, count_test_normm, count_test_norms)
		dev_gap_metric_mae(count_dev_pred_unnorm, count_dev_out_counts)
		dev_gap_metric_mse(count_dev_pred_unnorm, count_dev_out_counts)
		dev_gap_mae = dev_gap_metric_mae.result()
		dev_gap_mse = dev_gap_metric_mse.result()
		dev_gap_metric_mae.reset_states()
		dev_gap_metric_mse.reset_states()

		if best_dev_gap_mae > dev_gap_mae and patience <= epoch:
			best_dev_gap_mae = dev_gap_mae
			best_dev_epoch = epoch
			print('Saving model at epoch', epoch)
			model.save_weights(checkpoint_path)

		step_train_loss /= step_cnt
		print('Training loss after epoch %s: %s' %(epoch, float(step_train_loss)))
		print('MAE and MSE of Dev data %s: %s' \
			%(float(dev_gap_mae), float(dev_gap_mse)))
		train_losses.append(step_train_loss)

		dev_mae_list.append(dev_gap_mae)
		if epoch%10 == 0:
			_, [_, test_distribution_stddev] = model(count_test_in_counts, count_test_in_feats)
			stddev_sample.append(test_distribution_stddev[0])

	stddev_sample = np.array(stddev_sample)

	if num_epochs>0:
		for dec_idx in range(args.out_bin_sz):
			plt.plot(range(len(stddev_sample)), stddev_sample[:,dec_idx], label='dec_idx_'+str(dec_idx))

	plt.legend(loc='upper right')
	plt.savefig(os.path.join(
		args.output_dir,
		'count_model_test_var_'+distribution_name+'_'+args.current_dataset+'_test_id_0.png')
	)
	plt.close()

	plt.plot(range(len(train_losses)), train_losses)
	plt.savefig(os.path.join(
		args.output_dir,
		'count_model_train_loss_'+distribution_name+'_'+args.current_dataset+'.png'))
	plt.close()

	plt.plot(range(len(dev_mae_list)), dev_mae_list)
	plt.savefig(os.path.join(
		args.output_dir,
		'count_model_dev_mae_'+distribution_name+'_'+args.current_dataset+'.png'))
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	count_dev_pred, _ = model(count_dev_in_counts, count_dev_in_feats)		
	count_dev_pred_unnorm = utils.denormalize_data(count_dev_pred, count_test_normm, count_test_norms)
	dev_gap_metric_mae(count_dev_pred_unnorm, count_dev_out_counts)
	dev_gap_metric_mse(count_dev_pred_unnorm, count_dev_out_counts)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	print('MAE and MSE of Dev data %s: %s' \
		%(float(dev_gap_mae), float(dev_gap_mse)))

	test_bin_count_pred_norm, test_distribution_params = model(count_test_in_counts, count_test_in_feats)		
	test_bin_count_pred = utils.denormalize_data(test_bin_count_pred_norm, count_test_normm, count_test_norms)
	test_distribution_params[1] = utils.denormalize_data_stddev(test_distribution_params[1], count_test_normm, count_test_norms)
	return (
		model,
		{
			"count_all_means_pred": test_bin_count_pred.numpy(),
			"count_all_sigms_pred": test_distribution_params[1]
		}
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# WGAN Model
def run_wgan(args, data, test_data):
	'''
		Return a trained wgan model
		create HomogenouosPoisson sequences
			- length of sequence: span the forecast horizon
			- intensity: Intensity of the input sequence
			Divide training data into input and output
			Write expressions for D_loss and G_loss
			Write training loops for the model
			Add # WGAN Lipschitz constraint
	'''
	LAMBDA_LP = 0.1 # Penality for Lipschtiz divergence

	model = models.WGAN(g_state_size=args.hidden_layer_size,
						d_state_size=args.hidden_layer_size)

	G_optimizer = keras.optimizers.Adam(args.learning_rate)
	D_optimizer = keras.optimizers.Adam(args.learning_rate)
	pre_train_optimizer = keras.optimizers.Adam(args.learning_rate)

	[train_dataset_gaps,
	 nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types,
	 nc_event_dev_out_gaps, nc_event_dev_out_types,
	 train_norm_gaps] = data
	[event_train_norma, event_train_normd] = train_norm_gaps

	[event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_out_binend, event_test_in_lasttime, 
	 event_test_norma, event_test_normd] = test_data	
	enc_len = args.enc_len
	wgan_enc_len = args.wgan_enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size


	# Generating prior fake sequence in the range of forecast horizon
	# lambda0 = np.mean([len(item) for item in real_sequences])/T
	# intensityPoisson = IntensityHomogenuosPoisson(lambda0)
	# fake_sequences = generate_sample(intensityPoisson, T, 20000)
	train_z_seqs = list()
	for (gaps_batch, _, _, _, _, _) in train_dataset_gaps:
		wgan_dec_len = gaps_batch.shape[1] - wgan_enc_len
		times_batch = tf.cumsum(gaps_batch, axis=1)
		span_batch = times_batch[:, -1] - times_batch[:, 0]
		lambda0 = np.ones_like(span_batch) * gaps_batch.shape[1] / span_batch.numpy()
		intensityPoisson = IntensityHomogenuosPoisson(lambda0)
		output_span_batch = times_batch[:, -1] - times_batch[:, wgan_dec_len]
		train_z_seqs_batch = generate_sample(intensityPoisson, wgan_dec_len, lambda0.shape[0])
		train_z_seqs += train_z_seqs_batch
	train_z_seqs = tf.convert_to_tensor(train_z_seqs)

	dev_z_seqs = list()
	dev_data_gaps = nc_event_dev_in_gaps
	dev_data_feats = nc_event_dev_in_feats
	wgan_dec_len = dev_data_gaps.shape[1] - wgan_enc_len
	dev_data_times = tf.cumsum(dev_data_gaps, axis=1)
	dev_span = dev_data_times[:, wgan_enc_len-1] - dev_data_times[:, 0]
	lambda0 = np.ones_like(dev_span) * dev_data_gaps.shape[1] / dev_span.numpy()
	intensityPoisson = IntensityHomogenuosPoisson(lambda0)
	dev_z_seqs = generate_sample(intensityPoisson, wgan_dec_len, lambda0.shape[0])
	dev_z_seqs = tf.convert_to_tensor(dev_z_seqs)
	#dev_z_seqs = utils.normalize_avg_given_param(dev_z_seqs,
	#										event_train_norma,
	#										event_train_normd)

	os.makedirs('saved_models/training_wgan_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_wgan_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mse = np.inf
	best_dev_gap_mae = np.inf
	best_dev_epoch = 0

	# pre-train wgan model
	pre_train_losses = list()
	print('pre-training started')
	bch_cnt = 0
	#for sm_step, (gaps_batch, _) in enumerate(train_dataset_gaps):
	#	gaps_batch_in = gaps_batch[:, :wgan_enc_len]
	#	gaps_batch_out = gaps_batch[:, wgan_enc_len:]
	#	train_z_seqs_batch = train_z_seqs[sm_step*args.batch_size:(sm_step+1)*args.batch_size]
	#	with tf.GradientTape() as pre_train_tape:
	#		gaps_pred = model.generator(train_z_seqs_batch, gaps_batch_in)

	#		bch_pre_train_loss = tf.reduce_mean(tf.abs(gaps_pred - gaps_batch_out))
	#                    
	#		pre_train_losses.append(bch_pre_train_loss.numpy())
	#		
	#	pre_train_grads = pre_train_tape.gradient(bch_pre_train_loss, model.trainable_weights)
	#	pre_train_optimizer.apply_gradients(zip(pre_train_grads, model.trainable_weights))

	#	bch_cnt += 1
	#print('pre-training done, losses=', pre_train_losses)
	## pre-train done


	if args.use_wgan_d:
		print(' Training with discriminator')
	else:
		print(' Training without discriminator')

	train_losses = list()
	for epoch in range(args.epochs):
		print('Starting epoch', epoch)
		step_train_loss = 0.0
		step_cnt = 0
		next_initial_state = None
		for sm_step, (gaps_batch, feats_batch, _, _, _, _) \
				in enumerate(train_dataset_gaps):
			gaps_batch_in = gaps_batch[:, :wgan_enc_len]
			feats_batch_in = feats_batch[:, :wgan_enc_len]
			gaps_batch_out = gaps_batch[:, wgan_enc_len:]
			feats_batch_out = feats_batch[:, wgan_enc_len:]
			train_z_seqs_batch = train_z_seqs[sm_step*args.batch_size:(sm_step+1)*args.batch_size]
			with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
				gaps_pred = model.generator(train_z_seqs_batch,
											enc_inputs=gaps_batch_in,
											enc_feats=feats_batch_in)

				if args.use_wgan_d:
					D_pred = model.discriminator(gaps_batch_in, gaps_pred)
					D_true = model.discriminator(gaps_batch_in, gaps_batch_out)

					D_loss = tf.reduce_mean(D_pred) - tf.reduce_mean(D_true)
					G_loss = -tf.reduce_mean(D_pred)
	
					# Adding Lipschitz Constraint
					length_ = tf.minimum(tf.shape(gaps_batch_out)[1],tf.shape(gaps_pred)[1])
					lipschtiz_divergence = tf.abs(D_true-D_pred)/tf.sqrt(tf.reduce_sum(tf.square(gaps_batch_out[:,:length_,:]-gaps_pred[:,:length_,:]), axis=[1,2])+0.00001)
					lipschtiz_divergence = tf.reduce_mean((lipschtiz_divergence-1)**2)
					D_loss += LAMBDA_LP*lipschtiz_divergence
				else:
					length_ = tf.minimum(tf.shape(gaps_batch_out)[1],tf.shape(gaps_pred)[1])
					gaps_pred_cumsum = tf.cumsum(gaps_pred, axis=1)
					gaps_batch_out_cumsum = tf.cumsum(gaps_batch_out, axis=1)
					G_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(gaps_batch_out_cumsum[:,:length_,:]-gaps_pred_cumsum[:,:length_,:]), axis=[1,2])+0.00001))

				step_train_loss += G_loss.numpy()
				
			G_grads = g_tape.gradient(G_loss, model.trainable_weights)
			G_optimizer.apply_gradients(zip(G_grads, model.trainable_weights))
			if args.use_wgan_d:
				D_grads = d_tape.gradient(D_loss, model.trainable_weights)
				D_optimizer.apply_gradients(zip(D_grads, model.trainable_weights))

			train_gap_metric_mae(gaps_batch_out, gaps_pred)
			train_gap_metric_mse(gaps_batch_out, gaps_pred)
			train_gap_mae = train_gap_metric_mae.result()
			train_gap_mse = train_gap_metric_mse.result()
			train_gap_metric_mae.reset_states()
			train_gap_metric_mse.reset_states()

			# print(float(train_gap_mae), float(train_gap_mse))
			# print('Training loss (for one batch) at step %s: %s' %(sm_step, float(loss)))
			print('Training loss (for one batch) at step %s: %s' %(sm_step, float(G_loss)))
			step_cnt += 1
		
		# Dev calculations
		nc_event_dev_in_gaps = dev_data_gaps[:, :wgan_enc_len]
		nc_event_dev_in_feats = dev_data_feats[:, :wgan_enc_len]
		nc_event_dev_out_gaps = dev_data_gaps[:, wgan_enc_len:]
		dev_gaps_pred = model.generator(dev_z_seqs,
										enc_inputs=nc_event_dev_in_gaps,
										enc_feats=nc_event_dev_in_feats)
		dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
										event_train_norma, event_train_normd)
		dev_data_out_gaps_unnorm = utils.denormalize_avg(nc_event_dev_out_gaps, 
										event_train_norma, event_train_normd)
		
		dev_gap_metric_mae(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
		dev_gap_metric_mse(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
		dev_gap_mae = dev_gap_metric_mae.result()
		dev_gap_mse = dev_gap_metric_mse.result()
		dev_gap_metric_mae.reset_states()
		dev_gap_metric_mse.reset_states()
		#if best_dev_gap_mse > dev_gap_mse:
		if best_dev_gap_mae > dev_gap_mae:
			best_dev_gap_mae = dev_gap_mae
			best_dev_epoch = epoch
			print('Saving model at epoch', epoch)
			model.save_weights(checkpoint_path)

		step_train_loss /= step_cnt
		print('Training loss after epoch %s: %s' %(epoch, float(step_train_loss)))
		print('MAE and MSE of Dev data %s: %s' \
			%(float(dev_gap_mae), float(dev_gap_mse)))
		train_losses.append(step_train_loss)

	plt.plot(range(len(train_losses)), train_losses)
	plt.savefig(os.path.join(
		args.output_dir,
		'train_wgan_'+args.current_dataset+'_loss.png'))
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	nc_event_dev_in_gaps = dev_data_gaps[:, :wgan_enc_len]
	nc_event_dev_in_feats = dev_data_feats[:, :wgan_enc_len]
	nc_event_dev_out_gaps = dev_data_gaps[:, wgan_enc_len:]
	dev_gaps_pred = model.generator(dev_z_seqs,
									enc_inputs=nc_event_dev_in_gaps,
									enc_feats=nc_event_dev_in_feats)
	dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
									event_train_norma, event_train_normd)
	
	dev_data_out_gaps_unnorm = utils.denormalize_avg(nc_event_dev_out_gaps, 
									event_train_norma, event_train_normd)

	dev_gap_metric_mae(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
	dev_gap_metric_mse(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	print('Best MAE and MSE of Dev data %s: %s' \
		%(float(dev_gap_mae), float(dev_gap_mse)))

	return model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# WGAN Model
def run_seq2seq(args, data, test_data):
	'''
		Return a trained seq2seq model
			Divide training data into input and output
			Write expressions for D_loss and G_loss
			Write training loops for the model
			Add # WGAN Lipschitz constraint
	'''
	LAMBDA_LP = 10. # Penality for Lipschtiz divergence

	model = models.Seq2Seq(g_state_size=args.hidden_layer_size,
						   d_state_size=args.hidden_layer_size)

	G_optimizer = keras.optimizers.Adam(args.learning_rate)
	D_optimizer = keras.optimizers.Adam(args.learning_rate)
	pre_train_optimizer = keras.optimizers.Adam(args.learning_rate)

	[train_dataset_gaps,
	 nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types,
	 nc_event_dev_out_gaps,nc_event_dev_out_types,
	 train_norm_gaps] = data
	[event_train_norma, event_train_normd] = train_norm_gaps

	[event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_out_binend, event_test_in_lasttime, 
	 event_test_norma, event_test_normd] = test_data

	model_name = args.current_model
	enc_len = args.enc_len
	wgan_enc_len = args.wgan_enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size



	dev_data_gaps = nc_event_dev_in_gaps
	dev_data_feats = nc_event_dev_in_feats
	wgan_dec_len = dev_data_gaps.shape[1] - wgan_enc_len
	dev_data_times = tf.cumsum(dev_data_gaps, axis=1)

	os.makedirs('saved_models/training_'+model_name+'_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_"+model_name+"_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mse = np.inf
	best_dev_gap_mae = np.inf
	best_dev_epoch = 0

	# pre-train wgan model
	pre_train_losses = list()
	print('pre-training started')
	bch_cnt = 0

	if args.use_cwe_d:
		print(' Training with discriminator')
	else:
		print(' Training without discriminator')

	train_losses = list()
	for epoch in range(args.epochs):
		print('Starting epoch', epoch)
		step_train_loss = 0.0
		step_cnt = 0
		next_initial_state = None
		for sm_step, (gaps_batch, feats_batch, _, _, _, _) \
				in enumerate(train_dataset_gaps):
			gaps_batch_in = gaps_batch[:, :wgan_enc_len]
			feats_batch_in = feats_batch[:, :wgan_enc_len]
			gaps_batch_out = gaps_batch[:, wgan_enc_len:]
			feats_batch_out = feats_batch[:, wgan_enc_len:]
			with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
				gaps_pred = model.generator(gaps_batch_out,
											feats_batch_out,
											enc_inputs=gaps_batch_in,
											enc_feats=feats_batch_in)

				length_ = tf.minimum(tf.shape(gaps_batch_out)[1],tf.shape(gaps_pred)[1])
				G_mse_loss = tf.reduce_sum(
					tf.square(
						gaps_batch_out[:,:length_,:]-gaps_pred[:,:length_,:]
					)
				)
				if args.use_cwe_d:
					D_pred = model.discriminator(gaps_batch_in, gaps_pred)
					D_true = model.discriminator(gaps_batch_in, gaps_batch_out)

					D_loss = tf.reduce_sum(D_pred) - tf.reduce_sum(D_true)
					G_loss = 10. * G_mse_loss + (-tf.reduce_sum(D_pred))
	
					# Adding Lipschitz Constraint
					length_ = tf.minimum(tf.shape(gaps_batch_out)[1],tf.shape(gaps_pred)[1])
					#lipschtiz_divergence = tf.abs(D_true-D_pred)/tf.sqrt(tf.reduce_sum(tf.square(gaps_batch_out[:,:length_,:]-gaps_pred[:,:length_,:]), axis=[1,2])+0.00001)
					#lipschtiz_divergence = tf.reduce_mean((lipschtiz_divergence-1)**2)
					z = tf.random.uniform([1])
					with tf.GradientTape(watch_accessed_variables=False) as lipsch_tape:
						x_hat_out = z * gaps_batch_out[:,:length_,:] - (1.-z) * gaps_pred[:,:length_,:]
						lipsch_tape.watch(gaps_batch_in)
						lipsch_tape.watch(x_hat_out)
						D_x_hat = model.discriminator(gaps_batch_in, x_hat_out)
					D_x_grads = lipsch_tape.gradient(D_x_hat, model.rcnn_input)
					#lipschtiz_divergence = tf.abs(D_x_hat-1)
					lipschtiz_divergence = tf.abs(tf.reduce_sum(D_x_grads)-1)

					D_loss += LAMBDA_LP*lipschtiz_divergence

				step_train_loss += G_loss.numpy()
				
			G_grads = g_tape.gradient(G_loss, model.trainable_weights)
			G_optimizer.apply_gradients(zip(G_grads, model.trainable_weights))
			if args.use_cwe_d:
				D_grads = d_tape.gradient(D_loss, model.trainable_weights)
				D_optimizer.apply_gradients(zip(D_grads, model.trainable_weights))

			train_gap_metric_mae(gaps_batch_out, gaps_pred)
			train_gap_metric_mse(gaps_batch_out, gaps_pred)
			train_gap_mae = train_gap_metric_mae.result()
			train_gap_mse = train_gap_metric_mse.result()
			train_gap_metric_mae.reset_states()
			train_gap_metric_mse.reset_states()

			# print(float(train_gap_mae), float(train_gap_mse))
			print('Training loss (for one batch) at step %s: %s %s %s' %(sm_step, float(G_mse_loss), float(G_loss), float(D_loss)))
			step_cnt += 1
		
		# Dev calculations
		nc_event_dev_in_gaps = dev_data_gaps[:, :wgan_enc_len]
		nc_event_dev_in_feats = dev_data_feats[:, :wgan_enc_len]
		nc_event_dev_out_gaps = dev_data_gaps[:, wgan_enc_len:]
		dev_data_out_feats = dev_data_feats[:, wgan_enc_len:]
		dev_gaps_pred = model.generator(nc_event_dev_out_gaps,
										dev_data_out_feats,
										enc_inputs=nc_event_dev_in_gaps,
										enc_feats=nc_event_dev_in_feats)
		dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
										event_train_norma, event_train_normd)
		dev_data_out_gaps_unnorm = utils.denormalize_avg(nc_event_dev_out_gaps, 
										event_train_norma, event_train_normd)
		
		dev_gap_metric_mae(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
		dev_gap_metric_mse(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
		dev_gap_mae = dev_gap_metric_mae.result()
		dev_gap_mse = dev_gap_metric_mse.result()
		dev_gap_metric_mae.reset_states()
		dev_gap_metric_mse.reset_states()
		#if best_dev_gap_mse > dev_gap_mse:
		if best_dev_gap_mae > dev_gap_mae:
			best_dev_gap_mae = dev_gap_mae
			best_dev_epoch = epoch
			print('Saving model at epoch', epoch)
			model.save_weights(checkpoint_path)

		step_train_loss /= step_cnt
		print('Training loss after epoch %s: %s' %(epoch, float(step_train_loss)))
		print('MAE and MSE of Dev data %s: %s' \
			%(float(dev_gap_mae), float(dev_gap_mse)))
		train_losses.append(step_train_loss)

	plt.plot(range(len(train_losses)), train_losses)
	plt.savefig(os.path.join(
		args.output_dir,
		'train_wgan_'+args.current_dataset+'_loss.png'))
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	nc_event_dev_in_gaps = dev_data_gaps[:, :wgan_enc_len]
	nc_event_dev_in_feats = dev_data_feats[:, :wgan_enc_len]
	nc_event_dev_out_gaps = dev_data_gaps[:, wgan_enc_len:]
	dev_data_out_feats = dev_data_feats[:, wgan_enc_len:]
	dev_gaps_pred = model.generator(nc_event_dev_out_gaps,
									dev_data_out_feats,
									enc_inputs=nc_event_dev_in_gaps,
									enc_feats=nc_event_dev_in_feats)
	dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
									event_train_norma, event_train_normd)
	
	dev_data_out_gaps_unnorm = utils.denormalize_avg(nc_event_dev_out_gaps, 
									event_train_norma, event_train_normd)

	dev_gap_metric_mae(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
	dev_gap_metric_mse(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	print('Best MAE and MSE of Dev data %s: %s' \
		%(float(dev_gap_mae), float(dev_gap_mse)))


	return model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Transformer Model
def run_transformer(args, data, test_data):

	model = models.Transformer(
		num_types=args.num_types,
		d_model=args.d_model,
		d_rnn=args.d_rnn,
		d_inner=args.d_inner_hid,
		n_layers=args.n_layers,
		n_head=args.n_head,
		d_k=args.d_k,
		d_v=args.d_v,
		dropout=args.dropout,	
	)

	optimizer = keras.optimizers.Adam(args.learning_rate)

	[train_dataset_gaps,
	 nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types,
	 nc_event_dev_out_gaps, nc_event_dev_out_types,
	 train_norm_gaps] = data
	[event_train_norma, event_train_normd] = train_norm_gaps

	[event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_out_binend, event_test_in_lasttime, 
	 event_test_norma, event_test_normd] = test_data

	model_name = args.current_model
	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size

	os.makedirs('saved_models/training_'+model_name+'_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_"+model_name+"_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mse = np.inf
	best_dev_epoch = 0
	enc_len = args.enc_len
	comp_enc_len = args.comp_enc_len
	batch_size = args.batch_size
	stride_move = batch_size
	dataset_name = args.current_dataset


	train_losses = list()
	for epoch in range(args.epochs):
		print('Starting epoch', epoch)
		step_train_loss = 0.0
		step_cnt = 0
		next_initial_state = None
		st = time.time()
		for sm_step, (gaps_batch_in, feats_batch_in, types_batch_in,
					  gaps_batch_out, feats_batch_out, types_batch_out) \
						in enumerate(train_dataset_gaps):
			with tf.GradientTape() as tape:

				enc_out, (types_pred, gaps_pred) = model(
					gaps_batch_in, 
					feats_batch_in,
					types_batch_in)


				# Compute the loss for this minibatch.
				#TODO: type_loss_func not correctly mapped from torch to tf
				if args.smooth > 0:
					assert False, "LabelSmoothingLoss not well tested"
					type_loss_func = transformer_utils.LabelSmoothingLoss(args.smooth, args.num_types, ignore_index=-1)
				else:
					#type_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
					type_loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
						from_logits=True,
						reduction=tf.keras.losses.Reduction.NONE
					)

				types_batch_out = tf.cast(types_batch_out, tf.float32)
				event_ll, non_event_ll = transformer_utils.log_likelihood(
					model, enc_out,
					tf.squeeze(gaps_batch_out, axis=-1),
					types_batch_out)
				#gap_loss = -torch.sum(event_ll - non_event_ll)
				se = transformer_utils.time_loss(gaps_pred, gaps_batch_out)
				scale_se_loss = 10 # SE is usually large, scale it to stabilize training
				gap_loss = -tf.reduce_sum(event_ll - non_event_ll) + se / scale_se_loss
				type_loss, _ = transformer_utils.type_loss(types_pred, types_batch_out, type_loss_func)
				#import ipdb
				#ipdb.set_trace()

				loss = gap_loss + type_loss
				step_train_loss+=loss.numpy()

			grads = tape.gradient(loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))

			train_gap_metric_mae(gaps_batch_out, gaps_pred)
			train_gap_metric_mse(gaps_batch_out, gaps_pred)
			train_gap_mae = train_gap_metric_mae.result()
			train_gap_mse = train_gap_metric_mse.result()
			train_gap_metric_mae.reset_states()
			train_gap_metric_mse.reset_states()

			# print(float(train_gap_mae), float(train_gap_mse))
			print('Training loss (for one batch) at step %s: %s, %s, %s' %(sm_step, float(loss), float(gap_loss), float(type_loss)))
			step_cnt += 1
		et = time.time()
		print(model_name, 'time_reqd:', et-st)
		
		# Dev calculations
		enc_out, (dev_logits_pred, dev_gaps_pred) = model(nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types)
		dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
													 event_train_norma,
													 event_train_normd)
		
		dev_gap_metric_mae(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_metric_mse(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_mae = dev_gap_metric_mae.result()
		dev_gap_mse = dev_gap_metric_mse.result()
		dev_gap_metric_mae.reset_states()
		dev_gap_metric_mse.reset_states()

		if best_dev_gap_mse > dev_gap_mse:
			best_dev_gap_mse = dev_gap_mse
			best_dev_epoch = epoch
			print('Saving model at epoch', epoch)
			model.save_weights(checkpoint_path)

		step_train_loss /= step_cnt
		print('Training loss after epoch %s: %s' %(epoch, float(step_train_loss)))
		print('MAE and MSE of Dev data %s: %s' \
			%(float(dev_gap_mae), float(dev_gap_mse)))
		train_losses.append(step_train_loss)

	plt.plot(range(len(train_losses)), train_losses)
	plt.savefig(os.path.join(
		args.output_dir,
		'train_'+model_name+'_'+args.current_dataset+'_loss.png'))
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	_, (dev_logits_pred, dev_gaps_pred) = model(nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types)
	dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
												 event_train_norma,
												 event_train_normd)

	dev_gap_metric_mae(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_metric_mse(nc_event_dev_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	dev_types_acc = types_metric(nc_event_dev_out_types-1, dev_logits_pred)
	types_metric.reset_states()
	print('Best MAE, MSE, type_acc of Dev data: %s, %s, %s' \
		%(float(dev_gap_mae), float(dev_gap_mse), float(dev_types_acc)))

	return model
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#:


#####################################################
# 				Model Inference 					#
#####################################################

def simulate_fixed_cnt(model, gaps_in, sim_count, prev_hidden_state=None):
	gaps_pred = list()
	for i in range(sim_count):
		step_gaps_pred, _, _, prev_hidden_state, _ \
				= model(gaps_in, initial_state=prev_hidden_state)

		step_gaps_pred = step_gaps_pred[:,-1:]
		gaps_pred.append(tf.squeeze(step_gaps_pred, axis=-1))
		gaps_in = tf.concat([gaps_in[:, 1:], step_gaps_pred], axis=1)
	gaps_pred = tf.stack(gaps_pred, axis=1)
	return gaps_pred


def simulate_v2(model, times_in, gaps_in, bins_end_hrs, normalizers,
				embd_params,
				gaps_out_true,
				prev_hidden_state = None):
	#TODO: Check for this modification in functions which calls this def
	gaps_pred = list()
	times_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	# step_gaps_pred = gaps_in[:, -1]
	#step_gaps_pred, _, _, prev_hidden_state, _ \
	#		= model(gaps_in, initial_state=prev_hidden_state)

	#step_gaps_pred = step_gaps_pred[:,-1:]
	#gaps_in = tf.concat([gaps_in[:,1:], step_gaps_pred], axis=1)
	#step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
	#last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	#last_times_pred = times_in + last_gaps_pred_unnorm
	#gaps_pred.append(last_gaps_pred_unnorm)
	#times_pred.append(last_times_pred)

	simul_step = 0

	times_pred.append(times_in)
	#while any(times_pred[-1]<bins_end_hrs[:, -1]):
	while simul_step < gaps_out_true.shape[1]:
		step_gaps_pred, _, _, prev_hidden_state, _ \
				= model(gaps_in, initial_state=prev_hidden_state)

		step_gaps_pred = step_gaps_pred[:,-1:]
		gaps_in = step_gaps_pred
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		prev_hidden_state = model.hidden_states[:, -1]

		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm.numpy()
		gaps_pred.append(last_gaps_pred_unnorm)
		times_pred.append(last_times_pred)

		simul_step += 1

	gaps_pred = tf.concat(gaps_pred, axis=1)
	all_gaps_pred = gaps_pred.numpy()

	times_pred = times_pred[1:]
	times_pred = tf.concat(times_pred, axis=1)
	all_times_pred = times_pred.numpy()


	mask = (gaps_out_true>0).astype(np.float32)
	all_gaps_pred = all_gaps_pred * mask
	all_times_pred = all_times_pred * mask

	num_bins, num_grps, num_pos = embd_params
	all_times_pred_expand = tf.expand_dims(all_times_pred, axis=1)
	bin_ids = tf.reduce_sum(
		tf.cast(
			(all_times_pred_expand<bins_end_hrs),
			tf.float32
		),
		axis=1
	).numpy()
	bin_ids = (num_bins+1) - bin_ids
	bin_ids = np.where(bin_ids<=num_bins, bin_ids, np.zeros_like(bin_ids))

	bin_ranks = np.sum(
		np.stack(
			[
				np.cumsum(bin_ids==curr_b, axis=1)*(bin_ids==curr_b) for curr_b in range(1, num_bins+1)
			],
			axis=2
		),
		axis=-1
	)
	#grp_ids = np.ceil(bin_ranks/num_pos)
	#pos_ids = bin_ranks%(num_pos+1)
	grp_ids = (bin_ranks-1)//num_pos + 1
	pos_ids = (bin_ranks-1)%num_pos + 1
				

	return all_gaps_pred, all_times_pred, bin_ids, grp_ids, pos_ids


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate model until t_b_plus
def simulate_rmtpp(model, times_in, gaps_in, feats_in, types_in,
			 	   t_b_plus, normalizers, use_nowcast=False,
				   nc_gaps_in=None, nc_feats_in=None, nc_types_in=None,):
	#TODO: Check for this modification in functions which calls this def
	gaps_pred = list()
	types_pred = list()
	times_pred = list()
	D_pred, WT_pred = [], []
	data_norm_a, data_norm_d = normalizers
	
	# step_gaps_pred = gaps_in[:, -1]
	step_gaps_pred, step_types_logits, D, WT, prev_hidden_state \
			= model(gaps_in, feats_in, types_in)

	step_types_pred = tf.argmax(step_types_logits, axis=-1) + 1

	step_gaps_pred = step_gaps_pred[:,-1:]
	step_types_pred = step_types_pred[:,-1:]
	if not use_nowcast:
		gaps_in = step_gaps_pred
		types_in = step_types_pred
	else:
		gaps_in = nc_gaps_in[:, 0:1]
		types_in = nc_types_in[:, 0:1]
	step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	last_times_pred = times_in + last_gaps_pred_unnorm
	step_feats_pred = get_time_features(tf.expand_dims(last_times_pred, axis=-1))
	if not use_nowcast:
		feats_in = step_feats_pred
	else:
		feats_in = nc_feats_in[:, 0:1]
	#import ipdb
	#ipdb.set_trace()
	gaps_pred.append(last_gaps_pred_unnorm)
	types_pred.append(step_types_pred)
	times_pred.append(last_times_pred)
	D_pred.append(D[:, -1:])
	WT_pred.append(WT[:, -1:])

	simul_step = 0

	while any(times_pred[-1]<t_b_plus):
		simul_step += 1

		step_gaps_pred, step_types_logits, D, WT, prev_hidden_state \
				= model(gaps_in, feats_in, types_in, initial_state=prev_hidden_state)

		step_types_pred = tf.argmax(step_types_logits, axis=-1) + 1

		step_gaps_pred = step_gaps_pred[:,-1:]
		step_types_pred = step_types_pred[:,-1:]
		if not use_nowcast:
			gaps_in = step_gaps_pred
			types_in = step_types_pred
		else:
			gaps_in = nc_gaps_in[:, simul_step:simul_step+1]
			types_in = nc_types_in[:, simul_step:simul_step+1]

		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		step_feats_pred = get_time_features(tf.expand_dims(last_times_pred, axis=-1))
		if not use_nowcast:
			feats_in = step_feats_pred
		else:
			feats_in = nc_feats_in[:, simul_step:simul_step+1]
		gaps_pred.append(last_gaps_pred_unnorm)
		types_pred.append(step_types_pred)
		times_pred.append(last_times_pred)
		D_pred.append(D[:, -1:])
		WT_pred.append(WT[:, -1:])
		

	gaps_pred = tf.stack(gaps_pred, axis=1)
	types_pred = tf.squeeze(tf.stack(types_pred, axis=1), axis=2)
	all_gaps_pred = gaps_pred

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	D_pred = tf.squeeze(tf.concat(D_pred, axis=1), axis=-1)
	WT_pred = tf.squeeze(tf.concat(WT_pred, axis=1), axis=-1)

	return all_gaps_pred, all_times_pred, types_pred, prev_hidden_state, D_pred, WT_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate model until t_b_plus
def simulate_hierarchical(model, times_in, gaps_in, feats_in,
						  t_b_plus, normalizers, prev_hidden_state = None):

	#TODO:
	#	1. Incorporate features


	#TODO: Check for this modification in functions which calls this def
	gaps_pred_l1_lst = list()
	gaps_pred_l2_lst = list()
	times_pred_l1_lst = list()
	times_pred_l2_lst = list()
	data_norm_a, data_norm_d = normalizers

	# step_gaps_pred = gaps_in[:, -1]
	gaps_pred_l2, _, _, gaps_pred_l1, _, _, prev_hidden_state, _ \
			= model(gaps_in, initial_state=prev_hidden_state)

	step_gaps_pred_l2 = gaps_pred_l2[:,-1:]
	step_gaps_pred_l2 = tf.squeeze(step_gaps_pred_l2, axis=-1)
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred_l2, data_norm_a, data_norm_d)

	last_times_pred = times_in + last_gaps_pred_unnorm
	gaps_pred_l2_lst.append(last_gaps_pred_unnorm)
	times_pred_l2_lst.append(last_times_pred)

	actual_bin_start = times_in
	actual_bin_end = last_times_pred

	gaps_in = tf.concat([gaps_in[:,1:], gaps_pred_l1[:,-1:]], axis=1)

	step_gaps_pred_l1 = gaps_pred_l1[:,-1]
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred_l1, data_norm_a, data_norm_d)
	last_times_pred = tf.expand_dims(times_in, axis=-1) + tf.cumsum(last_gaps_pred_unnorm, axis=1)

	bin_start = times_in
	bin_end = last_times_pred[:,-1]

	last_times_pred_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
					 (tf.squeeze(last_times_pred, axis=-1) - bin_start)) + actual_bin_start
	
	last_gaps_pred_unnorm = last_times_pred_scaled - tf.concat([times_in, last_times_pred_scaled[:,:-1]], axis=1)

	gaps_pred_l1_lst.append(last_gaps_pred_unnorm)
	times_pred_l1_lst.append(last_times_pred_scaled)

	simul_step = 0

	while any(times_pred_l2_lst[-1]<t_b_plus):
		gaps_pred_l2, _, _, gaps_pred_l1, _, _, prev_hidden_state, _ \
				= model(gaps_in, initial_state=prev_hidden_state)

		step_gaps_pred_l2 = gaps_pred_l2[:,-1:]
		step_gaps_pred_l2 = tf.squeeze(step_gaps_pred_l2, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred_l2, data_norm_a, data_norm_d)
		times_in_tmp = times_pred_l2_lst[-1]

		last_times_pred = times_in_tmp + last_gaps_pred_unnorm
		gaps_pred_l2_lst.append(last_gaps_pred_unnorm)
		times_pred_l2_lst.append(last_times_pred)

		actual_bin_start = times_in_tmp
		actual_bin_end = last_times_pred

		gaps_in = tf.concat([gaps_in[:,1:], gaps_pred_l1[:,-1:]], axis=1)

		step_gaps_pred_l1 = gaps_pred_l1[:,-1]
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred_l1, data_norm_a, data_norm_d)
		last_times_pred = tf.expand_dims(times_in_tmp, axis=-1) + tf.cumsum(last_gaps_pred_unnorm, axis=1)

		bin_start = times_in_tmp
		bin_end = last_times_pred[:,-1]

		last_times_pred_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
						 (tf.squeeze(last_times_pred, axis=-1) - bin_start)) + actual_bin_start
		
		last_gaps_pred_unnorm = last_times_pred_scaled - tf.concat([times_in_tmp, last_times_pred_scaled[:,:-1]], axis=1)

		gaps_pred_l1_lst.append(last_gaps_pred_unnorm)
		times_pred_l1_lst.append(last_times_pred_scaled)
		
		simul_step += 1

	all_gaps_pred = tf.concat(gaps_pred_l1_lst, axis=1)
	all_times_pred = tf.concat(times_pred_l1_lst, axis=1)

	return all_gaps_pred, all_times_pred, None
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate model until t_b_plus
def simulate_for_D_WT(model, times_in, gaps_in, feats_in, types_in,
					  t_b_plus, normalizers):
	##### WARNING: This function ignores the types_in and hence can only be used
	##### for hierarchical mdoel

	#TODO: Check for this modification in functions which calls this def

	#TODO: Do we need to incorporate the variables
	# 	old_hidden_state, 
	# from simulate_with_counter(..) method?

	gaps_pred = list()
	D_pred = list()
	WT_pred = list()
	times_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	# step_gaps_pred = gaps_in[:, -1]
	step_gaps_pred, step_types_logits, step_D_pred, step_WT_pred, prev_hidden_state \
			= model(gaps_in, feats_in, types_in)

	step_gaps_pred = step_gaps_pred[:,-1:]
	D_pred.append(step_D_pred[:,-1:])
	WT_pred.append(step_WT_pred[:,-1:])
	gaps_in = step_gaps_pred
	step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	last_times_pred = times_in + last_gaps_pred_unnorm
	step_feats_pred = get_time_features(tf.expand_dims(last_times_pred, axis=-1))
	feats_in = step_feats_pred
	gaps_pred.append(last_gaps_pred_unnorm)
	times_pred.append(last_times_pred)

	simul_step = 0

	while any(times_pred[-1]<t_b_plus):
		step_gaps_pred, step_types_logits, step_D_pred, step_WT_pred, prev_hidden_state \
				= model(gaps_in, feats_in, types_in, initial_state=prev_hidden_state)

		step_gaps_pred = step_gaps_pred[:,-1:]
		D_pred.append(step_D_pred[:,-1:])
		WT_pred.append(step_WT_pred[:,-1:])
		gaps_in = step_gaps_pred
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		step_feats_pred = get_time_features(tf.expand_dims(last_times_pred, axis=-1))
		feats_in = step_feats_pred
		gaps_pred.append(last_gaps_pred_unnorm)
		times_pred.append(last_times_pred)

		simul_step += 1

	gaps_pred = tf.stack(gaps_pred, axis=1)
	all_gaps_pred = gaps_pred
	D_pred = tf.concat(D_pred, axis=1)
	WT_pred = tf.concat(WT_pred, axis=1)

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	return all_gaps_pred, all_times_pred, D_pred, WT_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate model until t_b_plus
def simulate_count_for_D_WT(model, times_in, gaps_in, out_gaps_count, normalizers, prev_hidden_state = None):
	#TODO: Check for this modification in functions which calls this def
	gaps_pred = list()
	times_pred = list()
	D_pred = list()
	WT_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	# step_gaps_pred = gaps_in[:, -1]
	step_gaps_pred, step_D_pred, step_WT_pred, prev_hidden_state, _ \
			= model(gaps_in, initial_state=prev_hidden_state)

	step_gaps_pred = step_gaps_pred[:,-1:]
	gaps_in = tf.concat([gaps_in[:,1:], step_gaps_pred], axis=1)
	step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	last_times_pred = times_in + last_gaps_pred_unnorm
	gaps_pred.append(last_gaps_pred_unnorm)
	times_pred.append(last_times_pred)

	step_D_pred = step_D_pred[:,-1:]
	step_WT_pred = step_WT_pred[:,-1:]
	D_pred.append(step_D_pred)
	WT_pred.append(step_WT_pred)

	simul_step = 0

	while any(simul_step < out_gaps_count):
		step_gaps_pred, step_D_pred, step_WT_pred, prev_hidden_state, _ \
				= model(gaps_in, initial_state=prev_hidden_state)

		step_gaps_pred = step_gaps_pred[:,-1:]
		gaps_in = tf.concat([gaps_in[:,1:], step_gaps_pred], axis=1)
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		gaps_pred.append(last_gaps_pred_unnorm)
		times_pred.append(last_times_pred)

		step_D_pred = step_D_pred[:,-1:]
		step_WT_pred = step_WT_pred[:,-1:]
		D_pred.append(step_D_pred)
		WT_pred.append(step_WT_pred)
		
		simul_step += 1

	gaps_pred = tf.stack(gaps_pred, axis=1)
	all_gaps_pred = gaps_pred

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	D_pred = tf.stack(D_pred, axis=1)
	WT_pred = tf.stack(WT_pred, axis=1)
	D_pred = tf.squeeze(D_pred, axis=-1)
	WT_pred = tf.squeeze(WT_pred, axis=-1)

	return all_gaps_pred, all_times_pred, D_pred, WT_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate model out_gaps_count times
def simulate_with_counter(model, times_in, gaps_in, feats_in, types_in,
						  out_gaps_count, normalizers, use_nowcast=False,
						  nc_gaps_in=None, nc_feats_in=None, nc_types_in=None,):
	gaps_pred = list()
	D_pred = list()
	WT_pred = list()
	types_pred = list()
	times_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	# step_gaps_pred = gaps_in[:, -1]
	step_gaps_pred, step_types_logits, D, WT, prev_hidden_state \
			= model(gaps_in, feats_in, types_in)

	step_types_pred = tf.argmax(step_types_logits, axis=-1) + 1

	step_gaps_pred = step_gaps_pred[:,-1:]
	step_types_pred = step_types_pred[:,-1:]
	D_pred.append(D[:, -1:])
	WT_pred.append(WT[:, -1:])
	if not use_nowcast:
		gaps_in = step_gaps_pred
		types_in = step_types_pred
	else:
		gaps_in = nc_gaps_in[:, 0:1]
		types_in = nc_types_in[:, 0:1]
	step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	last_times_pred = times_in + last_gaps_pred_unnorm
	step_feats_pred = get_time_features(tf.expand_dims(last_times_pred, axis=-1))
	if not use_nowcast:
		feats_in = step_feats_pred
	else:
		feats_in = nc_feats_in[:, 0:1]
	gaps_pred.append(last_gaps_pred_unnorm)
	types_pred.append(step_types_pred)
	times_pred.append(last_times_pred)

	simul_step = 0

	start_time = time.time()
	while any(simul_step < out_gaps_count):
		simul_step += 1

		print(simul_step, np.sum(simul_step<out_gaps_count), out_gaps_count[0])
		step_gaps_pred, step_types_logits, D, WT, prev_hidden_state \
				= model(gaps_in, feats_in, types_in, initial_state=prev_hidden_state)

		step_types_pred = tf.argmax(step_types_logits, axis=-1) + 1
			
		step_gaps_pred = step_gaps_pred[:,-1:]
		step_types_pred = step_types_pred[:,-1:]
		D_pred.append(D[:, -1:])
		WT_pred.append(WT[:, -1:])
		if not use_nowcast:
			gaps_in = step_gaps_pred
			types_in = step_types_pred
		else:
			gaps_in = nc_gaps_in[:, simul_step:simul_step+1]
			types_in = nc_types_in[:, simul_step:simul_step+1]
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		last_times_pred = (simul_step <= out_gaps_count) * last_times_pred
		step_feats_pred = get_time_features(tf.expand_dims(last_times_pred, axis=-1))
		if not use_nowcast:
			feats_in = step_feats_pred
		else:
			feats_in = nc_feats_in[:, simul_step:simul_step+1]
		gaps_pred.append(last_gaps_pred_unnorm)
		types_pred.append(step_types_pred)
		times_pred.append(last_times_pred)
		
	end_time = time.time()
	print('Time Reqd in simulate_with_counter:', end_time-start_time)

	gaps_pred = tf.stack(gaps_pred, axis=1)
	types_pred = tf.squeeze(tf.stack(types_pred, axis=1), axis=2)
	all_gaps_pred = gaps_pred
	D_pred = tf.concat(D_pred, axis=1)
	WT_pred = tf.concat(WT_pred, axis=1)

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	return all_gaps_pred, all_times_pred, types_pred, D_pred, WT_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate WGAN model till t_b_plus
def simulate_wgan(model, times_in, gaps_in, feats_in,
				  t_b_plus, normalizers, prev_hidden_state=None):
	'''
	Encode the input sequence.
	Generate output sequence in a loop until T_l^+ is not reached.
	Generate z_seqs on the fly until T_l^+ is reached.
		- That means, generator will process a step 
			at a time instead of entire sequence.
	TODO:

	'''
	gaps_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	step_gaps_pred = gaps_in[:, -1]

	times_pred = list()
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	
	last_times_pred = times_in + last_gaps_pred_unnorm
	times_pred.append(last_times_pred)
	
	simul_step = 0

	#TODO: Why we simulate one event at a time like RMTPP if we can get 20 events because of wgan
	times_in = tf.cumsum(gaps_in, axis=1)
	span_in = times_in[:, -1] - times_in[:, 0]
	lambda0 = np.ones_like(span_in) * gaps_in.shape[1] / span_in.numpy()
	intensityPoisson = IntensityHomogenuosPoisson(lambda0)

	if model.use_time_feats:
		feats_in = feats_in/24.
		enc_inputs = tf.concat([gaps_in, feats_in], axis=-1)
	else:
		enc_inputs = gaps_in
	_, g_init_state = model.run_encoder(enc_inputs)

	while any(times_pred[-1]<t_b_plus):

		z_seqs_in = generate_sample(intensityPoisson, 1, lambda0.shape[0])
		z_seqs_in = tf.convert_to_tensor(z_seqs_in)
		#z_seqs_in = utils.normalize_avg_given_param(z_seqs_in,
		#									data_norm_a,
		#									data_norm_d)

		step_gaps_pred \
				= model.generator(z_seqs_in, g_init_state=g_init_state)

		# step_gaps_pred = step_gaps_pred[:,-1:]
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		gaps_pred.append(last_gaps_pred_unnorm)
		times_pred.append(last_times_pred)

		g_init_state = model.g_state

		simul_step += 1

	gaps_pred = tf.stack(gaps_pred, axis=1)
	all_gaps_pred = gaps_pred

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	return all_gaps_pred, all_times_pred, prev_hidden_state
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate WGAN model till t_b_plus
def simulate_seq2seq(model, times_in, gaps_in, feats_in,
				  	 t_b_plus, normalizers, prev_hidden_state=None):
	'''
	Encode the input sequence.
	Generate output sequence in a loop until T_l^+ is not reached.
	Generate z_seqs on the fly until T_l^+ is reached.
		- That means, generator will process a step 
			at a time instead of entire sequence.
	TODO:

	'''
	gaps_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	step_gaps_pred = gaps_in[:, -1]

	times_pred = list()
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	
	last_times_pred = times_in + last_gaps_pred_unnorm
	times_pred.append(last_times_pred)
	
	simul_step = 0

	times_in = tf.cumsum(gaps_in, axis=1)
	span_in = times_in[:, -1] - times_in[:, 0]

	if model.use_time_feats:
		feats_in = feats_in/24.
		enc_inputs = tf.concat([gaps_in, feats_in], axis=-1)
	else:
		enc_inputs = gaps_in
	step_gaps_pred ,_, g_init_state = model.run_encoder(enc_inputs)
	gaps_in = step_gaps_pred[:, -1:]
	last_gaps_pred_unnorm = utils.denormalize_avg(gaps_in, data_norm_a, data_norm_d)
	feats_in = get_time_features(tf.expand_dims(times_pred[-1], axis=-1)+last_gaps_pred_unnorm)

	while any(times_pred[-1]<t_b_plus):

		step_gaps_pred \
				= model.generator(gaps_in, feats_in, dec_init_state=g_init_state)

		gaps_in = step_gaps_pred.numpy()
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		feats_in = get_time_features(
			tf.expand_dims(times_pred[-1], axis=-1)
			+ tf.expand_dims(last_gaps_pred_unnorm, axis=-1)
		)
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		gaps_pred.append(last_gaps_pred_unnorm)
		times_pred.append(last_times_pred)

		g_init_state = model.g_state

		simul_step += 1

	gaps_pred = tf.stack(gaps_pred, axis=1)
	all_gaps_pred = gaps_pred

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	return all_gaps_pred, all_times_pred, prev_hidden_state
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate model until t_b_plus
def simulate_transformer(
	model, times_in, gaps_in, feats_in, types_in,
	t_b_plus, normalizers, use_nowcast=False,
	nc_gaps_in=None, nc_feats_in=None, nc_types_in=None,
):
	#TODO: Check for this modification in functions which calls this def
	gaps_pred = list()
	types_pred = list()
	times_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	# step_gaps_pred = gaps_in[:, -1]
	enc_out, (step_types_logits, step_gaps_pred) = model(gaps_in, feats_in, types_in)

	step_types_pred = tf.argmax(step_types_logits, axis=-1) + 1

	step_gaps_pred = step_gaps_pred[:,-1:]
	step_types_pred = step_types_pred[:,-1:]
	if not use_nowcast:
		gaps_in = step_gaps_pred
		types_in = step_types_pred
	else:
		gaps_in = nc_gaps_in[:, 0:1]
		types_in = nc_types_in[:, 0:1]
	step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	last_times_pred = times_in + last_gaps_pred_unnorm
	step_feats_pred = get_time_features(tf.expand_dims(last_times_pred, axis=-1))
	if not use_nowcast:
		feats_in = step_feats_pred
	else:
		feats_in = nc_feats_in[:, 0:1]
	gaps_pred.append(last_gaps_pred_unnorm)
	types_pred.append(step_types_pred)
	times_pred.append(last_times_pred)

	simul_step = 0

	while any(times_pred[-1]<t_b_plus):
		simul_step += 1

		#print(np.sum(times_pred[-1]<t_b_plus), t_b_plus.shape)
		#print(np.squeeze(t_b_plus-times_pred[-1], axis=-1))
		#print(gaps_in[0, :, 0])
		enc_out, (step_types_logits, step_gaps_pred) = model(gaps_in, feats_in, types_in)

		step_types_pred = tf.argmax(step_types_logits, axis=-1) + 1

		step_gaps_pred = step_gaps_pred[:,-1:]
		step_types_pred = step_types_pred[:,-1:]
		if not use_nowcast:
			gaps_in = step_gaps_pred
			types_in = step_types_pred
		else:
			if simul_step>=nc_gaps_in.shape[1]:
				break
			gaps_in = nc_gaps_in[:, simul_step:simul_step+1]
			types_in = nc_types_in[:, simul_step:simul_step+1]
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		#print(np.squeeze(last_gaps_pred_unnorm, axis=-1))
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		step_feats_pred = get_time_features(tf.expand_dims(last_times_pred, axis=-1))
		if not use_nowcast:
			feats_in = step_feats_pred
		else:
			feats_in = nc_feats_in[:, simul_step:simul_step+1]
		gaps_pred.append(last_gaps_pred_unnorm)
		types_pred.append(step_types_pred)
		times_pred.append(last_times_pred)
		

	gaps_pred = tf.stack(gaps_pred, axis=1)
	types_pred = tf.squeeze(tf.stack(types_pred, axis=1), axis=2)
	all_gaps_pred = gaps_pred

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	return all_gaps_pred, all_times_pred, types_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp with state reinitialized 
# after each bin events preds and scaled, each bin has events
# whose count generated by count model
def run_rmtpp_count_reinit(args, models, data, test_data, rmtpp_type):
	model_cnt = models['count_model']
	if rmtpp_type=='nll':
		model_rmtpp = models['rmtpp_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = models['rmtpp_mse']
	elif rmtpp_type=='mse_var':
		model_rmtpp = models['rmtpp_mse_var']
	else:
		assert False, "rmtpp_type must be nll or mse"
	model_check = (model_cnt is not None) and (model_rmtpp is not None)
	assert model_check, "run_rmtpp_count_reinit requires count and RMTPP model"

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(count_test_in_counts, count_test_in_feats)
	if len(test_predictions_norm_cnt) == 2:
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, count_test_normm, count_test_norms)
	event_count_preds_cnt = np.clip(np.round(test_predictions_cnt), 1.0, np.inf)
	event_count_preds_cnt_min = np.min(event_count_preds_cnt, axis=0)
	event_count_preds_true = count_test_out_counts

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()

	bin_end = None
	for dec_idx in range(dec_len):
		all_gaps_pred, all_times_pred, _, _, _ = simulate_with_counter(model_rmtpp, 
												test_data_init_time, 
												test_data_input_gaps_bin,
												event_test_in_feats,
												output_event_count_pred[:,dec_idx],
												(event_test_norma, 
												event_test_normd),)

		bin_start = bin_end
		if bin_start is None:
			gaps_before_bin = all_times_pred[:,:1] - test_data_init_time
			gaps_before_bin = gaps_before_bin * np.random.uniform(size=gaps_before_bin.shape)
			bin_start = test_data_init_time + gaps_before_bin

		prev_test_data_init_time = test_data_init_time
		_, _, test_data_init_time, test_data_init_gaps = compute_event_in_bin(tf.expand_dims(all_times_pred, axis=-1), 
														 output_event_count_pred[:,dec_idx,0])
		test_data_init_time = np.expand_dims(test_data_init_time, axis=-1)
		test_data_init_gaps = np.expand_dims(test_data_init_gaps, axis=-1)
		gaps_after_bin = test_data_init_gaps
		gaps_after_bin = gaps_after_bin * np.random.uniform(size=gaps_after_bin.shape)
		bin_end = test_data_init_time + gaps_after_bin

		actual_bin_start = count_test_out_binend[:,dec_idx]-bin_size
		actual_bin_end = count_test_out_binend[:,dec_idx]

		all_times_pred, all_gaps_pred = scaled_points(actual_bin_start, actual_bin_end, bin_start, bin_end, all_times_pred)

		gaps_before_bins = all_times_pred[:,:1] - prev_test_data_init_time
		gaps_before_bins = tf.expand_dims(gaps_before_bins, axis=-1)
		all_gaps_pred = tf.concat([gaps_before_bins, all_gaps_pred], axis=1)

		event_in_bin_preds, _, test_data_init_time, _ = compute_event_in_bin(tf.expand_dims(all_times_pred, axis=-1), 
														 output_event_count_pred[:,dec_idx,0])
		test_data_init_time = np.expand_dims(test_data_init_time, axis=-1)
		
		
		all_gaps_pred_norm = utils.normalize_avg_given_param(all_gaps_pred,
												event_test_norma,
												event_test_normd)
		
		_, test_data_input_gaps_bin_full, _, _ = compute_event_in_bin(all_gaps_pred_norm, 
															  output_event_count_pred[:,dec_idx,0],
															  test_data_input_gaps_bin, 
															  enc_len+int(event_count_preds_cnt_min[dec_idx]))
		
		all_events_in_bin_pred.append(event_in_bin_preds)
		
		test_data_input_gaps_bin = np.expand_dims(test_data_input_gaps_bin_full[:,-enc_len:], axis=-1)
		test_data_input_gaps_bin = test_data_input_gaps_bin.astype(np.float32)

		test_data_input_gaps_bin_scaled = np.expand_dims(test_data_input_gaps_bin_full[:,:-enc_len], axis=-1)
		test_data_input_gaps_bin_scaled = test_data_input_gaps_bin_scaled.astype(np.float32)
		
		if test_data_input_gaps_bin_scaled.shape[1] == 0:
			next_hidden_state = None
		else:
			_, _, _, _, next_hidden_state \
						= model_rmtpp(test_data_input_gaps_bin_scaled, initial_state=scaled_rnn_hidden_state)

	all_events_in_bin_pred = np.array(all_events_in_bin_pred).T
	return event_count_preds_cnt, all_events_in_bin_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp with one rmtpp simulation untill 
# all events of bins generated then scale , each bin has events
# whose count generated by count model
def run_rmtpp_rescaling_model(args, models, data, test_data, rmtpp_type):
	model_cnt = models['count_model']
	if rmtpp_type=='nll':
		model_rmtpp = models['rmtpp_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = models['rmtpp_mse']
	elif rmtpp_type=='mse_var':
		model_rmtpp = models['rmtpp_mse_var']
	else:
		assert False, "rmtpp_type must be nll or mse"
	model_check = (model_cnt is not None) and (model_rmtpp is not None)
	assert model_check, "run_rmtpp_rescaling_model requires count and RMTPP model"

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(count_test_in_counts, count_test_in_feats)
	if len(test_predictions_norm_cnt) == 2:
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, count_test_normm, count_test_norms)
	event_count_preds_cnt = np.clip(np.round(test_predictions_cnt), 1.0, np.inf)

	#event_count_preds_cnt = count_test_out_counts
	event_count_preds_true = count_test_out_counts

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()
	
	output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_cnt, axis=-1).numpy()
	full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
	full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)

	(
		all_gaps_pred, all_times_pred, all_types_pred, D_pred, WT_pred,
	) = simulate_with_counter(
		model_rmtpp, 
		test_data_init_time, 
		test_data_input_gaps_bin,
		event_test_in_feats,
		event_test_in_types,
		full_cnt_event_all_bins_pred,
		(event_test_norma, 
		event_test_normd),
	)
	
	all_times_pred_lst = list()
	all_types_pred_lst = list()
	all_sigms_pred_lst = list()
	for batch_idx in range(len(all_gaps_pred)):
		event_past_cnt=0
		times_pred_all_bin_lst = list()
		batch_types_pred_lst = list()
		batch_sigms_pred_lst = list()

		gaps_before_bin = all_times_pred[batch_idx,:1] - test_data_init_time[batch_idx]
		gaps_before_bin = gaps_before_bin * np.random.uniform(size=gaps_before_bin.shape)
		next_bin_start = test_data_init_time[batch_idx] + gaps_before_bin

		for dec_idx in range(dec_len):

			times_pred_for_bin = all_times_pred[batch_idx,event_past_cnt:event_past_cnt+int(output_event_count_pred[batch_idx,dec_idx,0])]
			batch_bin_types_pred = all_types_pred[batch_idx, event_past_cnt:event_past_cnt+int(output_event_count_pred[batch_idx,dec_idx,0])]
			batch_bin_sigms_pred = WT_pred[batch_idx, event_past_cnt:event_past_cnt+int(output_event_count_pred[batch_idx,dec_idx,0]), 0]
			event_past_cnt += int(output_event_count_pred[batch_idx,dec_idx,0])

			bin_start = next_bin_start

			if event_past_cnt==0:
				test_data_init_time[batch_idx] = all_times_pred[batch_idx,0:1]
			else:
				test_data_init_time[batch_idx] = all_times_pred[batch_idx,event_past_cnt-1:event_past_cnt]

			gaps_after_bin = all_times_pred[batch_idx,event_past_cnt:event_past_cnt+1] - test_data_init_time[batch_idx]
			gaps_after_bin = gaps_after_bin * np.random.uniform(size=gaps_after_bin.shape)
			bin_end = test_data_init_time[batch_idx] + gaps_after_bin
			next_bin_start = bin_end
			
			actual_bin_start = count_test_out_binend[batch_idx,dec_idx]-bin_size
			actual_bin_end = count_test_out_binend[batch_idx,dec_idx]

			times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
							 (times_pred_for_bin - bin_start)) + actual_bin_start
			
			times_pred_all_bin_lst.append(times_pred_for_bin_scaled.numpy())
			batch_types_pred_lst.append(batch_bin_types_pred)
			batch_sigms_pred_lst.append(batch_bin_sigms_pred.numpy())



		all_times_pred_lst.append(times_pred_all_bin_lst)
		all_types_pred_lst.append(batch_types_pred_lst)
		all_sigms_pred_lst.append(batch_sigms_pred_lst)


	all_times_pred = np.array(all_times_pred_lst)
	all_types_pred = np.array(all_types_pred_lst)
	all_sigms_pred = np.array(all_sigms_pred_lst)

	all_types_pred_flatten = []
	for seq in all_types_pred:
		seq = [s.numpy().tolist() for s in seq]
		all_types_pred_flatten.append(np.array(flatten(seq)))
	all_types_pred = np.array(all_types_pred_flatten)

	all_times_pred_flatten = []
	for seq in all_times_pred:
		seq = [s.tolist() for s in seq]
		all_times_pred_flatten.append(np.array(flatten(seq)))
	all_times_pred = np.array(all_times_pred_flatten)

	all_means_pred = []
	for seq, beg_time in zip(all_times_pred, count_test_out_binend[:, 0] - args.bin_size):
		all_means_pred.append(
			utils.normalize_avg_given_param(
				seq - np.concatenate([beg_time, seq[:-1]]),
				event_test_norma, event_test_normd
			)
		)
	all_means_pred = np.array(all_means_pred)

	all_sigms_pred_flatten = []
	for seq in all_sigms_pred:
		seq = [s.tolist() for s in seq]
		all_sigms_pred_flatten.append(np.array(flatten(seq)))
	all_sigms_pred = np.array(all_sigms_pred_flatten)

	event_dist_params = [all_means_pred, all_sigms_pred]
	count_dist_params = [event_count_preds_cnt, None]

	return (
		event_count_preds_cnt, all_times_pred, all_types_pred,
		event_dist_params, count_dist_params,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp with one rmtpp simulation untill 
# all events of bins generated then scale , each bin has events
# whose count generated by rmtpp_comp model
def run_rmtpp_rescaling_model_comp(args, models, data, test_data, test_data_comp, rmtpp_type, rmtpp_type_comp):
	if rmtpp_type=='nll':
		model_rmtpp = models['rmtpp_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = models['rmtpp_mse']
	elif rmtpp_type=='mse_var':
		model_rmtpp = models['rmtpp_mse_var']
	else:
		assert False, "rmtpp_type must be nll or mse"

	if rmtpp_type_comp=='nll':
		model_rmtpp_comp = models['rmtpp_nll_comp']
	elif rmtpp_type_comp=='mse':
		model_rmtpp_comp = models['rmtpp_mse_comp']
	elif rmtpp_type_comp=='mse_var':
		model_rmtpp_comp = models['rmtpp_mse_var_comp']
	else:
		assert False, "rmtpp_type_comp must be nll_comp or mse_comp"

	model_check = (model_rmtpp is not None) and (model_rmtpp_comp is not None)
	assert model_check, "run_rmtpp_rescaling_model_comp requires RMTPP and RMTPP_comp model"

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data


	[comp_test_in_gaps, comp_test_in_feats, comp_test_in_types,
	 _, _,
	 comp_test_norma, comp_test_normd] = test_data_comp
	
	enc_len = args.enc_len
	comp_enc_len = args.comp_enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	comp_bin_sz = args.comp_bin_sz
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	test_data_input_gaps_bin_comp = comp_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()

	t_e_plus = count_test_out_binend[:,-1]
	(
		all_gaps_pred_comp, all_times_pred_comp,
		_, _, all_D_pred_comp, all_WT_pred_comp
	) = simulate_rmtpp(
		model_rmtpp_comp,
		test_data_init_time,
		test_data_input_gaps_bin_comp,
		comp_test_in_feats,
		comp_test_in_types,
		t_e_plus,
		(comp_test_norma,
		comp_test_normd),
	)
	event_count_preds_cnt_comp = np.ones_like(test_data_init_time) * all_times_pred_comp.shape[1] + 1
	(
		all_gaps_pred_comp, all_times_pred_comp, all_types_pred_comp,
		all_D_pred_comp, all_WT_pred_comp
	) = simulate_with_counter(
		model_rmtpp_comp,
		test_data_init_time,
		test_data_input_gaps_bin_comp,
		comp_test_in_feats,
		comp_test_in_types,
		event_count_preds_cnt_comp,
		(comp_test_norma,
		comp_test_normd),
	)

	event_count_preds_cnt = np.ones_like(test_data_init_time) * all_times_pred_comp.shape[1] * comp_bin_sz
	event_count_preds_true = count_test_out_counts
	(
		all_gaps_pred, all_times_pred, all_types_pred,
		D_pred, WT_pred,
	) = simulate_with_counter(
		model_rmtpp, 
		test_data_init_time, 
		test_data_input_gaps_bin,
		event_test_in_feats,
		event_test_in_types,
		event_count_preds_cnt,
		(event_test_norma, 
		event_test_normd),
	)
	
	all_times_pred_lst = list()
	all_types_pred_lst = list()
	all_sigms_pred_lst = list()
	for batch_idx in range(len(all_gaps_pred)):
		event_past_cnt=0
		times_pred_all_bin_lst=list()
		batch_types_pred_lst = list()
		batch_sigms_pred_lst = list()

		next_bin_start = test_data_init_time[batch_idx]
		next_actual_bin_start = test_data_init_time[batch_idx]

		for dec_idx in range(len(all_gaps_pred_comp[batch_idx])):

			if dec_idx==0 or all_times_pred_comp[batch_idx, dec_idx-1] < t_e_plus[batch_idx]:
				times_pred_for_bin = all_times_pred[batch_idx, event_past_cnt:event_past_cnt+comp_bin_sz]
				batch_bin_types_pred = all_types_pred[batch_idx, event_past_cnt:event_past_cnt+comp_bin_sz]
				batch_bin_sigms_pred = WT_pred[batch_idx, event_past_cnt:event_past_cnt+comp_bin_sz, 0]
	
				bin_start = next_bin_start
				bin_end = all_times_pred[batch_idx, event_past_cnt+comp_bin_sz-1:event_past_cnt+comp_bin_sz]
				next_bin_start = all_times_pred[batch_idx, event_past_cnt+comp_bin_sz-1:event_past_cnt+comp_bin_sz]
	
				actual_bin_start = next_actual_bin_start
				actual_bin_end = all_times_pred_comp[batch_idx,dec_idx]
				next_actual_bin_start = actual_bin_end
	
				times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
								 (times_pred_for_bin - bin_start)) + actual_bin_start

				times_pred_for_bin_scaled =  times_pred_for_bin_scaled.numpy()
				clip_te_cnt = np.sum(times_pred_for_bin_scaled<t_e_plus[batch_idx])
				times_pred_all_bin_lst.append(times_pred_for_bin_scaled[:clip_te_cnt])
				batch_types_pred_lst.append(batch_bin_types_pred[:clip_te_cnt])
				batch_sigms_pred_lst.append(batch_bin_sigms_pred.numpy()[:clip_te_cnt])
				event_past_cnt += clip_te_cnt
		
		all_times_pred_lst.append(times_pred_all_bin_lst)
		all_types_pred_lst.append(batch_types_pred_lst)
		all_sigms_pred_lst.append(batch_sigms_pred_lst)

	all_times_pred = np.array(all_times_pred_lst)
	all_types_pred = np.array(all_types_pred_lst)
	all_sigms_pred = np.array(all_sigms_pred_lst)

	all_times_pred_flatten = []
	for seq in all_times_pred:
		seq = [s.tolist() for s in seq]
		all_times_pred_flatten.append(np.array(flatten(seq)))
	all_times_pred = np.array(all_times_pred_flatten)

	all_types_pred_flatten = []
	for seq in all_types_pred:
		seq = [s.numpy().tolist() for s in seq]
		all_types_pred_flatten.append(np.array(flatten(seq)))
	all_types_pred = np.array(all_types_pred_flatten)

	all_counts_pred = []
	for dec_idx in range(dec_len):
		t_b_plus = count_test_out_binend[:, dec_idx] - bin_size
		t_e_plus = count_test_out_binend[:, dec_idx]
		all_counts_pred.append(
			count_events(all_times_pred, t_b_plus, t_e_plus)
		)
	all_counts_pred = np.stack(all_counts_pred, axis=1)

	all_means_pred = []
	for seq, beg_time in zip(all_times_pred, count_test_out_binend[:, 0] - args.bin_size):
		all_means_pred.append(
			utils.normalize_avg_given_param(
				seq - np.concatenate([beg_time, seq[:-1]]),
				event_test_norma, event_test_normd
			)
		)
	all_means_pred = np.array(all_means_pred)

	all_sigms_pred_flatten = []
	for seq in all_sigms_pred:
		seq = [s.tolist() for s in seq]
		all_sigms_pred_flatten.append(np.array(flatten(seq)))
	all_sigms_pred = np.array(all_sigms_pred_flatten)

	event_dist_params = [all_means_pred, all_sigms_pred]
	count_dist_params = [all_counts_pred, None]

	return (
		all_counts_pred, all_times_pred, all_types_pred,
		event_dist_params, count_dist_params,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp_count model with one rmtpp simulation untill 
# all events of bins generated then scale , each bin has events
# whose count generated by rmtpp_comp model
def run_pure_hierarchical_infer(args, models, data, test_data, test_data_comp, rmtpp_type='nll'):
	if rmtpp_type=='nll':
		model_rmtpp = models['pure_hierarchical_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = models['pure_hierarchical_mse']
	else:
		assert False, "rmtpp_type must be nll or mse"

	model_check = (model_rmtpp is not None)
	assert model_check, "run_pure_hierarchical_infer requires pure_hierarchical model"

	[count_test_in_counts, count_test_out_counts, count_test_out_binend,
	event_test_in_lasttime, event_test_in_gaps, count_test_normm, count_test_norms,
	event_test_norma, event_test_normd] = test_data

	[comp_test_in_gaps, _, _, comp_test_norma, comp_test_normd] =  test_data_comp
	
	enc_len = args.enc_len
	comp_enc_len = args.comp_enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	comp_bin_sz = args.comp_bin_sz
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None


	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	test_data_input_gaps_bin_comp = comp_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()


	t_e_plus = count_test_out_binend[:,-1]
	all_gaps_pred, all_times_pred, _ = simulate_hierarchical(model_rmtpp,
												test_data_init_time,
												test_data_input_gaps_bin_comp,
												t_e_plus,
												(comp_test_norma,
												comp_test_normd),
												prev_hidden_state=next_hidden_state)

	all_times_pred = tf.expand_dims(all_times_pred, axis=-1).numpy()
	return None, all_times_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run count_only model 
# Another baseline whose event prediction is entirely based on 
# count predicted from count model, 
# Gaps are generated uniformly to satisfy count constraints.
def run_count_only_model(args, models, data, test_data):
	model_cnt = models['count_model']
	model_check = (model_cnt is not None)
	assert model_check, "run_count_only_model requires count model"

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(count_test_in_counts, count_test_in_feats)
	if len(test_predictions_norm_cnt) == 2:
		count_all_means_pred = test_predictions_norm_cnt[0]
		count_all_sigms_pred = test_predictions_norm_cnt[1][1]


	test_predictions_cnt = utils.denormalize_data(count_all_means_pred, count_test_normm, count_test_norms)
	count_all_sigms_pred = utils.denormalize_data_stddev(count_all_sigms_pred, count_test_normm, count_test_norms)
	event_count_preds_cnt = np.clip(np.round(test_predictions_cnt), 1.0, np.inf).astype(np.int32)

	#event_count_preds_cnt = count_test_out_counts.astype(np.int32)
	event_count_preds_true = count_test_out_counts

	all_times_pred_lst = list()
	all_types_pred_lst = list()
	for batch_idx in range(event_count_preds_cnt.shape[0]):
		times_pred_per_bin_lst=list()
		batch_types_pred = list()
		old_last_gap = np.random.uniform(low=0.0, high=1.0)*np.random.uniform(low=0.0, high=1.0)
		for dec_idx in range(event_count_preds_cnt.shape[1]):
			actual_bin_start = count_test_out_binend[batch_idx,dec_idx]-bin_size
			actual_bin_end = count_test_out_binend[batch_idx,dec_idx]

			rand_uniform_gaps = np.random.uniform(low=0.0, high=1.0, size=event_count_preds_cnt[batch_idx,dec_idx]+1)
			rand_uniform_gaps[0] = old_last_gap

			last_gap = rand_uniform_gaps[-1]
			rand_uniform_gaps[-1] *= np.random.uniform(low=0.0, high=1.0)
			old_last_gap = last_gap - rand_uniform_gaps[-1]
			bin_start = np.array([0.0])
			bin_end = np.array([np.sum(rand_uniform_gaps)])
			rand_uniform_gaps = np.cumsum(rand_uniform_gaps)

			times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
							 (rand_uniform_gaps - bin_start)) + actual_bin_start
			
			times_pred_per_bin_lst.append(times_pred_for_bin_scaled[:-1])
			batch_types_pred.append(np.ones_like(times_pred_for_bin_scaled[:-1]))
		all_times_pred_lst.append(times_pred_per_bin_lst)
		all_types_pred_lst.append(batch_types_pred)
	all_times_pred = np.array(all_times_pred_lst)

	all_types_pred_flatten = []
	for seq in all_types_pred_lst:
		seq = [s.tolist() for s in seq]
		all_types_pred_flatten.append(np.array(flatten(seq)))
	all_types_pred = np.array(all_types_pred_flatten)

	all_times_pred_flatten = []
	for seq in all_times_pred:
		seq = [s.tolist() for s in seq]
		all_times_pred_flatten.append(np.array(flatten(seq)))
	all_times_pred = np.array(all_times_pred_flatten)

	all_means_pred = []
	for seq, beg_time in zip(all_times_pred, count_test_out_binend[:, 0] - args.bin_size):
		all_means_pred.append(
			utils.normalize_avg_given_param(
				seq - np.concatenate([beg_time, seq[:-1]]),
				event_test_norma, event_test_normd
			)
		)
	all_means_pred = np.array(all_means_pred)

	all_sigms_pred = []
	for seq in all_means_pred:
		all_sigms_pred.append(np.ones_like(seq)*1e-6)
	all_sigms_pred = np.array(all_sigms_pred)

	event_dist_params = [all_means_pred, all_sigms_pred]
	count_dist_params = [event_count_preds_cnt, count_all_sigms_pred]
	return (
		event_count_preds_cnt, all_times_pred, all_types_pred,
		event_dist_params, count_dist_params,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run hawkes_model model 
def run_hawkes_model(args, hawkes_timestamps_pred, data=None, test_data=None):

	if data is None:
		[count_test_in_counts, count_test_in_feats, count_test_out_counts, count_test_out_binend,
		 event_test_in_lasttime, event_test_in_gaps, event_test_in_feats,
		 count_test_normm, count_test_norms,
		 event_test_norma, event_test_normd] = test_data
	else:
		[event_test_in_gaps, event_test_in_feats, count_test_out_binend,
		 event_test_in_lasttime, 
		 event_test_norma, event_test_normd] = test_data

	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	all_times_pred = list()
	for batch_idx in range(len(count_test_out_binend)):
		batch_times_pred = list()
		for dec_idx in range(len(count_test_out_binend[batch_idx])):
			test_start_idx = bisect_right(hawkes_timestamps_pred, count_test_out_binend[batch_idx,dec_idx,0]-bin_size)
			test_end_idx = bisect_right(hawkes_timestamps_pred, count_test_out_binend[batch_idx,dec_idx,0])
			batch_times_pred.append(hawkes_timestamps_pred[test_start_idx:test_end_idx])
		all_times_pred.append(batch_times_pred)
	all_times_pred = np.array(all_times_pred)

	all_bins_count_pred_lst = list()
	for dec_idx in range(dec_len):
		t_b_plus = count_test_out_binend[:,dec_idx] - bin_size
		t_e_plus = count_test_out_binend[:,dec_idx]
		all_bins_count_pred_lst.append(np.array(count_events(all_times_pred[:,dec_idx], t_b_plus, t_e_plus)))
	all_bins_count_pred = np.array(all_bins_count_pred_lst).T

	return all_bins_count_pred, all_times_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp_count model with optimized gaps generated from rmtpp simulation untill 
# all events of bins generated then scale , each bin may have events
# whose count generated by count model because of optimization
def run_rmtpp_count_with_optimization(args, query_models, data, test_data):
	#model_cnt, model_rmtpp, _ = query_models
	model_cnt = query_models['count_model']
	model_rmtpp = query_models['rmtpp_nll']
	model_check = (model_cnt is not None) and (model_rmtpp is not None)
	assert model_check, "run_rmtpp_count_with_optimization requires count and RMTPP model"

	[count_test_in_counts, count_test_in_feats, count_test_out_counts, count_test_out_binend,
	 event_test_in_lasttime, event_test_in_gaps, event_test_in_feats,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(count_test_in_counts, count_test_in_feats)
	if len(test_predictions_norm_cnt) == 2:
		model_cnt_distribution_params = test_predictions_norm_cnt[1]
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, count_test_normm, count_test_norms)
	event_count_preds_cnt = np.clip(np.round(test_predictions_cnt), 1.0, np.inf)
	event_count_preds_true = count_test_out_counts

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()
	
	output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_cnt, axis=-1).numpy()
	full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
	full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)

	all_gaps_pred, all_times_pred, _, _, _ = simulate_with_counter(model_rmtpp, 
											test_data_init_time, 
											test_data_input_gaps_bin,
											event_test_in_feats,
											full_cnt_event_all_bins_pred,
											(event_test_norma, 
											event_test_normd),)
	
	for batch_idx in range(len(all_gaps_pred)):
		event_past_cnt=0
		times_pred_all_bin_lst=list()
		all_times_pred_lst = list()

		gaps_before_bin = all_times_pred[batch_idx,:1] - test_data_init_time[batch_idx]
		gaps_before_bin = gaps_before_bin * np.random.uniform(size=gaps_before_bin.shape)
		next_bin_start = test_data_init_time[batch_idx] + gaps_before_bin

		for dec_idx in range(dec_len):

			times_pred_for_bin = all_times_pred[batch_idx,event_past_cnt:event_past_cnt+int(output_event_count_pred[batch_idx,dec_idx,0])]
			event_past_cnt += int(output_event_count_pred[batch_idx,dec_idx,0])

			bin_start = next_bin_start

			if event_past_cnt==0:
				test_data_init_time[batch_idx] = all_times_pred[batch_idx,0:1]
			else:
				test_data_init_time[batch_idx] = all_times_pred[batch_idx,event_past_cnt-1:event_past_cnt]

			gaps_after_bin = all_times_pred[batch_idx,event_past_cnt:event_past_cnt+1] - test_data_init_time[batch_idx]
			gaps_after_bin = gaps_after_bin * np.random.uniform(size=gaps_after_bin.shape)
			bin_end = test_data_init_time[batch_idx] + gaps_after_bin
			next_bin_start = bin_end
			
			actual_bin_start = count_test_out_binend[batch_idx,dec_idx]-bin_size
			actual_bin_end = count_test_out_binend[batch_idx,dec_idx]

			times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
							 (times_pred_for_bin - bin_start)) + actual_bin_start
			
			times_pred_all_bin_lst.append(times_pred_for_bin_scaled.numpy())
		
		all_times_pred_lst.append(times_pred_all_bin_lst)
	all_times_pred = np.array(all_times_pred_lst)
	all_times_pred_before = all_times_pred

	test_data_init_time = event_test_in_lasttime.astype(np.float32)

	_, _, _, _, input_final_state = model_rmtpp(test_data_input_gaps_bin[:,:-1,:], initial_state=None)

	all_bins_gaps_pred = list()
	all_bins_D_pred = list()
	all_bins_WT_pred = list()
	all_input_final_state = list()
	for batch_idx in range(len(all_times_pred)):
		lst = list()
		d_lst = list()
		wt_lst = list()
		final_state_lst = list()
		batch_input_final_state = input_final_state[batch_idx:batch_idx+1]

		for idx in range(len(all_times_pred[batch_idx])):
			if idx==0:
				lst.append(utils.denormalize_avg(test_data_input_gaps_bin[batch_idx,-1:,0], event_test_norma, event_test_normd))
				lst.append(all_times_pred[batch_idx,idx]-np.concatenate([test_data_init_time[batch_idx],all_times_pred[batch_idx,idx][:-1]]))
			else:
				lst.append(all_times_pred[batch_idx,idx]-np.concatenate([all_times_pred[batch_idx,idx-1][-1:],all_times_pred[batch_idx,idx][:-1]]))

		merged_lst = list()
		for each in lst:
			merged_lst += each.tolist()

		test_bin_gaps_inp = np.array(merged_lst[:-1])
		test_bin_gaps_inp = np.expand_dims(np.expand_dims(test_bin_gaps_inp, axis=0), axis=-1).astype(np.float32)
		test_bin_gaps_inp = utils.normalize_avg_given_param(test_bin_gaps_inp, event_test_norma, event_test_normd)
		_, D, WT, _, batch_input_final_state = model_rmtpp(test_bin_gaps_inp, initial_state=input_final_state[batch_idx:batch_idx+1])

		all_bins_gaps_pred.append(np.array(merged_lst[1:]).astype(np.float32))
		all_bins_D_pred.append(D[0,:,0].numpy())
		all_bins_WT_pred.append(WT[0,:,0].numpy())

	all_bins_gaps_pred = np.array(all_bins_gaps_pred)
	all_bins_D_pred = np.array(all_bins_D_pred)
	all_bins_WT_pred = np.array(all_bins_WT_pred)
	model_rmtpp_params = [all_bins_D_pred, all_bins_WT_pred]

	bin_size = args.bin_size
	count_test_out_binend = count_test_out_binend.astype(np.float32)
	all_bins_end_time = tf.squeeze(count_test_out_binend, axis=-1)
	all_bins_start_time = all_bins_end_time - bin_size
	all_bins_mid_time = (all_bins_start_time+all_bins_end_time)/2

	events_count_per_batch = output_event_count_pred_cumm
	test_data_count_normalizer = [count_test_normm, count_test_norms]
	test_data_rmtpp_normalizer = [event_test_norma, event_test_normd]

	def fractional_belongingness(all_bins_gaps_pred,
								 all_bins_mid_time,
								 test_data_init_time):

		# frac_belong = tf.zeros_like(all_bins_mid_time)
		frac_belong = list()
		whole_belong = list()
		for batch_idx in range(test_data_init_time.shape[0]):
			batch_start_time = test_data_init_time[batch_idx,0]
			batch_per_bin_times = batch_start_time + tf.cumsum(all_bins_gaps_pred[batch_idx])
			batch_bins_mid_time = tf.expand_dims(all_bins_mid_time[batch_idx], axis=0)
			batch_per_bin_times = tf.expand_dims(batch_per_bin_times, axis=1)

			batch_bin_end_time = all_bins_end_time[batch_idx]
			batch_bin_start_time = all_bins_start_time[batch_idx]
			w_belong = tf.reduce_sum(tf.cast(batch_per_bin_times>batch_bin_start_time, tf.float32) \
				* tf.cast(batch_per_bin_times<batch_bin_end_time, tf.float32) \
				* tf.cast(tf.expand_dims(all_bins_gaps_pred[batch_idx]>0.0, axis=-1), dtype=tf.float32), axis=0)
			whole_belong.append(w_belong)

			time_diff = batch_per_bin_times - batch_bins_mid_time
			boundary_diff = tf.abs(tf.expand_dims(all_bins_end_time[batch_idx], axis=0) - batch_bins_mid_time)
			#TODO: Here we have used log of time diff for overflow issues 

			#rho_b_param = -tf.math.log(0.1)/tf.abs(boundary_diff)
			rho_b_param = -tf.math.log(0.1)/(boundary_diff**2)

			#f_belong = tf.nn.softmax(-(time_diff)**2 * rho_b_param, axis=1)
			#f_belong = tf.nn.softmax(-(time_diff**2), axis=1)
			#f_belong = tf.nn.softmax(-tf.abs(time_diff) * rho_b_param, axis=1)
			#f_belong = tf.nn.softmax(-tf.abs(time_diff), axis=1)
			f_belong = tf.math.exp(-(time_diff)**2 * rho_b_param)
			#f_belong = tf.math.exp(-(time_diff)**2)
			#f_belong = tf.math.exp(-tf.math.abs(time_diff) * rho_b_param)
			#f_belong = tf.math.exp(-tf.math.abs(time_diff))

			f_belong = f_belong * tf.cast(tf.expand_dims(all_bins_gaps_pred[batch_idx]>0.0, axis=-1), dtype=tf.float32)
			frac_belong.append(tf.reduce_sum(f_belong, axis=0))
			#whole_belong.append(tf.reduce_sum(tf.cast(tf.expand_dims(all_bins_gaps_pred[batch_idx]>0.0, axis=-1), dtype=tf.float32)))
		frac_belong = tf.stack(frac_belong, axis=0)
		whole_belong = tf.stack(whole_belong, axis=0)
		return frac_belong, whole_belong

	def rmtpp_loglikelihood_loss(gaps, D, WT, events_count_per_batch):
		rmtpp_loss = 0
		D = tf.sparse.to_dense(tf.ragged.constant(D).to_sparse())
		WT = tf.sparse.to_dense(tf.ragged.constant(WT).to_sparse())

		log_lambda_ = (D + (gaps * WT))
		lambda_ = tf.exp(tf.minimum(ETH, log_lambda_))
		log_f_star = (log_lambda_
					  + one_by(WT) * tf.exp(tf.minimum(ETH, D))
					  - one_by(WT) * lambda_)
		loss = -tf.reduce_mean(tf.reduce_sum(log_f_star, axis=1)/events_count_per_batch)
		return loss

	def joint_likelihood_loss(model_cnt_distribution_params,
							  model_rmtpp_params,
							  all_bins_gaps_pred,
							  all_bins_mid_time,
							  test_data_init_time,
							  events_count_per_batch,
							  test_data_count_normalizer,
							  test_data_rmtpp_normalizer):

		model_cnt_mu = model_cnt_distribution_params[0]
		model_cnt_stddev = model_cnt_distribution_params[1]
		model_rmtpp_D = model_rmtpp_params[0]
		model_rmtpp_WT = model_rmtpp_params[1]
		count_test_normm, count_test_norms = test_data_count_normalizer
		test_norm_a, test_norm_d = test_data_rmtpp_normalizer

		frac_belong, whole_belong = fractional_belongingness(all_bins_gaps_pred,
											   all_bins_mid_time,
											   test_data_init_time)
		count_loss_fn = models.NegativeLogLikelihood_CountModel(model_cnt_distribution_params, 'Gaussian')
		estimated_count_norm = utils.normalize_data_given_param(whole_belong, count_test_normm, count_test_norms)
		count_loss = count_loss_fn(estimated_count_norm, None)

		all_bins_gaps_pred = utils.normalize_avg_given_param(all_bins_gaps_pred, test_norm_a, test_norm_d)
		rmtpp_loss = rmtpp_loglikelihood_loss(all_bins_gaps_pred, model_rmtpp_D, model_rmtpp_WT, events_count_per_batch)

		return count_loss + rmtpp_loss, count_loss, rmtpp_loss

	class OPT(tf.keras.Model):
		def __init__(self,
					 model_cnt_distribution_params,
					 model_rmtpp_params,
					 likelihood_fn,
					 all_bins_gaps_pred,
					 all_bins_mid_time,
					 test_data_init_time,
					 events_count_per_batch,
					 test_data_count_normalizer,
					 test_data_rmtpp_normalizer,
					 name='opt',
					 **kwargs):
			super(OPT, self).__init__(name=name, **kwargs)
			
			gaps = tf.sparse.to_dense(tf.ragged.constant(all_bins_gaps_pred).to_sparse())
			self.gaps = tf.Variable(gaps, name='gaps', trainable=True)

			self.model_cnt_distribution_params = model_cnt_distribution_params
			self.model_rmtpp_params = model_rmtpp_params
			self.likelihood_fn = likelihood_fn
			self.all_bins_mid_time = all_bins_mid_time
			self.test_data_init_time = test_data_init_time
			self.events_count_per_batch = events_count_per_batch
			self.test_data_count_normalizer = test_data_count_normalizer
			self.test_data_rmtpp_normalizer = test_data_rmtpp_normalizer

		def __call__(self):
			return self.likelihood_fn(self.model_cnt_distribution_params,
									  self.model_rmtpp_params,
									  self.gaps,
									  self.all_bins_mid_time,
									  self.test_data_init_time,
									  self.events_count_per_batch,
									  self.test_data_count_normalizer,
									  self.test_data_rmtpp_normalizer)


	def optimize_gaps(model_cnt_distribution_params,
					  model_rmtpp_params,
					  joint_likelihood_loss,
					  all_bins_gaps_pred,
					  all_bins_mid_time,
					  test_data_init_time,
					  events_count_per_batch,
					  test_data_count_normalizer,
					  test_data_rmtpp_normalizer):


		model = OPT(model_cnt_distribution_params,
					model_rmtpp_params,
					joint_likelihood_loss,
					all_bins_gaps_pred,
					all_bins_mid_time,
					test_data_init_time,
					events_count_per_batch,
					test_data_count_normalizer,
					test_data_rmtpp_normalizer)

		#print(model.variables)

		optimizer = keras.optimizers.Adam(args.learning_rate)
	
		opt_losses = list()
		all_bins_gaps_pred = model.gaps.numpy()
		prev_nll, prev_count_loss, prev_rmtpp_loss = model()
		print('Loss before optimization:', prev_nll)
		nll = prev_nll
		e = 0
		while nll<=prev_nll:
			with tf.GradientTape() as tape:
				all_bins_gaps_pred = model.gaps.numpy()
				prev_nll = nll

				nll, count_loss, rmtpp_loss = model()
				grads = tape.gradient(nll, model.trainable_weights)
				optimizer.apply_gradients(zip(grads, model.trainable_weights))


				print('Itr ', e, ':', nll.numpy(), count_loss.numpy(), rmtpp_loss.numpy())
				e += 1
				opt_losses.append(nll)
				#if e%100==0:
				#    import ipdb
				#    ipdb.set_trace()
		
		print('Loss after optimization:', model())
	
		# Shape: list of 92 different length tensors
		return all_bins_gaps_pred
	
	
	all_bins_gaps_pred = optimize_gaps(model_cnt_distribution_params,
									   model_rmtpp_params,
									   joint_likelihood_loss,
									   all_bins_gaps_pred,
									   all_bins_mid_time,
									   test_data_init_time,
									   events_count_per_batch,
									   test_data_count_normalizer,
									   test_data_rmtpp_normalizer)

	all_times_pred = (test_data_init_time + tf.cumsum(all_bins_gaps_pred, axis=1)) * tf.cast(all_bins_gaps_pred>0., tf.float32)
	all_times_pred = all_times_pred.numpy()
	all_times_pred = np.array([seq[:int(cnt)] for seq, cnt in zip(all_times_pred, events_count_per_batch)])
	all_times_pred = np.expand_dims(all_times_pred, axis=1)

	return None, all_times_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp_count model with optimized gaps generated from rmtpp simulation untill 
# all events of bins generated then rescale
# For each possible count, find the best rmtpp_loss for that many gaps 
# by running optimization
# Select the final count as the count that produces best rmtpp_loss across
# all counts
def run_rmtpp_with_optimization_fixed_cnt(args, query_models, data, test_data):

	[count_test_in_counts, count_test_in_feats, count_test_out_counts, count_test_out_binend,
	 event_test_in_lasttime, event_test_in_gaps, event_test_in_feats,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	count_test_out_binend = count_test_out_binend.astype(np.float32)
	all_bins_end_time = tf.squeeze(count_test_out_binend, axis=-1)

	def rmtpp_loglikelihood_loss(gaps, D, WT, events_count_per_batch):
		rmtpp_loss = 0
		D = tf.sparse.to_dense(tf.ragged.constant(D).to_sparse())
		WT = tf.sparse.to_dense(tf.ragged.constant(WT).to_sparse())

		log_lambda_ = (D + (gaps * WT))
		lambda_ = tf.exp(tf.minimum(ETH, log_lambda_))
		log_f_star = (log_lambda_
					  + one_by(WT) * tf.exp(tf.minimum(ETH, D))
					  - one_by(WT) * lambda_)
		loss = -tf.reduce_mean(tf.reduce_sum(log_f_star, axis=1)/events_count_per_batch)
		return loss

	def joint_likelihood_loss(model_cnt_distribution_params,
							  model_rmtpp_params,
							  all_bins_gaps_pred,
							  all_bins_mid_time,
							  test_data_init_time,
							  events_count_per_batch,
							  test_data_count_normalizer,
							  test_data_rmtpp_normalizer):

		model_cnt_mu = model_cnt_distribution_params[0]
		model_cnt_stddev = model_cnt_distribution_params[1]
		model_rmtpp_D = model_rmtpp_params[0]
		model_rmtpp_WT = model_rmtpp_params[1]
		count_test_normm, count_test_norms = test_data_count_normalizer
		test_norm_a, test_norm_d = test_data_rmtpp_normalizer

		boundary_loss = []
		for batch_idx in range(len(all_bins_end_time)):
			g_seq = all_bins_gaps_pred[idx]
			cnt = events_count_per_batch[batch_idx]
			end_time = all_bins_end_time[batch_idx]
			if tf.cumsum(g_seq)[-1]>end_time:
				bl = tf.nn.sigmoid(g_seq[cnt-1]-end_time) - tf.nn.sigmoid(g_seq[cnt]-end_time)
				boundary_loss.append(bl)
				import ipdb
				ipdb.set_trace()
		boundary_loss = tf.reduce_sum(boundary_loss)
		all_bins_gaps_pred = utils.normalize_avg_given_param(all_bins_gaps_pred, test_norm_a, test_norm_d)
		rmtpp_loss = rmtpp_loglikelihood_loss(all_bins_gaps_pred, model_rmtpp_D, model_rmtpp_WT, events_count_per_batch)
		#import ipdb
		#ipdb.set_trace()

		return rmtpp_loss + boundary_loss, rmtpp_loss, boundary_loss

	class OPT(tf.keras.Model):
		def __init__(self,
					 model_cnt_distribution_params,
					 model_rmtpp_params,
					 likelihood_fn,
					 all_bins_gaps_pred,
					 all_bins_mid_time,
					 test_data_init_time,
					 events_count_per_batch,
					 test_data_count_normalizer,
					 test_data_rmtpp_normalizer,
					 name='opt',
					 **kwargs):
			super(OPT, self).__init__(name=name, **kwargs)
			
			gaps = tf.sparse.to_dense(tf.ragged.constant(all_bins_gaps_pred).to_sparse())
			self.gaps = tf.Variable(gaps, name='gaps', trainable=True)

			self.model_cnt_distribution_params = model_cnt_distribution_params
			self.model_rmtpp_params = model_rmtpp_params
			self.likelihood_fn = likelihood_fn
			self.all_bins_mid_time = all_bins_mid_time
			self.test_data_init_time = test_data_init_time
			self.events_count_per_batch = events_count_per_batch
			self.test_data_count_normalizer = test_data_count_normalizer
			self.test_data_rmtpp_normalizer = test_data_rmtpp_normalizer

		def __call__(self):
			return self.likelihood_fn(self.model_cnt_distribution_params,
									  self.model_rmtpp_params,
									  self.gaps,
									  self.all_bins_mid_time,
									  self.test_data_init_time,
									  self.events_count_per_batch,
									  self.test_data_count_normalizer,
									  self.test_data_rmtpp_normalizer)


	def optimize_gaps(model_cnt_distribution_params,
					  model_rmtpp_params,
					  joint_likelihood_loss,
					  all_bins_gaps_pred,
					  all_bins_mid_time,
					  test_data_init_time,
					  events_count_per_batch,
					  test_data_count_normalizer,
					  test_data_rmtpp_normalizer):

		
		model = OPT(model_cnt_distribution_params,
					model_rmtpp_params,
					joint_likelihood_loss,
					all_bins_gaps_pred,
					all_bins_mid_time,
					test_data_init_time,
					events_count_per_batch,
					test_data_count_normalizer,
					test_data_rmtpp_normalizer)

		#print(model.variables)

		optimizer = keras.optimizers.Adam(args.learning_rate)
	
		opt_losses = list()
		all_bins_gaps_pred = model.gaps.numpy()
		prev_nll, rmtpp_nll, boundary_nll = model()
		print('Loss before optimization:', prev_nll)
		nll = prev_nll
		e = 0
		#while nll<=prev_nll:
		while e<=200:
			with tf.GradientTape() as tape:
				all_bins_gaps_pred = model.gaps.numpy()
				prev_nll = nll

				nll, rmtpp_nll, boundary_nll = model()
				grads = tape.gradient(nll, model.trainable_weights)
				optimizer.apply_gradients(zip(grads, model.trainable_weights))


				print('Itr ', e, ':', nll.numpy(), rmtpp_nll.numpy(), boundary_nll.numpy())
				e += 1
				opt_losses.append(nll)
				#if e%100==0:
				#    import ipdb
				#    ipdb.set_trace()
		
		print('Loss after optimization:', model())
	
		# Shape: list of 92 different length tensors
		return all_bins_gaps_pred, prev_nll


	num_counts = 2 # Number of counts to take before and after mean
	#model_cnt, model_rmtpp, _ = query_models
	model_cnt = query_models['count_model']
	model_rmtpp = query_models['rmtpp_nll']
	model_check = (model_cnt is not None) and (model_rmtpp is not None)
	assert model_check, "run_rmtpp_count_with_optimization requires count and RMTPP model"


	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(count_test_in_counts, count_test_in_feats)
	if len(test_predictions_norm_cnt) == 2:
		model_cnt_distribution_params = test_predictions_norm_cnt[1]
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, count_test_normm, count_test_norms)
	event_count_preds_cnt = np.clip(np.round(test_predictions_cnt), 1.0, np.inf)
	event_count_preds_true = count_test_out_counts

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()
	
	output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_cnt, axis=-1).numpy()
	full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
	full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)
	full_cnt_event_all_bins_pred += num_counts

	all_gaps_pred, all_times_pred, _, _, _ = simulate_with_counter(model_rmtpp, 
											test_data_init_time, 
											test_data_input_gaps_bin,
											event_test_in_feats,
											full_cnt_event_all_bins_pred,
											(event_test_norma, 
											event_test_normd),)
	
	all_times_pred_simu = all_times_pred
	count2loss = dict()
	count2pred = dict()
	for nc in range(-num_counts, num_counts):
		all_times_pred_lst = list()
		output_event_count_curr = output_event_count_pred + nc
		test_data_init_time = event_test_in_lasttime.astype(np.float32)
		all_times_pred = all_times_pred_simu
		for batch_idx in range(len(all_gaps_pred)):
			event_past_cnt=0
			times_pred_all_bin_lst=list()
	
			gaps_before_bin = all_times_pred[batch_idx,:1] - test_data_init_time[batch_idx]
			gaps_before_bin = gaps_before_bin * np.random.uniform(size=gaps_before_bin.shape)
			next_bin_start = test_data_init_time[batch_idx] + gaps_before_bin
	
			for dec_idx in range(dec_len):
	
				times_pred_for_bin = all_times_pred[batch_idx,event_past_cnt:event_past_cnt+int(output_event_count_curr[batch_idx,dec_idx,0])]
				event_past_cnt += int(output_event_count_curr[batch_idx,dec_idx,0])
	
				bin_start = next_bin_start
	
				if event_past_cnt==0:
					test_data_init_time[batch_idx] = all_times_pred[batch_idx,0:1]
				else:
					test_data_init_time[batch_idx] = all_times_pred[batch_idx,event_past_cnt-1:event_past_cnt]
	
				gaps_after_bin = all_times_pred[batch_idx,event_past_cnt:event_past_cnt+1] - test_data_init_time[batch_idx]
				gaps_after_bin = gaps_after_bin * np.random.uniform(size=gaps_after_bin.shape)
				bin_end = test_data_init_time[batch_idx] + gaps_after_bin
				next_bin_start = bin_end
				
				actual_bin_start = count_test_out_binend[batch_idx,dec_idx]-bin_size
				actual_bin_end = count_test_out_binend[batch_idx,dec_idx]
	
				times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
								(times_pred_for_bin - bin_start)) + actual_bin_start
				
				times_pred_all_bin_lst.append(times_pred_for_bin_scaled.numpy())
			
			all_times_pred_lst.append(times_pred_all_bin_lst)
		all_times_pred = np.array(all_times_pred_lst)
		all_times_pred_before = all_times_pred
	
		test_data_init_time = event_test_in_lasttime.astype(np.float32)
	
		_, _, _, _, input_final_state = model_rmtpp(test_data_input_gaps_bin[:,:-1,:], initial_state=None)
	
		all_bins_gaps_pred = list()
		all_bins_D_pred = list()
		all_bins_WT_pred = list()
		all_input_final_state = list()
		for batch_idx in range(len(all_times_pred)):
			lst = list()
			d_lst = list()
			wt_lst = list()
			final_state_lst = list()
			batch_input_final_state = input_final_state[batch_idx:batch_idx+1]
	
			for idx in range(len(all_times_pred[batch_idx])):
				if idx==0:
					lst.append(utils.denormalize_avg(test_data_input_gaps_bin[batch_idx,-1:,0], event_test_norma, event_test_normd))
					lst.append(all_times_pred[batch_idx,idx]-np.concatenate([test_data_init_time[batch_idx],all_times_pred[batch_idx,idx][:-1]]))
				else:
					lst.append(all_times_pred[batch_idx,idx]-np.concatenate([all_times_pred[batch_idx,idx-1][-1:],all_times_pred[batch_idx,idx][:-1]]))
	
			merged_lst = list()
			for each in lst:
				merged_lst += each.tolist()
	
			test_bin_gaps_inp = np.array(merged_lst[:-1])
			test_bin_gaps_inp = np.expand_dims(np.expand_dims(test_bin_gaps_inp, axis=0), axis=-1).astype(np.float32)
			test_bin_gaps_inp = utils.normalize_avg_given_param(test_bin_gaps_inp, event_test_norma, event_test_normd)
			_, D, WT, _, batch_input_final_state = model_rmtpp(test_bin_gaps_inp, initial_state=input_final_state[batch_idx:batch_idx+1])
	
			all_bins_gaps_pred.append(np.array(merged_lst[1:]).astype(np.float32))
			all_bins_D_pred.append(D[0,:,0].numpy())
			all_bins_WT_pred.append(WT[0,:,0].numpy())
	
		all_bins_gaps_pred = np.array(all_bins_gaps_pred)
		all_bins_D_pred = np.array(all_bins_D_pred)
		all_bins_WT_pred = np.array(all_bins_WT_pred)
		model_rmtpp_params = [all_bins_D_pred, all_bins_WT_pred]
	
		bin_size = args.bin_size
		count_test_out_binend = count_test_out_binend.astype(np.float32)
		all_bins_end_time = tf.squeeze(count_test_out_binend, axis=-1)
		all_bins_start_time = all_bins_end_time - bin_size
		all_bins_mid_time = (all_bins_start_time+all_bins_end_time)/2
	
		events_count_per_batch = output_event_count_curr
		test_data_count_normalizer = [count_test_normm, count_test_norms]
		test_data_rmtpp_normalizer = [event_test_norma, event_test_normd]

	
	
		all_bins_gaps_pred, nc_loss = optimize_gaps(model_cnt_distribution_params,
										model_rmtpp_params,
										joint_likelihood_loss,
										all_bins_gaps_pred,
										all_bins_mid_time,
										test_data_init_time,
										events_count_per_batch,
										test_data_count_normalizer,
										test_data_rmtpp_normalizer)
	
		all_times_pred_nc = (test_data_init_time + tf.cumsum(all_bins_gaps_pred, axis=1)) * tf.cast(all_bins_gaps_pred>0., tf.float32)
		all_times_pred_nc = all_times_pred_nc.numpy()
		all_times_pred_nc = np.array([seq[:int(cnt)] for seq, cnt in zip(all_times_pred_nc, events_count_per_batch)])
		all_times_pred_nc = np.expand_dims(all_times_pred_nc, axis=1)

		print('nc:', nc, 'loss:', nc_loss)

		count2loss[nc] = nc_loss
		count2pred[nc] = all_times_pred_nc

	best_nc, best_all_times_pred = min(count2loss.items(), key=itemgetter(1))
	best_all_times_pred = count2pred[best_nc]
	import ipdb
	ipdb.set_trace()
	print('Best nc:', best_nc, 'Best all_times_pred', best_all_times_pred)

	return None, best_all_times_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp with optimized gaps generated from rmtpp simulation untill 
# all events of bins generated then rescale
# For each possible count, find the best rmtpp_loss for that many gaps 
# by using solver library
# Select the final count as the count that produces best rmtpp_loss across
# all counts
def run_rmtpp_optimizer_model(
	args,
	query_models,
	data,
	test_data,
	test_data_out_gaps_bin,
	dataset,
	rmtpp_type='nll'
):

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	model_cnt = query_models['count_model']
	if rmtpp_type=='nll':
		model_rmtpp = query_models['rmtpp_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = query_models['rmtpp_mse']
		if args.extra_var_model:
			rmtpp_var_model = query_models['rmtpp_var_model']
	elif rmtpp_type=='mse_var':
		model_rmtpp = query_models['rmtpp_mse_var']
	else:
		assert False, "rmtpp_type must be nll or mse"
	
	model_check = (model_cnt is not None) and (model_rmtpp is not None)
	assert model_check, "run_rmtpp_count_with_optimization requires count and RMTPP model"

	count_test_out_binend = count_test_out_binend.astype(np.float32)
	all_bins_end_time = tf.squeeze(count_test_out_binend, axis=-1)

	def rmtpp_loglikelihood_loss(gaps, D, WT, events_count_per_batch):

		rmtpp_loss = 0

		log_lambda_ = (D + cp.multiply(gaps, WT))
		lambda_ = cp.exp(log_lambda_)
		log_f_star = (log_lambda_
					  + cp.multiply(1./WT, cp.exp(D))
					  - cp.multiply(1./WT, lambda_))
		#loss = -tf.reduce_mean(tf.reduce_sum(log_f_star, axis=1)/events_count_per_batch)
		loss = log_f_star
		return loss

	def mse_loss(gaps, D, WT, events_count_per_batch):
		#return cp.multiply(cp.power(gaps-D, 2), cp.power(WT, -2))
		return cp.power(gaps-D, 2)

	def mse_loglikelihood_loss(gaps, D, WT, events_count_per_batch):
		return -(cp.log(1/(((2*np.pi)**0.5)*WT)) - (((gaps - D)**2) / (2*(WT)**2)))

	def optimize_gaps(model_rmtpp_params,
					  rmtpp_loglikelihood_loss,
					  model_cnt_distribution_params,
					  nc,
					  all_bins_gaps_pred,
					  all_bins_end_time,
					  test_data_init_time,
					  events_count_per_batch,
					  test_data_count_normalizer,
					  test_data_rmtpp_normalizer,
					  test_data_out_gaps_bin_batch,
					  unconstrained=False,
					  gaps_uc=None):

		gaps = cp.Variable(all_bins_gaps_pred.shape)
		gaps.value = all_bins_gaps_pred

		D, WT = model_rmtpp_params[0], model_rmtpp_params[1]
		#print(all_bins_gaps_pred.shape, D.shape, WT.shape)

		if rmtpp_type=='nll':
			opt_loss = -cp.sum(rmtpp_loglikelihood_loss(gaps, D, WT, events_count_per_batch))/all_bins_gaps_pred.shape[1]
		elif rmtpp_type=='mse':
			if args.extra_var_model:
				opt_loss = cp.sum(mse_loglikelihood_loss(gaps, D, WT, events_count_per_batch))/all_bins_gaps_pred.shape[1]
			else:
				WT = np.ones_like(WT)
				opt_loss = cp.sum(mse_loss(gaps, D, WT, events_count_per_batch))/all_bins_gaps_pred.shape[1]
		elif rmtpp_type=='mse_var':
			opt_loss = cp.sum(mse_loglikelihood_loss(gaps, D, WT, events_count_per_batch))/all_bins_gaps_pred.shape[1]

		rmtpp_loss_cont = opt_loss.value

		objective = cp.Minimize(opt_loss)


		test_norm_a, test_norm_d = test_data_rmtpp_normalizer
		init_end_diff = all_bins_end_time-test_data_init_time
		init_end_diff_norm = utils.normalize_avg_given_param(
			init_end_diff,
			test_norm_a,
			test_norm_d
		)
		first_gap_lb = utils.normalize_avg_given_param(
			(all_bins_end_time-args.bin_size)-test_data_init_time,
			test_norm_a,
			test_norm_d
		)
		constraints = [cp.sum(gaps[0, :nc])<=init_end_diff_norm-1e-2,
					   cp.sum(gaps[0, :nc+1])>=init_end_diff_norm+1e-2,
					   gaps[0, 0]>=first_gap_lb+1e-2,
					   gaps>=1e-3]

		if unconstrained==False and args.use_ratio_constraints:
			assert gaps_uc is not None
			for j in range(nc-1):
				constraints.append(
					(
						(cp.sum(gaps[0, :j+1])*cp.sum(gaps_uc[0, :j+2]))
						== (cp.sum(gaps[0, :j+2])*cp.sum(gaps_uc[0, :j+1]))
					)
				)

		if unconstrained:
			prob = cp.Problem(objective)
		else:
			prob = cp.Problem(objective, constraints)

		try:
			rmtpp_loss = prob.solve(warm_start=True)
		except cp.error.SolverError:
			rmtpp_loss = prob.solve(solver='SCS', warm_start=True)
		#rmtpp_loss = prob.solve(warm_start=True, solver=cp.OSQP)

		rmtpp_loss_opt = rmtpp_loss
		#if gaps.value is None:
		#	gaps.value = all_bins_gaps_pred
		#	rmtpp_loss = opt_loss.value

		count_test_normm, count_test_norms = test_data_count_normalizer
		nc_norm = utils.normalize_data_given_param(nc, count_test_normm, count_test_norms)
		mu, sigma = model_cnt_distribution_params[0], model_cnt_distribution_params[1]
		count_loss = -tfp.distributions.Normal(
				mu, sigma, validate_args=False, allow_nan_stats=True,
				name='Normal'
			).log_prob(nc_norm)
		count_loss = np.sum(count_loss)

		#import ipdb
		#ipdb.set_trace()


		if gaps.value is None:
			import ipdb
			ipdb.set_trace()
		#loss = rmtpp_loss + count_loss
		loss = rmtpp_loss + count_loss

		all_bins_gaps_pred = gaps.value[0:1, :nc]
		#print('Loss after optimization:', loss)
	
		# Shape: list of 92 different length tensors
		return (
			all_bins_gaps_pred,
			loss,
			rmtpp_loss_opt,
			rmtpp_loss_cont,
			np.array(count_loss),
		)


	num_counts = args.opt_num_counts # Number of counts to take before and after mean
	#model_cnt, model_rmtpp, _ = query_models
	# model_rmtpp = query_models['rmtpp_nll']


	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(count_test_in_counts, count_test_in_feats)
	if len(test_predictions_norm_cnt) == 2:
		model_cnt_distribution_params = test_predictions_norm_cnt[1]
		# test_predictions_norm_cnt = test_predictions_norm_cnt[0]
		test_predictions_norm_cnt = model_cnt_distribution_params[0]
		test_predictions_norm_stddev = model_cnt_distribution_params[1]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, count_test_normm, count_test_norms)
	test_predictions_cnt = np.clip(np.round(test_predictions_cnt), 1.0, np.inf).astype(np.int32)
	test_predictions_stddev = utils.denormalize_data_stddev(test_predictions_norm_stddev, count_test_normm, count_test_norms)
	event_count_preds_cnt = np.clip(np.round(test_predictions_cnt), 1.0, np.inf)
	event_count_preds_stddev = np.round(test_predictions_stddev)
	event_count_preds_stddev = np.maximum(event_count_preds_stddev, 1.) # stddev should not be less than 1.
	event_count_preds_true = count_test_out_counts

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()
	
	#output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_true, axis=-1).numpy() + 2.
	output_event_count_pred_cumm = event_count_preds_cnt[:, 0] + event_count_preds_stddev[:, 0]
	#output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_cnt, axis=-1).numpy() + tf.reduce_sum(event_count_preds_stddev, axis=-1).numpy()

	#import ipdb
	#ipdb.set_trace()

	full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
	full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)
	#full_cnt_event_all_bins_pred += num_counts
	#full_cnt_event_all_bins_pred += event_count_preds_stddev


	def get_optimized_gaps(
		batch_idx,
		dec_idx,
		curr_cnt,
		max_cnt,
		best_past_cnt,
		bin_start,
		batch_times_pred,
		batch_types_pred,
		unconstrained=False,
		gaps_uc=None
	):

		event_cnt = best_past_cnt + int(max_cnt)+1
		output_event_count_curr = np.zeros_like(output_event_count_pred) + max_cnt+1

		batch_bin_curr_cnt_times_pred = all_times_pred_simu[batch_idx,best_past_cnt:event_cnt]
		batch_bin_curr_cnt_types_pred = all_types_pred_simu[batch_idx,best_past_cnt:event_cnt]
		#batch_bin_curr_cnt_types_pred = np.squeeze(batch_bin_curr_cnt_types_pred, axis=-1)

		batch_bin_curr_cnt_times_pred_scaled = batch_bin_curr_cnt_times_pred.numpy()


		if dec_idx == 0:
			batch_temp_times_pred = [batch_bin_curr_cnt_times_pred_scaled]
			batch_temp_types_pred = [batch_bin_curr_cnt_types_pred]
		else:
			batch_temp_times_pred = batch_times_pred[:dec_idx]
			batch_temp_types_pred = batch_types_pred[:dec_idx]
			batch_temp_times_pred.append(batch_bin_curr_cnt_times_pred_scaled)
			batch_temp_types_pred.append(batch_bin_curr_cnt_types_pred)


		if args.no_rescale_rmtpp_params:
			D = D_pred[batch_idx:batch_idx+1]
			WT = WT_pred[batch_idx:batch_idx+1]
		else:
			_, D, WT, _, batch_input_final_state = model_rmtpp(
				batch_temp_gaps_pred,
				batch_temp_feats_pred,
				initial_state=input_final_state
			)

		if args.extra_var_model and rmtpp_type=='mse':
			batch_temp_times_pred_flatten \
				= np.concatenate(batch_temp_times_pred)

			batch_temp_times_pred_flatten = np.expand_dims(
				batch_temp_times_pred_flatten,
				axis=0
			)
			bin_ids = np.sum(
				(batch_temp_times_pred_flatten<count_test_out_binend[batch_idx]),
				axis=0
			)

			num_bins = args.out_bin_sz
			num_pos = args.num_pos
			bin_ids = (num_bins+1) - bin_ids
			bin_ids = np.where(bin_ids<=num_bins, bin_ids, np.zeros_like(bin_ids))
		
			bin_ranks = np.sum(
				np.stack(
					[
						np.cumsum(bin_ids==curr_b)*(bin_ids==curr_b) for curr_b in range(1, num_bins+1)
					],
					axis=1
				),
				axis=-1
			)
			grp_ids = (bin_ranks-1)//num_pos + 1
			pos_ids = (bin_ranks-1)%num_pos + 1

			rmtpp_var_input = tf.cumsum(batch_temp_gaps_pred_unnorm, axis=1)
			WT = rmtpp_var_model(rmtpp_var_input, bin_ids, grp_ids, pos_ids)
			WT = tf.expand_dims(WT, axis=0)

		batch_bin_curr_cnt_D_pred = D[:, best_past_cnt:event_cnt, 0].numpy()
		batch_bin_curr_cnt_WT_pred = WT[:, best_past_cnt:event_cnt, 0].numpy()

		model_rmtpp_params = [batch_bin_curr_cnt_D_pred, batch_bin_curr_cnt_WT_pred]
		model_count_params = [model_cnt_distribution_params[0][batch_idx, dec_idx],
							  model_cnt_distribution_params[1][batch_idx, dec_idx]]

		bin_size = args.bin_size
		batch_bin_end_time = tf.squeeze(count_test_out_binend[batch_idx:batch_idx+1, dec_idx:dec_idx+1].astype(np.float32), axis=-1)
		batch_bin_start_time = batch_bin_end_time - bin_size
		batch_bin_mid_time = (batch_bin_start_time+batch_bin_end_time)/2.

		events_count_per_batch = output_event_count_curr[batch_idx:batch_idx+1]
		test_data_count_normalizer = [count_test_normm, count_test_norms]
		test_data_rmtpp_normalizer = [event_test_norma, event_test_normd]

		if dec_idx == 0:
			test_data_init_time_batch = test_data_init_time[batch_idx:batch_idx+1]
		else:
			test_data_init_time_batch = batch_times_pred[-1][-1:]

		test_data_out_gaps_bin_batch \
			= test_data_out_gaps_bin[batch_idx][best_past_cnt:best_past_cnt+int(curr_cnt)]

		(
			batch_bin_curr_cnt_opt_gaps_pred,
			nc_loss,
			nc_loss_opt,
			nc_loss_cont,
			nc_count_loss,
		) = optimize_gaps(model_rmtpp_params,
							rmtpp_loglikelihood_loss,
							model_count_params,
							int(curr_cnt),
							batch_bin_curr_cnt_D_pred,
							batch_bin_end_time,
							test_data_init_time_batch,
							events_count_per_batch,
							test_data_count_normalizer,
							test_data_rmtpp_normalizer,
							test_data_out_gaps_bin_batch,
							unconstrained=unconstrained,
							gaps_uc=gaps_uc)

		batch_bin_curr_cnt_opt_gaps_pred = utils.denormalize_avg(batch_bin_curr_cnt_opt_gaps_pred,
												event_test_norma,
												event_test_normd)
		batch_bin_curr_cnt_opt_types_pred = batch_bin_curr_cnt_types_pred[:int(curr_cnt)]
		batch_bin_curr_cnt_opt_sigms = batch_bin_curr_cnt_WT_pred[0, :int(curr_cnt)]

		#import ipdb
		#ipdb.set_trace()

		#batch_bin_curr_cnt_opt_times_pred \
		#	= ((test_data_init_time_batch + tf.cast(tf.cumsum(batch_bin_curr_cnt_opt_gaps_pred, axis=1), tf.float64))
		#	   * tf.cast(batch_bin_curr_cnt_opt_gaps_pred>0., tf.float64))
		batch_bin_curr_cnt_opt_gaps_pred \
			= batch_bin_curr_cnt_opt_gaps_pred * tf.cast(batch_bin_curr_cnt_opt_gaps_pred>0., tf.float64)
		batch_bin_curr_cnt_opt_times_pred \
			= (test_data_init_time_batch + tf.cast(tf.cumsum(batch_bin_curr_cnt_opt_gaps_pred, axis=1), tf.float64))
		batch_bin_curr_cnt_opt_times_pred = batch_bin_curr_cnt_opt_times_pred[0].numpy()

		return (
			batch_bin_curr_cnt_opt_times_pred,
			batch_bin_curr_cnt_opt_gaps_pred,
			batch_bin_curr_cnt_opt_types_pred,
			batch_bin_curr_cnt_opt_sigms,
			nc_loss,
			nc_loss_opt,
			nc_loss_cont,
			nc_count_loss,
		)

	# TODO:
	#	1. Concatenate optimized gaps of previous bins to input gaps (Done)
	# 	2. set t_e depending on dec_idx
	#	3. Only simulate (mu_j + sigma_j) events
	#	4. Adjust the indexing in following functions since simulation is per-example (Done)


	def resimulate(
		model_rmtpp, test_data_init_time, test_data_input_gaps_bin,
		all_times_pred, all_types_pred, normalizers, all_best_cnt, dec_idx,
	):
		print('In resimulate dec_idx:', dec_idx)
		(event_test_norma, event_test_normd) = normalizers
		input_gaps = []
		input_feats = []
		input_types = []
		init_time = []
		for idx in range(len(test_data_input_gaps_bin)): 
			times_pred = np.concatenate(all_times_pred[idx], axis=-1)
			types_pred = np.concatenate(all_types_pred[idx], axis=-1)
			batch_gaps_pred = times_pred - np.concatenate([test_data_init_time[idx], times_pred[:-1]])
			batch_gaps_pred = np.expand_dims(batch_gaps_pred, axis=-1)
			batch_gaps_pred = utils.normalize_avg_given_param(
				batch_gaps_pred,
				event_test_norma,
				event_test_normd)
			input_gaps.append(np.concatenate([test_data_input_gaps_bin[idx], batch_gaps_pred], axis=0))
			init_time.append([times_pred[-1]])
			input_feats.append(
				np.concatenate(
					[event_test_in_feats[idx],
					 get_time_features(np.expand_dims(times_pred, axis=-1))],
					axis=0))
			input_types.append(np.concatenate([event_test_in_types[idx], types_pred], axis=0))

		init_time = np.array(init_time)
		input_gaps = tf.cast(tf.sparse.to_dense(tf.ragged.constant(input_gaps).to_sparse()), tf.float32)
		input_feats = tf.cast(tf.sparse.to_dense(tf.ragged.constant(input_feats).to_sparse()), tf.float32)
		input_types = tf.cast(tf.sparse.to_dense(tf.ragged.constant(input_types).to_sparse()), tf.int64)
		#input_gaps = tf.ragged.constant(input_gaps).to_sparse()
		#input_feats = tf.ragged.constant(input_feats).to_sparse()

		output_event_count_pred_cumm = event_count_preds_cnt[:, dec_idx] + event_count_preds_stddev[:, dec_idx]

		full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
		full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)

		(
			all_gaps_pred_simu, all_times_pred_simu, all_types_pred_simu,
			D_pred, WT_pred
		) = simulate_with_counter(
			model_rmtpp, 
			init_time, 
			input_gaps,
			input_feats,
			input_types,
			full_cnt_event_all_bins_pred,
			(event_test_norma,
			event_test_normd),
		)
		#all_types_pred_simu = [np.squeeze(seq, axis=-1) for seq in all_types_pred_simu]


		all_tps_adjst_lst, all_typs_adjst_lst, D_pred_adjst_lst, WT_pred_adjst_lst = [], [], [], []
		for idx, (tps_seq, typs_seq, D_seq, WT_seq) \
			in enumerate(zip(all_times_pred_simu, all_types_pred_simu, D_pred, WT_pred)):
			tps_seq_adjst = np.concatenate([np.zeros((all_best_cnt[idx])), tps_seq])
			typs_seq_adjst = np.concatenate([np.zeros((all_best_cnt[idx]), dtype=int), typs_seq])
			D_seq_adjst = np.concatenate([np.zeros((all_best_cnt[idx], 1)), D_seq], axis=0)
			WT_seq_adjst = np.concatenate([np.zeros((all_best_cnt[idx], 1)), WT_seq], axis=0)
			all_tps_adjst_lst.append(tps_seq_adjst)
			all_typs_adjst_lst.append(typs_seq_adjst)
			D_pred_adjst_lst.append(D_seq_adjst)
			WT_pred_adjst_lst.append(WT_seq_adjst)


		maxlen = max([len(seq) for seq in all_tps_adjst_lst])
		all_tps_adjst, all_typs_adjst, D_pred_adjst, WT_pred_adjst = [], [], [], []
		for idx, (tps_seq, typs_seq, D_seq, WT_seq) \
			in enumerate(zip(all_tps_adjst_lst, all_typs_adjst_lst, D_pred_adjst_lst, WT_pred_adjst_lst)):
			tps_seq_adjst = np.concatenate([tps_seq, np.zeros((maxlen+1-len(tps_seq)))])
			typs_seq_adjst = np.concatenate([typs_seq, np.zeros((maxlen+1-len(typs_seq)), dtype=int)])
			D_seq_adjst = np.concatenate([D_seq, np.zeros((maxlen+1-len(D_seq), 1))], axis=0)
			WT_seq_adjst = np.concatenate([WT_seq, np.zeros((maxlen+1-len(WT_seq), 1))], axis=0)
			all_tps_adjst.append(tps_seq_adjst)
			all_typs_adjst.append(typs_seq_adjst)
			D_pred_adjst.append(D_seq_adjst)
			WT_pred_adjst.append(WT_seq_adjst)

		all_tps_adjst = tf.constant(all_tps_adjst)
		all_typs_adjst = tf.constant(all_typs_adjst)
		D_pred_adjst = tf.constant(D_pred_adjst)
		WT_pred_adjst = tf.constant(WT_pred_adjst)

		#import ipdb
		#ipdb.set_trace()

		return all_tps_adjst, all_typs_adjst, D_pred_adjst, WT_pred_adjst


	count_sigma = args.opt_num_counts
	all_times_pred = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_types_pred = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_counts_pred = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_sigms_pred = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_best_opt_nc_losses = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_best_cont_nc_losses = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_best_nc_count_losses = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_best_cnt = [0 for _ in range(len(test_data_input_gaps_bin))]
	for dec_idx in range(dec_len):
		#event_cnt=0
		#best_past_cnt=0

		if dec_idx == 0:
			(
				all_gaps_pred_simu, all_times_pred_simu, all_types_pred_simu,
				D_pred, WT_pred
			) = simulate_with_counter(
				model_rmtpp, 
				test_data_init_time, 
				test_data_input_gaps_bin,
				event_test_in_feats,
				event_test_in_types,
				full_cnt_event_all_bins_pred,
				(event_test_norma,
				event_test_normd),
			)
		else:
			all_times_pred_simu, all_types_pred_simu, D_pred, WT_pred = resimulate(
				model_rmtpp, test_data_init_time, test_data_input_gaps_bin,
				all_times_pred, all_types_pred,
				(event_test_norma, event_test_normd),
				all_best_cnt, dec_idx,
			)

		for batch_idx in range(len(all_times_pred_simu)):

			print(batch_idx, dec_idx)

			best_past_cnt = all_best_cnt[batch_idx]

			batch_times_pred = all_times_pred[batch_idx]
			batch_types_pred = all_types_pred[batch_idx]
			batch_best_opt_nc_losses = all_best_opt_nc_losses[batch_idx]
			batch_best_cont_nc_losses = all_best_cont_nc_losses[batch_idx]
			batch_best_nc_count_losses = all_best_nc_count_losses[batch_idx]

			clipped_stddev = np.clip(event_count_preds_stddev[batch_idx, dec_idx], 1.0, count_sigma)
			min_cnt = event_count_preds_cnt[batch_idx, dec_idx] - clipped_stddev
			min_cnt = int(np.maximum(1., min_cnt))
			max_cnt = int(event_count_preds_cnt[batch_idx, dec_idx] + clipped_stddev)


			if dec_idx == 0:
				gaps_before_bin = all_times_pred_simu[batch_idx,:1] - test_data_init_time[batch_idx]
				gaps_before_bin = gaps_before_bin * np.random.uniform(size=gaps_before_bin.shape)
				bin_start = test_data_init_time[batch_idx] + gaps_before_bin
			else:
				gaps_before_bin = (all_times_pred_simu[batch_idx, best_past_cnt]
								   - all_times_pred_simu[batch_idx, best_past_cnt-1])
				gaps_before_bin = gaps_before_bin * np.random.uniform(size=gaps_before_bin.shape)
				bin_start = all_times_pred_simu[batch_idx, best_past_cnt-1] + gaps_before_bin

			#TODO Add flag for rescaling

			if args.no_rescale_rmtpp_params:	
				# 1. Get number of peaks in the bin by solving unconstrained problem
				# 2. Change the nc_range based on num_peaks_in_bin and mu
				# 3. use gaps_uc solution if args.use_ratio_constraints is True
				(
					batch_bin_cnrr_cnt_opt_times_pred_uc, 
					batch_bin_curr_cnt_opt_gaps_pred_uc,
					batch_bin_curr_cnt_opt_types_pred,
					batch_bin_curr_cnt_opt_sigms,
					nc_loss,
					nc_loss_opt,
					nc_loss_cont,
					nc_count_loss,
				) = get_optimized_gaps(
					batch_idx,
					dec_idx,
					max_cnt,
					max_cnt,
					best_past_cnt,
					bin_start,
					batch_times_pred,
					batch_types_pred,
					unconstrained=True,
				)
				bs = count_test_out_binend[batch_idx, dec_idx] - args.bin_size
				be = count_test_out_binend[batch_idx, dec_idx]
				bs_cnt = bisect_right(batch_bin_cnrr_cnt_opt_times_pred_uc, bs)
				be_cnt = bisect_right(batch_bin_cnrr_cnt_opt_times_pred_uc, be)
				num_peaks_in_bin = np.maximum(be_cnt - bs_cnt, 1)
				if num_peaks_in_bin <= event_count_preds_cnt[batch_idx, dec_idx]:
					min_cnt = int(num_peaks_in_bin)
					max_cnt = int(event_count_preds_cnt[batch_idx, dec_idx])
				else:
					min_cnt = int(event_count_preds_cnt[batch_idx, dec_idx])
					max_cnt = int(num_peaks_in_bin)

				if args.use_ratio_constraints:
					gaps_uc = batch_bin_curr_cnt_opt_gaps_pred_uc
					gaps_uc = utils.normalize_avg_given_param(
						gaps_uc,
						event_test_norma,
						event_test_normd
					)
				else:
					gaps_uc = None
			else:
				gaps_uc = None

			#min_cnt = dataset['count_test_out_counts'][batch_idx, dec_idx]
			#max_cnt = min_cnt
			if args.current_model in ['rmtpp_mse_coopt', 'rmtpp_mse_var_coopt']:
				min_cnt = int(event_count_preds_cnt[batch_idx, dec_idx])
				max_cnt = int(event_count_preds_cnt[batch_idx, dec_idx])
			if args.search == 1:
				min_cnt = int(event_count_preds_cnt[batch_idx, dec_idx] - event_count_preds_stddev[batch_idx, dec_idx])
				max_cnt = int(event_count_preds_cnt[batch_idx, dec_idx] + event_count_preds_stddev[batch_idx, dec_idx])
			nc_range = np.arange(min_cnt, max_cnt+1)
			def linear_search(counts_range, low, high):
				nc_loss_min = np.inf
				for mid_1 in range(len(counts_range)):
					(
						batch_bin_curr_cnt_opt_times_pred_mid_1,
						batch_bin_curr_cnt_opt_gaps_pred_mid_1,
						batch_bin_curr_cnt_opt_types_pred_mid_1,
						batch_bin_curr_cnt_opt_sigms_mid_1,
						nc_loss_mid_1,
						nc_loss_mid_1_opt,
						nc_loss_mid_1_cont,
						nc_count_loss_mid_1,
					) = get_optimized_gaps(
						batch_idx,
						dec_idx,
						counts_range[mid_1],
						max_cnt,
						best_past_cnt,
						bin_start,
						batch_times_pred,
						batch_types_pred,
						gaps_uc=gaps_uc,
					)
					if nc_loss_mid_1 <= nc_loss_min:
						min_c = mid_1,
						nc_loss_min = nc_loss_mid_1
						batch_bin_curr_cnt_opt_times_pred_min = batch_bin_curr_cnt_opt_times_pred_mid_1
						batch_bin_curr_cnt_opt_gaps_pred_min = batch_bin_curr_cnt_opt_gaps_pred_mid_1
						batch_bin_curr_cnt_opt_types_pred_min = batch_bin_curr_cnt_opt_types_pred_mid_1
						batch_bin_curr_cnt_opt_sigms_min = batch_bin_curr_cnt_opt_sigms_mid_1
						nc_loss_min_opt = nc_loss_mid_1_opt
						nc_loss_min_cont = nc_loss_mid_1_cont
						nc_count_loss_min = nc_count_loss_mid_1

				return (
					min_c,
					nc_loss_min,
					batch_bin_curr_cnt_opt_times_pred_min,
					batch_bin_curr_cnt_opt_gaps_pred_min,
					batch_bin_curr_cnt_opt_types_pred_min,
					batch_bin_curr_cnt_opt_sigms_min,
					counts_range[min_c],
					nc_loss_min_opt,
					nc_loss_min_cont,
					nc_count_loss_min,
				)
			def binary_search(counts_range, low, high):
				# print('low=', low, 'high=', high)

				mid_1 = (low + high) // 2
				mid_2 = mid_1 + 1
				(
					batch_bin_curr_cnt_opt_times_pred_mid_1,
					batch_bin_curr_cnt_opt_gaps_pred_mid_1,
					batch_bin_curr_cnt_opt_types_pred_mid_1,
					batch_bin_curr_cnt_opt_sigms_mid_1,
					nc_loss_mid_1,
					nc_loss_mid_1_opt,
					nc_loss_mid_1_cont,
					nc_count_loss_mid_1,
				) = get_optimized_gaps(
					batch_idx,
					dec_idx,
					counts_range[mid_1],
					max_cnt,
					best_past_cnt,
					bin_start,
					batch_times_pred,
					batch_types_pred,
					gaps_uc=gaps_uc,
				)

				if high > low:
					(
						batch_bin_curr_cnt_opt_times_pred_mid_2,
						batch_bin_curr_cnt_opt_gaps_pred_mid_2,
						batch_bin_curr_cnt_opt_types_pred_mid_2,
						batch_bin_curr_cnt_opt_sigms_mid_2,
						nc_loss_mid_2,
						nc_loss_mid2_opt,
						nc_loss_mid2_cont,
						nc_count_loss_mid_2,
					) = get_optimized_gaps(
						batch_idx,
						dec_idx,
						counts_range[mid_2],
						max_cnt,
						best_past_cnt,
						bin_start,
						batch_times_pred,
						batch_types_pred,
						gaps_uc=gaps_uc,
					)

					if nc_loss_mid_1 < nc_loss_mid_2:
						high = mid_1
					elif nc_loss_mid_1 > nc_loss_mid_2:
						low = mid_2

					return binary_search(counts_range, low, high)

				elif high == low:
					return (
						mid_1,
						nc_loss_mid_1,
						batch_bin_curr_cnt_opt_times_pred_mid_1,
						batch_bin_curr_cnt_opt_gaps_pred_mid_1,
						batch_bin_curr_cnt_opt_types_pred_mid_1,
						batch_bin_curr_cnt_opt_sigms_mid_1,
						counts_range[mid_1],
						nc_loss_mid_1_opt,
						nc_loss_mid_1_cont,
						nc_count_loss_mid_1,
					)

			if args.search == 0:
				(
					best_nc_idx, best_nc_loss,
					batch_bin_times_pred, batch_bin_gaps_pred,
					batch_bin_types_pred, batch_bin_sigms_pred,
					best_count, best_nc_loss_opt, best_nc_loss_cont,
					best_nc_count_loss,
				) = binary_search(nc_range, 0, len(nc_range)-1)
			elif args.search == 1:
				(
					best_nc_idx, best_nc_loss,
					batch_bin_times_pred, batch_bin_gaps_pred,
					batch_bin_types_pred, batch_bin_sigms_pred,
					best_count, best_nc_loss_opt, best_nc_loss_cont,
					best_nc_count_loss,
				) = linear_search(nc_range, 0, len(nc_range)-1)

			#batch_times_pred.append(batch_bin_times_pred)
			#batch_best_opt_nc_losses.append(best_nc_loss_opt)
			#batch_best_cont_nc_losses.append(best_nc_loss_cont)
			#batch_best_nc_count_losses.append(best_nc_count_loss)
			#best_past_cnt += best_count

			all_times_pred[batch_idx].append(batch_bin_times_pred)
			all_types_pred[batch_idx].append(batch_bin_types_pred)
			all_counts_pred[batch_idx].append(best_count)
			all_sigms_pred[batch_idx].append(batch_bin_sigms_pred)
			all_best_opt_nc_losses[batch_idx].append(best_nc_loss_opt)
			all_best_cont_nc_losses[batch_idx].append(best_nc_loss_cont)
			all_best_nc_count_losses[batch_idx].append(best_nc_count_loss)
			all_best_cnt[batch_idx] += best_count

			#print('Example:', batch_idx, 'dec_idx:', dec_idx, 'Best count:', \
			#	best_count, 'Mean:', event_count_preds_cnt[batch_idx, dec_idx])

		#batch_times_pred = [t for bin_list in batch_times_pred for t in bin_list]
		#all_times_pred.append(batch_times_pred)
		#all_best_opt_nc_losses.append(batch_best_opt_nc_losses)
		#all_best_cont_nc_losses.append(batch_best_cont_nc_losses)
		#all_best_nc_count_losses.append(batch_best_nc_count_losses)

	all_times_pred = np.array(all_times_pred)
	all_types_pred = np.array(all_types_pred)
	all_counts_pred = np.array(all_counts_pred)
	all_sigms_pred = np.array(all_sigms_pred)
	all_best_opt_nc_losses = np.array(all_best_opt_nc_losses)
	all_best_cont_nc_losses = np.array(all_best_cont_nc_losses)
	all_best_nc_count_losses = np.array(all_best_nc_count_losses)

	all_types_pred_flatten = []
	for seq in all_types_pred:
		seq = [s.numpy().tolist() for s in seq]
		all_types_pred_flatten.append(np.array(flatten(seq)))
	all_types_pred = np.array(all_types_pred_flatten)

	all_times_pred_flatten = []
	for seq in all_times_pred:
		seq = [s.tolist() for s in seq]
		all_times_pred_flatten.append(np.array(flatten(seq)))
	all_times_pred = np.array(all_times_pred_flatten)

	all_means_pred = []
	for seq, beg_time in zip(all_times_pred, count_test_out_binend[:, 0] - args.bin_size):
		all_means_pred.append(
			utils.normalize_avg_given_param(
				seq - np.concatenate([beg_time, seq[:-1]]),
				event_test_norma, event_test_normd
			)
		)
	all_means_pred = np.array(all_means_pred)

	all_sigms_pred_flatten = []
	for seq in all_sigms_pred:
		seq = [s.tolist() for s in seq]
		all_sigms_pred_flatten.append(np.array(flatten(seq)))
	all_sigms_pred = np.array(all_sigms_pred_flatten)

	event_dist_params = [all_means_pred, all_sigms_pred]
	count_dist_params = [all_counts_pred, None]

	return (
		all_times_pred, all_types_pred, all_counts_pred,
		event_dist_params, count_dist_params,
		all_best_opt_nc_losses, all_best_cont_nc_losses, all_best_nc_count_losses
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp with optimized gaps generated from rmtpp simulation untill 
# all events of bins generated then rescale
# For each possible count, find the best rmtpp_loss for that many gaps 
# by using solver library
# Select the final count as the count that produces best rmtpp_loss across
# all counts
def run_rmtpp_optimizer_model_comp(
	args,
	query_models,
	data,
	test_data,
	test_data_comp,
	test_data_out_gaps_bin,
	test_data_out_gaps_bin_comp,
	dataset,
	rmtpp_type='nll',
	rmtpp_type_comp='nll'
):

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data


	[comp_test_in_gaps, comp_test_in_feats, comp_test_in_types,
	 _, _,
	 comp_test_norma, comp_test_normd] = test_data_comp

	if rmtpp_type=='nll':
		model_rmtpp = query_models['rmtpp_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = query_models['rmtpp_mse']
	elif rmtpp_type=='mse_var':
		model_rmtpp = query_models['rmtpp_mse_var']
	else:
		assert False, "rmtpp_type must be nll or mse"

	if rmtpp_type_comp=='nll':
		model_rmtpp_comp = query_models['rmtpp_nll_comp']
	elif rmtpp_type_comp=='mse':
		model_rmtpp_comp = query_models['rmtpp_mse_comp']
	elif rmtpp_type_comp=='mse_var':
		model_rmtpp_comp = query_models['rmtpp_mse_var_comp']
	else:
		assert False, "rmtpp_type_comp must be nll_comp or mse_comp"

	model_check = (model_rmtpp is not None) and (model_rmtpp_comp is not None)
	assert model_check, "run_rmtpp_optimizer_model_comp requires RMTPP and RMTPP_comp model"

	enc_len = args.enc_len
	comp_enc_len = args.comp_enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	comp_bin_sz = args.comp_bin_sz
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	def rmtpp_loglikelihood_loss(gaps, D, WT):

		rmtpp_loss = 0

		log_lambda_ = (D + cp.multiply(gaps, WT))
		lambda_ = cp.exp(log_lambda_)
		log_f_star = (log_lambda_
					  + cp.multiply(1./WT, cp.exp(D))
					  - cp.multiply(1./WT, lambda_))
		#loss = -tf.reduce_mean(tf.reduce_sum(log_f_star, axis=1)/events_count_per_batch)
		loss = -log_f_star
		return loss

	def mse_loss(gaps, D, WT):
		#return cp.multiply(cp.power(gaps-D, 2), cp.power(WT, -2))
		return cp.power(gaps-D, 2)

	def mse_loglikelihood_loss(gaps, D, WT):
		return -(cp.log(1/(((2*np.pi)**0.5)*WT)) - (((gaps - D)**2) / (2*(WT)**2)))

	def optimize_gaps(model_rmtpp_params,
					  model_rmtpp_params_comp,
					  rmtpp_loglikelihood_loss,
					  comp_bin_sz,
					  test_data_init_time,
					  test_data_rmtpp_normalizer,
					  test_data_rmtpp_normalizer_comp):

		D, WT = model_rmtpp_params[0][:,:comp_bin_sz], model_rmtpp_params[1][:,:comp_bin_sz]
		D_comp, WT_comp = model_rmtpp_params_comp[0].numpy(), model_rmtpp_params_comp[1].numpy()

		gaps = cp.Variable(D.shape)
		gaps.value = D

		test_norm_a, test_norm_d = test_data_rmtpp_normalizer
		test_norm_a_comp, test_norm_d_comp = test_data_rmtpp_normalizer_comp

		gaps_sum = utils.normalize_avg_given_param(
			utils.denormalize_avg(
				cp.sum(gaps),
				test_norm_a,
				test_norm_d,
			),
			test_norm_a_comp,
			test_norm_d_comp,
		)

		if rmtpp_type=='nll':
			#WT = np.minimum(WT, ETH)
			opt_loss = cp.sum(rmtpp_loglikelihood_loss(gaps_sum, D_comp, WT_comp) + \
					cp.sum(rmtpp_loglikelihood_loss(gaps, D, WT))/D.shape[1])
		elif rmtpp_type=='mse':
			if args.extra_var_model:
				opt_loss = cp.sum(mse_loglikelihood_loss(cp.sum(gaps), D_comp, WT_comp) + \
					cp.sum(mse_loglikelihood_loss(gaps, D, WT))/D.shape[1])
			else:
				WT = np.ones_like(WT)
				opt_loss = cp.sum(mse_loss(gaps_sum, D_comp, WT_comp) + \
						cp.sum(mse_loss(gaps, D, WT))/D.shape[1])
		elif rmtpp_type=='mse_var':
			opt_loss = cp.sum(mse_loglikelihood_loss(gaps_sum, D_comp, WT_comp) + \
				cp.sum(mse_loglikelihood_loss(gaps, D, WT))/D.shape[1])

		objective = cp.Minimize(opt_loss)

		constraints = [gaps>=0]

		prob = cp.Problem(objective, constraints)

		try:
			rmtpp_loss = prob.solve(warm_start=True)
		except cp.error.SolverError:
			rmtpp_loss = prob.solve(solver='SCS', warm_start=True)

		#if gaps.value is None:
		#	gaps.value = all_bins_gaps_pred[:,:comp_bin_sz]
		#	rmtpp_loss = opt_loss.value

		#import ipdb
		#ipdb.set_trace()

		all_bins_gaps_pred = gaps.value[0:1, :comp_bin_sz]
		#print('Loss after optimization:', rmtpp_loss)
	
		return all_bins_gaps_pred, rmtpp_loss


	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)
	test_data_input_gaps_bin_comp = comp_test_in_gaps.astype(np.float32)
	all_events_in_bin_pred = list()

	t_e_plus = count_test_out_binend[:,-1]
	(
		all_gaps_pred_comp, all_times_pred_comp, 
		all_D_pred_comp, all_WT_pred_comp
	) = simulate_for_D_WT(
		model_rmtpp_comp,
		test_data_init_time,
		test_data_input_gaps_bin_comp,
		comp_test_in_feats,
		comp_test_in_types,
		t_e_plus,
		(comp_test_norma,
		comp_test_normd),
	)
	event_count_preds_cnt_comp = np.ones_like(test_data_init_time) * all_times_pred_comp.shape[1] + 1
	(
		all_gaps_pred_comp, all_times_pred_comp, all_types_pred_comp,
		all_D_pred_comp, all_WT_pred_comp,
	) = simulate_with_counter(
		model_rmtpp_comp, 
		test_data_init_time, 
		test_data_input_gaps_bin_comp,
		comp_test_in_feats,
		comp_test_in_types,
		event_count_preds_cnt_comp,
		(comp_test_norma, 
		comp_test_normd),
	)

	event_count_preds_cnt = np.ones_like(test_data_init_time) * all_times_pred_comp.shape[1] * comp_bin_sz
	event_count_preds_true = count_test_out_counts
	(
		all_gaps_pred_simu, all_times_pred_simu, all_types_pred_simu,
		D_pred, WT_pred
	) = simulate_with_counter(
		model_rmtpp, 
		test_data_init_time, 
		test_data_input_gaps_bin,
		event_test_in_feats,
		event_test_in_types,
		event_count_preds_cnt,
		(event_test_norma, 
		event_test_normd),
	)

	all_times_pred = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_types_pred = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_sigms_pred = [[] for _ in range(len(test_data_input_gaps_bin))]
	all_best_cnt = [0 for _ in range(len(test_data_input_gaps_bin))]
	#import ipdb
	#ipdb.set_trace()
	for dec_idx in range(all_times_pred_comp.shape[1]):

		for batch_idx in range(len(all_times_pred_simu)):

			if dec_idx==0 or all_times_pred_comp[batch_idx, dec_idx-1] < t_e_plus[batch_idx]:
				print(batch_idx, dec_idx)
	
				best_past_cnt = all_best_cnt[batch_idx]
	
				batch_times_pred = all_times_pred[batch_idx]
				batch_types_pred = all_types_pred[batch_idx]
	
				times_pred_for_bin = all_times_pred_simu[batch_idx:batch_idx+1, best_past_cnt:best_past_cnt+comp_bin_sz]
				types_pred_for_bin = all_types_pred_simu[batch_idx:batch_idx+1, best_past_cnt:best_past_cnt+comp_bin_sz]
	
	
				if args.no_rescale_rmtpp_params:
					D = D_pred[batch_idx:batch_idx+1]
					WT = WT_pred[batch_idx:batch_idx+1]
				else:
					raise NotImplementedError
	
				batch_bin_curr_cnt_D_pred = D[:, best_past_cnt:best_past_cnt+comp_bin_sz, 0].numpy()
				batch_bin_curr_cnt_WT_pred = WT[:, best_past_cnt:best_past_cnt+comp_bin_sz, 0].numpy()
	
				model_rmtpp_params = [batch_bin_curr_cnt_D_pred, batch_bin_curr_cnt_WT_pred]
				model_rmtpp_params_comp = [all_D_pred_comp[batch_idx, dec_idx], all_WT_pred_comp[batch_idx, dec_idx]]
				test_data_rmtpp_normalizer = [event_test_norma, event_test_normd]
				test_data_rmtpp_normalizer_comp = [comp_test_norma, comp_test_normd]
	
				if dec_idx == 0:
					test_data_init_time_batch = test_data_init_time[batch_idx:batch_idx+1]
				else:
					test_data_init_time_batch = np.array(flatten(batch_times_pred)[-1:])

				#import ipdb
				#ipdb.set_trace()
	
		
				batch_bin_curr_cnt_opt_gaps_pred, nc_loss \
					= optimize_gaps(model_rmtpp_params,
									model_rmtpp_params_comp,
									rmtpp_loglikelihood_loss,
									comp_bin_sz,
									test_data_init_time_batch,
									test_data_rmtpp_normalizer,
									test_data_rmtpp_normalizer_comp)
				batch_bin_curr_cnt_opt_gaps_pred = utils.denormalize_avg(batch_bin_curr_cnt_opt_gaps_pred,
														event_test_norma,
														event_test_normd)
			
				batch_bin_curr_cnt_opt_times_pred \
					= (test_data_init_time_batch + tf.cast(tf.cumsum(batch_bin_curr_cnt_opt_gaps_pred, axis=1), tf.float64))
				batch_bin_curr_cnt_opt_times_pred = batch_bin_curr_cnt_opt_times_pred[0].numpy()
				batch_bin_sigms_pred = batch_bin_curr_cnt_WT_pred[0]

				clip_te_cnt = np.sum(batch_bin_curr_cnt_opt_times_pred<t_e_plus[batch_idx])
				all_times_pred[batch_idx].append(batch_bin_curr_cnt_opt_times_pred[:clip_te_cnt])
				all_types_pred[batch_idx].append(types_pred_for_bin[0][:clip_te_cnt])
				all_sigms_pred[batch_idx].append(batch_bin_sigms_pred[:clip_te_cnt])
				all_best_cnt[batch_idx] += clip_te_cnt


	all_times_pred = np.array(all_times_pred)
	all_types_pred = np.array(all_types_pred)
	all_sigms_pred = np.array(all_sigms_pred)

	all_times_pred_flatten = []
	for seq in all_times_pred:
		seq = [s.tolist() for s in seq]
		all_times_pred_flatten.append(np.array(flatten(seq)))
	all_times_pred = np.array(all_times_pred_flatten)

	all_types_pred_flatten = []
	for seq in all_types_pred:
		seq = [s.numpy().tolist() for s in seq]
		all_types_pred_flatten.append(np.array(flatten(seq)))
	all_types_pred = np.array(all_types_pred_flatten)

	all_counts_pred = []
	for dec_idx in range(dec_len):
		t_b_plus = count_test_out_binend[:, dec_idx] - bin_size
		t_e_plus = count_test_out_binend[:, dec_idx]
		all_counts_pred.append(
			count_events(all_times_pred, t_b_plus, t_e_plus)
		)
	all_counts_pred = np.stack(all_counts_pred, axis=1)

	all_means_pred = []
	for seq, beg_time in zip(all_times_pred, count_test_out_binend[:, 0] - args.bin_size):
		all_means_pred.append(
			utils.normalize_avg_given_param(
				seq - np.concatenate([beg_time, seq[:-1]]),
				event_test_norma, event_test_normd
			)
		)
	all_means_pred = np.array(all_means_pred)

	all_sigms_pred_flatten = []
	for seq in all_sigms_pred:
		seq = [s.tolist() for s in seq]
		all_sigms_pred_flatten.append(np.array(flatten(seq)))
	all_sigms_pred = np.array(all_sigms_pred_flatten)

	event_dist_params = [all_means_pred, all_sigms_pred]
	count_dist_params = [all_counts_pred, None]

	return (
		all_times_pred, all_types_pred, all_counts_pred,
		event_dist_params, count_dist_params,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Plain rmtpp model to generate events independent of bin boundary
def run_rmtpp_simulation(args, models, data, test_data, rmtpp_type=None,
	use_nowcast=False, nc_gaps_in=None, nc_feats_in=None, nc_types_in=None,
):
	if rmtpp_type=='nll':
		model_rmtpp = models['rmtpp_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = models['rmtpp_mse']
	elif rmtpp_type=='mse_var':
		model_rmtpp = models['rmtpp_mse_var']
	else:
		assert False, "rmtpp_type must be nll or mse"
	model_check = (model_rmtpp is not None)
	assert model_check, "run_rmtpp_simulation requires RMTPP model"

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size

	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)

	t_b_plus = count_test_out_binend[:, 0] - bin_size
	t_e_plus = count_test_out_binend[:, -1]

	_, all_times_pred, all_types_pred, _, D_pred, WT_pred = simulate_rmtpp(
		model_rmtpp,
		test_data_init_time,
		test_data_input_gaps_bin,
		event_test_in_feats,
		event_test_in_types,
		t_e_plus,
		(event_test_norma,
		event_test_normd),
		use_nowcast=use_nowcast,
		nc_gaps_in=nc_gaps_in,
		nc_feats_in=nc_feats_in,
		nc_types_in=nc_types_in,
	)
	all_counts_pred = []
	for dec_idx in range(dec_len):
		t_b_plus = count_test_out_binend[:, dec_idx] - bin_size
		t_e_plus = count_test_out_binend[:, dec_idx]
		all_counts_pred.append(
			count_events(all_times_pred, t_b_plus, t_e_plus)
		)
	all_counts_pred = np.stack(all_counts_pred, axis=1)

	all_times_pred = all_times_pred.numpy()
	#all_times_pred = np.expand_dims(all_times_pred.numpy(), axis=-1)
	all_types_pred = all_types_pred.numpy()

	event_dist_params = (D_pred, WT_pred)
	count_dist_params = [all_counts_pred, None]
	return (
		all_counts_pred, all_times_pred, all_types_pred,
		event_dist_params, count_dist_params,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Plain wgan model to generate events independent of bin boundary
def run_wgan_simulation(args, models, data, test_data):

	model_wgan = models['wgan']
	model_check = (model_wgan is not None)
	assert model_check, "run_wgan_simulation requires WGAN model"

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	dec_len = args.out_bin_sz
	bin_size = args.bin_size

	next_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)

	t_b_plus = count_test_out_binend[:, 0] - bin_size
	t_e_plus = count_test_out_binend[:, -1]

	_, all_times_pred, _ = simulate_wgan(
		model_wgan,
		test_data_init_time,
		test_data_input_gaps_bin,
		event_test_in_feats,
		t_e_plus,
		(event_test_norma,
		event_test_normd),
		prev_hidden_state=next_hidden_state
	)

	all_counts_pred = []
	for dec_idx in range(dec_len):
		t_b_plus = count_test_out_binend[:, dec_idx] - bin_size
		t_e_plus = count_test_out_binend[:, dec_idx]
		all_counts_pred.append(
			count_events(all_times_pred, t_b_plus, t_e_plus)
		)
	all_counts_pred = np.stack(all_counts_pred, axis=1)

	all_times_pred = all_times_pred.numpy()

	all_types_pred = np.ones_like(all_times_pred)

	all_means_pred = []
	for seq, beg_time in zip(all_times_pred, count_test_out_binend[:, 0] - args.bin_size):
		all_means_pred.append(
			utils.normalize_avg_given_param(
				seq - np.concatenate([beg_time, seq[:-1]]),
				event_test_norma, event_test_normd
			)
		)
	all_means_pred = np.array(all_means_pred)

	all_sigms_pred = []
	for seq in all_means_pred:
		all_sigms_pred.append(np.ones_like(seq)*1e-6)
	all_sigms_pred = np.array(all_sigms_pred)

	event_dist_params = (all_means_pred, all_sigms_pred)
	count_dist_params = [all_counts_pred, None]

	return (
		all_counts_pred, all_times_pred, all_types_pred,
		event_dist_params, count_dist_params,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Plain wgan model to generate events independent of bin boundary
def run_seq2seq_simulation(args, models, data, test_data):

	model_seq2seq = models['seq2seq']
	model_check = (model_seq2seq is not None)
	assert model_check, "run_seq2seq_simulation requires Seq2Seq model"

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	dec_len = args.out_bin_sz
	bin_size = args.bin_size

	next_hidden_state = None

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)

	t_b_plus = count_test_out_binend[:, 0] - bin_size
	t_e_plus = count_test_out_binend[:, -1]

	_, all_times_pred, _ = simulate_seq2seq(
		model_seq2seq,
		test_data_init_time,
		test_data_input_gaps_bin,
		event_test_in_feats,
		t_e_plus,
		(event_test_norma,
		event_test_normd),
		prev_hidden_state=next_hidden_state
	)

	all_counts_pred = []
	for dec_idx in range(dec_len):
		t_b_plus = count_test_out_binend[:, dec_idx] - bin_size
		t_e_plus = count_test_out_binend[:, dec_idx]
		all_counts_pred.append(
			count_events(all_times_pred, t_b_plus, t_e_plus)
		)
	all_counts_pred = np.stack(all_counts_pred, axis=1)

	all_times_pred = all_times_pred.numpy()

	all_types_pred = np.ones_like(all_times_pred)

	all_means_pred = []
	for seq, beg_time in zip(all_times_pred, count_test_out_binend[:, 0] - args.bin_size):
		all_means_pred.append(
			utils.normalize_avg_given_param(
				seq - np.concatenate([beg_time, seq[:-1]]),
				event_test_norma, event_test_normd
			)
		)
	all_means_pred = np.array(all_means_pred)

	all_sigms_pred = []
	for seq in all_means_pred:
		all_sigms_pred.append(np.ones_like(seq)*1e-6)
	all_sigms_pred = np.array(all_sigms_pred)

	event_dist_params = (all_means_pred, all_sigms_pred)
	count_dist_params = [all_counts_pred, None]

	return (
		all_counts_pred, all_times_pred, all_types_pred,
		event_dist_params, count_dist_params,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Plain transformer model to generate events independent of bin boundary
def run_transformer_simulation(
	args, models, data, test_data, use_nowcast=False,
	nc_gaps_in=None, nc_feats_in=None, nc_types_in=None,
):

	model_transformer = models['transformer']
	model_check = (model_transformer is not None)
	assert model_check, "run_transformer_simulation requires transformer^ model"

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size

	test_data_init_time = event_test_in_lasttime.astype(np.float32)
	test_data_input_gaps_bin = event_test_in_gaps.astype(np.float32)

	t_b_plus = count_test_out_binend[:, 0] - bin_size
	t_e_plus = count_test_out_binend[:, -1]

	_, all_times_pred, all_types_pred, = simulate_transformer(
		model_transformer,
		test_data_init_time,
		test_data_input_gaps_bin,
		event_test_in_feats,
		event_test_in_types,
		t_e_plus,
		(event_test_norma,
		event_test_normd),
		use_nowcast=use_nowcast,
		nc_gaps_in=nc_gaps_in,
		nc_feats_in=nc_feats_in,
		nc_types_in=nc_types_in,
	)

	all_counts_pred = []
	for dec_idx in range(dec_len):
		t_b_plus = count_test_out_binend[:, dec_idx] - bin_size
		t_e_plus = count_test_out_binend[:, dec_idx]
		all_counts_pred.append(
			count_events(all_times_pred, t_b_plus, t_e_plus)
		)
	all_counts_pred = np.stack(all_counts_pred, axis=1)

	#all_times_pred = np.expand_dims(all_times_pred.numpy(), axis=-1)
	all_times_pred = all_times_pred.numpy()
	all_types_pred = all_types_pred.numpy()

	all_means_pred = []
	for seq, beg_time in zip(all_times_pred, count_test_out_binend[:, 0] - args.bin_size):
		all_means_pred.append(
			utils.normalize_avg_given_param(
				seq - np.concatenate([beg_time, seq[:-1]]),
				event_test_norma, event_test_normd
			)
		)
	all_means_pred = np.array(all_means_pred)

	all_sigms_pred = []
	for seq in all_means_pred:
		all_sigms_pred.append(np.ones_like(seq)*1e-6)
	all_sigms_pred = np.array(all_sigms_pred)

	event_dist_params = (all_means_pred, all_sigms_pred)
	count_dist_params = [all_counts_pred, None]

	return (
		all_counts_pred, all_times_pred, all_types_pred,
		event_dist_params, count_dist_params,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#:


#####################################################
# 				Utils Functions						#
#####################################################
def count_events(all_times_pred, t_b_plus, t_e_plus):
	times_out_indices_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_times_pred, t_b_plus)]
	times_out_indices_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_times_pred, t_e_plus)]
	event_count_preds = [times_out_indices_te[idx] - times_out_indices_tb[idx] for idx in range(len(t_b_plus))]
	return event_count_preds

def compute_event_in_bin(data, count, appender=None, size=40):
	count = tf.cast(count, tf.int32)
	event_bag = list()
	full_bag = list()
	end_event = list()
	end_gaps = list()
	for idx in range(len(count)):
		event_bag.append(data[idx,0:count[idx],0].numpy())
		end_event.append(data[idx,count[idx]-1,0])
		if appender is None:
			end_gaps.append(data[idx,count[idx],0] - data[idx,count[idx]-1,0])
		if appender is not None:
			amt = size-len(data[idx,0:count[idx],0].numpy().tolist())
			if amt<=0:
				full_bag.append(data[idx,-size:,0].numpy().tolist())
			else:
				full_bag.append(appender[idx,-amt:,0].tolist() + data[idx,0:count[idx],0].numpy().tolist())
	
	full_bag = np.array(full_bag)
	end_event = np.array(end_event)
	end_gaps = np.array(end_gaps)
	return event_bag, full_bag, end_event, end_gaps

def scaled_points(actual_bin_start, actual_bin_end, bin_start, bin_end, all_times_pred):
	all_times_pred_mask = np.ma.make_mask(all_times_pred)
	all_gaps_pred_mask = np.ma.make_mask(all_times_pred[:,1:])
	all_times_pred_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
							 (all_times_pred - bin_start)) + actual_bin_start
	all_times_pred_scaled = all_times_pred_scaled * all_times_pred_mask
	all_gaps_pred_scaled = all_times_pred_scaled[:,1:] - all_times_pred_scaled[:,:-1]
	all_gaps_pred_scaled = all_gaps_pred_scaled * all_gaps_pred_mask
	all_gaps_pred_scaled = tf.expand_dims(all_gaps_pred_scaled, axis=-1)
	return all_times_pred_scaled, all_gaps_pred_scaled

def scale_time_interval(data, t_start, t_end):
	t_start = tf.cast(t_start, tf.float32)
	t_end = tf.cast(t_end, tf.float32)
	N_bin = t_end - t_start
	scaled_time = ((data * N_bin * t_end) + t_start) / ((data * N_bin) + 1.0)
	return scaled_time

def before_time_event_count(events_in_bin_pred, timecheck):
	count=0
	for idx in range(len(events_in_bin_pred)):
		events_in_one_bin = events_in_bin_pred[idx]
		if timecheck <= events_in_one_bin[-1]:
			cnt=0
			while(timecheck >= events_in_one_bin[cnt]):
				cnt+=1
			count+=cnt
			return count
		else:
			count+=len(events_in_one_bin)
	return count

def compute_count_event_range(all_events_in_bin_pred, t_b_plus, t_e_plus):
	event_count = list()
	for batch_idx in range(len(all_events_in_bin_pred)):
		before_tb_event_count = before_time_event_count(all_events_in_bin_pred[batch_idx], t_b_plus[batch_idx])
		before_te_event_count = before_time_event_count(all_events_in_bin_pred[batch_idx], t_e_plus[batch_idx])
		event_count.append(before_te_event_count - before_tb_event_count)
	test_event_count_pred = np.array(event_count)
	return test_event_count_pred

def trim_evens_pred(all_times_pred_uncut, t_b_plus, t_e_plus):
	all_times_pred = list()
	for idx in range(len(all_times_pred_uncut)):
		lst = list()
		for each in all_times_pred_uncut[idx]:
			lst = lst+each.tolist()
		all_times_pred.append(lst)
	all_times_pred = np.array(all_times_pred)

	times_out_indices_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_times_pred, t_b_plus)]
	times_out_indices_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_times_pred, t_e_plus)]
	all_times_pred = [all_times_pred[idx][times_out_indices_tb[idx]:times_out_indices_te[idx]] for idx in range(len(t_b_plus))]
	return all_times_pred

def clean_dict_for_na_model(all_run_fun_pdf, run_model_flags):
	remove_item = list()
	for each in all_run_fun_pdf:
		if not (each in run_model_flags and run_model_flags[each]):
			remove_item.append(each)
	for each in remove_item:
		del all_run_fun_pdf[each]
	return all_run_fun_pdf
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def compute_count_metric(all_times_true, all_times_pred, t_b_plus, t_e_plus):
	idx_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_times_pred, t_b_plus)]
	idx_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_times_pred, t_e_plus)]
	all_counts_pred = [idx_te[idx] - idx_tb[idx] for idx in range(len(t_b_plus))]

	idx_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_times_true, t_b_plus)]
	idx_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_times_true, t_e_plus)]
	all_counts_true = [idx_te[idx] - idx_tb[idx] for idx in range(len(t_b_plus))]

	all_counts_true = np.array(all_counts_true)
	all_counts_pred = np.array(all_counts_pred)
	count_mae_rh = np.mean(np.abs(all_counts_true - all_counts_pred))
	count_mae_rh_pe = np.abs(all_counts_true - all_counts_pred)

	return count_mae_rh, count_mae_rh_pe


def compute_bleu_score(
	all_times_true, all_times_pred,
	all_types_true, all_types_pred,
	t_b_plus, t_e_plus
):

	idx_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_times_pred, t_b_plus)]
	idx_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_times_pred, t_e_plus)]
	all_types_pred = [all_types_pred[idx][idx_tb[idx]:idx_te[idx]] for idx in range(len(t_b_plus))]

	idx_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_times_true, t_b_plus)]
	idx_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_times_true, t_e_plus)]
	all_types_true = [all_types_true[idx][idx_tb[idx]:idx_te[idx]] for idx in range(len(t_b_plus))]


	true_types = all_types_true
	pred_types = [[seq] for seq in all_types_pred]
	bleu_score = corpus_bleu(pred_types, true_types)
	bleu_score_pe = []
	for true_seqs, pred_seqs in zip(all_types_true, all_types_pred):
		pred_seqs_ = [pred_seqs]
		true_seqs_ = true_seqs
		bleu_pe = sentence_bleu(pred_seqs_, true_seqs_)
		bleu_score_pe.append(bleu_pe)

	return bleu_score, bleu_score_pe

def compute_full_horizon_metrics(
	all_times_true,
	all_times_pred,
	all_types_true,
	all_types_pred,
	all_counts_true,
	all_counts_pred,
	count_test_out_binend, bin_size,
):
	count_mae_fh = np.mean(np.abs(all_counts_true - all_counts_pred))
	count_mae_fh_pe = np.mean(np.abs(all_counts_true - all_counts_pred), axis=1)
	count_mae_fh_per_bin = np.mean(np.abs(all_counts_true - all_counts_pred), axis=0)

	wass_dist_fh, wass_dist_fh_pe = compute_wasserstein_dist(
		all_times_pred, all_times_true,
		count_test_out_binend[:, 0] - bin_size,
		count_test_out_binend[:, -1],
	)

	#true_types = all_types_true
	#pred_types = [[seq] for seq in all_types_pred]
	#bleu_score_fh = corpus_bleu(pred_types, true_types-1)
	#bleu_score_fh_pe = []
	#for true_seqs, pred_seqs in zip(all_types_true, all_types_pred):
	#	pred_seqs_ = [pred_seqs]
	#	true_seqs_ = true_seqs
	#	bleu_pe = sentence_bleu(pred_seqs_, true_seqs_-1)
	#	bleu_score_fh_pe.append(bleu_pe)

	bleu_score_fh, bleu_score_fh_pe = compute_bleu_score(
		all_times_true, all_times_pred,
		all_types_true, all_types_pred,
		count_test_out_binend[:, 0] - bin_size,
		count_test_out_binend[:, -1],
	)

	wass_dist_fh_per_bin = []
	bleu_score_fh_per_bin = []
	for dec_idx in range(all_counts_true.shape[1]):
		t_b_plus = count_test_out_binend[:, dec_idx] - bin_size
		t_e_plus = count_test_out_binend[:, dec_idx]

		wass_dist_fh_dec_idx, _ = compute_wasserstein_dist(
			all_times_pred, all_times_true,
			t_b_plus, t_e_plus,
		)
		wass_dist_fh_per_bin.append(wass_dist_fh_dec_idx)

		bleu_score_fh_dec_idx, _ = compute_bleu_score(
			all_times_true, all_times_pred,
			all_types_true, all_types_pred,
			t_b_plus, t_e_plus,
		)
		bleu_score_fh_per_bin.append(bleu_score_fh_dec_idx)

	return (
		count_mae_fh,
		wass_dist_fh,
		bleu_score_fh,
		count_mae_fh_pe,
		wass_dist_fh_pe,
		bleu_score_fh_pe,
		count_mae_fh_per_bin,
		np.array(wass_dist_fh_per_bin),
		np.array(bleu_score_fh_per_bin),
	)


def compute_random_horizon_metrics(
	all_times_true, all_times_pred,
	all_types_true, all_types_pred,
	t_b_plus, t_e_plus,
):

	# Find all events in the range tb+ and te+
	# Count them to get true and predicted counts
	# evaluate	

	count_mae_rh, count_mae_rh_pe = compute_count_metric(
		all_times_true, all_times_pred, t_b_plus, t_e_plus
	)

	wass_dist_rh, wass_dist_rh_pe = compute_wasserstein_dist(
		all_times_pred, all_times_true, t_b_plus, t_e_plus,
	)

	bleu_score_rh, bleu_score_rh_pe = compute_bleu_score(
		all_times_true, all_times_pred,
		all_types_true, all_types_pred,
		t_b_plus, t_e_plus,
	)

	return (
		count_mae_rh,
		wass_dist_rh,
		bleu_score_rh,
		count_mae_rh_pe,
		wass_dist_rh_pe,
		bleu_score_rh_pe,
	)


def compute_full_model_acc(
	args,
	test_data,
	all_bins_count_pred,
	all_times_bin_pred,
	event_test_out_times,
	all_types_pred,
	test_data_out_types,
	dataset_name,
	model_name=None
):
	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	all_bins_count_true = count_test_out_counts


	all_times_pred = list()
	for idx in range(len(all_times_bin_pred)):
		lst = list()
		for each in all_times_bin_pred[idx]:
			lst = lst+each.tolist()
		all_times_pred.append(np.array(lst))
	all_times_pred = np.array(all_times_pred)

	if all_bins_count_pred is None:
		all_bins_count_pred_lst = list()
		for dec_idx in range(dec_len):
			t_b_plus = count_test_out_binend[:,dec_idx] - bin_size
			t_e_plus = count_test_out_binend[:,dec_idx]
			all_bins_count_pred_lst.append(np.array(count_events(all_times_pred, t_b_plus, t_e_plus)))
		all_bins_count_pred = np.array(all_bins_count_pred_lst).T

	if all_bins_count_pred is None:
		abcp = []
		for i in range(len(all_times_bin_pred)):
			batch_bins_count_pred = [len(times_pred) for times_pred in all_times_bin_pred[i]]
			abcp.append(batch_bins_count_pred)
		abcp = np.array(abcp)


	compute_depth = 10
	t_b_plus = count_test_out_binend[:,0] - bin_size
	t_e_plus = count_test_out_binend[:,-1]
	deep_mae, deep_mae_fh_pe = compute_hierarchical_mae_deep(
		all_times_pred,
		event_test_out_times,
		t_b_plus, t_e_plus,
		compute_depth
	)

	print("____________________________________________________________________")
	print(model_name, 'Full-eval: MAE for Count Prediction:', np.mean(np.abs(all_bins_count_true-all_bins_count_pred )))
	print(model_name, 'Full-eval: MAE for Count Prediction (per bin):', np.mean(np.abs(all_bins_count_true-all_bins_count_pred ), axis=0))
	print(model_name, 'Full-eval: Deep MAE for events Prediction:', deep_mae, 'at depth', compute_depth)
	print("____________________________________________________________________")

	count_mae_fh = np.mean(np.abs(all_bins_count_true-all_bins_count_pred))
	count_mae_fh_per_bin = np.mean(np.abs(all_bins_count_true-all_bins_count_pred), axis=0)
	deep_mae_fh = deep_mae

	count_mae_fh_pe = np.mean(np.abs(all_bins_count_true-all_bins_count_pred), axis=1)

	wasserstein_dist_fh, wass_dist_fh_pe = compute_wasserstein_dist(
		all_times_pred,
		event_test_out_times,
		t_b_plus,
		t_e_plus,
	)

	true_types = test_data_out_types
	pred_types = [[seq] for seq in all_types_pred]
	bleu_score_fh = corpus_bleu(pred_types, true_types-1)
	bleu_score_fh_pe = []
	for true_seqs, pred_seqs in zip(test_data_out_types, all_types_pred):
		pred_seqs_ = [pred_seqs]
		true_seqs_ = true_seqs
		bleu_pe = sentence_bleu(pred_seqs_, true_seqs_-1)
		bleu_score_fh_pe.append(bleu_pe)

	return (
		deep_mae_fh,
		count_mae_fh,
		count_mae_fh_per_bin,
		wasserstein_dist_fh,
		count_mae_fh_pe,
		deep_mae_fh_pe,
		wass_dist_fh_pe,
		event_test_out_times,
		all_times_pred,
		bleu_score_fh,
		bleu_score_fh_pe,
		test_data_out_types,
		all_types_pred,
	)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#



#####################################################
# 				Query Processing					#
#####################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Deep MAE Loss calculations
# Query 1
def compute_mae_cur_bound(all_event_pred, all_event_true, t_b_plus, t_e_plus):
	times_out_indices_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_event_pred, t_b_plus)]
	times_out_indices_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_event_pred, t_e_plus)]
	all_event_pred_count = [times_out_indices_te[idx] - times_out_indices_tb[idx] for idx in range(len(t_b_plus))]
	all_event_pred_count = np.array(all_event_pred_count)
	all_event_pred = [all_event_pred[idx][times_out_indices_tb[idx]:times_out_indices_te[idx]] for idx in range(len(t_b_plus))]

	times_out_indices_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_event_true, t_b_plus)]
	times_out_indices_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_event_true, t_e_plus)]
	all_event_true_count = [times_out_indices_te[idx] - times_out_indices_tb[idx] for idx in range(len(t_b_plus))]
	all_event_true_count = np.array(all_event_true_count)
	all_event_true = [all_event_true[idx][times_out_indices_tb[idx]:times_out_indices_te[idx]] for idx in range(len(t_b_plus))]
	mae = np.mean(np.abs(all_event_pred_count - all_event_true_count))
	mae_pe = np.abs(all_event_pred_count - all_event_true_count)

	return all_event_pred, all_event_true, mae, mae_pe

def compute_hierarchical_mae_deep(all_event_pred, all_event_true, t_b_plus, t_e_plus, compute_depth):
	if compute_depth == 0:
		return 0, np.zeros((len(all_event_pred)))
	
	all_event_pred, all_event_true, res, res_pe = compute_mae_cur_bound(
		all_event_pred, 
		all_event_true,
		t_b_plus, t_e_plus
	)

	t_b_e_mid = (t_b_plus + t_e_plus) / 2.0
	compute_depth -= 1

	res1, res1_pe = compute_hierarchical_mae_deep(all_event_pred, all_event_true, t_b_plus, t_b_e_mid, compute_depth)
	res2, res2_pe = compute_hierarchical_mae_deep(all_event_pred, all_event_true, t_b_e_mid, t_e_plus, compute_depth)
	return (
		res + res1 + res2,
		res_pe + res1_pe + res2_pe,
	)

def compute_hierarchical_mae(all_event_pred_uncut, query_data, all_event_true, compute_depth):
	[t_b_plus, t_e_plus, true_count] = query_data
	#TODO: Check for rmtpp and wgan model t_b_plus and t_e_plus index is first and last.
	all_event_pred = trim_evens_pred(all_event_pred_uncut, t_b_plus, t_e_plus)
	print('Event counts ', [len(x) for x in all_event_pred])
	test_event_count_pred = np.array([len(x) for x in all_event_pred]).astype(np.float32)
	event_count_mse = tf.keras.losses.MSE(true_count, test_event_count_pred).numpy()
	event_count_mae = tf.keras.losses.MAE(true_count, test_event_count_pred).numpy()
	print("MSE of event count in range:", event_count_mse)
	print("MAE of event count in range:", event_count_mae)

	wasserstein_dist_rh, wass_dist_rh_pe = compute_wasserstein_dist(
		all_event_pred,
		all_event_true,
		t_b_plus,
		t_e_plus
	)

	deep_mae_rh, deep_mae_rh_pe = compute_hierarchical_mae_deep(
		all_event_pred,
		all_event_true,
		t_b_plus,
		t_e_plus,
		compute_depth
	)

	return (
		deep_mae_rh,
		event_count_mae,
		event_count_mse,
		wasserstein_dist_rh,
		deep_mae_rh_pe,
		wass_dist_rh_pe,
	)

def compute_wasserstein_dist(all_event_pred, all_event_true, t_b_plus, t_e_plus):
	idx_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_event_pred, t_b_plus)]
	idx_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_event_pred, t_e_plus)]
	all_event_pred = [all_event_pred[idx][idx_tb[idx]:idx_te[idx]] for idx in range(len(t_b_plus))]

	idx_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_event_true, t_b_plus)]
	idx_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_event_true, t_e_plus)]
	all_event_true = [all_event_true[idx][idx_tb[idx]:idx_te[idx]] for idx in range(len(t_b_plus))]


	def w_dist(true_ts, pred_ts, B, E):
		true_ts = np.array(true_ts) * 1. / (E - B) # Normalize all times
		pred_ts = np.array(pred_ts) * 1. / (E - B) # Normalize all times
		E_norm = E * 1. / (E - B)
		B_norm = B * 1. / (E - B)
		min_len = min(len(true_ts), len(pred_ts))
		dist = 0.
		dist += np.sum(np.abs(true_ts[:min_len] - pred_ts[:min_len]))
		if len(true_ts) > len(pred_ts):
			leftover = true_ts[min_len:]
		elif len(pred_ts) > len(true_ts):
			leftover = pred_ts[min_len:]
		else:
			leftover = None
	
		if leftover is not None:
			dist += np.sum(np.abs(E_norm - leftover))
	
		dist = np.sum(dist)

		n = min_len
		if leftover is not None:
			n += len(leftover)
	
		return dist, n

	dist_lst = []
	n_lst = []
	for true_ts, pred_ts, B, E in zip(all_event_true, all_event_pred, t_b_plus, t_e_plus):
		dist, n = w_dist(
			true_ts,
			pred_ts,
			B, E,
		)
		dist_lst.append(dist)
		n_lst.append(n)

	sum_dist = np.sum(dist_lst)
	n = np.sum(n_lst)
	avg_dist = sum_dist / n * 1.

	return sum_dist, dist_lst

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def evaluate_query_2(
	event_dist_params, all_times_pred, all_times_true,
	horizon_start, horizon_end,
	interval_size, normalizers,
	less_threshold, more_threshold,
	use_dist=None,
):
	norm_a, norm_d = normalizers
	num_intervals = 1000

	idx_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_times_pred, horizon_start)]
	idx_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_times_pred, horizon_end)]
	event_dist_means = [event_dist_params[0][idx][idx_tb[idx]:idx_te[idx]] for idx in range(len(horizon_start))]
	event_dist_sigms = [event_dist_params[1][idx][idx_tb[idx]:idx_te[idx]] for idx in range(len(horizon_start))]

	idx_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(all_times_true, horizon_start)]
	idx_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(all_times_true, horizon_end)]
	all_times_true = [all_times_true[idx][idx_tb[idx]:idx_te[idx]] for idx in range(len(horizon_start))]

	horizon_end_norm = utils.normalize_avg_given_param(
		horizon_end - horizon_start,			
		norm_a, norm_d,
	)
	interval_size_norm = utils.normalize_avg_given_param(
		interval_size, norm_a, norm_d,
	)

	more_mismatch_pe, less_mismatch_pe = [], []
	for batch_idx in range(len(all_times_true)):
		batch_true_shifted = utils.normalize_avg_given_param(
			all_times_true[batch_idx] - horizon_start[batch_idx],
			norm_a, norm_d,
		)

		batch_means = event_dist_means[batch_idx]
		batch_sigms = event_dist_sigms[batch_idx]
		batch_sigms = np.where(batch_sigms==0., np.ones_like(batch_sigms)*1e-6, batch_sigms)

		all_intervals = np.linspace(
			0., horizon_end_norm[batch_idx] - interval_size_norm,
			num=num_intervals,
		)

		batch_means_shifted = np.cumsum(batch_means)
		if use_dist in ['poisson_binomial']:
			dist = tfd.Normal(loc=batch_means_shifted, scale=batch_sigms)
			interval_begin_cdf = dist.cdf(all_intervals)
			interval_end_cdf = dist.cdf(all_intervals + interval_size_norm)
			success_prob = interval_end_cdf - interval_begin_cdf # num_intervals x num_events
			pb_mean = tf.reduce_sum(success_prob, axis=1)
			pb_var = tf.maximum(tf.reduce_sum(success_prob*(1.-success_prob), axis=1), 1e-6)
			pb_skew = tf.pow(pb_var, -1.5) * tf.reduce_sum(success_prob*(1.-success_prob)*(1.-2*success_prob), axis=1)
			#pb_threshold_cdf = utils.refined_normal_approx(pb_mean, pb_var, pb_skew, more_threshold[batch_idx]) 
			#pb_threshold_cdf = utils.normal_approx(pb_mean, pb_var, more_threshold[batch_idx]) 
			pb_threshold_cdf = tfd.Normal(loc=pb_mean, scale=pb_var).cdf(more_threshold[batch_idx])
			#import ipdb
			#ipdb.set_trace()
			if np.sum(1.-pb_threshold_cdf) > 0.:
				more_pred_prob = (1.-pb_threshold_cdf) / tf.reduce_sum(1. - pb_threshold_cdf)
			else:
				more_pred_prob = (1.-pb_threshold_cdf)
			if np.sum(pb_threshold_cdf) > 0.:
				less_pred_prob = (pb_threshold_cdf) / tf.reduce_sum(pb_threshold_cdf)
			else:
				less_pred_prob = (pb_threshold_cdf)

			#import ipdb
			#ipdb.set_trace()

			true_counts = np.sum(
				np.logical_and(
					(batch_true_shifted - all_intervals)>0.,
					(batch_true_shifted - (all_intervals + interval_size_norm)<=0.)
				),
				axis=1
			)
			more_true = (true_counts >= more_threshold[batch_idx]).astype(np.float32)
			more_true_prob = more_true * 1. / np.sum(more_true)

			more_mismatch = (1.-more_pred_prob)*more_true + more_pred_prob*(1.-more_true)
			more_mismatch_pe.append(more_mismatch)


			less_true = (true_counts <= less_threshold[batch_idx]).astype(np.float32)
			less_true_prob = less_true * 1. / np.sum(less_true)

			less_mismatch = (1.-less_pred_prob)*less_true + less_pred_prob*(1.-less_true)
			less_mismatch_pe.append(less_mismatch)

			#if np.isnan(np.sum(more_mismatch_pe[-1])) or np.isnan(np.sum(less_mismatch_pe[-1])):
			#	import ipdb
			#	ipdb.set_trace()

		elif use_dist in ['binomial']:
			dist = tfd.Normal(loc=batch_means_shifted, scale=batch_sigms)
			interval_begin_cdf = dist.cdf(all_intervals)
			interval_end_cdf = dist.cdf(all_intervals + interval_size_norm)
			success_prob = interval_end_cdf - interval_begin_cdf # num_intervals x num_events
			success_prob = success_prob.numpy()
			success_prob[success_prob==1.] = 0.999999
			import ipdb
			ipdb.set_trace()
			bino_dist = tfd.inomial(total_count=4., probs=success_prob)
			pb_threshold_cdf = utils.normal_approx(pb_mean, pb_var, more_threshold[batch_idx])
			#import ipdb
			#ipdb.set_trace()
			more_pred_prob = (1.-pb_threshold_cdf) / tf.reduce_sum(1. - pb_threshold_cdf)
			less_pred_prob = (pb_threshold_cdf) / tf.reduce_sum(pb_threshold_cdf)

			true_counts = np.sum(
				np.logical_and(
					(batch_true_shifted - all_intervals)>0.,
					(batch_true_shifted - (all_intervals + interval_size_norm)<=0.)
				),
				axis=1
			)
			more_true = (true_counts >= more_threshold[batch_idx]).astype(np.float32)
			more_true_prob = more_true * 1. / np.sum(more_true)

			more_mismatch = (1.-more_pred_prob)*more_true + more_pred_prob*(1.-more_true)
			more_mismatch_pe.append(more_mismatch)

			#import ipdb
			#ipdb.set_trace()

			less_true = (true_counts <= less_threshold[batch_idx]).astype(np.float32)
			less_true_prob = less_true * 1. / np.sum(less_true)

			less_mismatch = (1.-less_pred_prob)*less_true + less_pred_prob*(1.-less_true)
			less_mismatch_pe.append(less_mismatch)

		else:
			dist = tfd.Normal(loc=batch_means_shifted, scale=batch_sigms)
			interval_begin_cdf = dist.cdf(all_intervals)
			interval_end_cdf = dist.cdf(all_intervals + interval_size_norm)
			pred_counts = np.sum(interval_end_cdf - interval_begin_cdf, axis=1)
	
			#import ipdb
			#ipdb.set_trace()
			dist_true = tfd.Normal(
				loc=batch_true_shifted,
				scale=(np.ones_like(batch_true_shifted)*1e-6).astype(np.float64)
			)
			interval_begin_cdf = dist_true.cdf(all_intervals)
			interval_end_cdf = dist_true.cdf(all_intervals + interval_size_norm)
			true_counts = np.sum(interval_end_cdf - interval_begin_cdf, axis=1)
	
			more_true = (true_counts >= more_threshold[batch_idx]).astype(np.float32)
			more_pred = (pred_counts >= more_threshold[batch_idx]).astype(np.float32)
			more_mismatch = np.sum(np.abs(more_true-more_pred))
			more_mismatch_pe.append(more_mismatch)
	
			less_true = (true_counts <= less_threshold[batch_idx]).astype(np.float32)
			less_pred = (pred_counts <= less_threshold[batch_idx]).astype(np.float32)
			less_mismatch = np.sum(np.abs(less_true-less_pred))
			less_mismatch_pe.append(less_mismatch)

	more_mismatch_pe = np.array(more_mismatch_pe)
	less_mismatch_pe = np.array(less_mismatch_pe)

	more_metric = np.sum(more_mismatch_pe) / (len(more_mismatch_pe) * num_intervals) * 1.
	less_metric = np.sum(less_mismatch_pe) / (len(less_mismatch_pe) * num_intervals) * 1.

	more_metric_pe = more_mismatch_pe / num_intervals
	less_metric_pe = less_mismatch_pe / num_intervals


	return (
		more_metric, less_metric,
		more_metric_pe, less_metric_pe
	)






#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Threshold MAE Loss calculations
# Query 2 and 3
def compute_threshold_loss(all_event_pred_uncut, query_data):
	[interval_range_count_less, interval_range_count_more,
	less_threshold, more_threshold, interval_size, _] = query_data

	all_times_pred = list()
	for idx in range(len(all_event_pred_uncut)):
		lst = list()
		for each in all_event_pred_uncut[idx]:
			lst = lst+each.tolist()
		all_times_pred.append(lst)
	all_times_pred = np.array(all_times_pred)

	interval_range_count_more_pred = utils.get_interval_count_more_than_threshold(all_times_pred, interval_size, more_threshold)

	lst = list()
	for idx in range(len(interval_range_count_more_pred)):
		if not(interval_range_count_more_pred[idx]==-1 or interval_range_count_more[idx]==-1):
			lst.append(np.abs(interval_range_count_more_pred[idx] - interval_range_count_more[idx]))
	threshold_mae_more = np.mean(np.array(lst))
	print()
	print('counting ', len(lst), 'testcase out of ', len(interval_range_count_more))
	print('MAE of computing range of more events than threshold:', threshold_mae_more)

	interval_range_count_less_pred = utils.get_interval_count_less_than_threshold(all_times_pred, interval_size, less_threshold)
	lst = list()
	for idx in range(len(interval_range_count_less_pred)):
		if not(interval_range_count_less_pred[idx]==-1 or interval_range_count_less[idx]==-1):
			lst.append(np.abs(interval_range_count_less_pred[idx] - interval_range_count_less[idx]))
	threshold_mae_less = np.mean(np.array(lst))
	print('counting ', len(lst), 'testcase out of ', len(interval_range_count_less))
	print('MAE of computing range of less events than threshold:', threshold_mae_less)
	print()

	return threshold_mae_less, threshold_mae_more
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Interval pdf prediction loss
# Query 2 and 3
def compute_time_range_pdf(all_run_fun_pdf, model_data, query_data, dataset_name):
	[arguments, models, data, test_data] = model_data
	[interval_range_count_less, interval_range_count_more, less_threshold,
	more_threshold, interval_size, event_test_out_times] = query_data

	[count_test_in_counts, count_test_in_feats,
	 count_test_out_counts, count_test_out_binend, event_test_in_lasttime,
	 event_test_in_gaps, event_test_in_feats, event_test_in_types,
	 count_test_normm, count_test_norms,
	 event_test_norma, event_test_normd] = test_data

	all_run_count_fun_name, all_run_count_fun, all_run_count_fun_rmtpp = list(), list(), list()
	for each in all_run_fun_pdf:
		all_run_count_fun_name.append(each)
		all_run_count_fun.append(all_run_fun_pdf[each][0])
		all_run_count_fun_rmtpp.append(all_run_fun_pdf[each][1])

	sample_count = 30
	no_points = 500
	test_plots_cnts=25

	x_range = np.round(np.array([(event_test_in_lasttime), (event_test_in_lasttime+(arguments.bin_size*arguments.out_bin_sz))]))[:,:,0].T.astype(int)

	interval_counts_more_true = np.zeros((len(event_test_in_lasttime), no_points))
	interval_counts_less_true = np.zeros((len(event_test_in_lasttime), no_points))

	for batch_idx in range(len(event_test_in_lasttime)):
		all_begins = np.linspace(x_range[batch_idx][0], x_range[batch_idx][1], no_points)
		for begin_idx in range(len(all_begins)):
			interval_start_cand = np.array([all_begins[begin_idx]])
			interval_end_cand = np.array([all_begins[begin_idx] + interval_size])
			interval_count_in_range_pred = count_events(np.expand_dims(np.array(event_test_out_times[batch_idx]), axis=0), 
														interval_start_cand, interval_end_cand)
			if more_threshold[batch_idx] <= interval_count_in_range_pred:
				interval_counts_more_true[batch_idx][begin_idx]+=1.0

			if less_threshold[batch_idx] >= interval_count_in_range_pred:
				interval_counts_less_true[batch_idx][begin_idx]+=1.0

	# interval_counts_more_true = interval_counts_more_true / np.expand_dims(np.sum(interval_counts_more_true, axis=1), axis=-1)
	# interval_counts_less_true = interval_counts_less_true / np.expand_dims(np.sum(interval_counts_less_true, axis=1), axis=-1)

	more_results = list()
	less_results = list()
	more_results_rank = list()
	less_results_rank = list()

	for run_count_fun_idx in range(len(all_run_count_fun)):
		print('Running for model', all_run_count_fun_name[run_count_fun_idx])
		interval_counts_more = np.zeros((len(event_test_in_lasttime), no_points))
		interval_counts_less = np.zeros((len(event_test_in_lasttime), no_points))
		interval_counts_more_rank = np.zeros((len(event_test_in_lasttime), no_points))
		interval_counts_less_rank = np.zeros((len(event_test_in_lasttime), no_points))

		for each_sim_idx in range(sample_count):
			# print('Simulating sample number', each_sim_idx)

			if all_run_count_fun_name[run_count_fun_idx] == 'wgan_simu':
				simul_end_time = x_range[:,1]
				_, all_event_pred_uncut = all_run_count_fun[run_count_fun_idx](arguments, models, data, test_data, simul_end_time=simul_end_time)

			elif (all_run_count_fun_name[run_count_fun_idx] == 'rmtpp_mse_simu' or \
				 all_run_count_fun_name[run_count_fun_idx] == 'rmtpp_mse_var_simu' or \
				 all_run_count_fun_name[run_count_fun_idx] == 'rmtpp_nll_simu'):

				simul_end_time = x_range[:,1]
				_, all_event_pred_uncut = all_run_count_fun[run_count_fun_idx](arguments, models, data, test_data,
																				simul_end_time=simul_end_time, 
																				rmtpp_type=all_run_count_fun_rmtpp[run_count_fun_idx])

			else:
				_, all_event_pred_uncut = all_run_count_fun[run_count_fun_idx](arguments, models, data, test_data, 
																				rmtpp_type=all_run_count_fun_rmtpp[run_count_fun_idx])

			all_times_pred = list()
			for idx in range(len(all_event_pred_uncut)):
				lst = list()
				for each in all_event_pred_uncut[idx]:
					lst = lst+each.tolist()
				all_times_pred.append(lst)
			all_times_pred = np.array(all_times_pred)

			for batch_idx in range(len(all_times_pred)):
				all_begins = np.linspace(x_range[batch_idx][0], x_range[batch_idx][1], no_points)
				interval_position_more = 1.0
				interval_position_less = 1.0
				for begin_idx in range(len(all_begins)):
					interval_start_cand = np.array([all_begins[begin_idx]])
					interval_end_cand = np.array([all_begins[begin_idx] + interval_size])

					interval_count_in_range_pred = count_events(all_times_pred[batch_idx:batch_idx+1], interval_start_cand, interval_end_cand)
					if more_threshold[batch_idx] <= interval_count_in_range_pred:
						interval_counts_more[batch_idx][begin_idx]+=1.0
						interval_counts_more_rank[batch_idx][begin_idx]+=(1.0/interval_position_more)
						interval_position_more += 1.0

					if less_threshold[batch_idx] >= interval_count_in_range_pred:
						interval_counts_less[batch_idx][begin_idx]+=1.0
						interval_counts_less_rank[batch_idx][begin_idx]+=(1.0/interval_position_less)
						interval_position_less += 1.0

		more_results.append(interval_counts_more)
		less_results.append(interval_counts_less)
		more_results_rank.append(interval_counts_more_rank)
		less_results_rank.append(interval_counts_less_rank)

	more_results = np.array(more_results)
	less_results = np.array(less_results)
	more_results_rank = np.array(more_results_rank)
	less_results_rank = np.array(less_results_rank)

	all_begins = np.linspace(x_range[:,0], x_range[:,1], no_points).T
	crps_loss_more = -1*np.ones((len(all_run_count_fun)))
	crps_loss_less = -1*np.ones((len(all_run_count_fun)))
	cross_entropy_more = -1*np.ones((len(all_run_count_fun)))
	cross_entropy_less = -1*np.ones((len(all_run_count_fun)))
	for run_count_fun_idx in range(len(all_run_count_fun)):
		# CRPS calculations
		all_counts_sum_more = np.sum(more_results_rank[run_count_fun_idx], axis=1)
		all_counts_sum_less = np.sum(less_results_rank[run_count_fun_idx], axis=1)

		more_results_rank[run_count_fun_idx][(all_counts_sum_more == 0)] = np.ones_like(more_results_rank[run_count_fun_idx][(all_counts_sum_more == 0)])
		less_results_rank[run_count_fun_idx][(all_counts_sum_less == 0)] = np.ones_like(less_results_rank[run_count_fun_idx][(all_counts_sum_less == 0)])

		all_counts_sum_more = np.expand_dims(np.sum(more_results_rank[run_count_fun_idx], axis=1), axis=-1)
		all_counts_sum_less = np.expand_dims(np.sum(less_results_rank[run_count_fun_idx], axis=1), axis=-1)

		crps_loss_more[run_count_fun_idx] = np.mean(ps.crps_ensemble(interval_range_count_more, all_begins, weights=more_results_rank[run_count_fun_idx]/all_counts_sum_more))
		crps_loss_less[run_count_fun_idx] = np.mean(ps.crps_ensemble(interval_range_count_less, all_begins, weights=less_results_rank[run_count_fun_idx]/all_counts_sum_more))

		# Cross Entropy calculations
		all_counts_sum_more = np.sum(more_results[run_count_fun_idx], axis=1)
		all_counts_sum_less = np.sum(less_results[run_count_fun_idx], axis=1)

		more_results[run_count_fun_idx][(all_counts_sum_more == 0)] = np.ones_like(more_results[run_count_fun_idx][(all_counts_sum_more == 0)])
		less_results[run_count_fun_idx][(all_counts_sum_less == 0)] = np.ones_like(less_results[run_count_fun_idx][(all_counts_sum_less == 0)])

		all_counts_sum_more = np.expand_dims(np.sum(more_results[run_count_fun_idx], axis=1), axis=-1)
		all_counts_sum_less = np.expand_dims(np.sum(less_results[run_count_fun_idx], axis=1), axis=-1)

		cross_entropy_more[run_count_fun_idx] = np.mean((interval_counts_more_true * (1.0 - more_results[run_count_fun_idx]/all_counts_sum_more)) +\
												((1.0 - interval_counts_more_true) * more_results[run_count_fun_idx]/all_counts_sum_more))

		cross_entropy_less[run_count_fun_idx] = np.mean((interval_counts_less_true * (1.0 - less_results[run_count_fun_idx]/all_counts_sum_less)) +\
												((1.0 - interval_counts_less_true) * less_results[run_count_fun_idx]/all_counts_sum_less))

	print("CRPS for More")
	for run_count_fun_idx in range(len(all_run_count_fun)):
		print("Model", all_run_count_fun_name[run_count_fun_idx], ": Score =", crps_loss_more[run_count_fun_idx])

	print("CRPS for Less")
	for run_count_fun_idx in range(len(all_run_count_fun)):
		print("Model", all_run_count_fun_name[run_count_fun_idx], ": Score =", crps_loss_less[run_count_fun_idx])

	print("Cross Entropy for More")
	for run_count_fun_idx in range(len(all_run_count_fun)):
		print("Model", all_run_count_fun_name[run_count_fun_idx], ": Score =", cross_entropy_more[run_count_fun_idx])

	print("Cross Entropy for Less")
	for run_count_fun_idx in range(len(all_run_count_fun)):
		print("Model", all_run_count_fun_name[run_count_fun_idx], ": Score =", cross_entropy_less[run_count_fun_idx])

	# Plots
	if test_plots_cnts is None:
		test_plots_cnts = len(event_test_in_lasttime)

	os.makedirs(os.path.join(args.output_dir, dataset_name+'_threshold_less/'), exist_ok=True)
	os.makedirs(os.path.join(args.output_dir, dataset_name+'_threshold_more/'), exist_ok=True)
	os.makedirs(os.path.join(args.output_dir, dataset_name+'_threshold_less_rank/'), exist_ok=True)
	os.makedirs(os.path.join(args.output_dir, dataset_name+'_threshold_more_rank/'), exist_ok=True)
	for batch_idx in range(test_plots_cnts):
		all_begins = np.linspace(x_range[batch_idx][0], x_range[batch_idx][1], no_points)
		plt.plot(all_begins, (interval_counts_more_true[batch_idx] / np.sum(interval_counts_more_true[batch_idx])), 
				 label='True Preds')
		for run_count_fun_idx in range(len(all_run_count_fun)):
			all_counts_sum = max(1, np.sum(more_results[run_count_fun_idx,batch_idx]))
			plt.plot(all_begins, (more_results[run_count_fun_idx,batch_idx] / all_counts_sum), 
					 label=all_run_count_fun_name[run_count_fun_idx])
		plt.xlabel('timeline')
		plt.ylabel('pdf_threshold_more')
		plt.axvline(x=interval_range_count_more[batch_idx], color='red', linestyle='--')
		img_name_cnt = args.output_dir+'/'+dataset_name+'_threshold_more/'+dataset_name+'_threshold_more_'+str(batch_idx)+'.svg'
		plt.legend(loc='upper right')
		plt.savefig(img_name_cnt, format='svg', dpi=1200)
		plt.close()

		plt.plot(all_begins, (interval_counts_less_true[batch_idx] / np.sum(interval_counts_less_true[batch_idx])), 
				 label='True Preds')
		for run_count_fun_idx in range(len(all_run_count_fun)):
			all_counts_sum = max(1, np.sum(less_results[run_count_fun_idx,batch_idx]))
			plt.plot(all_begins, (less_results[run_count_fun_idx,batch_idx] / all_counts_sum), 
					 label=all_run_count_fun_name[run_count_fun_idx])
		plt.xlabel('timeline')
		plt.ylabel('pdf_threshold_less')
		plt.axvline(x=interval_range_count_less[batch_idx], color='red', linestyle='--')
		img_name_cnt = args.output_dir+'/'+dataset_name+'_threshold_less/'+dataset_name+'_threshold_less_'+str(batch_idx)+'.svg'
		plt.legend(loc='upper right')
		plt.savefig(img_name_cnt, format='svg', dpi=1200)
		plt.close()

		for run_count_fun_idx in range(len(all_run_count_fun)):
			all_counts_sum = max(1, np.sum(more_results_rank[run_count_fun_idx,batch_idx]))
			plt.plot(all_begins, (more_results_rank[run_count_fun_idx,batch_idx] / all_counts_sum), 
					 label=all_run_count_fun_name[run_count_fun_idx])
		plt.xlabel('timeline')
		plt.ylabel('pdf_threshold_more_rank')
		plt.axvline(x=interval_range_count_more[batch_idx], color='red', linestyle='--')
		img_name_cnt = args.output_dir+'/'+dataset_name+'_threshold_more_rank/'+dataset_name+'_threshold_more_rank_'+str(batch_idx)+'.svg'
		plt.legend(loc='upper right')
		plt.savefig(img_name_cnt, format='svg', dpi=1200)
		plt.close()

		for run_count_fun_idx in range(len(all_run_count_fun)):
			all_counts_sum = max(1, np.sum(less_results_rank[run_count_fun_idx,batch_idx]))
			plt.plot(all_begins, (less_results_rank[run_count_fun_idx,batch_idx] / all_counts_sum), 
					 label=all_run_count_fun_name[run_count_fun_idx])
		plt.xlabel('timeline')
		plt.ylabel('pdf_threshold_less_rank')
		plt.axvline(x=interval_range_count_less[batch_idx], color='red', linestyle='--')
		img_name_cnt = args.output_dir+'/'+dataset_name+'_threshold_less_rank/'+dataset_name+'_threshold_less_rank_'+str(batch_idx)+'.svg'
		plt.legend(loc='upper right')
		plt.savefig(img_name_cnt, format='svg', dpi=1200)
		plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#####################################################
# 			Model Init and Run handler				#
#####################################################
def run_model(dataset_name, model_name, dataset, args, results, prev_models=None, run_model_flags=None):
	print("Running for model", model_name, "on dataset", dataset_name)

	print('BIN SIZE:', args.bin_size)

	tf.random.set_seed(args.seed)
	count_test_out_counts = dataset['count_test_out_counts']
	event_count_preds_true = count_test_out_counts
	batch_size = args.batch_size
	model=None
	result=None
	count_dist_params = {}


	if model_name == 'hierarchical':
		count_train_in_counts = dataset['count_train_in_counts']
		count_train_in_feats = dataset['count_train_in_feats']
		count_train_out_counts = dataset['count_train_out_counts']
		count_test_in_counts = dataset['count_test_in_counts']
		count_test_in_feats = dataset['count_test_in_feats']
		count_test_out_counts = dataset['count_test_out_counts']
		count_test_normm = dataset['count_test_normm']
		count_test_norms = dataset['count_test_norms']

		data = [count_train_in_counts, count_train_out_counts]
		test_data = [count_test_in_counts, count_test_in_feats, count_test_out_counts, count_test_normm, count_test_norms]
		event_count_preds_cnt = run_hierarchical(args, data, test_data)
		model, result = event_count_preds_cnt

	if model_name == 'count_model':
		count_train_in_counts = dataset['count_train_in_counts']
		count_train_in_feats = dataset['count_train_in_feats']
		count_train_out_counts = dataset['count_train_out_counts']
		count_dev_in_counts = dataset['count_dev_in_counts']
		count_dev_in_feats = dataset['count_dev_in_feats']
		count_dev_out_counts = dataset['count_dev_out_counts']
		count_test_in_counts = dataset['count_test_in_counts']
		count_test_in_feats = dataset['count_test_in_feats']
		count_test_out_counts = dataset['count_test_out_counts']
		count_test_normm = dataset['count_test_normm']
		count_test_norms = dataset['count_test_norms']

		data = [count_train_in_counts, count_train_in_feats, count_train_out_counts,
				count_dev_in_counts, count_dev_in_feats, count_dev_out_counts]
		test_data = [count_test_in_counts, count_test_in_feats,
					 count_test_out_counts, count_test_normm, count_test_norms]
		model, count_dist_params = run_count_model(args, data, test_data)

	if model_name in ['rmtpp_nll', 'rmtpp_mse', 'rmtpp_mse_var', 
					  'rmtpp_nll_comp', 'rmtpp_mse_comp', 'rmtpp_mse_var_comp',
					  'pure_hierarchical_nll', 'pure_hierarchical_mse', 
					  'wgan', 'seq2seq', 'transformer',
					  'inference_models', 'hawkes_model']:
					  
		nc_event_train_in_gaps = dataset['nc_event_train_in_gaps'].astype(np.float32)
		nc_event_train_in_feats = dataset['nc_event_train_in_feats'].astype(np.float32)
		nc_event_train_in_types = dataset['nc_event_train_in_types']
		nc_event_train_out_gaps = dataset['nc_event_train_out_gaps'].astype(np.float32)
		nc_event_train_out_feats = dataset['nc_event_train_out_feats'].astype(np.float32)
		nc_event_train_out_types = dataset['nc_event_train_out_types']
		train_dataset_gaps = tf.data.Dataset.from_tensor_slices(
			(nc_event_train_in_gaps, nc_event_train_in_feats, nc_event_train_in_types,
			 nc_event_train_out_gaps, nc_event_train_out_feats, nc_event_train_out_types)
		).batch(
			batch_size,
			drop_remainder=True
		)
		nc_event_dev_in_gaps = dataset['nc_event_dev_in_gaps']
		nc_event_dev_in_feats = dataset['nc_event_dev_in_feats'].astype(np.float32)
		nc_event_dev_in_types = dataset['nc_event_dev_in_types']
		nc_event_dev_out_gaps = dataset['nc_event_dev_out_gaps']
		nc_event_dev_out_types = dataset['nc_event_dev_out_types']
		event_train_norma = dataset['event_train_norma']
		event_train_normd = dataset['event_train_normd']

		event_test_in_gaps = dataset['event_test_in_gaps']
		event_test_in_feats = dataset['event_test_in_feats'].astype(np.float32)
		event_test_in_types = dataset['event_test_in_types']
		count_test_out_binend = dataset['count_test_out_binend'] 
		event_test_in_lasttime = dataset['event_test_in_lasttime']
		event_test_norma = dataset['event_test_norma'] 
		event_test_normd = dataset['event_test_normd']

		test_data = [event_test_in_gaps, event_test_in_feats, event_test_in_types,
					 count_test_out_binend, event_test_in_lasttime,
					 event_test_norma, event_test_normd]
		train_norm_gaps = [event_train_norma ,event_train_normd]
		data = [train_dataset_gaps,
				nc_event_dev_in_gaps, nc_event_dev_in_feats, nc_event_dev_in_types,
				nc_event_dev_out_gaps, nc_event_dev_out_types,
				train_norm_gaps]


		nc_comp_train_in_gaps = dataset['nc_comp_train_in_gaps']
		nc_comp_train_in_feats = dataset['nc_comp_train_in_feats'].astype(np.float32)
		nc_comp_train_in_types = dataset['nc_comp_train_in_types']
		nc_comp_train_out_gaps = dataset['nc_comp_train_out_gaps']
		nc_comp_train_out_feats = dataset['nc_comp_train_out_feats'].astype(np.float32)
		nc_comp_train_out_types = dataset['nc_comp_train_out_types']
		train_dataset_gaps_comp = tf.data.Dataset.from_tensor_slices(
			(nc_comp_train_in_gaps, nc_comp_train_in_feats, nc_comp_train_in_types,
			 nc_comp_train_out_gaps, nc_comp_train_out_feats, nc_comp_train_out_types)
		).batch(
			batch_size,
			drop_remainder=True
		)
		nc_comp_dev_in_gaps = dataset['nc_comp_dev_in_gaps']
		nc_comp_dev_in_feats = dataset['nc_comp_dev_in_feats']
		nc_comp_dev_in_types = dataset['nc_comp_dev_in_types']
		nc_comp_dev_out_gaps = dataset['nc_comp_dev_out_gaps']
		nc_comp_dev_out_types = dataset['nc_comp_dev_out_types']
		comp_train_norma = dataset['comp_train_norma']
		comp_train_normd = dataset['comp_train_normd']

		comp_test_in_gaps = dataset['comp_test_in_gaps']
		comp_test_in_feats = dataset['comp_test_in_feats'].astype(np.float32)
		comp_test_in_types = dataset['comp_test_in_types']
		comp_test_norma = dataset['comp_test_norma']
		comp_test_normd = dataset['comp_test_normd']

		test_data_comp = [comp_test_in_gaps, comp_test_in_feats, comp_test_in_types,
						  count_test_out_binend, event_test_in_lasttime,
						  comp_test_norma, comp_test_normd]
		train_norm_gaps_comp = [comp_train_norma ,comp_train_normd]
		data_comp = [train_dataset_gaps_comp,
					 nc_comp_dev_in_gaps, nc_comp_dev_in_feats, nc_comp_dev_in_types,
					 nc_comp_dev_out_gaps, nc_comp_dev_out_types,
					 train_norm_gaps_comp]

		var_data = None

		if model_name == 'wgan':
			model = run_wgan(args, data, test_data)

		if model_name == 'seq2seq':
			model = run_seq2seq(args, data, test_data)

		if model_name == 'transformer':
			model = run_transformer(args, data, test_data)

		if model_name == 'rmtpp_nll':
			model, _ = run_rmtpp_init(
				args,
				data,
				test_data,
				var_data,
				NLL_loss=True,
				rmtpp_type='nll'
			)

		if model_name == 'rmtpp_mse':
			model, rmtpp_var_model = run_rmtpp_init(
				args,
				data,
				test_data,
				var_data,
				NLL_loss=False,
				rmtpp_type='mse'
			)

		if model_name == 'rmtpp_mse_var':
			model, _ = run_rmtpp_init(
				args,
				data,
				test_data,
				var_data,
				NLL_loss=False,
				use_var_model=True,
				rmtpp_type='mse_var'
			)

		if model_name == 'rmtpp_nll_comp':
			model, rmtpp_var_model = run_rmtpp_comp_init(
				args,
				data_comp,
				test_data_comp,
				var_data,
				NLL_loss=True,
				rmtpp_type='nll'
			)

		if model_name == 'rmtpp_mse_comp':
			model, rmtpp_var_model = run_rmtpp_comp_init(
				args,
				data_comp,
				test_data_comp,
				var_data,
				NLL_loss=False,
				rmtpp_type='mse'
			)

		if model_name == 'rmtpp_mse_var_comp':
			model, rmtpp_var_model = run_rmtpp_comp_init(
				args,
				data_comp,
				test_data_comp,
				var_data,
				NLL_loss=False,
				use_var_model=True,
				rmtpp_type='mse_var'
			)

		if model_name == 'pure_hierarchical_mse':
			model, _ = run_pure_hierarchical_init(args, data_comp_full, test_data_comp_full, NLL_loss=False)

		if model_name == 'pure_hierarchical_nll':
			model, _ = run_pure_hierarchical_init(args, data_comp_full, test_data_comp_full, NLL_loss=True)

		if model_name == 'hawkes_model':
			hawkes_timestamps_pred = dataset['hawkes_timestamps_pred']
			result, _ = run_hawkes_model(args, hawkes_timestamps_pred, data, test_data)
			
		# This block contains all the inference models that returns
		# answers to various queries
		if model_name == 'inference_models':

			event_test_out_gaps = dataset['event_test_out_gaps']
			event_test_out_types = dataset['event_test_out_types']
			event_test_out_times = dataset['event_test_out_times']

			count_test_in_counts = dataset['count_test_in_counts']
			count_test_in_feats = dataset['count_test_in_feats']
			count_test_out_counts = dataset['count_test_out_counts']
			count_test_normm = dataset['count_test_normm']
			count_test_norms = dataset['count_test_norms']

			test_time_out_tb_plus = dataset['test_time_out_tb_plus']
			test_time_out_te_plus = dataset['test_time_out_te_plus']
			test_out_event_count_true = dataset['test_out_event_count_true']
			test_out_all_event_true = dataset['test_out_all_event_true']

			interval_range_count_less = dataset['interval_range_count_less']
			interval_range_count_more = dataset['interval_range_count_more']
			less_threshold = dataset['less_threshold']
			more_threshold = dataset['more_threshold']
			interval_size = dataset['interval_size']
			all_times_true = dataset['event_test_out_times']
			all_types_true = dataset['event_test_out_types']
			for ts_seq, ty_seq in zip(all_times_true, all_types_true):
				assert len(ts_seq) == len(ty_seq)

			#model_cnt, model_rmtpp, model_wgan = prev_models['count_model'], prev_models['rmtpp_mse'], prev_models['wgan']
			models = prev_models
			count_test_in_counts = count_test_in_counts.astype(np.float32)
			count_test_out_counts = count_test_out_counts.astype(np.float32)
			test_data = [count_test_in_counts, count_test_in_feats,
						 count_test_out_counts, count_test_out_binend,
						 event_test_in_lasttime,
						 event_test_in_gaps, event_test_in_feats, event_test_in_types,
						 count_test_normm, count_test_norms,
						 event_test_norma, event_test_normd]
			all_times_true = np.array(all_times_true)


			data = None
			compute_depth = 5

			query_1_data = [test_time_out_tb_plus, test_time_out_te_plus, test_out_event_count_true]
			query_2_data = [interval_range_count_less, interval_range_count_more, 
							less_threshold, more_threshold, interval_size, all_times_true]

			#old_stdout = sys.stdout
			#sys.stdout=open("Outputs/count_model_"+dataset_name+".txt","a")
			print("____________________________________________________________________")
			print("True counts")
			print(test_out_event_count_true)
			print("____________________________________________________________________")
			print("")


			# ----- Start: Stale models, not updated according to latest code ----- #
			if 'run_rmtpp_count_with_optimization' in run_model_flags and run_model_flags['run_rmtpp_count_with_optimization']:
				print("Prediction for run_rmtpp_count_with_optimization model")
				_, all_times_pred = run_rmtpp_count_with_optimization(args, models, data, test_data)
				(
					deep_mae_rh,
					count_mae_rh,
					count_mse_rh,
					wass_dist_rh,
					deep_mae_rh_pe,
					wass_dist_rh_pe,
				) = compute_hierarchical_mae(
					all_times_pred,
					query_1_data,
					test_out_all_event_true,
					compute_depth
				)
				threshold_mae = compute_threshold_loss(all_times_pred, query_2_data)
				print("deep_mae", deep_mae_rh)
				deep_mae_fh, count_mae_fh, count_mae_fh_per_bin = compute_full_model_acc(
					args,
					test_data,
					None,
					all_times_pred,
					event_test_out_times,
					dataset_name,
					'run_rmtpp_count_with_optimization'
				)
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_with_optimization_fixed_cnt' in run_model_flags and run_model_flags['run_rmtpp_with_optimization_fixed_cnt']:
				print("Prediction for run_rmtpp_with_optimization_fixed_cnt model")
				_, all_times_pred = run_rmtpp_with_optimization_fixed_cnt(args, models, data, test_data)
				(
					deep_mae_rh,
					count_mae_rh,
					count_mse_rh,
					wass_dist_rh,
					deep_mae_rh_pe,
					wass_dist_rh_pe,
				) = compute_hierarchical_mae(
					all_times_pred,
					query_1_data,
					test_out_all_event_true,
					compute_depth
				)
				threshold_mae = compute_threshold_loss(all_times_pred, query_2_data)
				print("deep_mae", deep_mae_rh)
				deep_mae_fh, count_mae_fh, count_mae_fh_per_bin = compute_full_model_acc(
					args,
					test_data,
					None,
					all_times_pred,
					event_test_out_times,
					dataset_name,
					'run_rmtpp_with_optimization_fixed_cnt'
				)
				print("____________________________________________________________________")
				print("")
			# ----- End: Stale models, not updated according to latest code ----- #

			# ----- Start: Variants of Hierarchical model ----- #
			# (i) Pure hierarchical: Both compound and simple layer jointly trained
			# (ii) Hierarchical with count: Count loss also included in the optimizer of hierarchical model
			# Pure hierarchical model: jointly trained both simple layer and compound layer
			if 'run_pure_hierarchical_infer_nll' in run_model_flags and run_model_flags['run_pure_hierarchical_infer_nll']:
				print("Prediction for run_pure_hierarchical_infer_nll model")
				_, all_times_bin_pred = run_pure_hierarchical_infer(args, prev_models, data, test_data, test_data_comp_full, rmtpp_type='nll')

				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred, event_test_out_times, dataset_name, 'run_pure_hierarchical_infer_nll')
				print("____________________________________________________________________")
				print("")

			# Pure hierarchical model: jointly trained both simple layer and compound layer
			if 'run_pure_hierarchical_infer_mse' in run_model_flags and run_model_flags['run_pure_hierarchical_infer_mse']:
				print("Prediction for run_pure_hierarchical_infer_mse model")
				_, all_times_bin_pred = run_pure_hierarchical_infer(args, prev_models, data, test_data, test_data_comp_full, rmtpp_type='mse')

				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred, event_test_out_times, dataset_name, 'run_pure_hierarchical_infer_mse')
				print("____________________________________________________________________")
				print("")

			# Using both compound model and count model during inference
			if 'run_rmtpp_with_joint_optimization_fixed_cnt_solver_nll_comp' in run_model_flags and run_model_flags['run_rmtpp_with_joint_optimization_fixed_cnt_solver_nll_comp']:
				print("Prediction for run_rmtpp_with_joint_optimization_fixed_cnt_solver_nll_comp model")
				_, all_times_pred = run_rmtpp_with_joint_optimization_fixed_cnt_solver_comp(args, prev_models, data, test_data, test_data_comp, rmtpp_type='nll', rmtpp_type_comp='nll')

				deep_mae = compute_hierarchical_mae(all_times_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_pred, event_test_out_times, dataset_name, 'run_rmtpp_with_joint_optimization_fixed_cnt_solver_nll_comp')
				print("____________________________________________________________________")
				print("")

			# Using both compound model and count model during inference
			if 'run_rmtpp_with_joint_optimization_fixed_cnt_solver_mse_comp' in run_model_flags and run_model_flags['run_rmtpp_with_joint_optimization_fixed_cnt_solver_mse_comp']:
				print("Prediction for run_rmtpp_with_joint_optimization_fixed_cnt_solver_mse_comp model")
				_, all_times_pred = run_rmtpp_with_joint_optimization_fixed_cnt_solver_comp(args, prev_models, data, test_data, test_data_comp, rmtpp_type='mse', rmtpp_type_comp='mse')

				deep_mae = compute_hierarchical_mae(all_times_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_pred, event_test_out_times, dataset_name, 'run_rmtpp_with_joint_optimization_fixed_cnt_solver_mse_comp')
				print("____________________________________________________________________")
				print("")
			# ----- End: Variants of Hierarchical model ----- #

			dist_params_dict = dict()
			for inference_model_name in run_model_flags.keys():
				start_time = time.time()
				print("Running Inference Model: "+inference_model_name)
				test_data_out_gaps_bin = dataset['event_test_out_gaps']
				#if 'rmtpp_mse_opt' in run_model_flags and run_model_flags['rmtpp_mse_opt']:
				all_best_opt_nc_losses = 0.
				all_best_cont_nc_losses = 0.
				all_best_nc_count_losses = 0.
				if inference_model_name in \
					['rmtpp_nll_opt', 'rmtpp_mse_opt',
					 'rmtpp_mse_var_opt', 'rmtpp_mse_coopt',
					 'rmtpp_mse_var_coopt'] \
					and run_model_flags[inference_model_name]:
					(
						all_times_pred,
						all_types_pred,
						all_counts_pred,
						event_dist_params,
						count_dist_params,
						all_best_opt_nc_losses,
						all_best_cont_nc_losses,
						all_best_nc_count_losses,
					) = run_rmtpp_optimizer_model(
						args,
						models,
						data,
						test_data,
						test_data_out_gaps_bin,
						dataset,
						rmtpp_type=run_model_flags[inference_model_name]['rmtpp_type']
					)

				if inference_model_name in ['rmtpp_nll_cont', 'rmtpp_mse_cont', 'rmtpp_mse_var_cont'] \
					and run_model_flags[inference_model_name]:
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_rmtpp_rescaling_model(
						args, models, data, test_data,
						rmtpp_type=run_model_flags[inference_model_name]['rmtpp_type']
					)

				if inference_model_name in ['rmtpp_nll_reinit', 'rmtpp_mse_reinit', 'rmtpp_mse_var_reinit'] \
					and run_model_flags[inference_model_name]:
					(
						all_counts_pred, all_times_pred, all_types_pred
					)= run_rmtpp_count_reinit(
						args, models, data, test_data,
						rmtpp_type=run_model_flags[inference_model_name]['rmtpp_type']
					)

				if inference_model_name in ['rmtpp_nll_opt_comp', 'rmtpp_mse_opt_comp', 'rmtpp_mse_var_opt_comp'] \
					and run_model_flags[inference_model_name]:
					#test_data_out_gaps_bin = dataset['test_data_out_gaps_bin']
					#test_data_out_gaps_bin_comp = dataset['test_data_out_gaps_bin_comp']
					test_data_out_gaps_bin_comp = None
					(
						all_times_pred, all_types_pred, all_counts_pred,
						event_dist_params, count_dist_params,
					) = run_rmtpp_optimizer_model_comp(
						args,
						models,
						data,
						test_data,
						test_data_comp,
						test_data_out_gaps_bin,
						test_data_out_gaps_bin_comp,
						dataset,
						rmtpp_type=run_model_flags[inference_model_name]['rmtpp_type'],
						rmtpp_type_comp=run_model_flags[inference_model_name]['rmtpp_type_comp']
					)

				if inference_model_name in ['rmtpp_nll_cont_comp', 'rmtpp_mse_cont_comp', 'rmtpp_mse_var_cont_comp'] \
					and run_model_flags[inference_model_name]:
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_rmtpp_rescaling_model_comp(
						args,
						models,
						data,
						test_data,
						test_data_comp,
						rmtpp_type=run_model_flags[inference_model_name]['rmtpp_type'],
						rmtpp_type_comp=run_model_flags[inference_model_name]['rmtpp_type_comp']
					)

				if inference_model_name=='hawkes_simu' and run_model_flags[inference_model_name]:
					hawkes_timestamps_pred = dataset['hawkes_timestamps_pred']
					_, all_times_pred = run_hawkes_model(args, hawkes_timestamps_pred, None, test_data)

				if inference_model_name=='count_only' and run_model_flags['count_only']:
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_count_only_model(args, models, data, test_data)

				if inference_model_name in ['rmtpp_nll_simu', 'rmtpp_mse_simu', 'rmtpp_mse_var_simu'] \
					and run_model_flags[inference_model_name]:
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_rmtpp_simulation(
						args, models, data, test_data,
						rmtpp_type=run_model_flags[inference_model_name]['rmtpp_type']
					)
				if inference_model_name in ['rmtpp_nll_simu_nc', 'rmtpp_mse_simu_nc', 'rmtpp_mse_var_simu_nc'] \
					and run_model_flags[inference_model_name]:
					nc_gaps_in = np.expand_dims(pad_sequences(event_test_out_gaps, padding='post'), axis=-1).astype(np.float32)
					nc_types_in = pad_sequences(event_test_out_types, padding='post').astype(np.int64)
					nc_feats_in = np.expand_dims(get_time_features(pad_sequences(event_test_out_times, padding='post')), axis=-1).astype(np.float32)
					#nc_times_in = pad_sequences(event_test_out_times, padding='post').astype(np.float32)
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_rmtpp_simulation(
						args, models, data, test_data,
						rmtpp_type=run_model_flags[inference_model_name]['rmtpp_type'],
						use_nowcast=True,
						nc_gaps_in=nc_gaps_in,
						nc_feats_in=nc_feats_in,
						nc_types_in=nc_types_in,
					)

				if inference_model_name=='wgan_simu' and run_model_flags[inference_model_name]:
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_wgan_simulation(args, models, data, test_data)
	
				if inference_model_name=='seq2seq_simu' and run_model_flags[inference_model_name]:
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_seq2seq_simulation(args, models, data, test_data)
	
				if inference_model_name=='transformer_simu' and run_model_flags[inference_model_name]:
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_transformer_simulation(args, models, data, test_data)
				if inference_model_name=='transformer_simu_nc' and run_model_flags[inference_model_name]:
					nc_gaps_in = np.expand_dims(pad_sequences(event_test_out_gaps, padding='post'), axis=-1).astype(np.float32)
					nc_types_in = pad_sequences(event_test_out_types, padding='post').astype(np.int64)
					nc_feats_in = np.expand_dims(get_time_features(pad_sequences(event_test_out_times, padding='post')), axis=-1).astype(np.float32)
					#nc_times_in = pad_sequences(event_test_out_times, padding='post').astype(np.float32)
					(
						all_counts_pred, all_times_pred, all_types_pred,
						event_dist_params, count_dist_params,
					) = run_transformer_simulation(
						args, models, data, test_data,
						use_nowcast=True,
						nc_gaps_in=nc_gaps_in,
						nc_feats_in=nc_feats_in,
						nc_types_in=nc_types_in,
					)
				end_time = time.time()

				all_counts_true = count_test_out_counts
				for ts_seq, ty_seq in zip(all_times_pred, all_types_pred):
					assert len(ts_seq) == len(ty_seq)
				(
					count_mae_fh,
					wass_dist_fh,
					bleu_score_fh,
					count_mae_fh_pe,
					wass_dist_fh_pe,
					bleu_score_fh_pe,
					count_mae_fh_per_bin,
					wass_dist_fh_per_bin,
					bleu_score_fh_per_bin,
				) = compute_full_horizon_metrics(
					all_times_true,
					all_times_pred,
					all_types_true,
					all_types_pred,
					all_counts_true,
					all_counts_pred,
					count_test_out_binend,
					args.bin_size,
				)

				(
					count_mae_rh,
					wass_dist_rh,
					bleu_score_rh,
					count_mae_rh_pe,
					wass_dist_rh_pe,
					bleu_score_rh_pe,
				) = compute_random_horizon_metrics(
					all_times_true,
					all_times_pred,
					all_types_true,
					all_types_pred,
					test_time_out_tb_plus,
					test_time_out_te_plus,
				)

				# Evaluate Query 2
				#query_2_data = [interval_range_count_less, interval_range_count_more, 
				#less_threshold, more_threshold, interval_size, all_times_true]
				(
					more_metric, less_metric,
					more_metric_pe, less_metric_pe,
				) = evaluate_query_2(
					event_dist_params,
					all_times_pred,
					all_times_true,
					count_test_out_binend[:, 0] - args.bin_size,
					count_test_out_binend[:, -1],
					args.interval_size,
					(event_test_norma, event_test_normd),
					less_threshold, more_threshold,
					use_dist='poisson_binomial',
				)


				#(
				#	deep_mae_fh,
				#	count_mae_fh,
				#	count_mae_fh_per_bin,
				#	wass_dist_fh,
				#	count_mae_fh_pe,
				#	deep_mae_fh_pe,
				#	wass_dist_fh_pe,
				#	all_times_true,
				#	_,
				#	bleu_score_fh,
				#	bleu_score_fh_pe,
				#	all_types_true,
				#	all_types_pred,
				#) = compute_full_model_acc(
				#	args,
				#	test_data,
				#	None,
				#	all_times_pred,
				#	all_times_true,
				#	all_types_pred,
				#	all_types_true,
				#	dataset_name,
				#	inference_model_name
				#)
				#(
				#	deep_mae_rh,
				#	count_mae_rh,
				#	count_mse_rh,
				#	wass_dist_rh,
				#	deep_mae_rh_pe,
				#	wass_dist_rh_pe,
				#) = compute_hierarchical_mae(
				#	all_times_pred,
				#	query_1_data,
				#	test_out_all_event_true,
				#	compute_depth
				#)
				results = add_metrics_to_dict(
					results,
					inference_model_name,
					count_mae_fh,
					wass_dist_fh,
					bleu_score_fh,
					count_mae_rh,
					wass_dist_rh,
					bleu_score_rh,
					count_mae_fh_per_bin,
					wass_dist_fh_per_bin,
					bleu_score_fh_per_bin,
					more_metric,
					less_metric,
					end_time - start_time,
					np.mean(all_best_opt_nc_losses),
					np.mean(all_best_cont_nc_losses),
					np.mean(all_best_nc_count_losses),
				)
				write_arr_to_file(
					os.path.join(
						args.output_dir,
						args.current_dataset+'__'+inference_model_name,
					),
					all_times_true,
					all_times_pred,
					all_types_true,
					all_types_pred,
				)
				write_pe_metrics_to_file(
					os.path.join(
						args.output_dir,
						args.current_dataset+'__'+inference_model_name,
					),
					count_mae_fh_pe, wass_dist_fh_pe, bleu_score_fh_pe,
					more_metric_pe, less_metric_pe,
				)
				write_opt_losses_to_file(
					os.path.join(
						args.output_dir,
						args.current_dataset + '__' + inference_model_name,
					),
					all_best_opt_nc_losses,
					all_best_cont_nc_losses,
					all_best_nc_count_losses,
				)
				#threshold_mae = compute_threshold_loss(all_times_pred, query_2_data)
				print("____________________________________________________________________")
				print("")

				# Save count_dist_params for bin-count plots
				dist_params_dict[inference_model_name] = {'count_dist_params':count_dist_params}


			print("")

			if 'compute_time_range_pdf' in run_model_flags and run_model_flags['compute_time_range_pdf']:
				print("Running threshold query to generate pdf for all models")
				model_data = [args, models, data, test_data]
				all_run_fun_pdf = {
					'rmtpp_nll_opt': [run_rmtpp_optimizer_model, 'nll'],
					'rmtpp_mse_opt': [run_rmtpp_optimizer_model, 'mse'],
					'rmtpp_mse_var_opt': [run_rmtpp_optimizer_model, 'mse_var'],

					'rmtpp_nll_cont': [run_rmtpp_rescaling_model, 'nll'],
					'rmtpp_mse_cont': [run_rmtpp_rescaling_model, 'mse'],
					'rmtpp_mse_var_cont': [run_rmtpp_rescaling_model, 'mse_var'],
					'rmtpp_nll_reinit': [run_rmtpp_count_reinit, 'nll'],
					'rmtpp_mse_reinit': [run_rmtpp_count_reinit, 'mse'],
					'rmtpp_mse_var_reinit': [run_rmtpp_count_reinit, 'mse_var'],

					'rmtpp_nll_simu': [run_rmtpp_simulation, 'nll'],
					'rmtpp_mse_simu': [run_rmtpp_simulation, 'mse'],
					'rmtpp_mse_var_simu': [run_rmtpp_simulation, 'mse_var'],
					'wgan_simu': [run_wgan_simulation, None],
					'transformer_simu': [run_transformer_simulation, None],
					'count_only': [run_count_only_model, None],
				}
				clean_dict_for_na_model(all_run_fun_pdf, run_model_flags)
				compute_time_range_pdf(all_run_fun_pdf, model_data, query_2_data, dataset_name)
				print("____________________________________________________________________")
				print("")

			print("")
			#sys.stdout.close()
			#sys.stdout = old_stdout

			# ----- Start: Plot bin-counts ----- # 
			in_bin_sz_plt = 10
			for idx in range(10):
				inp_count = utils.denormalize_data(
					count_test_in_counts[idx, -in_bin_sz_plt:],
					count_test_normm, count_test_norms
				)
				x_range = np.arange(1, in_bin_sz_plt+args.out_bin_sz+1)
				out_count = count_test_out_counts[idx]
				plt.plot(x_range, np.concatenate([inp_count, out_count]), 'ko-', label='true counts')
				for inference_model_name in run_model_flags.keys():
					if inference_model_name not in ['transformer_simu']:
						out_mean = dist_params_dict[inference_model_name]['count_dist_params'][0][idx]
						plt.plot(x_range[in_bin_sz_plt:], out_mean, '*-', label=inference_model_name)
	
						out_std = dist_params_dict[inference_model_name]['count_dist_params'][1]
						if out_std is not None:
							out_std = out_std[idx]
							out_std_down = out_mean - out_std
							out_std_up = out_mean + out_std
							plt.fill_between(x_range[in_bin_sz_plt:], out_std_down, out_std_up, color='gray', alpha=0.5)
				
				plt.axvline(x=in_bin_sz_plt+.5, color='k', linestyle='--')
				plt.legend(loc='upper left')
				plt.savefig(args.output_dir+'/'+dataset_name+'_'+str(idx)+'.svg', format='svg', dpi=1200)
				plt.close()
			# ----- End: Plot bin-counts ----- # 

	if model_name != 'rmtpp_mse':
		rmtpp_var_model = None

	return model, count_dist_params, rmtpp_var_model, results

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


