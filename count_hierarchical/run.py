import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

import cvxpy as cp
import numpy as np
import pandas as pd
from itertools import chain
from bisect import bisect_right
from multiprocessing import Pool
import matplotlib.pyplot as plt
import properscoring as ps
from operator import itemgetter

import models
import utils
import os
import sys

from utils import IntensityHomogenuosPoisson, generate_sample

train_gap_metric_mae = tf.keras.metrics.MeanAbsoluteError()
train_gap_metric_mse = tf.keras.metrics.MeanSquaredError()
dev_gap_metric_mae = tf.keras.metrics.MeanAbsoluteError()
dev_gap_metric_mse = tf.keras.metrics.MeanSquaredError()
test_gap_metric_mae = tf.keras.metrics.MeanAbsoluteError()
test_gap_metric_mse = tf.keras.metrics.MeanSquaredError()

ETH = 10.0
one_by = tf.math.reciprocal_no_nan


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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# RMTPP model
def run_rmtpp(args, model, optimizer, data, NLL_loss, rmtpp_epochs=10, use_var_model=False):
	[train_dataset_gaps, dev_data_in_gaps, dev_data_out_gaps, train_norm_gaps] = data
	[train_norm_a_gaps, train_norm_d_gaps] = train_norm_gaps
	model_name = args.current_model

	os.makedirs('saved_models/training_'+model_name+'_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_"+model_name+"_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mse = np.inf
	best_dev_epoch = 0

	train_losses = list()
	for epoch in range(rmtpp_epochs):
		print('Starting epoch', epoch)
		step_train_loss = 0.0
		step_cnt = 0
		next_initial_state = None
		for sm_step, (gaps_batch_in, gaps_batch_out) in enumerate(train_dataset_gaps):
			with tf.GradientTape() as tape:
				# TODO: Make sure to pass correct next_stat
				gaps_pred, D, WT, next_initial_state, _ = model(gaps_batch_in, 
						initial_state=next_initial_state, 
						next_state_sno=args.batch_size)

				# Compute the loss for this minibatch.
				if use_var_model:
					gap_loss_fn = Gaussian_MSE(D, WT)
				elif NLL_loss:
					gap_loss_fn = NegativeLogLikelihood(D, WT)
				else:
					gap_loss_fn = MeanSquareLoss()
				
				gap_loss = gap_loss_fn(gaps_batch_out, gaps_pred)
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
		dev_gaps_pred, _, _, _, _ = model(dev_data_in_gaps)
		dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
										train_norm_a_gaps, train_norm_d_gaps)
		
		dev_gap_metric_mae(dev_data_out_gaps, dev_gaps_pred_unnorm)
		dev_gap_metric_mse(dev_data_out_gaps, dev_gaps_pred_unnorm)
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
	plt.savefig('Outputs/train_rmtpp_'+args.current_dataset+'_loss.png')
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	dev_gaps_pred, _, _, _, _ = model(dev_data_in_gaps)
	dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
									train_norm_a_gaps, train_norm_d_gaps)
	
	dev_gap_metric_mae(dev_data_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_metric_mse(dev_data_out_gaps, dev_gaps_pred_unnorm)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	print('Best MAE and MSE of Dev data %s: %s' \
		%(float(dev_gap_mae), float(dev_gap_mse)))
		
	return train_losses
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# RMTPP run initialize with loss function for run_rmtpp
def run_rmtpp_init(args, data, test_data, NLL_loss=False, use_var_model=False):
	[test_data_in_gaps_bin, test_end_hr_bins, test_data_in_time_end_bin, 
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] =  test_data	
	rmtpp_epochs = args.epochs
	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size

	use_intensity = True
	if not NLL_loss:
		use_intensity = False
	model, optimizer = models.build_rmtpp_model(args, use_intensity, use_var_model)
	model.summary()
	if use_var_model:
		print('\nTraining Model with MSE Loss with Variance')
	elif NLL_loss:
		print('\nTraining Model with NLL Loss')
	else:
		print('\nTraining Model with Mean Square Loss')
	train_loss = run_rmtpp(args, model, optimizer, data, NLL_loss=NLL_loss, 
							rmtpp_epochs=rmtpp_epochs, use_var_model=use_var_model)

	next_hidden_state = None
	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)
	all_event_count_preds = list()
	all_times_pred_from_beg = None
	#TODO: Should we pass more event than 80 for better prediction
	for dec_idx in range(dec_len):
		print('Simulating dec_idx', dec_idx)
		all_gaps_pred, all_times_pred, next_hidden_state = simulate(model, 
												test_data_init_time, 
												test_data_input_gaps_bin,
												test_end_hr_bins[:,dec_idx], 
												(test_gap_in_bin_norm_a, 
												test_gap_in_bin_norm_d),
												prev_hidden_state=next_hidden_state)
		
		if all_times_pred_from_beg is not None:
			all_times_pred_from_beg = tf.concat([all_times_pred_from_beg, all_times_pred], axis=1)
		else:
			all_times_pred_from_beg = all_times_pred

		event_count_preds = count_events(all_times_pred_from_beg, 
											 test_end_hr_bins[:,dec_idx]-bin_size, 
											 test_end_hr_bins[:,dec_idx])
		all_event_count_preds.append(event_count_preds)
		
		test_data_init_time = all_times_pred[:,-1:].numpy()
		
		all_gaps_pred_norm = utils.normalize_avg_given_param(all_gaps_pred,
												test_gap_in_bin_norm_a,
												test_gap_in_bin_norm_d)
		all_prev_gaps_pred = tf.concat([test_data_input_gaps_bin, all_gaps_pred_norm], axis=1)
		test_data_input_gaps_bin = all_prev_gaps_pred[:,-enc_len:].numpy()

	event_count_preds = np.array(all_event_count_preds).T
	return model, event_count_preds
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Count Model with bin count from plain FF network
def run_hierarchical(args, data, test_data):
	validation_split = 0.2
	num_epochs = args.epochs * 100

	train_data_in_bin, train_data_out_bin = data
	test_data_in_bin, test_data_out_bin, test_mean_bin, test_std_bin = test_data
	batch_size = args.batch_size

	model_name = args.current_model
	os.makedirs('saved_models/training_'+model_name+'_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_"+model_name+"_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"

	model_cnt = models.hierarchical_model(args)
	model_cnt.summary()

	if num_epochs > 0:
		history_cnt = model_cnt.fit(train_data_in_bin, train_data_out_bin, batch_size=batch_size,
						epochs=num_epochs, validation_split=validation_split, verbose=0)
		model_cnt.save_weights(checkpoint_path)
		hist = pd.DataFrame(history_cnt.history)
		hist['epoch'] = history_cnt.epoch
		print(hist)
	else:
		model_cnt.load_weights(checkpoint_path)

	test_data_out_norm = utils.normalize_data_given_param(test_data_out_bin, test_mean_bin, test_std_bin)
	loss, mae, mse = model_cnt.evaluate(test_data_in_bin, test_data_out_norm, verbose=0)
	print('Normalized loss, mae, mse', loss, mae, mse)

	test_predictions_norm_cnt = model_cnt.predict(test_data_in_bin)
	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, 
											test_mean_bin, test_std_bin)
	event_count_preds_cnt = test_predictions_cnt
	return model_cnt, event_count_preds_cnt
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Count model with NB or Gaussian distribution
def run_count_model(args, data, test_data):
	validation_split = 0.2
	num_epochs = args.epochs * 100
	patience = args.patience * 100
	distribution_name = 'Gaussian'
	#distribution_name = 'var_model'

	train_data_in_bin, train_data_out_bin = data
	test_data_in_bin, test_data_out_bin, test_mean_bin, test_std_bin = test_data

	dataset_size = len(train_data_in_bin)
	train_data_size = dataset_size - round(validation_split*dataset_size)

	train_data_in_bin = train_data_in_bin.astype(np.float32)
	train_data_out_bin = train_data_out_bin.astype(np.float32)
	test_data_in_bin = test_data_in_bin.astype(np.float32)
	test_data_out_bin = test_data_out_bin.astype(np.float32)

	dev_data_in_bin = train_data_in_bin[train_data_size:]
	dev_data_out_bin = train_data_out_bin[train_data_size:]
	train_data_in_bin = train_data_in_bin[:train_data_size]
	train_data_out_bin = train_data_out_bin[:train_data_size]

	batch_size = args.batch_size
	train_dataset = tf.data.Dataset.from_tensor_slices((train_data_in_bin,
													train_data_out_bin)).batch(batch_size,
													drop_remainder=True)

	model, optimizer = models.build_count_model(args, distribution_name)
	model.summary()

	os.makedirs('saved_models/training_count_'+args.current_dataset+'/', exist_ok=True)
	checkpoint_path = "saved_models/training_count_"+args.current_dataset+"/cp_"+args.current_dataset+".ckpt"
	best_dev_gap_mse = np.inf
	best_dev_epoch = 0

	train_losses = list()
	stddev_sample = list()
	dev_mae_list = list()
	for epoch in range(num_epochs):
		#print('Starting epoch', epoch)
		step_train_loss = 0.0
		step_cnt = 0
		for sm_step, (bin_count_batch_in, bin_count_batch_out) in enumerate(train_dataset):
			with tf.GradientTape() as tape:
				bin_counts_pred, distribution_params = model(bin_count_batch_in)

				loss_fn = models.NegativeLogLikelihood_CountModel(distribution_params, distribution_name)
				loss = loss_fn(bin_count_batch_out, bin_counts_pred)

				step_train_loss+=loss.numpy()
				
				grads = tape.gradient(loss, model.trainable_weights)
				optimizer.apply_gradients(zip(grads, model.trainable_weights))

			# print('Training loss (for one batch) at step %s: %s' %(sm_step, float(loss)))
			step_cnt += 1
		
		# Dev calculations
		dev_bin_count_pred, _ = model(dev_data_in_bin)		
		dev_gap_metric_mae(dev_bin_count_pred, dev_data_out_bin)
		dev_gap_metric_mse(dev_bin_count_pred, dev_data_out_bin)
		dev_gap_mae = dev_gap_metric_mae.result()
		dev_gap_mse = dev_gap_metric_mse.result()
		dev_gap_metric_mae.reset_states()
		dev_gap_metric_mse.reset_states()

		if best_dev_gap_mse > dev_gap_mse and patience <= epoch:
			best_dev_gap_mse = dev_gap_mse
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
			_, [_, test_distribution_stddev] = model(test_data_in_bin)
			stddev_sample.append(test_distribution_stddev[0])
	
	stddev_sample = np.array(stddev_sample)
	for dec_idx in range(args.out_bin_sz//2):
		plt.plot(range(len(stddev_sample)), stddev_sample[:,dec_idx], label='dec_idx_'+str(dec_idx))

	plt.legend(loc='upper right')
	plt.savefig('Outputs/count_model_test_var_'+distribution_name+'_'+args.current_dataset+'_test_id_0.png')
	plt.close()

	plt.plot(range(len(train_losses)), train_losses)
	plt.savefig('Outputs/count_model_train_loss_'+distribution_name+'_'+args.current_dataset+'.png')
	plt.close()

	plt.plot(range(len(dev_mae_list)), dev_mae_list)
	plt.savefig('Outputs/count_model_dev_mae_'+distribution_name+'_'+args.current_dataset+'.png')
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	dev_bin_count_pred, _ = model(dev_data_in_bin)		
	dev_gap_metric_mae(dev_bin_count_pred, dev_data_out_bin)
	dev_gap_metric_mse(dev_bin_count_pred, dev_data_out_bin)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	print('MAE and MSE of Dev data %s: %s' \
		%(float(dev_gap_mae), float(dev_gap_mse)))

	test_bin_count_pred_norm, test_distribution_params = model(test_data_in_bin)		
	test_bin_count_pred = utils.denormalize_data(test_bin_count_pred_norm, test_mean_bin, test_std_bin)
	test_distribution_params[1] = utils.denormalize_data_stddev(test_distribution_params[1], test_mean_bin, test_std_bin)
	return model, {"count_preds": test_bin_count_pred.numpy(), "count_var": test_distribution_params[1]}
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

	[train_dataset_gaps, dev_data_in_gaps, dev_data_out_gaps, train_norm_gaps] = data
	[train_norm_a_gaps, train_norm_d_gaps] = train_norm_gaps

	[test_data_in_gaps_bin, test_end_hr_bins, test_data_in_time_end_bin, 
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] =  test_data	
	wgan_enc_len = args.wgan_enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size


	# Generating prior fake sequence in the range of forecast horizon
	# lambda0 = np.mean([len(item) for item in real_sequences])/T
	# intensityPoisson = IntensityHomogenuosPoisson(lambda0)
	# fake_sequences = generate_sample(intensityPoisson, T, 20000)
	train_z_seqs = list()
	for (gaps_batch, _) in train_dataset_gaps:
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
	dev_data_gaps = dev_data_in_gaps
	wgan_dec_len = dev_data_gaps.shape[1] - wgan_enc_len
	dev_data_times = tf.cumsum(dev_data_gaps, axis=1)
	dev_span = dev_data_times[:, wgan_enc_len-1] - dev_data_times[:, 0]
	lambda0 = np.ones_like(dev_span) * dev_data_gaps.shape[1] / dev_span.numpy()
	intensityPoisson = IntensityHomogenuosPoisson(lambda0)
	dev_z_seqs = generate_sample(intensityPoisson, wgan_dec_len, lambda0.shape[0])
	dev_z_seqs = tf.convert_to_tensor(dev_z_seqs)
	#dev_z_seqs = utils.normalize_avg_given_param(dev_z_seqs,
	#										train_norm_a_gaps,
	#										train_norm_d_gaps)

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
		for sm_step, (gaps_batch, _) \
				in enumerate(train_dataset_gaps):
			gaps_batch_in = gaps_batch[:, :wgan_enc_len]
			gaps_batch_out = gaps_batch[:, wgan_enc_len:]
			train_z_seqs_batch = train_z_seqs[sm_step*args.batch_size:(sm_step+1)*args.batch_size]
			with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
				gaps_pred = model.generator(train_z_seqs_batch, gaps_batch_in)

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
			step_cnt += 1
		
		# Dev calculations
		dev_data_in_gaps = dev_data_gaps[:, :wgan_enc_len]
		dev_data_out_gaps = dev_data_gaps[:, wgan_enc_len:]
		dev_gaps_pred = model.generator(dev_z_seqs, dev_data_in_gaps)
		dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
										train_norm_a_gaps, train_norm_d_gaps)
		dev_data_out_gaps_unnorm = utils.denormalize_avg(dev_data_out_gaps, 
										train_norm_a_gaps, train_norm_d_gaps)
		
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
	plt.savefig('Outputs/train_wgan_'+args.current_dataset+'_loss.png')
	plt.close()

	print("Loading best model from epoch", best_dev_epoch)
	model.load_weights(checkpoint_path)
	dev_data_in_gaps = dev_data_gaps[:, :wgan_enc_len]
	dev_data_out_gaps = dev_data_gaps[:, wgan_enc_len:]
	dev_gaps_pred = model.generator(dev_z_seqs, dev_data_in_gaps)
	dev_gaps_pred_unnorm = utils.denormalize_avg(dev_gaps_pred, 
									train_norm_a_gaps, train_norm_d_gaps)
	
	dev_data_out_gaps_unnorm = utils.denormalize_avg(dev_data_out_gaps, 
									train_norm_a_gaps, train_norm_d_gaps)

	dev_gap_metric_mae(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
	dev_gap_metric_mse(dev_data_out_gaps_unnorm, dev_gaps_pred_unnorm)
	dev_gap_mae = dev_gap_metric_mae.result()
	dev_gap_mse = dev_gap_metric_mse.result()
	dev_gap_metric_mae.reset_states()
	dev_gap_metric_mse.reset_states()
	print('Best MAE and MSE of Dev data %s: %s' \
		%(float(dev_gap_mae), float(dev_gap_mse)))

	# Test Data results
	event_count_preds = None
	next_hidden_state = None
	#event_count_preds of size [Test_example_numbers, bin_out_sz]

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)
	all_event_count_preds = list()
	all_times_pred_from_beg = None
	for dec_idx in range(dec_len):
		print('Simulating dec_idx', dec_idx)
		all_gaps_pred, all_times_pred, next_hidden_state = simulate_wgan(model, 
												test_data_init_time, 
												test_data_input_gaps_bin,
												test_end_hr_bins[:,dec_idx], 
												(test_gap_in_bin_norm_a, 
												test_gap_in_bin_norm_d),
												prev_hidden_state=next_hidden_state)
		
		if all_times_pred_from_beg is not None:
			all_times_pred_from_beg = tf.concat([all_times_pred_from_beg, all_times_pred], axis=1)
		else:
			all_times_pred_from_beg = all_times_pred

		event_count_preds = count_events(all_times_pred_from_beg, 
											 test_end_hr_bins[:,dec_idx]-bin_size, 
											 test_end_hr_bins[:,dec_idx])
		all_event_count_preds.append(event_count_preds)
		
		test_data_init_time = all_times_pred[:,-1:].numpy()
		
		all_gaps_pred_norm = utils.normalize_avg_given_param(all_gaps_pred,
												test_gap_in_bin_norm_a,
												test_gap_in_bin_norm_d)
		all_prev_gaps_pred = tf.concat([test_data_input_gaps_bin, all_gaps_pred_norm], axis=1)
		test_data_input_gaps_bin = all_prev_gaps_pred[:,-wgan_enc_len:].numpy()

	event_count_preds = np.array(all_event_count_preds).T
	return model, event_count_preds
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#####################################################
# 				Model Inference 					#
#####################################################
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate model until t_b_plus
def simulate(model, times_in, gaps_in, t_b_plus, normalizers, prev_hidden_state = None):
	#TODO: Check for this modification in functions which calls this def
	gaps_pred = list()
	times_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	# step_gaps_pred = gaps_in[:, -1]
	step_gaps_pred, _, _, prev_hidden_state, _ \
			= model(gaps_in, initial_state=prev_hidden_state)

	step_gaps_pred = step_gaps_pred[:,-1:]
	gaps_in = tf.concat([gaps_in[:,1:], step_gaps_pred], axis=1)
	step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	last_times_pred = times_in + last_gaps_pred_unnorm
	gaps_pred.append(last_gaps_pred_unnorm)
	times_pred.append(last_times_pred)

	simul_step = 0

	while any(times_pred[-1]<t_b_plus):
		step_gaps_pred, _, _, prev_hidden_state, _ \
				= model(gaps_in, initial_state=prev_hidden_state)

		step_gaps_pred = step_gaps_pred[:,-1:]
		gaps_in = tf.concat([gaps_in[:,1:], step_gaps_pred], axis=1)
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		gaps_pred.append(last_gaps_pred_unnorm)
		times_pred.append(last_times_pred)
		
		simul_step += 1

	gaps_pred = tf.stack(gaps_pred, axis=1)
	all_gaps_pred = gaps_pred

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	return all_gaps_pred, all_times_pred, prev_hidden_state
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate model out_gaps_count times
def simulate_with_counter(model, times_in, gaps_in, out_gaps_count, normalizers, prev_hidden_state = None):
	gaps_pred = list()
	times_pred = list()
	data_norm_a, data_norm_d = normalizers
	
	# step_gaps_pred = gaps_in[:, -1]
	step_gaps_pred, _, _, prev_hidden_state, _ \
			= model(gaps_in, initial_state=prev_hidden_state)

	step_gaps_pred = step_gaps_pred[:,-1:]
	gaps_in = tf.concat([gaps_in[:,1:], step_gaps_pred], axis=1)
	step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
	last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
	last_times_pred = times_in + last_gaps_pred_unnorm
	gaps_pred.append(last_gaps_pred_unnorm)
	times_pred.append(last_times_pred)

	simul_step = 0
	old_hidden_state = None

	while any(simul_step < out_gaps_count):
		step_gaps_pred, _, _, prev_hidden_state, _ \
				= model(gaps_in, initial_state=prev_hidden_state)
		
		if old_hidden_state is not None:
			prev_hidden_state = (simul_step < out_gaps_count) * prev_hidden_state + \
								(simul_step >= out_gaps_count) * old_hidden_state
			step_gaps_pred = np.expand_dims((simul_step < out_gaps_count), axis=-1) * step_gaps_pred
			
		old_hidden_state = prev_hidden_state
		step_gaps_pred = step_gaps_pred[:,-1:]
		gaps_in = tf.concat([gaps_in[:,1:], step_gaps_pred], axis=1)
		step_gaps_pred = tf.squeeze(step_gaps_pred, axis=-1)
		last_gaps_pred_unnorm = utils.denormalize_avg(step_gaps_pred, data_norm_a, data_norm_d)
		last_times_pred = times_pred[-1] + last_gaps_pred_unnorm
		last_times_pred = (simul_step < out_gaps_count) * last_times_pred
		gaps_pred.append(last_gaps_pred_unnorm)
		times_pred.append(last_times_pred)
		
		simul_step += 1

	gaps_pred = tf.stack(gaps_pred, axis=1)
	all_gaps_pred = gaps_pred

	times_pred = tf.squeeze(tf.stack(times_pred, axis=1), axis=2)
	all_times_pred = times_pred

	return all_gaps_pred, all_times_pred, prev_hidden_state
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Simulate WGAN model till t_b_plus
def simulate_wgan(model, times_in, gaps_in, t_b_plus, normalizers, prev_hidden_state=None):
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

	_, g_init_state = model.run_encoder(gaps_in)

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
# Run rmtpp_count model with state reinitialized 
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

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(test_data_in_bin)
	if len(test_predictions_norm_cnt) == 2:
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, test_mean_bin, test_std_bin)
	event_count_preds_cnt = np.round(test_predictions_cnt)
	event_count_preds_cnt_min = np.min(event_count_preds_cnt, axis=0)
	event_count_preds_true = test_data_out_bin

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()

	bin_end = None
	for dec_idx in range(dec_len):
		all_gaps_pred, all_times_pred, _ = simulate_with_counter(model_rmtpp, 
												test_data_init_time, 
												test_data_input_gaps_bin,
												output_event_count_pred[:,dec_idx],
												(test_gap_in_bin_norm_a, 
												test_gap_in_bin_norm_d),
												prev_hidden_state=next_hidden_state)

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

		actual_bin_start = test_end_hr_bins[:,dec_idx]-bin_size
		actual_bin_end = test_end_hr_bins[:,dec_idx]

		all_times_pred, all_gaps_pred = scaled_points(actual_bin_start, actual_bin_end, bin_start, bin_end, all_times_pred)

		gaps_before_bins = all_times_pred[:,:1] - prev_test_data_init_time
		gaps_before_bins = tf.expand_dims(gaps_before_bins, axis=-1)
		all_gaps_pred = tf.concat([gaps_before_bins, all_gaps_pred], axis=1)

		event_in_bin_preds, _, test_data_init_time, _ = compute_event_in_bin(tf.expand_dims(all_times_pred, axis=-1), 
														 output_event_count_pred[:,dec_idx,0])
		test_data_init_time = np.expand_dims(test_data_init_time, axis=-1)
		
		
		all_gaps_pred_norm = utils.normalize_avg_given_param(all_gaps_pred,
												test_gap_in_bin_norm_a,
												test_gap_in_bin_norm_d)
		
		_, test_data_input_gaps_bin_full, _, _ = compute_event_in_bin(all_gaps_pred_norm, 
															  output_event_count_pred[:,dec_idx,0],
															  test_data_input_gaps_bin, 
															  enc_len+int(event_count_preds_cnt_min[dec_idx]))
		
		all_events_in_bin_pred.append(event_in_bin_preds)
		
		test_data_input_gaps_bin = np.expand_dims(test_data_input_gaps_bin_full[:,-enc_len:], axis=-1)
		test_data_input_gaps_bin = test_data_input_gaps_bin.astype(np.float32)

		test_data_input_gaps_bin_scaled = np.expand_dims(test_data_input_gaps_bin_full[:,:-enc_len], axis=-1)
		test_data_input_gaps_bin_scaled = test_data_input_gaps_bin_scaled.astype(np.float32)
		
		_, _, _, _, next_hidden_state \
					= model_rmtpp(test_data_input_gaps_bin_scaled, initial_state=scaled_rnn_hidden_state)

	all_events_in_bin_pred = np.array(all_events_in_bin_pred).T
	return event_count_preds_cnt, all_events_in_bin_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Run rmtpp_count model with one rmtpp simulation untill 
# all events of bins generated then scale , each bin has events
# whose count generated by count model
def run_rmtpp_count_cont_rmtpp(args, models, data, test_data, rmtpp_type):
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
	assert model_check, "run_rmtpp_count_cont_rmtpp requires count and RMTPP model"

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(test_data_in_bin)
	if len(test_predictions_norm_cnt) == 2:
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, test_mean_bin, test_std_bin)
	event_count_preds_cnt = np.round(test_predictions_cnt)
	event_count_preds_true = test_data_out_bin

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()
	
	output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_cnt, axis=-1).numpy()
	full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
	full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)

	all_gaps_pred, all_times_pred, _ = simulate_with_counter(model_rmtpp, 
											test_data_init_time, 
											test_data_input_gaps_bin,
											full_cnt_event_all_bins_pred,
											(test_gap_in_bin_norm_a, 
											test_gap_in_bin_norm_d),
											prev_hidden_state=next_hidden_state)
	
	all_times_pred_lst = list()
	for batch_idx in range(len(all_gaps_pred)):
		event_past_cnt=0
		times_pred_all_bin_lst=list()

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
			
			actual_bin_start = test_end_hr_bins[batch_idx,dec_idx]-bin_size
			actual_bin_end = test_end_hr_bins[batch_idx,dec_idx]

			times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
							 (times_pred_for_bin - bin_start)) + actual_bin_start
			
			times_pred_all_bin_lst.append(times_pred_for_bin_scaled.numpy())
		
		all_times_pred_lst.append(times_pred_all_bin_lst)
	all_times_pred = np.array(all_times_pred_lst)
	return event_count_preds_cnt, all_times_pred
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

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(test_data_in_bin)
	if len(test_predictions_norm_cnt) == 2:
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, test_mean_bin, test_std_bin)
	event_count_preds_cnt = np.round(test_predictions_cnt).astype(np.int32)
	event_count_preds_true = test_data_out_bin

	all_times_pred_lst = list()
	for batch_idx in range(event_count_preds_cnt.shape[0]):
		times_pred_per_bin_lst=list()
		old_last_gap = np.random.uniform(low=0.0, high=1.0)*np.random.uniform(low=0.0, high=1.0)
		for dec_idx in range(event_count_preds_cnt.shape[1]):
			actual_bin_start = test_end_hr_bins[batch_idx,dec_idx]-bin_size
			actual_bin_end = test_end_hr_bins[batch_idx,dec_idx]

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
		all_times_pred_lst.append(times_pred_per_bin_lst)
	all_times_pred = np.array(all_times_pred_lst)
	return event_count_preds_cnt, all_times_pred
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

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(test_data_in_bin)
	if len(test_predictions_norm_cnt) == 2:
		model_cnt_distribution_params = test_predictions_norm_cnt[1]
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, test_mean_bin, test_std_bin)
	event_count_preds_cnt = np.round(test_predictions_cnt)
	event_count_preds_true = test_data_out_bin

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()
	
	output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_cnt, axis=-1).numpy()
	full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
	full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)

	all_gaps_pred, all_times_pred, _ = simulate_with_counter(model_rmtpp, 
											test_data_init_time, 
											test_data_input_gaps_bin,
											full_cnt_event_all_bins_pred,
											(test_gap_in_bin_norm_a, 
											test_gap_in_bin_norm_d),
											prev_hidden_state=next_hidden_state)
	
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
			
			actual_bin_start = test_end_hr_bins[batch_idx,dec_idx]-bin_size
			actual_bin_end = test_end_hr_bins[batch_idx,dec_idx]

			times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
							 (times_pred_for_bin - bin_start)) + actual_bin_start
			
			times_pred_all_bin_lst.append(times_pred_for_bin_scaled.numpy())
		
		all_times_pred_lst.append(times_pred_all_bin_lst)
	all_times_pred = np.array(all_times_pred_lst)
	all_times_pred_before = all_times_pred

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)

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
				lst.append(utils.denormalize_avg(test_data_input_gaps_bin[batch_idx,-1:,0], test_gap_in_bin_norm_a, test_gap_in_bin_norm_d))
				lst.append(all_times_pred[batch_idx,idx]-np.concatenate([test_data_init_time[batch_idx],all_times_pred[batch_idx,idx][:-1]]))
			else:
				lst.append(all_times_pred[batch_idx,idx]-np.concatenate([all_times_pred[batch_idx,idx-1][-1:],all_times_pred[batch_idx,idx][:-1]]))

		merged_lst = list()
		for each in lst:
			merged_lst += each.tolist()

		test_bin_gaps_inp = np.array(merged_lst[:-1])
		test_bin_gaps_inp = np.expand_dims(np.expand_dims(test_bin_gaps_inp, axis=0), axis=-1).astype(np.float32)
		test_bin_gaps_inp = utils.normalize_avg_given_param(test_bin_gaps_inp, test_gap_in_bin_norm_a, test_gap_in_bin_norm_d)
		_, D, WT, _, batch_input_final_state = model_rmtpp(test_bin_gaps_inp, initial_state=input_final_state[batch_idx:batch_idx+1])

		all_bins_gaps_pred.append(np.array(merged_lst[1:]).astype(np.float32))
		all_bins_D_pred.append(D[0,:,0].numpy())
		all_bins_WT_pred.append(WT[0,:,0].numpy())

	all_bins_gaps_pred = np.array(all_bins_gaps_pred)
	all_bins_D_pred = np.array(all_bins_D_pred)
	all_bins_WT_pred = np.array(all_bins_WT_pred)
	model_rmtpp_params = [all_bins_D_pred, all_bins_WT_pred]

	bin_size = args.bin_size
	test_end_hr_bins = test_end_hr_bins.astype(np.float32)
	all_bins_end_time = tf.squeeze(test_end_hr_bins, axis=-1)
	all_bins_start_time = all_bins_end_time - bin_size
	all_bins_mid_time = (all_bins_start_time+all_bins_end_time)/2

	events_count_per_batch = output_event_count_pred_cumm
	test_data_count_normalizer = [test_mean_bin, test_std_bin]
	test_data_rmtpp_normalizer = [test_gap_in_bin_norm_a, test_gap_in_bin_norm_d]

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
		test_mean_bin, test_std_bin = test_data_count_normalizer
		test_norm_a, test_norm_d = test_data_rmtpp_normalizer

		frac_belong, whole_belong = fractional_belongingness(all_bins_gaps_pred,
											   all_bins_mid_time,
											   test_data_init_time)
		count_loss_fn = models.NegativeLogLikelihood_CountModel(model_cnt_distribution_params, 'Gaussian')
		estimated_count_norm = utils.normalize_data_given_param(whole_belong, test_mean_bin, test_std_bin)
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

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	test_end_hr_bins = test_end_hr_bins.astype(np.float32)
	all_bins_end_time = tf.squeeze(test_end_hr_bins, axis=-1)

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
		test_mean_bin, test_std_bin = test_data_count_normalizer
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

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(test_data_in_bin)
	if len(test_predictions_norm_cnt) == 2:
		model_cnt_distribution_params = test_predictions_norm_cnt[1]
		test_predictions_norm_cnt = test_predictions_norm_cnt[0]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, test_mean_bin, test_std_bin)
	event_count_preds_cnt = np.round(test_predictions_cnt)
	event_count_preds_true = test_data_out_bin

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()
	
	output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_cnt, axis=-1).numpy()
	full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
	full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)
	full_cnt_event_all_bins_pred += num_counts

	all_gaps_pred, all_times_pred, _ = simulate_with_counter(model_rmtpp, 
											test_data_init_time, 
											test_data_input_gaps_bin,
											full_cnt_event_all_bins_pred,
											(test_gap_in_bin_norm_a, 
											test_gap_in_bin_norm_d),
											prev_hidden_state=next_hidden_state)
	
	all_times_pred_simu = all_times_pred
	count2loss = dict()
	count2pred = dict()
	for nc in range(-num_counts, num_counts):
		all_times_pred_lst = list()
		output_event_count_curr = output_event_count_pred + nc
		test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
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
				
				actual_bin_start = test_end_hr_bins[batch_idx,dec_idx]-bin_size
				actual_bin_end = test_end_hr_bins[batch_idx,dec_idx]
	
				times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
							 	(times_pred_for_bin - bin_start)) + actual_bin_start
				
				times_pred_all_bin_lst.append(times_pred_for_bin_scaled.numpy())
			
			all_times_pred_lst.append(times_pred_all_bin_lst)
		all_times_pred = np.array(all_times_pred_lst)
		all_times_pred_before = all_times_pred
	
		test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	
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
					lst.append(utils.denormalize_avg(test_data_input_gaps_bin[batch_idx,-1:,0], test_gap_in_bin_norm_a, test_gap_in_bin_norm_d))
					lst.append(all_times_pred[batch_idx,idx]-np.concatenate([test_data_init_time[batch_idx],all_times_pred[batch_idx,idx][:-1]]))
				else:
					lst.append(all_times_pred[batch_idx,idx]-np.concatenate([all_times_pred[batch_idx,idx-1][-1:],all_times_pred[batch_idx,idx][:-1]]))
	
			merged_lst = list()
			for each in lst:
				merged_lst += each.tolist()
	
			test_bin_gaps_inp = np.array(merged_lst[:-1])
			test_bin_gaps_inp = np.expand_dims(np.expand_dims(test_bin_gaps_inp, axis=0), axis=-1).astype(np.float32)
			test_bin_gaps_inp = utils.normalize_avg_given_param(test_bin_gaps_inp, test_gap_in_bin_norm_a, test_gap_in_bin_norm_d)
			_, D, WT, _, batch_input_final_state = model_rmtpp(test_bin_gaps_inp, initial_state=input_final_state[batch_idx:batch_idx+1])
	
			all_bins_gaps_pred.append(np.array(merged_lst[1:]).astype(np.float32))
			all_bins_D_pred.append(D[0,:,0].numpy())
			all_bins_WT_pred.append(WT[0,:,0].numpy())
	
		all_bins_gaps_pred = np.array(all_bins_gaps_pred)
		all_bins_D_pred = np.array(all_bins_D_pred)
		all_bins_WT_pred = np.array(all_bins_WT_pred)
		model_rmtpp_params = [all_bins_D_pred, all_bins_WT_pred]
	
		bin_size = args.bin_size
		test_end_hr_bins = test_end_hr_bins.astype(np.float32)
		all_bins_end_time = tf.squeeze(test_end_hr_bins, axis=-1)
		all_bins_start_time = all_bins_end_time - bin_size
		all_bins_mid_time = (all_bins_start_time+all_bins_end_time)/2
	
		events_count_per_batch = output_event_count_curr
		test_data_count_normalizer = [test_mean_bin, test_std_bin]
		test_data_rmtpp_normalizer = [test_gap_in_bin_norm_a, test_gap_in_bin_norm_d]

	
	
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
# Run rmtpp_count model with optimized gaps generated from rmtpp simulation untill 
# all events of bins generated then rescale
# For each possible count, find the best rmtpp_loss for that many gaps 
# by using solver library
# Select the final count as the count that produces best rmtpp_loss across
# all counts
def run_rmtpp_with_optimization_fixed_cnt_solver(args, query_models, data, test_data, rmtpp_type='nll'):

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	model_cnt = query_models['count_model']
	if rmtpp_type=='nll':
		model_rmtpp = query_models['rmtpp_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = query_models['rmtpp_mse']
	elif rmtpp_type=='mse_var':
		model_rmtpp = query_models['rmtpp_mse_var']
	else:
		assert False, "rmtpp_type must be nll or mse"
	
	model_check = (model_cnt is not None) and (model_rmtpp is not None)
	assert model_check, "run_rmtpp_count_with_optimization requires count and RMTPP model"

	test_end_hr_bins = test_end_hr_bins.astype(np.float32)
	all_bins_end_time = tf.squeeze(test_end_hr_bins, axis=-1)

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
					  test_data_rmtpp_normalizer):

		gaps = cp.Variable(all_bins_gaps_pred.shape)
		gaps.value = all_bins_gaps_pred

		D, WT = model_rmtpp_params[0], model_rmtpp_params[1]
		if rmtpp_type=='nll':
			objective = cp.Minimize(-cp.sum(rmtpp_loglikelihood_loss(gaps, D, WT, events_count_per_batch))/all_bins_gaps_pred.shape[1])
		elif rmtpp_type=='mse':
			WT = np.ones_like(WT)
			D = all_bins_gaps_pred
			objective = cp.Minimize(cp.sum(mse_loss(gaps, D, WT, events_count_per_batch))/all_bins_gaps_pred.shape[1])
		elif rmtpp_type=='mse_var':
			objective = cp.Minimize(cp.sum(mse_loglikelihood_loss(gaps, D, WT, events_count_per_batch))/all_bins_gaps_pred.shape[1])


		test_norm_a, test_norm_d = test_data_rmtpp_normalizer
		init_end_diff = all_bins_end_time-test_data_init_time
		init_end_diff_norm = utils.normalize_avg_given_param(init_end_diff, test_norm_a, test_norm_d)
		constraints = [cp.sum(gaps[0, :nc])<=init_end_diff_norm,
					   cp.sum(gaps[0, :nc+1])>=init_end_diff_norm,
					   gaps>=0]
		# TODO Need normalizer for constraints
		prob = cp.Problem(objective, constraints)
		#print('D:')
		#print(D)
		#print('WT:')
		#print(WT)
		#print('all_bins_gaps_pred:')
		#print(all_bins_gaps_pred)
		#print('all_bins_end_time:')
		#print(all_bins_end_time)
		#print('test_data_init_time:')
		#print(test_data_init_time)
		#print('\n'),

		try:
			rmtpp_loss = prob.solve(warm_start=True)
		except cp.error.SolverError:
			rmtpp_loss = prob.solve(solver='SCS', warm_start=True)

		test_mean_bin, test_std_bin = test_data_count_normalizer
		nc_norm = utils.normalize_data_given_param(nc, test_mean_bin, test_std_bin)
		mu, sigma = model_cnt_distribution_params[0], model_cnt_distribution_params[1]
		count_loss = -tfp.distributions.Normal(
                mu, sigma, validate_args=False, allow_nan_stats=True, 
                name='Normal'
            ).log_prob(nc_norm)
		count_loss = np.sum(count_loss)


		loss = rmtpp_loss + count_loss

		all_bins_gaps_pred = gaps.value[:nc]
		#print('Loss after optimization:', loss)
	
		# Shape: list of 92 different length tensors
		return all_bins_gaps_pred, loss


	num_counts = 2 # Number of counts to take before and after mean
	#model_cnt, model_rmtpp, _ = query_models
	# model_rmtpp = query_models['rmtpp_nll']


	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	
	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)
	all_events_in_bin_pred = list()

	test_predictions_norm_cnt = model_cnt(test_data_in_bin)
	if len(test_predictions_norm_cnt) == 2:
		model_cnt_distribution_params = test_predictions_norm_cnt[1]
		# test_predictions_norm_cnt = test_predictions_norm_cnt[0]
		test_predictions_norm_cnt = model_cnt_distribution_params[0]
		test_predictions_norm_stddev = model_cnt_distribution_params[1]

	test_predictions_cnt = utils.denormalize_data(test_predictions_norm_cnt, test_mean_bin, test_std_bin)
	test_predictions_stddev = utils.denormalize_data_stddev(test_predictions_norm_stddev, test_mean_bin, test_std_bin)
	event_count_preds_cnt = np.round(test_predictions_cnt)
	event_count_preds_stddev = np.round(test_predictions_stddev)
	event_count_preds_stddev = np.maximum(event_count_preds_stddev, 1.) # stddev should not be less than 1.
	event_count_preds_true = test_data_out_bin

	output_event_count_pred = tf.expand_dims(event_count_preds_cnt, axis=-1).numpy()
	
	output_event_count_pred_cumm = tf.reduce_sum(event_count_preds_cnt, axis=-1).numpy()
	full_cnt_event_all_bins_pred = max(output_event_count_pred_cumm) * np.ones_like(output_event_count_pred_cumm)
	full_cnt_event_all_bins_pred = np.expand_dims(full_cnt_event_all_bins_pred, axis=-1)
	#full_cnt_event_all_bins_pred += num_counts
	full_cnt_event_all_bins_pred += event_count_preds_stddev

	all_gaps_pred, all_times_pred, _ = simulate_with_counter(model_rmtpp, 
											test_data_init_time, 
											test_data_input_gaps_bin,
											full_cnt_event_all_bins_pred,
											(test_gap_in_bin_norm_a, 
											test_gap_in_bin_norm_d),
											prev_hidden_state=next_hidden_state)
	
	all_times_pred_simu = all_times_pred

	count_sigma = 2.
	best_all_times_pred = []
	for batch_idx in range(len(all_gaps_pred)):
		nc_loss_lst = []
		all_times_pred_nc_lst = []
		clipped_stddev = np.minimum(event_count_preds_stddev[batch_idx], count_sigma)
		min_cnt = event_count_preds_cnt[batch_idx] - clipped_stddev
		min_cnt = np.maximum(1., min_cnt)
		max_cnt = event_count_preds_cnt[batch_idx] + clipped_stddev + 1.
		nc_range = np.arange(min_cnt, max_cnt)
		for nc in nc_range:
			event_past_cnt=0
			all_times_pred = all_times_pred_simu
			times_pred_all_bin_lst=list()
			all_times_pred_lst = list()
			output_event_count_curr = np.zeros_like(output_event_count_pred) + max_cnt-1
			test_data_init_time = test_data_in_time_end_bin.astype(np.float32)

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
				
				actual_bin_start = test_end_hr_bins[batch_idx,dec_idx]-bin_size
				actual_bin_end = test_end_hr_bins[batch_idx,dec_idx]
	
				# times_pred_for_bin_scaled = times_pred_for_bin
				times_pred_for_bin_scaled = (((actual_bin_end - actual_bin_start)/(bin_end - bin_start)) * \
							 	(times_pred_for_bin - bin_start)) + actual_bin_start
				
				times_pred_all_bin_lst.append(times_pred_for_bin_scaled.numpy())
			
			all_times_pred_lst.append(times_pred_all_bin_lst)
			#all_times_pred = np.array(all_times_pred_lst)
			#all_times_pred_before = all_times_pred
			all_times_pred_arr = np.array(all_times_pred_lst)
	
			test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	
			_, _, _, _, input_final_state \
				= model_rmtpp(test_data_input_gaps_bin[batch_idx:batch_idx+1,:-1,:],
							  initial_state=None)
	
			all_bins_gaps_pred = list()
			all_bins_D_pred = list()
			all_bins_WT_pred = list()
			all_input_final_state = list()

			lst = list()
			d_lst = list()
			wt_lst = list()
			final_state_lst = list()
			batch_input_final_state = input_final_state[batch_idx:batch_idx+1]
	
			for idx in range(len(all_times_pred_arr[0])):
				if idx==0:
					lst.append(utils.denormalize_avg(test_data_input_gaps_bin[batch_idx,-1:,0], test_gap_in_bin_norm_a, test_gap_in_bin_norm_d))
					lst.append(all_times_pred_arr[0,idx]-np.concatenate([test_data_init_time[batch_idx],all_times_pred_arr[0,idx][:-1]]))
					# TODO  First entry in the list appears to be negative
				else:
					lst.append(all_times_pred_arr[0,idx]-np.concatenate([all_times_pred_arr[0,idx-1][-1:],all_times_pred_arr[0,idx][:-1]]))
	
			merged_lst = list()
			for each in lst:
				merged_lst += each.tolist()
	
			test_bin_gaps_inp = np.array(merged_lst[:-1])
			test_bin_gaps_inp = np.expand_dims(np.expand_dims(test_bin_gaps_inp, axis=0), axis=-1).astype(np.float32)
			test_bin_gaps_inp = utils.normalize_avg_given_param(test_bin_gaps_inp, test_gap_in_bin_norm_a, test_gap_in_bin_norm_d)
			_, D, WT, _, batch_input_final_state = model_rmtpp(test_bin_gaps_inp, initial_state=input_final_state)
	
			all_bins_gaps_pred.append(np.array(merged_lst[1:]).astype(np.float32))
			all_bins_D_pred.append(D[0,:,0].numpy())
			all_bins_WT_pred.append(WT[0,:,0].numpy())
	
			all_bins_gaps_pred = np.array(all_bins_gaps_pred)
			all_bins_D_pred = np.array(all_bins_D_pred)
			all_bins_WT_pred = np.array(all_bins_WT_pred)
			model_rmtpp_params = [all_bins_D_pred, all_bins_WT_pred]
			model_count_params = [model_cnt_distribution_params[0][batch_idx], model_cnt_distribution_params[1][batch_idx]]
	
			bin_size = args.bin_size
			all_bins_end_time = tf.squeeze(test_end_hr_bins[batch_idx:batch_idx+1].astype(np.float32), axis=-1)
			all_bins_start_time = all_bins_end_time - bin_size
			all_bins_mid_time = (all_bins_start_time+all_bins_end_time)/2
	
			events_count_per_batch = output_event_count_curr[batch_idx:batch_idx+1]
			test_data_count_normalizer = [test_mean_bin, test_std_bin]
			test_data_rmtpp_normalizer = [test_gap_in_bin_norm_a, test_gap_in_bin_norm_d]
			test_data_init_time_batch = test_data_init_time[batch_idx:batch_idx+1]
	
			all_bins_gaps_pred = utils.normalize_avg_given_param(all_bins_gaps_pred,
																 test_gap_in_bin_norm_a,
																 test_gap_in_bin_norm_d)

			all_bins_gaps_pred, nc_loss \
				= optimize_gaps(model_rmtpp_params,
								rmtpp_loglikelihood_loss,
								model_count_params,
								int(nc),
								all_bins_gaps_pred,
								all_bins_end_time,
								test_data_init_time_batch,
								events_count_per_batch,
								test_data_count_normalizer,
								test_data_rmtpp_normalizer)
			all_bins_gaps_pred = utils.denormalize_avg(all_bins_gaps_pred,
													   test_gap_in_bin_norm_a,
													   test_gap_in_bin_norm_d)
		
			all_times_pred_nc = (test_data_init_time_batch + tf.cumsum(all_bins_gaps_pred, axis=1)) * tf.cast(all_bins_gaps_pred>0., tf.float64)
			all_times_pred_nc = all_times_pred_nc[0].numpy()
			#all_times_pred_nc = np.array([seq[:int(cnt)] for seq, cnt in zip(all_times_pred_nc, events_count_per_batch)])
			#all_times_pred_nc = np.expand_dims(all_times_pred_nc, axis=1)
	
			print('Example:', batch_idx, 'nc:', nc, 'loss:', nc_loss, 'Mean:', event_count_preds_cnt[batch_idx])
	
			all_times_pred_nc_lst.append(all_times_pred_nc)
			nc_loss_lst.append(nc_loss)
		best_nc_idx, best_nc_loss = min(enumerate(nc_loss_lst), key=itemgetter(1))
		best_all_times_pred_nc = all_times_pred_nc_lst[best_nc_idx]
		best_all_times_pred.append(best_all_times_pred_nc)
		#print('Best count:', best_all_times_pred_nc.shape)

	best_all_times_pred = np.expand_dims(np.array(best_all_times_pred), axis=1)
	# [99, dec_len, ...] Currently supports only dec_len=1

	return None, best_all_times_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Plain rmtpp model to generate events independent of bin boundary
def run_rmtpp_for_count(args, models, data, test_data, query_data=None, simul_end_time=None, rmtpp_type='mse'):
	if rmtpp_type=='nll':
		model_rmtpp = models['rmtpp_nll']
	elif rmtpp_type=='mse':
		model_rmtpp = models['rmtpp_mse']
	elif rmtpp_type=='mse_var':
		model_rmtpp = models['rmtpp_mse_var']
	else:
		assert False, "rmtpp_type must be nll or mse"
	model_check = (model_rmtpp is not None)
	assert model_check, "run_rmtpp_for_count requires RMTPP model"

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size

	next_hidden_state = None
	scaled_rnn_hidden_state = None

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)

	if simul_end_time is None and query_data is None:
		t_e_plus = test_end_hr_bins[:,-1]
	elif simul_end_time is None:
		[t_b_plus, t_e_plus, true_count] = query_data
		t_b_plus = np.expand_dims(t_b_plus, axis=-1)
		t_e_plus = np.expand_dims(t_e_plus, axis=-1)
	else:
		t_e_plus = np.expand_dims(simul_end_time, axis=-1)

	_, all_times_pred, _ = simulate(model_rmtpp,
										test_data_init_time,
										test_data_input_gaps_bin,
										t_e_plus,
										(test_gap_in_bin_norm_a,
										test_gap_in_bin_norm_d),
										prev_hidden_state=next_hidden_state)
	test_event_count_pred = None
	if query_data is not None:
		[t_b_plus, t_e_plus, true_count] = query_data
		t_b_plus = np.expand_dims(t_b_plus, axis=-1)
		t_e_plus = np.expand_dims(t_e_plus, axis=-1)
		test_event_count_pred = count_events(all_times_pred, t_b_plus, t_e_plus)
		test_event_count_pred = np.array(test_event_count_pred)

		event_count_mse = tf.keras.losses.MSE(true_count, test_event_count_pred).numpy()
		event_count_mae = tf.keras.losses.MAE(true_count, test_event_count_pred).numpy()
		#print("MSE of event count in range:", event_count_mse)
		#print("MAE of event count in range:", event_count_mae)

	all_times_pred = np.expand_dims(all_times_pred.numpy(), axis=-1)
	return test_event_count_pred, all_times_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
# Plain wgan model to generate events independent of bin boundary
def run_wgan_for_count(args, models, data, test_data, query_data=None, simul_end_time=None):
	'''
	TODO:
		Call the simulate_wgan method with condition:
			Simulate the wgan model until T_l^+ is not reached.

	'''
	# return test_event_count_pred, all_times_pred
	# test_event_count_pred shape=[test_example_size, 1]
	# all_times_pred shape=[test_example_size] 
	# It will be array/lists of lists where each list contains events in output window

	model_wgan = models['wgan']
	model_check = (model_wgan is not None)
	assert model_check, "run_rmtpp_count_cont_rmtpp requires WGAN model"

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	dec_len = args.out_bin_sz
	bin_size = args.bin_size

	next_hidden_state = None

	test_data_init_time = test_data_in_time_end_bin.astype(np.float32)
	test_data_input_gaps_bin = test_data_in_gaps_bin.astype(np.float32)

	if simul_end_time is None and query_data is None:
		t_e_plus = test_end_hr_bins[:,-1]
	elif simul_end_time is None:
		[t_b_plus, t_e_plus, true_count] = query_data
		t_b_plus = np.expand_dims(t_b_plus, axis=-1)
		t_e_plus = np.expand_dims(t_e_plus, axis=-1)
	else:
		t_e_plus = np.expand_dims(simul_end_time, axis=-1)

	_, all_times_pred, _ = simulate_wgan(model_wgan,
										test_data_init_time,
										test_data_input_gaps_bin,
										t_e_plus,
										(test_gap_in_bin_norm_a,
										test_gap_in_bin_norm_d),
										prev_hidden_state=next_hidden_state)
	test_event_count_pred = None
	if query_data is not None:
		[t_b_plus, t_e_plus, true_count] = query_data
		t_b_plus = np.expand_dims(t_b_plus, axis=-1)
		t_e_plus = np.expand_dims(t_e_plus, axis=-1)
		test_event_count_pred = count_events(all_times_pred, t_b_plus, t_e_plus)
		test_event_count_pred = np.array(test_event_count_pred)

		event_count_mse = tf.keras.losses.MSE(true_count, test_event_count_pred).numpy()
		event_count_mae = tf.keras.losses.MAE(true_count, test_event_count_pred).numpy()
		#print("MSE of event count in range:", event_count_mse)
		#print("MAE of event count in range:", event_count_mae)

	all_times_pred = np.expand_dims(all_times_pred.numpy(), axis=-1)
	return test_event_count_pred, all_times_pred
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


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


def compute_full_model_acc(args, test_data, all_bins_count_pred, all_times_bin_pred, test_out_times_in_bin, dataset_name, model_name=None):
	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	enc_len = args.enc_len
	dec_len = args.out_bin_sz
	bin_size = args.bin_size
	all_bins_count_true = test_data_out_bin


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
			t_b_plus = test_end_hr_bins[:,dec_idx] - bin_size
			t_e_plus = test_end_hr_bins[:,dec_idx]
			all_bins_count_pred_lst.append(np.array(count_events(all_times_pred, t_b_plus, t_e_plus)))
		all_bins_count_pred = np.array(all_bins_count_pred_lst).T

	compute_depth = 10
	t_b_plus = test_end_hr_bins[:,0] - bin_size
	t_e_plus = test_end_hr_bins[:,-1]
	deep_mae = compute_hierarchical_mae_deep(all_times_pred, test_out_times_in_bin, t_b_plus, t_e_plus, compute_depth)

	old_stdout = sys.stdout
	sys.stdout=open("Outputs/count_model_"+dataset_name+".txt","a")
	print("____________________________________________________________________")
	print(model_name, 'Full-eval: MAE for Count Prediction:', np.mean(np.abs(all_bins_count_true-all_bins_count_pred )))
	print(model_name, 'Full-eval: MAE for Count Prediction (per bin):', np.mean(np.abs(all_bins_count_true-all_bins_count_pred ), axis=0))
	print(model_name, 'Full-eval: Deep MAE for events Prediction:', deep_mae, 'at depth', compute_depth)
	print("____________________________________________________________________")
	sys.stdout.close()
	sys.stdout = old_stdout
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

	return all_event_pred, all_event_true, mae 

def compute_hierarchical_mae_deep(all_event_pred, all_event_true, t_b_plus, t_e_plus, compute_depth):
	if compute_depth == 0:
		return 0
	
	all_event_pred, all_event_true, res = compute_mae_cur_bound(all_event_pred, 
											all_event_true, t_b_plus, t_e_plus)

	t_b_e_mid = (t_b_plus + t_e_plus) / 2.0
	compute_depth -= 1

	res1 = compute_hierarchical_mae_deep(all_event_pred, all_event_true, t_b_plus, t_b_e_mid, compute_depth)
	res2 = compute_hierarchical_mae_deep(all_event_pred, all_event_true, t_b_e_mid, t_e_plus, compute_depth)
	return res + res1 + res2

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
	return compute_hierarchical_mae_deep(all_event_pred, all_event_true, t_b_plus, t_e_plus, compute_depth)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

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
	more_threshold, interval_size, test_out_times_in_bin] = query_data

	[test_data_in_bin, test_data_out_bin, test_end_hr_bins,
	test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = test_data

	all_run_count_fun_name, all_run_count_fun, all_run_count_fun_rmtpp = list(), list(), list()
	for each in all_run_fun_pdf:
		all_run_count_fun_name.append(each)
		all_run_count_fun.append(all_run_fun_pdf[each][0])
		all_run_count_fun_rmtpp.append(all_run_fun_pdf[each][1])

	sample_count = 30
	no_points = 500
	test_plots_cnts=25

	x_range = np.round(np.array([(test_data_in_time_end_bin), (test_data_in_time_end_bin+(arguments.bin_size*arguments.out_bin_sz))]))[:,:,0].T.astype(int)

	interval_counts_more_true = np.zeros((len(test_data_in_time_end_bin), no_points))
	interval_counts_less_true = np.zeros((len(test_data_in_time_end_bin), no_points))

	for batch_idx in range(len(test_data_in_time_end_bin)):
		all_begins = np.linspace(x_range[batch_idx][0], x_range[batch_idx][1], no_points)
		for begin_idx in range(len(all_begins)):
			interval_start_cand = np.array([all_begins[begin_idx]])
			interval_end_cand = np.array([all_begins[begin_idx] + interval_size])
			interval_count_in_range_pred = count_events(np.expand_dims(np.array(test_out_times_in_bin[batch_idx]), axis=0), 
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
		interval_counts_more = np.zeros((len(test_data_in_time_end_bin), no_points))
		interval_counts_less = np.zeros((len(test_data_in_time_end_bin), no_points))
		interval_counts_more_rank = np.zeros((len(test_data_in_time_end_bin), no_points))
		interval_counts_less_rank = np.zeros((len(test_data_in_time_end_bin), no_points))

		for each_sim_idx in range(sample_count):
			# print('Simulating sample number', each_sim_idx)

			if all_run_count_fun_name[run_count_fun_idx] == 'run_wgan_for_count':
				simul_end_time = x_range[:,1]
				_, all_event_pred_uncut = all_run_count_fun[run_count_fun_idx](arguments, models, data, test_data, simul_end_time=simul_end_time)

			elif (all_run_count_fun_name[run_count_fun_idx] == 'run_rmtpp_for_count_with_mse' or \
				 all_run_count_fun_name[run_count_fun_idx] == 'run_rmtpp_for_count_with_mse_var' or \
				 all_run_count_fun_name[run_count_fun_idx] == 'run_rmtpp_for_count_with_nll'):

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
		test_plots_cnts = len(test_data_in_time_end_bin)

	os.makedirs('Outputs/'+dataset_name+'_threshold_less/', exist_ok=True)
	os.makedirs('Outputs/'+dataset_name+'_threshold_more/', exist_ok=True)
	os.makedirs('Outputs/'+dataset_name+'_threshold_less_rank/', exist_ok=True)
	os.makedirs('Outputs/'+dataset_name+'_threshold_more_rank/', exist_ok=True)
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
		img_name_cnt = 'Outputs/'+dataset_name+'_threshold_more/'+dataset_name+'_threshold_more_'+str(batch_idx)+'.svg'
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
		img_name_cnt = 'Outputs/'+dataset_name+'_threshold_less/'+dataset_name+'_threshold_less_'+str(batch_idx)+'.svg'
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
		img_name_cnt = 'Outputs/'+dataset_name+'_threshold_more_rank/'+dataset_name+'_threshold_more_rank_'+str(batch_idx)+'.svg'
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
		img_name_cnt = 'Outputs/'+dataset_name+'_threshold_less_rank/'+dataset_name+'_threshold_less_rank_'+str(batch_idx)+'.svg'
		plt.legend(loc='upper right')
		plt.savefig(img_name_cnt, format='svg', dpi=1200)
		plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#


#####################################################
# 			Model Init and Run handler				#
#####################################################
def run_model(dataset_name, model_name, dataset, args, prev_models=None, run_model_flags=None):
	print("Running for model", model_name, "on dataset", dataset_name)

	tf.random.set_seed(args.seed)
	test_data_out_bin = dataset['test_data_out_bin']
	event_count_preds_true = test_data_out_bin
	batch_size = args.batch_size
	model=None
	result=None

	if model_name == 'hierarchical':
		train_data_in_bin = dataset['train_data_in_bin']
		train_data_out_bin = dataset['train_data_out_bin']
		test_data_in_bin = dataset['test_data_in_bin']
		test_data_out_bin = dataset['test_data_out_bin']
		test_mean_bin = dataset['test_mean_bin']
		test_std_bin = dataset['test_std_bin']

		data = [train_data_in_bin, train_data_out_bin]
		test_data = [test_data_in_bin, test_data_out_bin, test_mean_bin, test_std_bin]
		event_count_preds_cnt = run_hierarchical(args, data, test_data)
		model, result = event_count_preds_cnt

	if model_name == 'count_model':
		train_data_in_bin = dataset['train_data_in_bin']
		train_data_out_bin = dataset['train_data_out_bin']
		test_data_in_bin = dataset['test_data_in_bin']
		test_data_out_bin = dataset['test_data_out_bin']
		test_mean_bin = dataset['test_mean_bin']
		test_std_bin = dataset['test_std_bin']

		data = [train_data_in_bin, train_data_out_bin]
		test_data = [test_data_in_bin, test_data_out_bin, test_mean_bin, test_std_bin]
		event_count_preds_cnt = run_count_model(args, data, test_data)
		model, result = event_count_preds_cnt

	if model_name in ['rmtpp_mse', 'rmtpp_nll', 'rmtpp_mse_var', 'wgan', 'rmtpp_count']:
		train_data_in_gaps = dataset['train_data_in_gaps']
		train_data_out_gaps = dataset['train_data_out_gaps']
		train_dataset_gaps = tf.data.Dataset.from_tensor_slices((train_data_in_gaps,
														train_data_out_gaps)).batch(batch_size,
														drop_remainder=True)
		dev_data_in_gaps = dataset['dev_data_in_gaps']
		dev_data_out_gaps = dataset['dev_data_out_gaps']
		train_norm_a_gaps = dataset['train_norm_a_gaps']
		train_norm_d_gaps = dataset['train_norm_d_gaps']

		test_data_in_gaps_bin = dataset['test_data_in_gaps_bin']
		test_end_hr_bins = dataset['test_end_hr_bins'] 
		test_data_in_time_end_bin = dataset['test_data_in_time_end_bin']
		test_gap_in_bin_norm_a = dataset['test_gap_in_bin_norm_a'] 
		test_gap_in_bin_norm_d = dataset['test_gap_in_bin_norm_d']

		test_data = [test_data_in_gaps_bin, test_end_hr_bins, test_data_in_time_end_bin, 
					test_gap_in_bin_norm_a, test_gap_in_bin_norm_d]
		train_norm_gaps = [train_norm_a_gaps ,train_norm_d_gaps]
		data = [train_dataset_gaps, dev_data_in_gaps, dev_data_out_gaps, train_norm_gaps]

		if model_name == 'wgan':
			model, result = run_wgan(args, data, test_data)

		if model_name == 'rmtpp_mse':
			event_count_preds_mse = run_rmtpp_init(args, data, test_data, NLL_loss=False)
			model, result = event_count_preds_mse

		if model_name == 'rmtpp_nll':
			event_count_preds_nll = run_rmtpp_init(args, data, test_data, NLL_loss=True)
			model, result = event_count_preds_nll

		if model_name == 'rmtpp_mse_var':
			event_count_preds_mse_var = run_rmtpp_init(args, data, test_data, NLL_loss=False, use_var_model=True)
			model, result = event_count_preds_mse_var

		# This block contains all the inference models that returns
		# answers to various queries
		if model_name == 'rmtpp_count':
			test_data_in_bin = dataset['test_data_in_bin']
			test_data_out_bin = dataset['test_data_out_bin']
			test_mean_bin = dataset['test_mean_bin']
			test_std_bin = dataset['test_std_bin']

			test_time_out_tb_plus = dataset['test_time_out_tb_plus']
			test_time_out_te_plus = dataset['test_time_out_te_plus']
			test_out_event_count_true = dataset['test_out_event_count_true']
			test_out_all_event_true = dataset['test_out_all_event_true']

			interval_range_count_less = dataset['interval_range_count_less']
			interval_range_count_more = dataset['interval_range_count_more']
			less_threshold = dataset['less_threshold']
			more_threshold = dataset['more_threshold']
			interval_size = dataset['interval_size']
			test_out_times_in_bin = dataset['test_out_times_in_bin']

			#model_cnt, model_rmtpp, model_wgan = prev_models['count_model'], prev_models['rmtpp_mse'], prev_models['wgan']
			models = prev_models
			test_data_in_bin = test_data_in_bin.astype(np.float32)
			test_data_out_bin = test_data_out_bin.astype(np.float32)
			test_data = [test_data_in_bin, test_data_out_bin, test_end_hr_bins,
			test_data_in_time_end_bin, test_data_in_gaps_bin, test_mean_bin, test_std_bin,
			test_gap_in_bin_norm_a, test_gap_in_bin_norm_d]
			test_out_times_in_bin = np.array(test_out_times_in_bin)

			data = None
			compute_depth = 5

			query_1_data = [test_time_out_tb_plus, test_time_out_te_plus, test_out_event_count_true]
			query_2_data = [interval_range_count_less, interval_range_count_more, 
							less_threshold, more_threshold, interval_size, test_out_times_in_bin]

			old_stdout = sys.stdout
			sys.stdout=open("Outputs/count_model_"+dataset_name+".txt","a")
			print("____________________________________________________________________")
			print("True counts")
			print(test_out_event_count_true)
			print("____________________________________________________________________")
			print("")

			if 'run_rmtpp_count_with_optimization' in run_model_flags and run_model_flags['run_rmtpp_count_with_optimization']:
				print("Prediction for run_rmtpp_count_with_optimization model")
				_, all_times_bin_pred_opt = run_rmtpp_count_with_optimization(args, models, data, test_data)
				deep_mae = compute_hierarchical_mae(all_times_bin_pred_opt, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred_opt, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred_opt, test_out_times_in_bin, dataset_name, 'run_rmtpp_count_with_optimization')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_with_optimization_fixed_cnt' in run_model_flags and run_model_flags['run_rmtpp_with_optimization_fixed_cnt']:
				print("Prediction for run_rmtpp_with_optimization_fixed_cnt model")
				_, all_times_bin_pred_opt = run_rmtpp_with_optimization_fixed_cnt(args, models, data, test_data)
				deep_mae = compute_hierarchical_mae(all_times_bin_pred_opt, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred_opt, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred_opt, test_out_times_in_bin, dataset_name, 'run_rmtpp_with_optimization_fixed_cnt')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_with_optimization_fixed_cnt_solver_with_nll' in run_model_flags and run_model_flags['run_rmtpp_with_optimization_fixed_cnt_solver_with_nll']:
				print("Prediction for run_rmtpp_with_optimization_fixed_cnt_solver_with_nll model")
				_, all_times_bin_pred_opt = run_rmtpp_with_optimization_fixed_cnt_solver(args, models, data, test_data, rmtpp_type='nll')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred_opt, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred_opt, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred_opt, test_out_times_in_bin, dataset_name, 'run_rmtpp_with_optimization_fixed_cnt_solver_with_nll')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_with_optimization_fixed_cnt_solver_with_mse' in run_model_flags and run_model_flags['run_rmtpp_with_optimization_fixed_cnt_solver_with_mse']:
				print("Prediction for run_rmtpp_with_optimization_fixed_cnt_solver_with_mse model")
				_, all_times_bin_pred_opt = run_rmtpp_with_optimization_fixed_cnt_solver(args, models, data, test_data, rmtpp_type='mse')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred_opt, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred_opt, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred_opt, test_out_times_in_bin, dataset_name, 'run_rmtpp_with_optimization_fixed_cnt_solver_with_mse')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_with_optimization_fixed_cnt_solver_with_mse_var' in run_model_flags and run_model_flags['run_rmtpp_with_optimization_fixed_cnt_solver_with_mse_var']:
				print("Prediction for run_rmtpp_with_optimization_fixed_cnt_solver_with_mse_var model")
				_, all_times_bin_pred_opt = run_rmtpp_with_optimization_fixed_cnt_solver(args, models, data, test_data, rmtpp_type='mse_var')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred_opt, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred_opt, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred_opt, test_out_times_in_bin, dataset_name, 'run_rmtpp_with_optimization_fixed_cnt_solver_with_mse_var')
				print("____________________________________________________________________")
				print("")


			print("")

			if 'run_rmtpp_count_cont_rmtpp_with_nll' in run_model_flags and run_model_flags['run_rmtpp_count_cont_rmtpp_with_nll']:
				print("Prediction for run_rmtpp_count_cont_rmtpp model with rmtpp_nll")
				all_bins_count_pred, all_times_bin_pred = run_rmtpp_count_cont_rmtpp(args, models, data, test_data, rmtpp_type='nll')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, all_bins_count_pred, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_count_cont_rmtpp_with_nll')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_count_cont_rmtpp_with_mse' in run_model_flags and run_model_flags['run_rmtpp_count_cont_rmtpp_with_mse']:
				print("Prediction for run_rmtpp_count_cont_rmtpp model with rmtpp_mse")
				all_bins_count_pred, all_times_bin_pred = run_rmtpp_count_cont_rmtpp(args, models, data, test_data, rmtpp_type='mse')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, all_bins_count_pred, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_count_cont_rmtpp_with_mse')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_count_cont_rmtpp_with_mse_var' in run_model_flags and run_model_flags['run_rmtpp_count_cont_rmtpp_with_mse_var']:
				print("Prediction for run_rmtpp_count_cont_rmtpp model with rmtpp_mse_var")
				all_bins_count_pred, all_times_bin_pred = run_rmtpp_count_cont_rmtpp(args, models, data, test_data, rmtpp_type='mse_var')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, all_bins_count_pred, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_count_cont_rmtpp_with_mse_var')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_count_reinit_with_nll' in run_model_flags and run_model_flags['run_rmtpp_count_reinit_with_nll']:
				print("Prediction for run_rmtpp_count_reinit model with rmtpp_nll")
				all_bins_count_pred, all_times_bin_pred = run_rmtpp_count_reinit(args, models, data, test_data, rmtpp_type='nll')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, all_bins_count_pred, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_count_reinit_with_nll')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_count_reinit_with_mse' in run_model_flags and run_model_flags['run_rmtpp_count_reinit_with_mse']:
				print("Prediction for run_rmtpp_count_reinit model with rmtpp_mse")
				all_bins_count_pred, all_times_bin_pred = run_rmtpp_count_reinit(args, models, data, test_data, rmtpp_type='mse')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, all_bins_count_pred, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_count_reinit_with_mse')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_count_reinit_with_mse_var' in run_model_flags and run_model_flags['run_rmtpp_count_reinit_with_mse_var']:
				print("Prediction for run_rmtpp_count_reinit model with rmtpp_mse_var")
				all_bins_count_pred, all_times_bin_pred = run_rmtpp_count_reinit(args, models, data, test_data, rmtpp_type='mse_var')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, all_bins_count_pred, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_count_reinit_with_mse_var')
				print("____________________________________________________________________")
				print("")

			print("")

			if 'run_count_only_model' in run_model_flags and run_model_flags['run_count_only_model']:
				print("Prediction for run_count_only_model model with rmtpp_mse")
				all_bins_count_pred, all_times_bin_pred = run_count_only_model(args, models, data, test_data)
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, all_bins_count_pred, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_count_only_model')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_for_count_with_mse' in run_model_flags and run_model_flags['run_rmtpp_for_count_with_mse']:
				print("Prediction for plain_rmtpp_count model with rmtpp_mse")
				result, all_times_bin_pred = run_rmtpp_for_count(args, models, data, test_data, rmtpp_type='mse')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_for_count_with_mse')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_for_count_with_mse_var' in run_model_flags and run_model_flags['run_rmtpp_for_count_with_mse_var']:
				print("Prediction for plain_rmtpp_count model with rmtpp_mse_var")
				result, all_times_bin_pred = run_rmtpp_for_count(args, models, data, test_data, rmtpp_type='mse_var')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_for_count_with_mse_var')
				print("____________________________________________________________________")
				print("")

			if 'run_rmtpp_for_count_with_nll' in run_model_flags and run_model_flags['run_rmtpp_for_count_with_nll']:
				print("Prediction for plain_rmtpp_count model with rmtpp_nll")
				result, all_times_bin_pred = run_rmtpp_for_count(args, models, data, test_data, rmtpp_type='nll')
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_rmtpp_for_count_with_nll')
				print("____________________________________________________________________")
				print("")

			if 'run_wgan_for_count' in run_model_flags and run_model_flags['run_wgan_for_count']:
				print("Prediction for plain_wgan_count model")
				result, all_times_bin_pred = run_wgan_for_count(args, models, data, test_data)
				deep_mae = compute_hierarchical_mae(all_times_bin_pred, query_1_data, test_out_all_event_true, compute_depth)
				threshold_mae = compute_threshold_loss(all_times_bin_pred, query_2_data)
				print("deep_mae", deep_mae)
				compute_full_model_acc(args, test_data, None, all_times_bin_pred, test_out_times_in_bin, dataset_name, 'run_wgan_for_count')
				print("____________________________________________________________________")
				print("")

			print("")

			if 'compute_time_range_pdf' in run_model_flags and run_model_flags['compute_time_range_pdf']:
				print("Running threshold query to generate pdf for all models")
				model_data = [args, models, data, test_data]
				all_run_fun_pdf = {
					'run_rmtpp_with_optimization_fixed_cnt_solver_with_nll': [run_rmtpp_with_optimization_fixed_cnt_solver, 'nll'],
					'run_rmtpp_with_optimization_fixed_cnt_solver_with_mse': [run_rmtpp_with_optimization_fixed_cnt_solver, 'mse'],
					'run_rmtpp_with_optimization_fixed_cnt_solver_with_mse_var': [run_rmtpp_with_optimization_fixed_cnt_solver, 'mse_var'],

					'run_rmtpp_count_cont_rmtpp_with_nll': [run_rmtpp_count_cont_rmtpp, 'nll'],
					'run_rmtpp_count_cont_rmtpp_with_mse': [run_rmtpp_count_cont_rmtpp, 'mse'],
					'run_rmtpp_count_cont_rmtpp_with_mse_var': [run_rmtpp_count_cont_rmtpp, 'mse_var'],
					'run_rmtpp_count_reinit_with_nll': [run_rmtpp_count_reinit, 'nll'],
					'run_rmtpp_count_reinit_with_mse': [run_rmtpp_count_reinit, 'mse'],
					'run_rmtpp_count_reinit_with_mse_var': [run_rmtpp_count_reinit, 'mse_var'],

					'run_rmtpp_for_count_with_nll': [run_rmtpp_for_count, 'nll'],
					'run_rmtpp_for_count_with_mse': [run_rmtpp_for_count, 'mse'],
					'run_rmtpp_for_count_with_mse_var': [run_rmtpp_for_count, 'mse_var'],
					'run_wgan_for_count': [run_wgan_for_count, None],
					'run_count_only_model': [run_count_only_model, None],
				}
				clean_dict_for_na_model(all_run_fun_pdf, run_model_flags)
				compute_time_range_pdf(all_run_fun_pdf, model_data, query_2_data, dataset_name)
				print("____________________________________________________________________")
				print("")

			print("")
			sys.stdout.close()
			sys.stdout = old_stdout
			
	return model, result

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
