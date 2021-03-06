import numpy as np
import os, sys
import abc
import matplotlib.pyplot as plt
from bisect import bisect_right
from modules import Hawkes as hk
import time
from scipy.stats import entropy
from collections import Counter

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras.preprocessing.sequence import pad_sequences

flatten = lambda l: [item for sublist in l for item in sublist]

class Intensity(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getValue(self, t):
        return

class IntensityHomogenuosPoisson(Intensity):

    def __init__(self, lam):
        self.lam = lam

    def getValue(self, t):
        return self.lam

    def getUpperBound(self, from_t, to_t):
        return self.lam

def generate_sample(intensity, T, n):
    Sequnces = []
    i = 0
    while True:
        seq = []
        t = 0
        while len(seq)<T:
            intens1 = intensity.getUpperBound(t,T)
            intens1 = intens1[i]
            dt = np.random.exponential(1/intens1)
            new_t = t + dt
            #if new_t > T:
            #    break

            intens2 = intensity.getValue(new_t)
            intens2 = intens2[i]
            u = np.random.uniform()
            if intens2/intens1 >= u:
                #seq.append(new_t)
                seq.append(dt)
            t = new_t
        #if len(seq)>1:
        if len(seq):
            Sequnces.append(seq)
            i+=1
        if i==n:
            break
    return Sequnces

def add_metrics_to_dict(
	metrics_dict,
	model_name,
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
	reqd_time,
	opt_loss=0.,
	cont_loss=0.,
	count_loss=0.,
):
	if model_name not in metrics_dict:
		metrics_dict[model_name] = dict()
	metrics_dict[model_name]['count_mae_fh'] = count_mae_fh
	metrics_dict[model_name]['wass_dist_fh'] = wass_dist_fh
	metrics_dict[model_name]['bleu_score_fh'] = bleu_score_fh

	metrics_dict[model_name]['count_mae_rh'] = count_mae_rh
	metrics_dict[model_name]['wass_dist_rh'] = wass_dist_rh
	metrics_dict[model_name]['bleu_score_rh'] = bleu_score_rh

	for i in range(len(count_mae_fh_per_bin)):
		metrics_dict[model_name]['count_mae_fh_bin_'+str(i)] = count_mae_fh_per_bin[i]
		metrics_dict[model_name]['wass_dist_fh_per_bin'+str(i)] = wass_dist_fh_per_bin[i]
		metrics_dict[model_name]['bleu_score_fh_per_bin'+str(i)] = bleu_score_fh_per_bin[i]

	metrics_dict[model_name]['more_metric'] = more_metric
	metrics_dict[model_name]['less_metric'] = less_metric

	metrics_dict[model_name]['opt_loss'] = opt_loss
	metrics_dict[model_name]['cont_loss'] = cont_loss
	metrics_dict[model_name]['count_loss'] = count_loss

	metrics_dict[model_name]['reqd_time'] = reqd_time

	print(model_name, 'count_mae_fh', count_mae_fh)
	print(model_name, 'wass_dist_fh', wass_dist_fh)
	print(model_name, 'bleu_score_fh', bleu_score_fh)

	print(model_name, 'count_mae_rh', count_mae_rh)
	print(model_name, 'wass_dist_rh', wass_dist_rh)
	print(model_name, 'bleu_score_rh', bleu_score_rh)

	for i in range(len(count_mae_fh_per_bin)):
		print(model_name, 'count_mae_fh_bin_'+str(i), count_mae_fh_per_bin[i])
		print(model_name, 'wass_dist_fh_per_bin'+str(i), wass_dist_fh_per_bin[i])
		print(model_name, 'bleu_score_fh_per_bin'+str(i), bleu_score_fh_per_bin[i])

	print(model_name, 'more_metric', more_metric)
	print(model_name, 'less_metric', less_metric)

	print('Time required for ', model_name, ':', reqd_time)

	return metrics_dict

def write_arr_to_file(
	output_dir, current_dataset, inference_model_name,
	arr_true, arr_pred, types_true, types_pred,
	counts_true, counts_pred, counts_sigms,
	counts_input,
):
	
	output_path = os.path.join(
		output_dir, current_dataset+'__'+inference_model_name,
	)
	# Files are saved in .npy format
	np.save(
		output_path + '__' + 'fh_times_true',
		arr_true,
	)
	np.save(
		output_path + '__' + 'fh_times_pred',
		arr_pred,
	)
	np.save(
		output_path + '__' + 'fh_types_true',
		types_true,
	)
	np.save(
		output_path + '__' + 'fh_types_pred',
		types_pred,
	)

	for fname in os.listdir(output_dir):
	    if fname.endswith(current_dataset+'__fh_counts_true'):
	        break
	else:
		np.save(
			output_path + '__' + 'fh_counts_true',
			counts_true,
		)
	np.save(
		output_path + '__' + 'fh_counts_pred',
		counts_pred,
	)
	for fname in os.listdir(output_dir):
	    if fname.endswith(current_dataset+'__counts_input'):
	        break
	else:
		np.save(
			os.path.join(
				output_dir, current_dataset + '__' + 'counts_input',
			),
			counts_input,
		)
	if inference_model_name == 'count_only':
		np.save(
			output_path + '__' + 'fh_counts_sigms',
			counts_sigms,
		)

def write_pe_metrics_to_file(
	output_path,
	count_mae_fh_pe, wass_dist_fh_pe, bleu_score_fh_pe,
	more_metric_pe, less_metric_pe
):
	np.save(
		output_path + '__' + 'count_mae_fh_pe',
		count_mae_fh_pe,
	)
	np.save(
		output_path + '__' + 'wass_dist_fh_pe',
		wass_dist_fh_pe,
	)
	np.save(
		output_path + '__' + 'bleu_score_fh_pe',
		wass_dist_fh_pe,
	)
	np.save(
		output_path + '__' + 'more_metric_pe',
		more_metric_pe,
	)
	np.save(
		output_path + '__' + 'less_metric_pe',
		less_metric_pe,
	)

def write_opt_losses_to_file(
	output_path,
	opt_losses,
	cont_losses,
	count_losses,
):
	np.save(
		output_path + '__' + 'opt_losses',
		opt_losses,
	)
	np.save(
		output_path + '__' + 'cont_losses',
		cont_losses,
	)
	np.save(
		output_path + '__' + 'count_losses',
		count_losses,
	)

def normal_approx(pb_mean, pb_var, threshold):
	unit_normal_dist = tfd.Normal(loc=tf.zeros_like(pb_mean), scale=tf.ones_like(pb_mean))
	x = (threshold + 0.5 - pb_mean) / tf.sqrt(pb_var)
	pb_threshold_cdf = (unit_normal_dist.cdf(x))
	return pb_threshold_cdf

def refined_normal_approx(pb_mean, pb_var, pb_skew, threshold):
	unit_normal_dist = tfd.Normal(loc=tf.zeros_like(pb_mean), scale=tf.ones_like(pb_mean))
	x = (threshold + 0.5 - pb_mean) / tf.sqrt(pb_var)
	pb_threshold_cdf = (unit_normal_dist.prob(x)) + pb_skew * (1-x*x) * unit_normal_dist.prob(x) / 6.

	return pb_threshold_cdf

def normalize_data(data):
	mean = np.mean(data)
	std = np.std(data)
	return (data - mean)/std, mean, std

def normalize_data_given_param(data, mean, std):
	return (data - mean)/std

def denormalize_data(data, mean, std):
	return (data * std) + mean

def denormalize_data_stddev(data, mean, std):
	return (data * std)

def normalize_avg(data):
	norm_a = 0.0
	norm_d = np.mean(data)
	return data/norm_d, norm_a, norm_d

def normalize_avg_given_param(data, norm_a, norm_d):
	return data/norm_d

def denormalize_avg(data, norm_a, norm_d):
	return data*norm_d

def get_bins(timestamps, binsize):
   	cnt=0
   	bincounts=[]
   	t_b=0
   	t_e=t_b+binsize
   	for ts in timestamps:
   	    if ts<=t_e:
   	        cnt += 1
   	    else:
   	        bincounts.append(cnt)
   	        cnt=0
   	        t_b = t_e
   	        t_e = t_b + binsize
   	return np.array(bincounts)

def get_optimal_bin_size(dataset_name):
	timestamps = np.loadtxt('data/'+dataset_name+'.txt')
	time_interval = timestamps[-1]-timestamps[0]
	events_count = len(timestamps)
	event_count = 60
	if dataset_name in ['911_ems']:
		event_count=100
	if dataset_name in ['911_traffic']:
		event_count=70
	if dataset_name in ['taxi']:
		event_count=250

	opt_bin_sz = int(round((time_interval*event_count) / events_count))
	hr_scale = round(opt_bin_sz/3600)

	if hr_scale == 0:
		min_scale = round(opt_bin_sz/60)
		print('Bins are at cycle of', min_scale, 'mins')
		opt_bin_sz = min_scale * 60
	elif hr_scale <= 12:
		opt_bin_sz = hr_scale*3600
		print('Bins are at cycle of', hr_scale, 'hours')
	else:
		day_scale = round(opt_bin_sz/(3600*24))
		print('Bins are at cycle of', day_scale, 'days')
		opt_bin_sz = day_scale * (3600*24)
	return opt_bin_sz

def find_best_bin_size(dataset_name):
	'''
		Find smallest bin s.t. each bin contains at least one event
		and each consecutive in_bin_sz bins contains at least 80 events
	'''
	timestamps = np.loadtxt('data/'+dataset_name+'.txt')
	#for bin_size in np.arange(1, 24+1)*3600.:
	for bin_size in np.array([1., 2, 3, 4., 6, 8, 12, 24.])*3600.:
		bincounts = get_bins(timestamps, bin_size)
		#import ipdb
		#ipdb.set_trace()
		print(bin_size, np.sum(bincounts==0.), bincounts.shape)
		if np.sum(bincounts==0) == 0:
			return bin_size



def generate_plots(args, dataset_name, dataset, per_model_count, test_sample_idx=1, count_var=None):
	inp_seq_len_plot = 10
	dec_len = args.out_bin_sz

	true_pred = per_model_count['true']
	rmtpp_mse_pred = true_pred
	rmtpp_nll_pred = true_pred
	hierarchical_pred = true_pred
	count_model_pred = true_pred
	wgan_pred = true_pred
	transformer_pred = true_pred
	hawkes_pred = true_pred
	if 'rmtpp_mse' in per_model_count:
		rmtpp_mse_pred = per_model_count['rmtpp_mse']
		event_count_preds_mse = rmtpp_mse_pred
		rmtpp_mse_pred = event_count_preds_mse[test_sample_idx].astype(np.float32)
	if 'rmtpp_nll' in per_model_count:
		rmtpp_nll_pred = per_model_count['rmtpp_nll']
		event_count_preds_nll = rmtpp_nll_pred
		rmtpp_nll_pred = event_count_preds_nll[test_sample_idx].astype(np.float32)
	if 'hierarchical' in per_model_count:
		hierarchical_pred = per_model_count['hierarchical']
		event_count_preds_cnt = hierarchical_pred
		hierarchical_pred = event_count_preds_cnt[test_sample_idx].astype(np.float32)
	if 'count_model' in per_model_count:
		count_model_pred = per_model_count['count_model']
		event_count_preds_count = count_model_pred
		count_model_pred = event_count_preds_count[test_sample_idx].astype(np.float32)
	if 'wgan' in per_model_count:
		wgan_pred = per_model_count['wgan']
		event_count_preds_wgan = wgan_pred
		wgan_pred = event_count_preds_wgan[test_sample_idx].astype(np.float32)
	if 'transformer' in per_model_count:
		transformer_pred = per_model_count['transformer']
		event_count_preds_transformer = transformer_pred
		transformer_pred = event_count_preds_transformer[test_sample_idx].astype(np.float32)
	if 'hawkes_model' in per_model_count:
		hawkes_pred = per_model_count['hawkes_model']
		event_count_preds_hawkes = hawkes_pred
		hawkes_pred = event_count_preds_hawkes[test_sample_idx].astype(np.float32)

	event_count_preds_true = true_pred
	true_pred = event_count_preds_true[test_sample_idx].astype(np.float32)

	count_test_in_counts = np.squeeze(dataset['count_test_in_counts'], axis=-1)
	count_test_normm = dataset['count_test_normm']
	count_test_norms = dataset['count_test_norms']
	
	true_inp_bins = denormalize_data(count_test_in_counts[test_sample_idx,inp_seq_len_plot:], 
							count_test_normm, count_test_norms)
	true_inp_bins = true_inp_bins.astype(np.float32)
	x = np.arange(inp_seq_len_plot+dec_len)
	if count_var is not None:
		count_model_std = count_var[test_sample_idx].astype(np.float32)
		count_model_std_up = np.concatenate((true_inp_bins, count_model_pred))
		count_model_std_down = np.concatenate((true_inp_bins, count_model_pred))
		count_model_std_up[-dec_len:] = count_model_pred+count_model_std
		count_model_std_down[-dec_len:] = count_model_pred-count_model_std

	if 'hierarchical' in per_model_count:
		plt.plot(x, true_inp_bins.tolist()+hierarchical_pred.tolist(), label='hierarchical_pred')
	if 'count_model' in per_model_count:
		plt.plot(x, true_inp_bins.tolist()+count_model_pred.tolist(), label='count_model_pred')
	if 'wgan' in per_model_count:
		plt.plot(x, true_inp_bins.tolist()+wgan_pred.tolist(), label='wgan_pred')
	if 'transformer' in per_model_count:
		plt.plot(x, true_inp_bins.tolist()+transformer_pred.tolist(), label='transformer_pred')
	if 'rmtpp_mse' in per_model_count:
		plt.plot(x, true_inp_bins.tolist()+rmtpp_mse_pred.tolist(), label='rmtpp_mse_pred')
	if 'rmtpp_nll' in per_model_count:
		plt.plot(x, true_inp_bins.tolist()+rmtpp_nll_pred.tolist(), label='rmtpp_nll_pred')
	if 'hawkes_model' in per_model_count:
		plt.plot(x, true_inp_bins.tolist()+hawkes_pred.tolist(), label='hawkes_pred')
	plt.plot(x, true_inp_bins.tolist()+true_pred.tolist(), label='true_pred')

	if count_var is not None:
		plt.fill_between(x, count_model_std_down, count_model_std_up, color='gray', alpha=0.5)

	plt.axvline(x=inp_seq_len_plot-1, color='k', linestyle='--')
	plt.legend(loc='upper left')
	plt.savefig(os.path.join(args.output_dir, dataset_name+'_'+str(test_sample_idx)+'.svg'), format='svg', dpi=1200)
	plt.close()

def create_bin(times, types, bin_size, num_bins):
	"""

	Args:
		times (list): A sequence of raw timestamps
		bin_size (int): Length of the bin

	Returns:
		cnt_bin (list): Number of events in each bin
		end_hr_bin (list): End time of each bin
		bintotimes (list of lists): list of lists of timestamps in each bin
	"""
	end_hr = 0
	cnt_bin = []
	end_hr_bin = []
	bintotimes = []
	bintogaps = []
	bintotypes = []
	last_ind= [0]
	ind = 0
	bin_id = 0
	while bin_id < num_bins:

		end_hr = end_hr + bin_size
		times_saver, gaps_saver, types_saver = [], [], []

		while times[ind]<=end_hr:
			if ind>0:
				gaps_saver.append(times[ind]-times[ind-1])
			else:
				gaps_saver.append(0.)
			times_saver.append(times[ind])
			types_saver.append(types[ind])
			ind += 1
			if ind>=len(times):
				break

		if bin_id < num_bins:
			cnt_bin.append(ind-last_ind[-1])
			last_ind.append(ind)
			end_hr_bin.append(end_hr)
			bintotimes.append(times_saver)
			bintogaps.append(gaps_saver)
			bintotypes.append(types_saver)
			bin_id += 1


	print('Total bins generated', len(bintotimes))
	print('Each bin has Average', int(round(np.mean(cnt_bin))), 'timestamps')
	return cnt_bin, end_hr_bin, bintotimes, bintogaps, bintotypes

def generate_train_dev_test_data(
	count_counts, count_binend, bintotimes, bintogaps,
	in_bin_sz, bintotypes=None,
):
	train_per = 0.8 - 0.16
	dev_per = 0.16
	data_sz = len(count_counts)

	count_train_counts = count_counts[:int(train_per*data_sz)]
	count_train_binend = count_binend[:int(train_per*data_sz)]
	bintotimes_train = bintotimes[:int(train_per*data_sz)]
	bintogaps_train = bintogaps[:int(train_per*data_sz)]

	count_dev_counts = count_counts[int(train_per*data_sz)-in_bin_sz:int((train_per+dev_per)*data_sz)]
	count_dev_binend = count_binend[int(train_per*data_sz)-in_bin_sz:int((train_per+dev_per)*data_sz)]
	bintotimes_dev = bintotimes[int(train_per*data_sz)-in_bin_sz:int((train_per+dev_per)*data_sz)]
	bintogaps_dev = bintogaps[int(train_per*data_sz)-in_bin_sz:int((train_per+dev_per)*data_sz)]

	count_test_counts = count_counts[int((train_per+dev_per)*data_sz)-in_bin_sz:]
	count_test_binend = count_binend[int((train_per+dev_per)*data_sz)-in_bin_sz:]
	bintotimes_test = bintotimes[int((train_per+dev_per)*data_sz)-in_bin_sz:]
	bintogaps_test = bintogaps[int((train_per+dev_per)*data_sz)-in_bin_sz:]

	if bintotypes is not None:
		bintotypes_train = bintotypes[:int(train_per*data_sz)]
		bintotypes_dev = bintotypes[int(train_per*data_sz)-in_bin_sz:int((train_per+dev_per)*data_sz)]
		bintotypes_test = bintotypes[int((train_per+dev_per)*data_sz)-in_bin_sz:]
	else:
		bintotypes_train, bintotypes_dev, bintotypes_test = None, None, None


	return  (count_train_counts, count_dev_counts, count_test_counts,
			 count_train_binend, count_dev_binend, count_test_binend,
			 bintogaps_train, bintogaps_dev, bintogaps_test,
			 bintotimes_train, bintotimes_dev, bintotimes_test,
			 bintotypes_train, bintotypes_dev, bintotypes_test)

def get_data_in_next_n_bins(n, inp, out, bin_size):
	out_n_bins = []
	for i in range(len(out)):
		if (out[i]-inp[-1]) > (bin_size*n):
			break
		out_n_bins.append(out[i])
	return out_n_bins

def create_nowcast_io_seqs(data, chunk_len, stride):

	data_in, data_out = [], []
	for idx in range(0, len(data), stride):
		if idx+chunk_len < len(data):
			data_in.append(data[idx:idx+chunk_len])
			data_out.append(data[idx+1:idx+1+chunk_len])

	data_in = np.array(data_in)
	data_out = np.array(data_out)
	return data_in, data_out

def create_forecast_io_seqs(data, enc_len, dec_len, stride):

	data_in, data_out = [], []
	for idx in range(0, len(data), stride):
		if idx+enc_len+dec_len < len(data):
			data_in.append(data[idx:idx+enc_len])
			data_out.append(data[idx+enc_len:idx+enc_len+dec_len])

	data_in = np.array(data_in)
	data_out = np.array(data_out)
	return data_in, data_out



def make_seq_from_data(data, enc_len, in_bin_sz, out_bin_sz, batch_size,
					   types=None,
					   is_it_bins=True, is_it_var=False,
					   bintotimes=None, bintotypes=None,
					   count_binend=None,
					   stride_len=1, count_strid_len=1, dataset_name=None,
					   bin_size=None):
	inp_seq_lst = list()
	out_seq_lst = list()
	inp_times_in_bin = list()
	out_times_in_bin = list()
	inp_types_in_bin = list()
	out_types_in_bin = list()
	in_end_hr_bins = list()
	out_end_hr_bins = list()
	inp_types_lst = list()
	oup_types_lst = list()
	#count_strid_len = 1
	#rmtpp_strid_len = 1
	#if dataset_name in ['taxi', '911_traffic', '911_ems']:
	#	rmtpp_strid_len = stride_len
	#	count_strid_len = 1
	#if dataset_name in ['taxi', '911_traffic', '911_ems', 'Trump'] \
	#		and bintotimes is not None:
	#	count_strid_len = count_strid_len
	

	iter_range = len(data)-enc_len-out_bin_sz
	#strid_len = rmtpp_strid_len
	strid_len = stride_len
	if is_it_bins:
		strid_len = count_strid_len
		iter_range = len(data)-in_bin_sz-out_bin_sz
	if is_it_var:
		strid_len = stride_len

	for idx in range(0,iter_range,strid_len):
		
		if is_it_bins:
			inp_seq_lst.append(data[idx:idx+in_bin_sz])
			if bintotimes is not None:
				sm_time_in_bin = []
				for r_idx in range(in_bin_sz):
					sm_time_in_bin+=bintotimes[idx+r_idx]
				inp_times_in_bin.append(sm_time_in_bin)
			if bintotypes is not None:
				sm_type_in_bin = []
				for r_idx in range(in_bin_sz):
					sm_type_in_bin+=bintotypes[idx+r_idx]
				inp_types_in_bin.append(sm_type_in_bin)
			if count_binend is not None:
				in_end_hr_bins.append(count_binend[idx:idx+in_bin_sz])
		elif is_it_var:
			inp_times_in_bin.append(data[idx:idx+enc_len])
		else:
			inp_seq_lst.append(data[idx:idx+enc_len])
			if types is not None:
				inp_types_lst.append(types[idx:idx+enc_len])
		
		if is_it_bins:
			out_seq_lst.append(data[idx+in_bin_sz:idx+in_bin_sz+out_bin_sz])
			if bintotimes is not None:
				sm_time_in_bin = []
				for dec_idx in range(out_bin_sz):
					sm_time_in_bin+=bintotimes[idx+in_bin_sz+dec_idx]
				out_times_in_bin.append(sm_time_in_bin)
			if bintotypes is not None:
				sm_type_in_bin = []
				for dec_idx in range(out_bin_sz):
					sm_type_in_bin+=bintotypes[idx+in_bin_sz+dec_idx]
				out_types_in_bin.append(sm_type_in_bin)
			if count_binend is not None:
				out_end_hr_bins.append(count_binend[idx+in_bin_sz:idx+in_bin_sz+out_bin_sz])
		elif is_it_var:
			out_times_in_bin.append(
				get_data_in_next_n_bins(
					out_bin_sz,
					data[idx:idx+enc_len],
					data[idx+enc_len:],
					bin_size
				)
			)
			out_end_hr_bins.append(
				data[idx+enc_len]+np.arange(1, out_bin_sz+1)*bin_size
			)
		else:
			out_seq_lst.append(data[idx+1:idx+enc_len+1])
			if types is not None:
				oup_types_lst.append(types[idx+1:idx+enc_len+1])

	inp_seq_lst = np.array(inp_seq_lst)
	out_seq_lst = np.array(out_seq_lst)
	return (inp_seq_lst, out_seq_lst,
			inp_times_in_bin, out_times_in_bin,
			in_end_hr_bins, out_end_hr_bins,
			inp_types_lst, oup_types_lst,
			inp_types_in_bin, out_types_in_bin,)

def generate_dev_data(
	nc_event_train_in_gaps, nc_event_train_out_gaps,
	train_data_in_timestamps, train_data_out_timestamps,
	event_train_norma, event_train_normd,
	nc_event_train_in_types=None, nc_event_train_out_types=None,
):
	train_per = 0.8
	data_sz = nc_event_train_in_gaps.shape[0]
	nc_event_dev_in_gaps = nc_event_train_in_gaps[int(train_per*data_sz):]
	nc_event_dev_out_gaps = nc_event_train_out_gaps[int(train_per*data_sz):]
	nc_event_train_in_gaps = nc_event_train_in_gaps[:int(train_per*data_sz)]
	nc_event_train_out_gaps = nc_event_train_out_gaps[:int(train_per*data_sz)]

	dev_data_in_timestamps = train_data_in_timestamps[int(train_per*data_sz):]
	dev_data_out_timestamps = train_data_out_timestamps[int(train_per*data_sz):]
	train_data_in_timestamps = train_data_in_timestamps[:int(train_per*data_sz)]
	train_data_out_timestamps = train_data_out_timestamps[:int(train_per*data_sz)]

	if nc_event_train_in_types is not None and nc_event_train_out_types is not None:
		nc_event_dev_in_types = nc_event_train_in_types[int(train_per*data_sz):]
		nc_event_dev_out_types = nc_event_train_out_types[int(train_per*data_sz):]
		nc_event_train_in_types = nc_event_train_in_types[:int(train_per*data_sz)]
		nc_event_train_out_types = nc_event_train_out_types[:int(train_per*data_sz)]
	else:
		nc_event_dev_in_types, nc_event_dev_out_types = None, None

	nc_event_dev_out_gaps = denormalize_data(nc_event_dev_out_gaps,
										 event_train_norma,
										 event_train_normd)
	return [nc_event_train_in_gaps, nc_event_train_out_gaps,
			nc_event_dev_in_gaps, nc_event_dev_out_gaps,
			train_data_in_timestamps, train_data_out_timestamps,
			dev_data_in_timestamps, dev_data_out_timestamps,
			nc_event_train_in_types, nc_event_train_out_types,
			nc_event_dev_in_types, nc_event_dev_out_types]

def stabalize_data(data_in_gaps, data_out_gaps,
				   data_in_timestamps, data_out_timestamps,
				   batch_size,
				   data_in_types=None, data_out_types=None):
	data_sz = data_in_gaps.shape[0]
	data_sz_rem = data_sz % batch_size
	if data_sz_rem != 0:
		data_in_gaps = data_in_gaps[:(-data_sz_rem)]
		data_out_gaps = data_out_gaps[:(-data_sz_rem)]
		data_in_timestamps = data_in_timestamps[:(-data_sz_rem)]
		data_out_timestamps = data_out_timestamps[:(-data_sz_rem)]
		if data_in_types is not None and data_out_types is not None:
			data_in_types = data_in_types[:(-data_sz_rem)]
			data_out_types = data_out_types[:(-data_sz_rem)]
		
	data_in_gaps = np.expand_dims(data_in_gaps, axis=-1)
	data_out_gaps = np.expand_dims(data_out_gaps, axis=-1)
	data_in_timestamps = np.expand_dims(data_in_timestamps, axis=-1)
	data_out_timestamps = np.expand_dims(data_out_timestamps, axis=-1)
	data_in_types = np.array(data_in_types)
	data_out_types = np.array(data_out_types)
	return (data_in_gaps, data_out_gaps,
			data_in_timestamps, data_out_timestamps,
			data_in_types, data_out_types,)

def get_end_time_from_bins(test_inp_times_in_bin, event_test_out_times,
						   count_test_out_binend, enc_len=80,
						   test_inp_types=None, test_out_types=None):
	test_data_in_gaps_bin_lst = list()
	test_data_in_time_end_bin_lst = list()
	test_data_out_gaps_bin_lst = list()
	test_data_in_times_bin_lst = list()
	test_data_out_times_bin_lst = list()
	test_data_in_types_in_bin_lst = list()
	test_data_out_types_in_bin_lst = list()
	for idx in range(len(count_test_out_binend)):
		A1 = np.array(test_inp_times_in_bin[idx][1:])
		A2 = np.array(test_inp_times_in_bin[idx][:-1])
		test_data_in_time_end_bin_lst.append(test_inp_times_in_bin[idx][-1])
		test_data_in_gaps_bin_lst.append((A1-A2)[-enc_len:])
		test_data_in_times_bin_lst.append(A1[-enc_len:])
		if test_inp_types is not None and test_out_types is not None:
			test_data_in_types_in_bin_lst.append(test_inp_types[idx][-enc_len:])
			test_data_out_types_in_bin_lst.append(test_out_types[idx])

		A1 = np.array(event_test_out_times[idx])
		A2 = np.array([test_inp_times_in_bin[idx][-1]] + event_test_out_times[idx][:-1])
		test_data_out_gaps_bin_lst.append(A1-A2)
		test_data_out_times_bin_lst.append(A1)

	event_test_in_lasttime = np.array(test_data_in_time_end_bin_lst)
	event_test_in_gaps = np.array(test_data_in_gaps_bin_lst)
	count_test_out_binend = np.array(count_test_out_binend)
	test_data_out_gaps_bin = np.array(test_data_out_gaps_bin_lst)
	test_data_in_times_bin = np.array(test_data_in_times_bin_lst)
	test_data_out_times_bin = np.array(test_data_out_times_bin_lst)
	test_data_in_types_in_bin = np.array(test_data_in_types_in_bin_lst)
	test_data_out_types_in_bin = np.array(test_data_out_types_in_bin_lst)

	event_test_in_gaps, event_test_norma, event_test_normd \
		= normalize_avg(event_test_in_gaps)
	return [event_test_in_gaps, test_data_out_gaps_bin,
			test_data_in_times_bin, test_data_out_times_bin,
			event_test_in_lasttime, count_test_out_binend,
			event_test_norma, event_test_normd,
			test_data_in_types_in_bin, test_data_out_types_in_bin]

def get_end_time_from_bins_comp(test_inp_times_in_bin,
								count_test_out_binend, enc_len=20, comp_bin_sz=10):
	test_data_in_gaps_bin_lst = list()
	test_data_in_times_bin_lst = list()
	for idx in range(len(count_test_out_binend)):
		cur_test_inp_times_in_bin = test_inp_times_in_bin[idx]
		cur_test_inp_times_in_bin = cur_test_inp_times_in_bin[(len(cur_test_inp_times_in_bin)-1)%comp_bin_sz:][::comp_bin_sz]
		A1 = np.array(cur_test_inp_times_in_bin[1:])
		A2 = np.array(cur_test_inp_times_in_bin[:-1])
		test_data_in_gaps_bin_lst.append((A1-A2)[-enc_len:])
		test_data_in_times_bin_lst.append((A1-A2)[-enc_len:])
	event_test_in_gaps = np.array(test_data_in_gaps_bin_lst)
	test_data_in_times_bin = np.array(test_data_in_times_bin_lst)

	event_test_in_gaps, event_test_norma, event_test_normd = normalize_avg(event_test_in_gaps)
	return [event_test_in_gaps,
			test_data_in_times_bin,
			event_test_norma, event_test_normd]

def get_end_time_from_bins_comp_full(test_inp_times_in_bin,
									 count_test_out_binend, enc_len=20, comp_bin_sz=10):
	test_data_in_gaps_bin_full_lst = list()
	test_data_in_times_bin_full_lst = list()
	for idx in range(len(count_test_out_binend)):
		cur_test_inp_times_in_bin = test_inp_times_in_bin[idx]
		init_pad = (len(cur_test_inp_times_in_bin))%comp_bin_sz
		if init_pad == 0:
			init_pad+=comp_bin_sz

		cur_test_inp_times_in_bin_simp = cur_test_inp_times_in_bin[init_pad-1:]

		lst_g, lst_t = list(), list()
		for seq_idx in range((len(cur_test_inp_times_in_bin_simp)-1)//comp_bin_sz):
			A1 = np.array(cur_test_inp_times_in_bin_simp[(seq_idx*comp_bin_sz)+1:((seq_idx+1)*comp_bin_sz)+1])
			A2 = np.array(cur_test_inp_times_in_bin_simp[(seq_idx*comp_bin_sz):((seq_idx+1)*comp_bin_sz)])
			lst_g.append(A1-A2)
			lst_t.append(A1)
		test_data_in_gaps_bin_full_lst.append(lst_g[-enc_len:])
		test_data_in_times_bin_full_lst.append(lst_t[-enc_len:])

	test_data_in_gaps_bin_full = np.array(test_data_in_gaps_bin_full_lst)
	test_data_in_times_bin_full = np.array(test_data_in_times_bin_full_lst)

	test_data_in_gaps_bin_full, test_gap_in_bin_norm_a_comp_full, test_gap_in_bin_norm_d_comp_full = normalize_avg(test_data_in_gaps_bin_full)
	return [test_data_in_gaps_bin_full,
			test_data_in_times_bin_full_lst,
			test_gap_in_bin_norm_a_comp_full, test_gap_in_bin_norm_d_comp_full]

def get_rand_interval_count(event_test_out_times):
	test_time_out_interval = [np.random.uniform(low=item[0], high=item[-1], size=2) for item in event_test_out_times]
	test_time_out_tb_plus = [min(item) for item in test_time_out_interval]
	test_time_out_te_plus = [max(item) for item in test_time_out_interval]
	times_out_indices_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(event_test_out_times, test_time_out_tb_plus)]
	times_out_indices_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(event_test_out_times, test_time_out_te_plus)]
	test_out_event_count_true = [times_out_indices_te[idx] - times_out_indices_tb[idx] for idx in range(len(test_time_out_tb_plus))]
	test_out_all_event_true = [event_test_out_times[idx][times_out_indices_tb[idx]:times_out_indices_te[idx]] for idx in range(len(test_time_out_tb_plus))]
	test_time_out_tb_plus = np.array(test_time_out_tb_plus)
	test_time_out_te_plus = np.array(test_time_out_te_plus)
	test_out_event_count_true = np.array(test_out_event_count_true)
	return test_time_out_tb_plus, test_time_out_te_plus, test_out_event_count_true, test_out_all_event_true

def get_interval_count_more_than_threshold(times_out, interval_size, threshold):
	threshold = threshold.astype(int)
	interval_range_count_more = np.ones(len(times_out)) * -1
	for batch_idx in range(len(times_out)):
		events_count = threshold[batch_idx]
		for idx in range(events_count, len(times_out[batch_idx]), 1):
			if (times_out[batch_idx][idx]-interval_size <= times_out[batch_idx][idx-events_count]):
				interval_range_count_more[batch_idx] = \
					max(times_out[batch_idx][idx]-interval_size, times_out[batch_idx][0])
				break

	return interval_range_count_more

def get_interval_count_less_than_threshold(times_out, interval_size, threshold):
	threshold = threshold.astype(int)
	interval_range_count_less = np.ones(len(times_out)) * -1
	for batch_idx in range(len(times_out)):
		events_count = threshold[batch_idx]
		for idx in range(len(times_out[batch_idx])-events_count):
			if (times_out[batch_idx][idx]+interval_size <= times_out[batch_idx][idx+events_count]):
				interval_range_count_less[batch_idx] = times_out[batch_idx][idx]
				break

	return interval_range_count_less

def get_interval_count_with_threshold(event_test_out_times, interval_size, dataset_name, threshold=None):
	test_sample_count = len(event_test_out_times)

	if threshold == -1:
		interval_range_count_more = None
		interval_range_count_less = None
		more_thresh = 1
		less_thresh = 1
		for thresh in range(1, len(event_test_out_times)):
			threshold = np.ones(test_sample_count) * thresh
			threshold = threshold.astype(int)
			interval_range_count_more_tmp \
				= get_interval_count_more_than_threshold(event_test_out_times,
														 interval_size,
														 threshold)
			if np.any(interval_range_count_more_tmp == -1):
				break

			interval_range_count_more = interval_range_count_more_tmp
			more_thresh = thresh

		for thresh in range(len(event_test_out_times), 1, -1):
			threshold = np.ones(test_sample_count) * thresh
			threshold = threshold.astype(int)
			interval_range_count_less_tmp \
				= get_interval_count_less_than_threshold(event_test_out_times,
														 interval_size,
														 threshold)
			if np.any(interval_range_count_less_tmp == -1):
				break

			interval_range_count_less = interval_range_count_less_tmp
			less_thresh = thresh

		less_thresh = np.ones(test_sample_count) * less_thresh
		more_thresh = np.ones(test_sample_count) * more_thresh
		return (interval_range_count_less, interval_range_count_more,
			    less_thresh, more_thresh)

	threshold_more = None
	threshold_less = None
	if threshold is not None:
		threshold = np.ones(test_sample_count) * threshold
		threshold_more = threshold
		threshold_less = threshold
	else:
		threshold_more = np.ones(test_sample_count)
		threshold_less = np.ones(test_sample_count)
		more_factor = {
			'sin': 1.1,
			'hawkes': 1.1,
			'sin_hawkes_overlay': 1.05,
			'Trump': 1.05,
			'Verdict': 1.6,
			'Delhi': 1.4,
			'taxi': 1.25,
			'911_traffic': 1.8,
			'911_ems': 1.8,
		}

		less_factor = {
			'sin': 0.85,
			'hawkes': 0.85,
			'sin_hawkes_overlay': 0.85,
			'Trump': 0.85,
			'Verdict': 0.4,
			'Delhi': 0.6,
			'taxi': 0.8,
			'911_traffic': 0.3,
			'911_ems': 0.3,
		}

		for idx in range(test_sample_count):
			bins_count_more = np.ceil((event_test_out_times[idx][-1] - event_test_out_times[idx][0])/interval_size)
			bins_count_less = np.floor((event_test_out_times[idx][-1] - event_test_out_times[idx][0])/interval_size)
			bins_count_less = max(1, bins_count_less)
			avg_events_count_more = (len(event_test_out_times[idx])/bins_count_more)
			avg_events_count_less = (len(event_test_out_times[idx])/bins_count_less)
			avg_events_count_more = round(avg_events_count_more * more_factor[dataset_name])
			avg_events_count_more = max(1, avg_events_count_more)
			avg_events_count_less = round(avg_events_count_less * less_factor[dataset_name])
			avg_events_count_less = max(1, avg_events_count_less)
			threshold_more[idx] = avg_events_count_more
			threshold_less[idx] = avg_events_count_less
		threshold_more = np.array(threshold_more)
		threshold_less = np.array(threshold_less)

	interval_range_count_more \
		= get_interval_count_more_than_threshold(event_test_out_times,
												 interval_size,
												 threshold_more)

	interval_range_count_less \
		= get_interval_count_less_than_threshold(event_test_out_times,
												 interval_size,
												 threshold_less)

	begin = np.array([x[0] for x in event_test_out_times])
	print('event_test_out_times', np.array([x[0] for x in event_test_out_times]) )
	print('interval_range_count_more', interval_range_count_more - begin)
	print('interval_range_count_less', interval_range_count_less - begin)
	# print('more', sum((interval_range_count_more - begin)==0.0))
	# print('less', sum((interval_range_count_less - begin)==0.0))
	# print('all', sum((interval_range_count_less - begin)>=0.0))
	# assert np.all(interval_range_count_more>=0.0), 'No range found in t_b range'
	# assert np.all(interval_range_count_less>=0.0), 'No range found in t_b range'
	return (interval_range_count_less, interval_range_count_more,
			threshold_less, threshold_more)

def get_time_features(times):
#    time_feature_hour = (times // 3600) % 24
#    time_feature_minute = ((times-(times//3600)*3600) // 60) % 60
#    time_feature_seconds = (times-(times//60)*60) % 60
#
#    time_feature = (time_feature_hour * 3600.0 
#                 + time_feature_minute * 60.0
#                 + time_feature_seconds)
#    time_feature = time_feature / 3600.0

    time_feature = (times/3600.)%24
    # time_feature = time_feature*1./24.
    return time_feature

def reset_indices(types):
	types_new = []
	num_types = len(np.unique(types))
	unique_types = sorted(np.unique(types))
	type2id = dict()
	cur_id = 1
	for t in unique_types:
		if type2id.get(t, -1) == -1:
			type2id[t] = cur_id
			cur_id += 1
	for t in types:
		types_new.append(type2id[t])
	types_new = np.array(types_new)
	return types_new, type2id

def set_comp_bin_sz(bin_counts):
	return int(np.round(np.mean(bin_counts))/2)

def get_num_bins(timestamps, bin_size):
	num_bins = (timestamps[-1] - timestamps[0]) // bin_size
	return num_bins


def get_processed_data(dataset_name, args):

	bin_size = args.bin_size
	in_bin_sz = args.in_bin_sz
	out_bin_sz = args.out_bin_sz
	enc_len = args.enc_len
	batch_size = args.batch_size
	comp_enc_len = args.comp_enc_len
	comp_bin_sz = args.comp_bin_sz

	if dataset_name in ['Trump']:
		comp_enc_len = 25

	timestamps = np.loadtxt('data/'+dataset_name+'.txt')
	if os.path.isfile('data/'+dataset_name+'_types.txt'):
		types = np.loadtxt('data/'+dataset_name+'_types.txt')
	else:
		types = np.ones_like(timestamps)
	gaps = timestamps[1:] - timestamps[:-1]
	gaps = gaps.astype(np.float32)
	args.num_types = len(np.unique(types))
	types, _ = reset_indices(types) # Make sure type-indieces are in the range [Y]
	num_bins = get_num_bins(timestamps, bin_size)
	count_counts, count_binend, bintotimes, bintogaps, bintotypes = create_bin(timestamps, types, bin_size, num_bins)

	args.comp_bin_sz = set_comp_bin_sz(count_counts)
	comp_bin_sz = args.comp_bin_sz

	timestamps_comp_full = list()
	gaps_comp_full = list()
	for idx in range(0,(len(timestamps)-comp_bin_sz-1),comp_bin_sz):
		timestamps_comp_full.append(timestamps[idx+1:idx+comp_bin_sz+1])
		gaps_comp_full.append(timestamps[idx+1:idx+comp_bin_sz+1] - timestamps[idx:idx+comp_bin_sz])
	timestamps_comp_full = np.array(timestamps_comp_full)
	gaps_comp_full = np.array(gaps_comp_full)

	timestamps_comp = timestamps[::comp_bin_sz]
	gaps_comp = timestamps_comp[1:] - timestamps_comp[:-1]
	gaps_comp = gaps_comp.astype(np.float32)
	types_comp = np.ones_like(types)
	_, _, bintotimes_comp, bintogaps_comp, bintotypes_comp = create_bin(timestamps_comp, types_comp, bin_size, num_bins)


	#TODO Resolve these plots
#	os.makedirs('data/plots', exist_ok=True)
#	plt.plot(range(len(count_counts[:500])), count_counts[:500])
#	plt.ylabel('bin_counts')
#	plt.xlabel(dataset_name+' Dataset')
#	plt.savefig('data/plots/bin_count_'+dataset_name+'.png')
#	plt.close()
#
#	plt.plot(range(len(gaps[:500])), gaps[:500])
#	plt.ylabel('gaps')
#	plt.xlabel(dataset_name+' Dataset')
#	plt.savefig('data/plots/gaps_'+dataset_name+'.png')
#	plt.close()
#
#	plt.plot(range(len(gaps_comp[:500])), gaps_comp[:500])
#	plt.ylabel('gaps_comp')
#	plt.xlabel(dataset_name+' Dataset')
#	plt.savefig('data/plots/gaps_comp_'+dataset_name+'.png')
#	plt.close()


	(
		count_train_counts, count_dev_counts, count_test_counts,
		count_train_binend, count_dev_binend, count_test_binend,
		bintogaps_train, bintogaps_dev, bintogaps_test,
		bintotimes_train, bintotimes_dev, bintotimes_test,
		bintotypes_train, bintotypes_dev, bintotypes_test
	) = generate_train_dev_test_data(
	 	count_counts, count_binend, bintotimes, bintogaps, args.in_bin_sz,
	 	bintotypes=bintotypes
	)
	print('Data Statistics:')
	print('Total Number of events:', len(flatten(bintotimes)))
	print('Number of events in training set:', len(flatten(bintotimes_train)))
	print('Number of events in dev set:', len(flatten(bintotimes_dev)))
	print('Number of events in test set:', len(flatten(bintotimes_test)))
	print('Average gap:', np.mean(flatten(bintogaps_train)))
	print('Variance of gaps:', np.std(flatten(bintogaps_train)))
	print('Number of Types:', len(np.unique(flatten(bintotypes))))
	print('Entropy of types distributions:', entropy([i for i in Counter(flatten(bintotypes)).values()]))
	#import ipdb
	#ipdb.set_trace()


	(
		_, _, _,
		_, _, _,
		bintogaps_train_comp, bintogaps_dev_comp, bintogaps_test_comp,
		bintotimes_train_comp, bintotimes_dev_comp, bintotimes_test_comp,
		bintotypes_train_comp, bintotypes_dev_comp, bintotypes_test_comp
	) = generate_train_dev_test_data(
		count_counts, count_binend, bintotimes_comp, bintogaps_comp, args.in_bin_sz,
		bintotypes=bintotypes_comp
	)


#	hawkes_timestamps_pred = event_test_times
#	train_times_gaps_norm, gaps_norm_a, gaps_norm_d \
#		= normalize_avg(event_train_gaps)
#	train_times_timestamps_norm \
#		= (np.cumsum(train_times_gaps_norm)
#		   + (event_train_times[0]-event_train_gaps[0]))
#	test_times_gaps_norm = normalize_avg_given_param(event_test_gaps,
#													 gaps_norm_a,
#													 gaps_norm_d)
#	test_times_timestamps_norm \
#		= (np.cumsum(test_times_gaps_norm)
#		   + (train_times_timestamps_norm[-1]))
#
#	#TODO: Put hawkes process at appropriate place
#	print("")
#	print("Training for Hawkes Model")
#	model = hk.estimator().set_kernel('exp').set_baseline('const')
#	start_idx = min(30000, len(train_times_timestamps_norm))
#	itv = [train_times_timestamps_norm[-start_idx], train_times_timestamps_norm[-1]]
#	train_times_timestamps_norm = train_times_timestamps_norm.astype(np.float64)
#	model.fit(train_times_timestamps_norm[-start_idx:],itv)
#	print("Parameters generated by Hawkes Model")
#	print("Parameter:",model.parameter)
#	print("AIC:",model.AIC)
#	print("")
#	hawkes_timestamps_pred = model.predict(test_times_timestamps_norm[-1]+100,1)[0]
#	test_times_gaps_norm \
#		= (hawkes_timestamps_pred
#		   - np.concatenate([train_times_timestamps_norm[-1:],
#		   					 hawkes_timestamps_pred[:-1]]))
#	event_test_gaps = denormalize_avg(test_times_gaps_norm, gaps_norm_a, gaps_norm_d)
#	hawkes_timestamps_pred = np.cumsum(event_test_gaps) + event_train_times[-1]
#	# Plots for Hawkes Predictions
#	os.makedirs('Outputs/hawkes_model_'+dataset_name, exist_ok=True)
#	model.plot_N_pred()
#	plt.savefig('Outputs/hawkes_model_'+dataset_name+'/count_pred.png')
#	plt.close()
#	model.plot_KS()
#	plt.savefig('Outputs/hawkes_model_'+dataset_name+'/KS_plot.png')
#	plt.close()



	count_train_in_counts, count_train_out_counts = create_forecast_io_seqs(
		count_train_counts, args.in_bin_sz, args.out_bin_sz, 1,
	)
	count_train_in_binend, count_train_out_binend = create_forecast_io_seqs(
		count_train_binend, args.in_bin_sz, args.out_bin_sz, 1,
	)

	count_dev_in_counts, count_dev_out_counts = create_forecast_io_seqs(
		count_dev_counts, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)
	count_dev_in_binend, count_dev_out_binend = create_forecast_io_seqs(
		count_dev_binend, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)

	count_test_in_counts, count_test_out_counts = create_forecast_io_seqs(
		count_test_counts, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)
	count_test_in_binend, count_test_out_binend = create_forecast_io_seqs(
		count_test_binend, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)

	count_train_in_counts, count_train_normm, count_train_norms \
		= normalize_data(count_train_in_counts)
	count_train_out_counts = normalize_data_given_param(count_train_out_counts,
													count_train_normm,
													count_train_norms)
	count_dev_in_counts = normalize_data_given_param(count_dev_in_counts,
													count_train_normm,
													count_train_norms)
	count_test_in_counts = normalize_data_given_param(count_test_in_counts,
													count_train_normm,
													count_train_norms)
	#count_test_in_counts, count_test_normm, count_test_norms = normalize_data(count_test_in_counts)
	count_test_normm, count_test_norms = count_train_normm, count_train_norms

	count_train_in_feats = get_time_features(count_train_in_binend-bin_size/2.).astype(np.float32)
	count_dev_in_feats = get_time_features(count_dev_in_binend-bin_size/2.).astype(np.float32)
	count_test_in_feats = get_time_features(count_test_in_binend-bin_size/2.).astype(np.float32)
	count_train_out_feats = get_time_features(count_train_out_binend-bin_size/2.).astype(np.float32)
	count_test_out_feats = get_time_features(count_test_out_binend-bin_size/2.).astype(np.float32)


	nc_event_train_in_gaps, nc_event_train_out_gaps = create_nowcast_io_seqs(
		flatten(bintogaps_train), enc_len, args.stride_len,
	)
	nc_event_train_in_types, nc_event_train_out_types = create_nowcast_io_seqs(
		flatten(bintotypes_train), enc_len, args.stride_len,
	)
	nc_event_train_in_times, nc_event_train_out_times = create_nowcast_io_seqs(
		flatten(bintotimes_train), enc_len, args.stride_len,
	)

	nc_event_dev_in_gaps, nc_event_dev_out_gaps = create_nowcast_io_seqs(
		flatten(bintogaps_dev), enc_len, args.stride_len,
	)
	nc_event_dev_in_types, nc_event_dev_out_types = create_nowcast_io_seqs(
		flatten(bintotypes_dev), enc_len, args.stride_len,
	)
	nc_event_dev_in_times, nc_event_dev_out_times = create_nowcast_io_seqs(
		flatten(bintotimes_dev), enc_len, args.stride_len,
	)

	event_test_in_gaps, event_test_out_gaps = create_forecast_io_seqs(
		bintogaps_test, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)
	event_test_in_types, event_test_out_types = create_forecast_io_seqs(
		bintotypes_test, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)
	event_test_in_times, event_test_out_times = create_forecast_io_seqs(
		bintotimes_test, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)

	#event_test_in_gaps = np.array(pad_sequences([flatten(seq) for seq in event_test_in_gaps], padding='post'))
	#event_test_in_types = np.array(pad_sequences([flatten(seq) for seq in event_test_in_types], padding='post'))
	#event_test_in_times = np.array(pad_sequences([flatten(seq) for seq in event_test_in_times], padding='post'))
	event_test_in_gaps = np.array([flatten(seq)[-enc_len:] for seq in event_test_in_gaps])
	event_test_in_types = np.array([flatten(seq)[-enc_len:] for seq in event_test_in_types])
	event_test_in_times = np.array([flatten(seq)[-enc_len:] for seq in event_test_in_times])
	event_test_out_gaps = np.array([flatten(seq) for seq in event_test_out_gaps])
	event_test_out_types = np.array([flatten(seq) for seq in event_test_out_types])
	event_test_out_times = np.array([flatten(seq) for seq in event_test_out_times])

	event_test_out_mask = tf.sequence_mask([len(s) for s in event_test_out_times], dtype=tf.float32)
	event_test_out_seqlens = [len(s) for s in event_test_out_times]
	#event_test_out_gaps = np.expand_dims(pad_sequences(event_test_out_gaps, padding='post'), axis=-1).astype(np.float32)
	#event_test_out_types = pad_sequences(event_test_out_types, padding='post').astype(np.int64)
	#event_test_out_times = pad_sequences(event_test_out_times, padding='post').astype(np.float32)

	event_test_in_lasttime = np.array([seq[-1] for seq in event_test_in_times])

	nc_event_train_in_gaps, event_train_norma, event_train_normd \
		= normalize_avg(nc_event_train_in_gaps)
	nc_event_train_out_gaps = normalize_avg_given_param(
		nc_event_train_out_gaps, event_train_norma, event_train_normd
	)
	nc_event_dev_in_gaps = normalize_avg_given_param(nc_event_dev_in_gaps, event_train_norma, event_train_normd)
	#event_test_in_gaps, event_test_norma, event_test_normd = normalize_avg(event_test_in_gaps)
	event_test_in_gaps = normalize_avg_given_param(event_test_in_gaps, event_train_norma, event_train_normd)
	event_test_norma, event_test_normd = event_train_norma, event_train_normd

	nc_event_train_in_feats = get_time_features(nc_event_train_in_times)
	nc_event_train_out_feats = get_time_features(nc_event_train_out_times)
	nc_event_dev_in_feats = get_time_features(nc_event_dev_in_times)
	event_test_in_feats = get_time_features(event_test_in_times)
	#event_test_out_feats = np.expand_dims(get_time_features(pad_sequences(event_test_out_times, padding='post')), axis=-1).astype(np.float32)

	nc_comp_train_in_gaps, nc_comp_train_out_gaps = create_nowcast_io_seqs(
		flatten(bintogaps_train_comp), enc_len, args.stride_len,
	)
	nc_comp_train_in_types, nc_comp_train_out_types = create_nowcast_io_seqs(
		flatten(bintotypes_train_comp), enc_len, args.stride_len,
	)
	nc_comp_train_in_times, nc_comp_train_out_times = create_nowcast_io_seqs(
		flatten(bintotimes_train_comp), enc_len, args.stride_len,
	)

	nc_comp_dev_in_gaps, nc_comp_dev_out_gaps = create_nowcast_io_seqs(
		flatten(bintogaps_dev_comp), enc_len, args.stride_len,
	)
	nc_comp_dev_in_types, nc_comp_dev_out_types = create_nowcast_io_seqs(
		flatten(bintotypes_dev_comp), enc_len, args.stride_len,
	)
	nc_comp_dev_in_times, nc_comp_dev_out_times = create_nowcast_io_seqs(
		flatten(bintotimes_dev_comp), enc_len, args.stride_len,
	)

	comp_test_in_gaps, comp_test_out_gaps = create_forecast_io_seqs(
		bintogaps_test_comp, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)
	comp_test_in_types, comp_test_out_types = create_forecast_io_seqs(
		bintotypes_test_comp, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)
	comp_test_in_times, comp_test_out_times = create_forecast_io_seqs(
		bintotimes_test_comp, args.in_bin_sz, args.out_bin_sz, args.out_bin_sz,
	)
	#comp_test_in_gaps = np.array(pad_sequences([flatten(seq) for seq in comp_test_in_gaps], padding='post'))
	#comp_test_in_types = np.array(pad_sequences([flatten(seq) for seq in comp_test_in_types], padding='post'))
	#comp_test_in_times = np.array(pad_sequences([flatten(seq) for seq in comp_test_in_times], padding='post'))
	mx_enc_len = min([len(flatten(x)) for x in comp_test_in_gaps])
	#mx_enc_len = min(comp_enc_len, ((mx_enc_len-1)//comp_bin_sz))
	comp_test_in_gaps = np.array([flatten(seq)[-mx_enc_len:] for seq in comp_test_in_gaps])
	comp_test_in_types = np.array([flatten(seq)[-mx_enc_len:] for seq in comp_test_in_types])
	comp_test_in_times = np.array([flatten(seq)[-mx_enc_len:] for seq in comp_test_in_times])
	comp_test_out_gaps = np.array([flatten(seq) for seq in comp_test_out_gaps])
	comp_test_out_types = np.array([flatten(seq) for seq in comp_test_out_types])
	comp_test_out_times = np.array([flatten(seq) for seq in comp_test_out_times])


	nc_comp_train_in_gaps, comp_train_norma, comp_train_normd \
		= normalize_avg(nc_comp_train_in_gaps)
	nc_comp_train_out_gaps = normalize_avg_given_param(
		nc_comp_train_out_gaps, comp_train_norma, comp_train_normd
	)
	#comp_test_in_gaps, comp_test_norma, comp_test_normd = normalize_avg(comp_test_in_gaps)
	comp_test_in_gaps = normalize_avg_given_param(comp_test_in_gaps, comp_train_norma, comp_train_normd)
	comp_test_norma, comp_test_normd = comp_train_norma, comp_train_normd

	nc_comp_train_in_feats = get_time_features(nc_comp_train_in_times)
	nc_comp_train_out_feats = get_time_features(nc_comp_train_out_times)
	nc_comp_dev_in_feats = get_time_features(nc_comp_dev_in_times)
	comp_test_in_feats = get_time_features(comp_test_in_times)

	print('')
	print('Dataset size:')
	print('')
	print('(Bin wise)')
	print('Train in', count_train_in_counts.shape)
	print('Train out', count_train_out_counts.shape)
	print('Test in', count_test_in_counts.shape)
	print('Test out', count_test_out_counts.shape)
	print('')
	print('(Gaps wise)')
	print('Train in', nc_event_train_in_gaps.shape)
	print('Train out', nc_event_train_out_gaps.shape)
	print('Dev in', nc_event_dev_in_gaps.shape)
	print('Dev out', nc_event_dev_out_gaps.shape)
	# print('Test in', test_data_in_gaps.shape)
	# print('Test out', test_data_out_gaps.shape)


	print('Test in', event_test_in_gaps.shape)
	print('')
	print('(Compound gaps wise)')
	print('Comp Train in', nc_comp_train_in_gaps.shape)
	print('Comp Train out', nc_comp_train_out_gaps.shape)
	print('Comp Dev in', nc_comp_dev_in_gaps.shape)
	print('Comp Dev out', nc_comp_dev_out_gaps.shape)
	print('')
	print('')

	[test_time_out_tb_plus, test_time_out_te_plus, 
	 test_out_event_count_true, test_out_all_event_true] \
	 	= get_rand_interval_count(event_test_out_times)

	interval_size = args.interval_size
	if interval_size==0:
		interval_size = args.bin_size
	[interval_range_count_less, interval_range_count_more,
	 less_threshold, more_threshold] \
	 	= get_interval_count_with_threshold(event_test_out_times,
	 										interval_size,
	 										dataset_name)


	#count_train_in_counts = np.expand_dims(count_train_in_counts, axis=-1).astype(np.float32)
	#count_dev_in_counts = np.expand_dims(count_dev_in_counts, axis=-1).astype(np.float32)
	#count_test_in_counts = np.expand_dims(count_test_in_counts, axis=-1).astype(np.float32)
	#count_train_in_binend = np.expand_dims(count_train_in_binend, axis=-1).astype(np.float32)
	#count_dev_in_binend = np.expand_dims(count_dev_in_binend, axis=-1).astype(np.float32)
	#count_test_in_binend = np.expand_dims(count_test_in_binend, axis=-1).astype(np.float32)
	#count_train_out_binend = np.expand_dims(count_train_out_binend, axis=-1).astype(np.float32)
	#count_dev_out_binend = np.expand_dims(count_dev_out_binend, axis=-1).astype(np.float32)


	print(count_train_in_counts.shape)
	print(count_test_in_counts.shape)
	print(nc_event_train_in_gaps.shape)
	print(event_test_in_gaps.shape)


	# ----- Define dummy types for _comp datasets ----- #
	#nc_comp_dev_in_types = np.array([np.ones_like(seq) for seq in nc_comp_dev_in_gaps])
	#nc_comp_dev_out_types = np.array([np.ones_like(seq) for seq in nc_comp_dev_out_gaps])
	#nc_comp_train_in_types = np.squeeze(np.array([np.ones_like(seq) for seq in nc_comp_train_in_gaps]), axis=-1)
	#nc_comp_train_out_types = np.squeeze(np.array([np.ones_like(seq) for seq in nc_comp_train_out_gaps]), axis=-1)


#	# ----- Start: Data Augmentation to counter skewness in the data ----- #
#	if dataset_name in ['taxi']:
#		event_train_in_times = np.cumsum(nc_event_train_in_gaps[:,:,0], axis=1)
#		span = event_train_in_times[:,-1]-event_train_in_times[:,0]
#		ge_100 = np.sum(span>100.)
#		indices_l = np.random.choice(np.where(span<100)[0], size=ge_100)
#		indices_g = np.where(span>100.)[0]
#		indices = np.array(sorted(np.concatenate([indices_l, indices_g])))
#	
#		nc_event_train_in_gaps = nc_event_train_in_gaps[indices]
#		nc_event_train_out_gaps = nc_event_train_out_gaps[indices]
#		nc_event_train_in_feats = nc_event_train_in_feats[indices]
#		nc_event_train_out_feats = nc_event_train_out_feats[indices]
#		nc_event_train_in_types = nc_event_train_in_types[indices]
#		nc_event_train_out_types = nc_event_train_out_types[indices]
#	# ----- End: Data Augmentation to counter skewness in the data ----- #

	nc_event_train_in_gaps = np.expand_dims(nc_event_train_in_gaps, axis=-1).astype(np.float32)
	nc_event_train_in_feats = np.expand_dims(nc_event_train_in_feats, axis=-1).astype(np.float32)
	nc_event_train_out_gaps = np.expand_dims(nc_event_train_out_gaps, axis=-1).astype(np.float32)
	nc_event_train_out_feats = np.expand_dims(nc_event_train_out_feats, axis=-1).astype(np.float32)

	nc_event_dev_in_gaps = np.expand_dims(nc_event_dev_in_gaps, axis=-1).astype(np.float32)
	nc_event_dev_in_feats = np.expand_dims(nc_event_dev_in_feats, axis=-1).astype(np.float32)
	nc_event_dev_out_gaps = np.expand_dims(nc_event_dev_out_gaps, axis=-1).astype(np.float32)


	event_test_in_gaps = np.expand_dims(event_test_in_gaps, axis=-1).astype(np.float32)
	event_test_in_feats = np.expand_dims(event_test_in_feats, axis=-1).astype(np.float32)
	count_test_out_binend = np.expand_dims(count_test_out_binend, axis=-1).astype(np.float32)
	event_test_in_lasttime = np.expand_dims(event_test_in_lasttime, axis=-1).astype(np.float32)
	#event_test_out_times = np.expand_dims(event_test_out_times, axis=-1).astype(np.float32)

	nc_comp_train_in_gaps = np.expand_dims(nc_comp_train_in_gaps, axis=-1).astype(np.float32)
	nc_comp_train_in_feats = np.expand_dims(nc_comp_train_in_feats, axis=-1).astype(np.float32)
	nc_comp_train_out_gaps = np.expand_dims(nc_comp_train_out_gaps, axis=-1).astype(np.float32)
	nc_comp_train_out_feats = np.expand_dims(nc_comp_train_out_feats, axis=-1).astype(np.float32)


	nc_comp_dev_in_gaps = np.expand_dims(nc_comp_dev_in_gaps, axis=-1).astype(np.float32)
	nc_comp_dev_in_feats = np.expand_dims(nc_comp_dev_in_feats, axis=-1).astype(np.float32)
	nc_comp_dev_out_gaps = np.expand_dims(nc_comp_dev_out_gaps, axis=-1).astype(np.float32)

	comp_test_in_gaps = np.expand_dims(comp_test_in_gaps, axis=-1).astype(np.float32)
	comp_test_in_feats = np.expand_dims(comp_test_in_feats, axis=-1).astype(np.float32)
	comp_test_out_times = np.expand_dims(comp_test_out_times, axis=-1)

	#import ipdb
	#ipdb.set_trace()

	dataset = {
		# Now-casting event training data
		'nc_event_train_in_gaps': nc_event_train_in_gaps,
		'nc_event_train_in_types': nc_event_train_in_types,
		'nc_event_train_in_feats': nc_event_train_in_feats,
		'nc_event_train_out_gaps': nc_event_train_out_gaps,
		'nc_event_train_out_types': nc_event_train_out_types,
		'nc_event_train_out_feats': nc_event_train_out_feats,

		'event_train_norma': event_train_norma,
		'event_train_normd': event_train_normd,

		# Now-casting event dev data
		'nc_event_dev_in_gaps': nc_event_dev_in_gaps,
		'nc_event_dev_in_types': nc_event_dev_in_types,
		'nc_event_dev_in_feats': nc_event_dev_in_feats,
		'nc_event_dev_out_gaps': nc_event_dev_out_gaps,
		'nc_event_dev_out_types': nc_event_dev_out_types,

		# Count train data
		'count_train_in_counts': count_train_in_counts,
		'count_train_in_feats': count_train_in_feats,
		'count_train_out_counts': count_train_out_counts,

		# Count dev data
		'count_dev_in_counts': count_dev_in_counts,
		'count_dev_in_feats': count_dev_in_feats,
		'count_dev_out_counts': count_dev_out_counts,

		# Count test data
		'count_test_in_counts': count_test_in_counts,
		'count_test_in_feats': count_test_in_feats,
		'count_test_out_counts': count_test_out_counts,

		'count_test_normm': count_test_normm,
		'count_test_norms': count_test_norms,

		# Forecasting event test data
		'event_test_in_gaps': event_test_in_gaps,
		'event_test_in_types': event_test_in_types,
		'event_test_in_feats': event_test_in_feats,
		'count_test_out_binend': count_test_out_binend,
		'event_test_in_lasttime': event_test_in_lasttime,
		'event_test_out_times': event_test_out_times,
		'event_test_out_gaps': event_test_out_gaps,
		#'event_test_out_feats': event_test_out_feats,
		'event_test_out_types': event_test_out_types,

		'event_test_norma': event_test_norma,
		'event_test_normd': event_test_normd,

	 	# Now-casting comp training data
		'nc_comp_train_in_gaps': nc_comp_train_in_gaps,
		'nc_comp_train_in_types': nc_comp_train_in_types,
		'nc_comp_train_in_feats': nc_comp_train_in_feats,
		'nc_comp_train_out_gaps': nc_comp_train_out_gaps,
		'nc_comp_train_out_types': nc_comp_train_out_types,
		'nc_comp_train_out_feats': nc_comp_train_out_feats,

		'comp_train_norma': comp_train_norma,
		'comp_train_normd': comp_train_normd,

		# Now-casting comp dev data
		'nc_comp_dev_in_gaps': nc_comp_dev_in_gaps,
		'nc_comp_dev_in_types': nc_comp_dev_in_types,
		'nc_comp_dev_in_feats': nc_comp_dev_in_feats,
		'nc_comp_dev_out_gaps': nc_comp_dev_out_gaps,
		'nc_comp_dev_out_types': nc_comp_dev_out_types,

		# Forecasting comp test data
		'comp_test_in_gaps': comp_test_in_gaps,
		'comp_test_in_types': comp_test_in_types,
		'comp_test_in_feats': comp_test_in_feats,
		'comp_test_out_times': comp_test_out_times,
		'comp_test_out_types': comp_test_out_types,

		'comp_test_norma': comp_test_norma,
		'comp_test_normd': comp_test_normd,

		# Data required for Queries
		'test_time_out_tb_plus': test_time_out_tb_plus,
		'test_time_out_te_plus': test_time_out_te_plus,
		'test_out_event_count_true': test_out_event_count_true,
		'test_out_all_event_true': test_out_all_event_true,

		'interval_range_count_less': interval_range_count_less,
		'interval_range_count_more': interval_range_count_more,
		'less_threshold': less_threshold,
		'more_threshold': more_threshold,
		'interval_size': interval_size,
	}

	return dataset
