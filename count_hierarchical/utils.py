import numpy as np
import os, sys
import abc
import matplotlib.pyplot as plt
from bisect import bisect_right
from modules import Hawkes as hk

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

def generate_plots(args, dataset_name, dataset, per_model_count, test_sample_idx=1, count_var=None):
	inp_seq_len_plot = 10
	dec_len = args.out_bin_sz

	true_pred = per_model_count['true']
	rmtpp_mse_pred = true_pred
	rmtpp_nll_pred = true_pred
	hierarchical_pred = true_pred
	count_model_pred = true_pred
	wgan_pred = true_pred
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
	if 'hawkes_model' in per_model_count:
		hawkes_pred = per_model_count['hawkes_model']
		event_count_preds_hawkes = hawkes_pred
		hawkes_pred = event_count_preds_hawkes[test_sample_idx].astype(np.float32)

	event_count_preds_true = true_pred
	true_pred = event_count_preds_true[test_sample_idx].astype(np.float32)

	test_data_in_bin = np.squeeze(dataset['test_data_in_bin'], axis=-1)
	test_mean_bin = dataset['test_mean_bin']
	test_std_bin = dataset['test_std_bin']
	
	true_inp_bins = denormalize_data(test_data_in_bin[test_sample_idx,inp_seq_len_plot:], 
							test_mean_bin, test_std_bin)
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
	plt.savefig('Outputs/'+dataset_name+'_'+str(test_sample_idx)+'.svg', format='svg', dpi=1200)
	plt.close()

def create_bin(times, bin_size):
	"""

	Args:
		times (list): A sequence of raw timestamps
		bin_size (int): Length of the bin

	Returns:
		cnt_bin (list): Number of events in each bin
		end_hr_bin (list): End time of each bin
		times_in_bin (list of lists): list of lists of timestamps in each bin
	"""
	end_hr = 0
	cnt_bin = []
	end_hr_bin = []
	times_in_bin = []
	last_ind= [0]
	ind=0
	while ind<len(times):
		end_hr = end_hr + bin_size
		times_saver = []
		while ind<len(times) and times[ind]<=end_hr:
			times_saver.append(times[ind])
			ind+=1
		if ind<len(times):
			cnt_bin.append(ind-last_ind[-1])
			last_ind.append(ind)
			end_hr_bin.append(end_hr)
			times_in_bin.append(times_saver)

	print('Total bins generated', len(cnt_bin))
	print('Each bin has Average', int(round(np.mean(cnt_bin))), 'timestamps')
	return cnt_bin, end_hr_bin, times_in_bin

def generate_train_test_data(timestamps, gaps, data_bins, end_hr_bins, times_in_bin):
	train_per = 0.8
	data_sz = len(data_bins)
	train_times_bin = data_bins[:int(train_per*data_sz)]
	test_times_bin = data_bins[int(train_per*data_sz):]
	train_times_bin_end = end_hr_bins[:int(train_per*data_sz)]
	test_times_bin_end = end_hr_bins[int(train_per*data_sz):]
	train_seq_times_in_bin = times_in_bin[:int(train_per*data_sz)]
	test_seq_times_in_bin = times_in_bin[int(train_per*data_sz):]

	data_sz = len(gaps)
	train_times_gaps = gaps[:int(train_per*data_sz)]
	test_times_gaps = gaps[int(train_per*data_sz):]
	train_times_timestamps = timestamps[1:int(train_per*data_sz)+1]
	test_times_timestamps = timestamps[int(train_per*data_sz)+1:]

	print("Length of train and test bins", \
			len(train_times_bin), len(test_times_bin))
	print("Length of train and test gaps", \
			len(train_times_gaps), len(test_times_gaps))
	print("Length of train and test timestamps", \
			len(train_times_timestamps), len(test_times_timestamps))

	return  (train_times_bin, test_times_bin,
			 train_times_bin_end, test_times_bin_end,
			 train_seq_times_in_bin, test_seq_times_in_bin,
			 train_times_gaps, test_times_gaps,
			 train_times_timestamps, test_times_timestamps)

def get_data_in_next_n_bins(n, inp, out, bin_size):
	out_n_bins = []
	for i in range(len(out)):
		if (out[i]-inp[-1]) > (bin_size*n):
			break
		out_n_bins.append(out[i])
	return out_n_bins

def make_seq_from_data(data, enc_len, in_bin_sz, out_bin_sz, batch_size,
					   is_it_bins=True, is_it_var=False,
					   times_in_bin=None, end_hr_bins=None,
					   stride_len=1, count_strid_len=1, dataset_name=None,
					   bin_size=None):
	inp_seq_lst = list()
	out_seq_lst = list()
	inp_times_in_bin = list()
	out_times_in_bin = list()
	in_end_hr_bins = list()
	out_end_hr_bins = list()
	count_strid_len = 1
	rmtpp_strid_len = 1
	if dataset_name in ['taxi', '911_traffic', '911_ems']:
		rmtpp_strid_len = stride_len
		count_strid_len = 1
	if dataset_name in ['taxi', '911_traffic', '911_ems', 'Trump'] \
			and times_in_bin is not None:
		count_strid_len = count_strid_len
	

	iter_range = len(data)-enc_len-out_bin_sz
	strid_len = rmtpp_strid_len
	if is_it_bins:
		strid_len = count_strid_len
		iter_range = len(data)-in_bin_sz-out_bin_sz
	if is_it_var:
		strid_len = stride_len

	for idx in range(0,iter_range,strid_len):
		
		if is_it_bins:
			inp_seq_lst.append(data[idx:idx+in_bin_sz])
			if times_in_bin is not None:
				sm_time_in_bin = []
				for r_idx in range(in_bin_sz):
					sm_time_in_bin+=times_in_bin[idx+r_idx]
				inp_times_in_bin.append(sm_time_in_bin)
			if end_hr_bins is not None:
				in_end_hr_bins.append(end_hr_bins[idx:idx+in_bin_sz])
		elif is_it_var:
			inp_times_in_bin.append(data[idx:idx+enc_len])
		else:
			inp_seq_lst.append(data[idx:idx+enc_len])
		
		if is_it_bins:
			out_seq_lst.append(data[idx+in_bin_sz:idx+in_bin_sz+out_bin_sz])
			if times_in_bin is not None:
				sm_time_in_bin = []
				for dec_idx in range(out_bin_sz):
					sm_time_in_bin+=times_in_bin[idx+in_bin_sz+dec_idx]
				out_times_in_bin.append(sm_time_in_bin)
			if end_hr_bins is not None:
				out_end_hr_bins.append(end_hr_bins[idx+in_bin_sz:idx+in_bin_sz+out_bin_sz])
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

	inp_seq_lst = np.array(inp_seq_lst)
	out_seq_lst = np.array(out_seq_lst)
	return inp_seq_lst, out_seq_lst, inp_times_in_bin, out_times_in_bin, in_end_hr_bins, out_end_hr_bins

def generate_dev_data(data):
	[train_data_in_gaps, train_data_out_gaps, \
	train_data_in_timestamps, train_data_out_timestamps, \
	train_norm_a_gaps, train_norm_d_gaps] = data
	train_per = 0.8
	data_sz = train_data_in_gaps.shape[0]
	dev_data_in_gaps = train_data_in_gaps[int(train_per*data_sz):]
	dev_data_out_gaps = train_data_out_gaps[int(train_per*data_sz):]
	train_data_in_gaps = train_data_in_gaps[:int(train_per*data_sz)]
	train_data_out_gaps = train_data_out_gaps[:int(train_per*data_sz)]

	dev_data_in_timestamps = train_data_in_timestamps[int(train_per*data_sz):]
	dev_data_out_timestamps = train_data_out_timestamps[int(train_per*data_sz):]
	train_data_in_timestamps = train_data_in_timestamps[:int(train_per*data_sz)]
	train_data_out_timestamps = train_data_out_timestamps[:int(train_per*data_sz)]

	dev_data_out_gaps = denormalize_data(dev_data_out_gaps,
										 train_norm_a_gaps,
										 train_norm_d_gaps)
	return [train_data_in_gaps, train_data_out_gaps,
			dev_data_in_gaps, dev_data_out_gaps,
			train_data_in_timestamps, train_data_out_timestamps,
			dev_data_in_timestamps, dev_data_out_timestamps]

def stabalize_data(data_in_gaps, data_out_gaps,
				   data_in_timestamps, data_out_timestamps,
				   batch_size):
	data_sz = data_in_gaps.shape[0]
	data_sz_rem = data_sz % batch_size
	if data_sz_rem != 0:
		data_in_gaps = data_in_gaps[:(-data_sz_rem)]
		data_out_gaps = data_out_gaps[:(-data_sz_rem)]
		data_in_timestamps = data_in_timestamps[:(-data_sz_rem)]
		data_out_timestamps = data_out_timestamps[:(-data_sz_rem)]
		
	data_in_gaps = np.expand_dims(data_in_gaps, axis=-1)
	data_out_gaps = np.expand_dims(data_out_gaps, axis=-1)
	data_in_timestamps = np.expand_dims(data_in_timestamps, axis=-1)
	data_out_timestamps = np.expand_dims(data_out_timestamps, axis=-1)
	return data_in_gaps, data_out_gaps, data_in_timestamps, data_out_timestamps

def get_end_time_from_bins(test_inp_times_in_bin, test_out_times_in_bin,
						   test_end_hr_bins, enc_len=80):
	test_data_in_gaps_bin_lst = list()
	test_data_in_time_end_bin_lst = list()
	test_data_out_gaps_bin_lst = list()
	test_data_in_times_bin_lst = list()
	test_data_out_times_bin_lst = list()
	for idx in range(len(test_end_hr_bins)):
		A1 = np.array(test_inp_times_in_bin[idx][1:])
		A2 = np.array(test_inp_times_in_bin[idx][:-1])
		test_data_in_time_end_bin_lst.append(test_inp_times_in_bin[idx][-1])
		test_data_in_gaps_bin_lst.append((A1-A2)[-enc_len:])
		test_data_in_times_bin_lst.append(A1[-enc_len:])

		A1 = np.array(test_out_times_in_bin[idx])
		A2 = np.array([test_inp_times_in_bin[idx][-1]] + test_out_times_in_bin[idx][:-1])
		test_data_out_gaps_bin_lst.append(A1-A2)
		test_data_out_times_bin_lst.append(A1)

	test_data_in_time_end_bin = np.array(test_data_in_time_end_bin_lst)
	test_data_in_gaps_bin = np.array(test_data_in_gaps_bin_lst)
	test_end_hr_bins = np.array(test_end_hr_bins)
	test_data_out_gaps_bin = np.array(test_data_out_gaps_bin_lst)
	test_data_in_times_bin = np.array(test_data_in_times_bin_lst)
	test_data_out_times_bin = np.array(test_data_out_times_bin_lst)

	test_data_in_gaps_bin, test_gap_in_bin_norm_a, test_gap_in_bin_norm_d \
		= normalize_avg(test_data_in_gaps_bin)
	return [test_data_in_gaps_bin, test_data_out_gaps_bin,
			test_data_in_times_bin, test_data_out_times_bin,
			test_data_in_time_end_bin, test_end_hr_bins,
			test_gap_in_bin_norm_a, test_gap_in_bin_norm_d]

def get_rand_interval_count(test_out_times_in_bin):
	test_time_out_interval = [np.random.uniform(low=item[0], high=item[-1], size=2) for item in test_out_times_in_bin]
	test_time_out_tb_plus = [min(item) for item in test_time_out_interval]
	test_time_out_te_plus = [max(item) for item in test_time_out_interval]
	times_out_indices_tb = [bisect_right(t_out, t_b) for t_out, t_b in zip(test_out_times_in_bin, test_time_out_tb_plus)]
	times_out_indices_te = [bisect_right(t_out, t_e) for t_out, t_e in zip(test_out_times_in_bin, test_time_out_te_plus)]
	test_out_event_count_true = [times_out_indices_te[idx] - times_out_indices_tb[idx] for idx in range(len(test_time_out_tb_plus))]
	test_out_all_event_true = [test_out_times_in_bin[idx][times_out_indices_tb[idx]:times_out_indices_te[idx]] for idx in range(len(test_time_out_tb_plus))]
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

def get_interval_count_with_threshold(test_out_times_in_bin, interval_size, dataset_name, threshold=None):
	test_sample_count = len(test_out_times_in_bin)

	if threshold == -1:
		interval_range_count_more = None
		interval_range_count_less = None
		more_thresh = 1
		less_thresh = 1
		for thresh in range(1, len(test_out_times_in_bin)):
			threshold = np.ones(test_sample_count) * thresh
			threshold = threshold.astype(int)
			interval_range_count_more_tmp \
				= get_interval_count_more_than_threshold(test_out_times_in_bin,
														 interval_size,
														 threshold)
			if np.any(interval_range_count_more_tmp == -1):
				break

			interval_range_count_more = interval_range_count_more_tmp
			more_thresh = thresh

		for thresh in range(len(test_out_times_in_bin), 1, -1):
			threshold = np.ones(test_sample_count) * thresh
			threshold = threshold.astype(int)
			interval_range_count_less_tmp \
				= get_interval_count_less_than_threshold(test_out_times_in_bin,
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
			bins_count_more = np.ceil((test_out_times_in_bin[idx][-1] - test_out_times_in_bin[idx][0])/interval_size)
			bins_count_less = np.floor((test_out_times_in_bin[idx][-1] - test_out_times_in_bin[idx][0])/interval_size)
			bins_count_less = max(1, bins_count_less)
			avg_events_count_more = (len(test_out_times_in_bin[idx])/bins_count_more)
			avg_events_count_less = (len(test_out_times_in_bin[idx])/bins_count_less)
			avg_events_count_more = round(avg_events_count_more * more_factor[dataset_name])
			avg_events_count_more = max(1, avg_events_count_more)
			avg_events_count_less = round(avg_events_count_less * less_factor[dataset_name])
			avg_events_count_less = max(1, avg_events_count_less)
			threshold_more[idx] = avg_events_count_more
			threshold_less[idx] = avg_events_count_less
		threshold_more = np.array(threshold_more)
		threshold_less = np.array(threshold_less)

	interval_range_count_more \
		= get_interval_count_more_than_threshold(test_out_times_in_bin,
												 interval_size,
												 threshold_more)

	interval_range_count_less \
		= get_interval_count_less_than_threshold(test_out_times_in_bin,
												 interval_size,
												 threshold_less)

	begin = np.array([x[0] for x in test_out_times_in_bin])
	print('test_out_times_in_bin', np.array([x[0] for x in test_out_times_in_bin]) )
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
    time_feature_hour = (times // 3600) % 24
    time_feature_minute = ((times-(times//3600)*3600) // 60) % 60
    time_feature_seconds = (times-(times//60)*60) % 60

    time_feature = (time_feature_hour * 3600.0 
                 + time_feature_minute * 60.0
                 + time_feature_seconds)
    time_feature = time_feature / 3600.0

    time_feature = (times/3600.)%24
    return time_feature

def get_processed_data(dataset_name, args):

	bin_size = args.bin_size
	in_bin_sz = args.in_bin_sz
	out_bin_sz = args.out_bin_sz
	enc_len = args.enc_len
	batch_size = args.batch_size

	timestamps = np.loadtxt('data/'+dataset_name+'.txt')
	gaps = timestamps[1:] - timestamps[:-1]
	gaps = gaps.astype(np.float32)
	data_bins, end_hr_bins, times_in_bin = create_bin(timestamps, bin_size)
	plt.plot(range(len(data_bins[:100])), data_bins[:100])
	plt.ylabel('bin_counts')
	plt.xlabel('911 Dataset')
	plt.savefig('data/bin_count_'+dataset_name+'.png')
	plt.close()

	[train_times_bin, test_times_bin,
	 train_times_bin_end, test_times_bin_end,
	 train_seq_times_in_bin, test_seq_times_in_bin,
	 train_times_gaps, test_times_gaps,
	 train_times_timestamps, test_times_timestamps] \
	 	= generate_train_test_data(timestamps, gaps, data_bins,
	 							   end_hr_bins, times_in_bin)

	hawkes_timestamps_pred = test_times_timestamps
	train_times_gaps_norm, gaps_norm_a, gaps_norm_d \
		= normalize_avg(train_times_gaps)
	train_times_timestamps_norm \
		= (np.cumsum(train_times_gaps_norm)
		   + (train_times_timestamps[0]-train_times_gaps[0]))
	test_times_gaps_norm = normalize_avg_given_param(test_times_gaps,
													 gaps_norm_a,
													 gaps_norm_d)
	test_times_timestamps_norm \
		= (np.cumsum(test_times_gaps_norm)
		   + (train_times_timestamps_norm[-1]))

	model = hk.estimator().set_kernel('exp').set_baseline('const')
	start_idx = min(30000, len(train_times_timestamps_norm))
	itv = [train_times_timestamps_norm[-start_idx], train_times_timestamps_norm[-1]]
	train_times_timestamps_norm = train_times_timestamps_norm.astype(np.float64)
	model.fit(train_times_timestamps_norm[-start_idx:],itv)
	print("Parameters generated by Hawkes Model")
	print("Parameter:",model.parameter)
	print("AIC:",model.AIC)
	hawkes_timestamps_pred = model.predict(test_times_timestamps_norm[-1]+100,1)[0]
	test_times_gaps_norm \
		= (hawkes_timestamps_pred
		   - np.concatenate([train_times_timestamps_norm[-1:],
		   					 hawkes_timestamps_pred[:-1]]))
	test_times_gaps = denormalize_avg(test_times_gaps_norm, gaps_norm_a, gaps_norm_d)
	hawkes_timestamps_pred = np.cumsum(test_times_gaps) + train_times_timestamps[-1]
	# Plots for Hawkes Predictions
	os.makedirs('Outputs/hawkes_model_'+dataset_name, exist_ok=True)
	model.plot_N_pred()
	plt.savefig('Outputs/hawkes_model_'+dataset_name+'/count_pred.png')
	plt.close()
	model.plot_KS()
	plt.savefig('Outputs/hawkes_model_'+dataset_name+'/KS_plot.png')
	plt.close()

	[train_data_in_bin, train_data_out_bin,
	 _, _,
	 train_in_end_hr_bins, train_end_hr_bins] \
		= make_seq_from_data(train_times_bin, enc_len, in_bin_sz, out_bin_sz,
							 batch_size,
							 is_it_bins=True,
							 stride_len=args.stride_len,
							 count_strid_len=1,
	 						 times_in_bin=train_seq_times_in_bin,
	 						 end_hr_bins=train_times_bin_end,
							 dataset_name=dataset_name)
	[test_data_in_bin, test_data_out_bin,
	 test_inp_times_in_bin, test_out_times_in_bin, 
	 test_in_end_hr_bins, test_end_hr_bins] \
	 	= make_seq_from_data(test_times_bin, enc_len, in_bin_sz, out_bin_sz,
	 						 batch_size,
	 						 is_it_bins=True,
	 						 stride_len=args.stride_len,
	 						 count_strid_len=args.out_bin_sz,
	 						 times_in_bin=test_seq_times_in_bin,
	 						 end_hr_bins=test_times_bin_end,
	 						 dataset_name=dataset_name)

	train_data_in_bin, train_mean_bin, train_std_bin \
		= normalize_data(train_data_in_bin)
	train_data_out_bin = normalize_data_given_param(train_data_out_bin,
													train_mean_bin,
													train_std_bin)
	test_data_in_bin, test_mean_bin, test_std_bin = normalize_data(test_data_in_bin)

	train_data_in_gaps, train_data_out_gaps, _, _, _, _ \
		= make_seq_from_data(train_times_gaps, enc_len, in_bin_sz, out_bin_sz,
							 batch_size,
							 is_it_bins=False,
							 stride_len=args.stride_len,
							 dataset_name=dataset_name)
	test_data_in_gaps, test_data_out_gaps, _, _, _, _ \
		= make_seq_from_data(test_times_gaps, enc_len, in_bin_sz, out_bin_sz,
							 batch_size,
							 is_it_bins=False,
							 stride_len=args.stride_len,
							 dataset_name=dataset_name)
	train_data_in_timestamps, train_data_out_timestamps, _, _, _, _ \
		= make_seq_from_data(train_times_timestamps, enc_len, in_bin_sz, out_bin_sz,
							 batch_size,
							 is_it_bins=False,
							 stride_len=args.stride_len,
							 dataset_name=dataset_name)
	test_data_in_timestamps, test_data_out_timestamps, _, _, _, _ \
		= make_seq_from_data(test_times_timestamps, enc_len, in_bin_sz, out_bin_sz,
							 batch_size,
					   		 is_it_bins=False,
					   		 stride_len=args.stride_len,
					   		 dataset_name=dataset_name)

	(_, _,
	 train_inp_times_in_bin, train_out_times_in_bin,
	 _, train_end_hr_bins_relative) \
		= make_seq_from_data(train_times_timestamps, enc_len, in_bin_sz, out_bin_sz,
							 batch_size,
							 is_it_bins=False,
							 is_it_var=True,
							 stride_len=int(enc_len/4),
							 dataset_name=dataset_name,
							 bin_size=args.bin_size)

	train_data_in_gaps, train_norm_a_gaps, train_norm_d_gaps \
		= normalize_avg(train_data_in_gaps)
	train_data_out_gaps = normalize_avg_given_param(train_data_out_gaps,
													train_norm_a_gaps,
													train_norm_d_gaps)
	test_data_in_gaps, test_norm_a_gaps, test_norm_d_gaps \
		= normalize_avg(test_data_in_gaps)


	[train_data_in_gaps, train_data_out_gaps,
	 dev_data_in_gaps, dev_data_out_gaps,
	 train_data_in_timestamps, train_data_out_timestamps,
	 dev_data_in_timestamps, dev_data_out_timestamps] \
	 	= generate_dev_data([train_data_in_gaps, train_data_out_gaps,
	 						 train_data_in_timestamps, train_data_out_timestamps,
	 						 train_norm_a_gaps, train_norm_d_gaps])


	[train_data_in_gaps, train_data_out_gaps,
	 train_data_in_timestamps, train_data_out_timestamps] \
	 	= stabalize_data(train_data_in_gaps, train_data_out_gaps,
	 				  	 train_data_in_timestamps, train_data_out_timestamps,
	 				  	 batch_size)

	[dev_data_in_gaps, dev_data_out_gaps,
	 dev_data_in_timestamps, dev_data_out_timestamps] \
	 	= stabalize_data(dev_data_in_gaps, dev_data_out_gaps,
	 					 dev_data_in_timestamps, dev_data_out_timestamps,
	 					 batch_size)

	[test_data_in_gaps, test_data_out_gaps,
	 test_data_in_timestamps, test_data_out_timestamps] \
	 	= stabalize_data(test_data_in_gaps, test_data_out_gaps, 
						 test_data_in_timestamps, test_data_out_timestamps,
						 batch_size)

	train_data_in_feats = get_time_features(train_data_in_timestamps)
	train_data_out_feats = get_time_features(train_data_out_timestamps)
	dev_data_in_feats = get_time_features(dev_data_in_timestamps)

	print('')
	print('Dataset size')
	print('Train in', train_data_in_gaps.shape)
	print('Train out', train_data_out_gaps.shape)
	print('Dev in', dev_data_in_gaps.shape)
	print('Dev out', dev_data_out_gaps.shape)
	print('Test in', test_data_in_gaps.shape)
	print('Test out', test_data_out_gaps.shape)
	print('')


	[train_data_in_gaps_bin, train_data_out_gaps_bin,
	 train_data_in_times_bin, train_data_out_times_bin,
	 train_data_in_time_end_bin, train_end_hr_bins_relative,
	 train_gap_in_bin_norm_a, train_gap_in_bin_norm_d] \
		= get_end_time_from_bins(train_inp_times_in_bin, train_out_times_in_bin,
								 train_end_hr_bins_relative, enc_len)
	train_data_out_gaps_bin = normalize_avg_given_param(train_data_out_gaps_bin,
									  					train_norm_a_gaps,
									  					train_norm_d_gaps)

	[test_data_in_gaps_bin, test_data_out_gaps_bin,
	 test_data_in_times_bin, test_data_out_times_bin,
	 test_data_in_time_end_bin, test_end_hr_bins,
	 test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] \
		= get_end_time_from_bins(test_inp_times_in_bin, test_out_times_in_bin,
								 test_end_hr_bins, enc_len)

	test_data_in_feats_bin = get_time_features(test_data_in_times_bin)

	[test_time_out_tb_plus, test_time_out_te_plus, 
	 test_out_event_count_true, test_out_all_event_true] \
	 	= get_rand_interval_count(test_out_times_in_bin)

	interval_size = args.interval_size
	if interval_size==0:
		interval_size = args.bin_size
	[interval_range_count_less, interval_range_count_more,
	 less_threshold, more_threshold] \
	 	= get_interval_count_with_threshold(test_out_times_in_bin,
	 										interval_size,
	 										dataset_name)

	test_end_hr_bins = np.expand_dims(test_end_hr_bins, axis=-1)
	test_data_in_time_end_bin = np.expand_dims(test_data_in_time_end_bin, axis=-1)
	test_data_in_gaps_bin = np.expand_dims(test_data_in_gaps_bin, axis=-1)
	test_data_in_feats_bin = np.expand_dims(test_data_in_feats_bin, axis=-1)
	train_end_hr_bins_relative = np.expand_dims(train_end_hr_bins_relative, axis=-1)
	train_data_in_time_end_bin = np.expand_dims(train_data_in_time_end_bin, axis=-1)
	train_data_in_gaps_bin = np.expand_dims(train_data_in_gaps_bin, axis=-1)


	train_data_in_bin = np.expand_dims(train_data_in_bin, axis=-1).astype(np.float32)
	test_data_in_bin = np.expand_dims(test_data_in_bin, axis=-1).astype(np.float32)
	train_in_end_hr_bins = np.expand_dims(train_in_end_hr_bins, axis=-1).astype(np.float32)
	test_in_end_hr_bins = np.expand_dims(test_in_end_hr_bins, axis=-1).astype(np.float32)
	train_data_in_bin_feats = get_time_features(train_in_end_hr_bins-bin_size/2.)
	test_data_in_bin_feats = get_time_features(test_in_end_hr_bins-bin_size/2.)


	print(train_data_in_bin.shape)
	print(test_data_in_bin.shape)
	print(train_data_in_gaps.shape)
	print(test_data_in_gaps_bin.shape)


	dataset = {
		'train_data_in_gaps': train_data_in_gaps,
		'train_data_out_gaps': train_data_out_gaps,
		'dev_data_in_gaps': dev_data_in_gaps,
		'dev_data_out_gaps': dev_data_out_gaps,
		'train_norm_a_gaps': train_norm_a_gaps,
		'train_norm_d_gaps': train_norm_d_gaps,
		'train_data_in_bin': train_data_in_bin,
		'train_data_out_bin': train_data_out_bin,

		'test_data_in_gaps_bin': test_data_in_gaps_bin,
		'test_end_hr_bins': test_end_hr_bins,
		'test_data_in_time_end_bin': test_data_in_time_end_bin,
		'test_gap_in_bin_norm_a': test_gap_in_bin_norm_a,
		'test_gap_in_bin_norm_d': test_gap_in_bin_norm_d,
		'test_data_in_bin': test_data_in_bin,
		'test_data_out_bin': test_data_out_bin,
		'test_mean_bin': test_mean_bin,
		'test_std_bin': test_std_bin,
		'test_data_in_bin': test_data_in_bin,
		'test_mean_bin': test_mean_bin,
		'test_std_bin': test_std_bin,

		'test_time_out_tb_plus': test_time_out_tb_plus,
		'test_time_out_te_plus': test_time_out_te_plus,
		'test_out_event_count_true': test_out_event_count_true,
		'test_out_all_event_true': test_out_all_event_true,

		'interval_range_count_less': interval_range_count_less,
		'interval_range_count_more': interval_range_count_more,
		'less_threshold': less_threshold,
		'more_threshold': more_threshold,
		'interval_size': interval_size,
		'test_out_times_in_bin': test_out_times_in_bin,

		'train_data_in_gaps_bin': train_data_in_gaps_bin,
		'train_data_out_gaps_bin': train_data_out_gaps_bin,
		'train_data_in_time_end_bin': train_data_in_time_end_bin,
	 	'train_end_hr_bins_relative': train_end_hr_bins_relative,
	 	'train_gap_in_bin_norm_a': train_gap_in_bin_norm_a,
	 	'train_gap_in_bin_norm_d': train_gap_in_bin_norm_d,

		'hawkes_timestamps_pred': hawkes_timestamps_pred,

		'test_data_out_gaps_bin': test_data_out_gaps_bin,

		'train_data_in_feats': train_data_in_feats,
		'train_data_out_feats': train_data_out_feats,
		'dev_data_in_feats': dev_data_in_feats,
		'test_data_in_feats_bin': test_data_in_feats_bin,

		'train_data_in_bin_feats': train_data_in_bin_feats,
		'test_data_in_bin_feats': test_data_in_bin_feats,
	}

	return dataset
