import numpy as np
import os, sys
import matplotlib.pyplot as plt
from bisect import bisect_right

def normalize_data(data):
	mean = np.mean(data)
	std = np.std(data)
	return (data - mean)/std, mean, std

def normalize_data_given_param(data, mean, std):
	return (data - mean)/std

def denormalize_data(data, mean, std):
	return (data * std) + mean

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
	return int(round((time_interval*50) / events_count))

def generate_plots(args, dataset_name, dataset, per_model_count, test_sample_idx=1):
	inp_seq_len_plot = 10
	dec_len = args.out_bin_sz

	true_pred = per_model_count['true']
	rmtpp_mse_pred = true_pred
	rmtpp_nll_pred = true_pred
	hierarchical_pred = true_pred
	count_model_pred = true_pred
	if 'rmtpp_mse' in per_model_count:
		rmtpp_mse_pred = per_model_count['rmtpp_mse']
	if 'rmtpp_nll' in per_model_count:
		rmtpp_nll_pred = per_model_count['rmtpp_nll']
	if 'hierarchical' in per_model_count:
		hierarchical_pred = per_model_count['hierarchical']
	if 'count_model' in per_model_count:
		count_model_pred = per_model_count['count_model']


	event_count_preds_true = true_pred
	event_count_preds_mse = rmtpp_mse_pred
	event_count_preds_nll = rmtpp_nll_pred
	event_count_preds_cnt = hierarchical_pred
	event_count_preds_count = count_model_pred

	test_data_in_bin = dataset['test_data_in_bin']
	test_mean_bin = dataset['test_mean_bin']
	test_std_bin = dataset['test_std_bin']
	
	true_inp_bins = denormalize_data(test_data_in_bin[test_sample_idx,inp_seq_len_plot:], 
							test_mean_bin, test_std_bin)
	true_inp_bins = true_inp_bins.astype(np.float32)
	x = np.arange(inp_seq_len_plot+dec_len)
	true_pred = event_count_preds_true[test_sample_idx].astype(np.float32)
	rmtpp_mse_pred = event_count_preds_mse[test_sample_idx].astype(np.float32)
	rmtpp_nll_pred = event_count_preds_nll[test_sample_idx].astype(np.float32)
	hierarchical_pred = event_count_preds_cnt[test_sample_idx].astype(np.float32)
	count_model_pred = event_count_preds_count[test_sample_idx].astype(np.float32)

	plt.plot(x, true_inp_bins.tolist()+hierarchical_pred.tolist(), label='hierarchical_pred')
	plt.plot(x, true_inp_bins.tolist()+count_model_pred.tolist(), label='count_model_pred')
	plt.plot(x, true_inp_bins.tolist()+rmtpp_mse_pred.tolist(), label='rmtpp_mse_pred')
	plt.plot(x, true_inp_bins.tolist()+rmtpp_nll_pred.tolist(), label='rmtpp_nll_pred')
	plt.plot(x, true_inp_bins.tolist()+true_pred.tolist(), label='true_pred')

	plt.axvline(x=inp_seq_len_plot-1, color='k', linestyle='--')
	plt.legend(loc='upper left')
	plt.savefig('Outputs/'+dataset_name+'_'+str(test_sample_idx)+'.png')
	plt.close()

def create_bin(times, bin_size):
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
	test_times_bin_end = end_hr_bins[int(train_per*data_sz):]
	test_seq_times_in_bin = times_in_bin[int(train_per*data_sz):]

	data_sz = len(gaps)
	train_times_gaps = gaps[:int(train_per*data_sz)]
	test_times_gaps = gaps[int(train_per*data_sz):]
	train_times_timestamps = timestamps[1:int(train_per*data_sz)+1]
	test_times_timestamps = timestamps[int(train_per*data_sz)+1:]

	print("Length of train and test bins", len(train_times_bin), len(test_times_bin))
	print("Length of train and test gaps", len(train_times_gaps), len(test_times_gaps))
	print("Length of train and test timestamps", len(train_times_timestamps), len(test_times_timestamps))
	return  train_times_bin, test_times_bin, test_times_bin_end, test_seq_times_in_bin, \
			train_times_gaps ,test_times_gaps ,train_times_timestamps ,test_times_timestamps

def make_seq_from_data(data, enc_len, in_bin_sz, out_bin_sz, 
	is_it_bins=True, times_in_bin=None, end_hr_bins=None):
	inp_seq_lst = list()
	out_seq_lst = list()
	inp_times_in_bin = list()
	out_times_in_bin = list()
	out_end_hr_bins = list()

	iter_range = len(data)-enc_len-out_bin_sz
	if is_it_bins:
		iter_range = len(data)-in_bin_sz-out_bin_sz

	for idx in range(iter_range):
		
		if is_it_bins:
			inp_seq_lst.append(data[idx:idx+in_bin_sz])
			if times_in_bin is not None:
				sm_time_in_bin = []
				for r_idx in range(in_bin_sz):
					sm_time_in_bin+=times_in_bin[idx+r_idx]
				inp_times_in_bin.append(sm_time_in_bin)
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
		else:
			out_seq_lst.append(data[idx+1:idx+enc_len+1])

	inp_seq_lst = np.array(inp_seq_lst)
	out_seq_lst = np.array(out_seq_lst)
	return inp_seq_lst, out_seq_lst, inp_times_in_bin, out_times_in_bin, out_end_hr_bins

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

	dev_data_out_gaps = denormalize_data(dev_data_out_gaps, train_norm_a_gaps, train_norm_d_gaps)
	return [train_data_in_gaps, train_data_out_gaps, dev_data_in_gaps, dev_data_out_gaps, 
			train_data_in_timestamps, train_data_out_timestamps, dev_data_in_timestamps, 
			dev_data_out_timestamps]

def stabalize_data(data_in_gaps, data_out_gaps, data_in_timestamps, data_out_timestamps, batch_size):
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

def get_end_time_from_bins(test_inp_times_in_bin, test_end_hr_bins, enc_len=80):
	test_data_in_gaps_bin_lst = list()
	test_data_in_time_end_bin_lst = list()
	for idx in range(len(test_end_hr_bins)):
		A1 = np.array(test_inp_times_in_bin[idx][1:])
		A2 = np.array(test_inp_times_in_bin[idx][:-1])
		test_data_in_time_end_bin_lst.append(test_inp_times_in_bin[idx][-1])
		test_data_in_gaps_bin_lst.append((A1-A2)[-enc_len:])
	test_data_in_time_end_bin = np.array(test_data_in_time_end_bin_lst)
	test_data_in_gaps_bin = np.array(test_data_in_gaps_bin_lst)
	test_end_hr_bins = np.array(test_end_hr_bins)

	test_data_in_gaps_bin, test_gap_in_bin_norm_a, test_gap_in_bin_norm_d = normalize_avg(test_data_in_gaps_bin)
	return [test_data_in_gaps_bin, test_data_in_time_end_bin, test_end_hr_bins,
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
			interval_range_count_more_tmp = get_interval_count_more_than_threshold(test_out_times_in_bin,
																				interval_size,
																				threshold)
			if np.any(interval_range_count_more_tmp == -1):
				break

			interval_range_count_more = interval_range_count_more_tmp
			more_thresh = thresh

		for thresh in range(len(test_out_times_in_bin), 1, -1):
			threshold = np.ones(test_sample_count) * thresh
			threshold = threshold.astype(int)
			interval_range_count_less_tmp = get_interval_count_less_than_threshold(test_out_times_in_bin,
																				interval_size,
																				threshold)
			if np.any(interval_range_count_less_tmp == -1):
				break

			interval_range_count_less = interval_range_count_less_tmp
			less_thresh = thresh

		less_thresh = np.ones(test_sample_count) * less_thresh
		more_thresh = np.ones(test_sample_count) * more_thresh
		return interval_range_count_less, interval_range_count_more, less_thresh, more_thresh

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
			'sin': 1.0,
			'hawkes': 1.2,
			'sin_hawkes_overlay': 1.05,
			'Trump': 1.1,
			'Verdict': 1.6,
			'Delhi': 1.4,
			'taxi': 1.05,
		}

		less_factor = {
			'sin': 0.85,
			'hawkes': 0.85,
			'sin_hawkes_overlay': 0.85,
			'Trump': 0.95,
			'Verdict': 0.4,
			'Delhi': 0.6,
			'taxi': 0.95,
		}

		for idx in range(test_sample_count):
			bins_count = round((test_out_times_in_bin[idx][-1] - test_out_times_in_bin[idx][0])/interval_size)
			avg_events_count = (len(test_out_times_in_bin[idx])/bins_count)
			avg_events_count_more = round(avg_events_count * more_factor[dataset_name])
			avg_events_count_more = max(1, avg_events_count_more)
			avg_events_count_less = round(avg_events_count * less_factor[dataset_name])
			avg_events_count_less = max(1, avg_events_count_less)
			threshold_more[idx] = avg_events_count_more
			threshold_less[idx] = avg_events_count_less
		threshold_more = np.array(threshold_more)
		threshold_less = np.array(threshold_less)

	interval_range_count_more = get_interval_count_more_than_threshold( test_out_times_in_bin,
																		interval_size,
																		threshold_more)

	interval_range_count_less = get_interval_count_less_than_threshold( test_out_times_in_bin,
																		interval_size,
																		threshold_less)

	return interval_range_count_less, interval_range_count_more, threshold_less, threshold_more

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

	[train_times_bin, test_times_bin, test_times_bin_end, test_seq_times_in_bin, \
	train_times_gaps ,test_times_gaps ,train_times_timestamps ,test_times_timestamps] = \
	generate_train_test_data(timestamps, gaps, data_bins, end_hr_bins, times_in_bin)

	train_data_in_bin, train_data_out_bin, _, _, _ = \
	make_seq_from_data(train_times_bin, enc_len, in_bin_sz, out_bin_sz, True)
	[test_data_in_bin, test_data_out_bin, test_inp_times_in_bin, test_out_times_in_bin, 
	test_end_hr_bins] = \
	make_seq_from_data( test_times_bin, enc_len, in_bin_sz, out_bin_sz, True, 
						times_in_bin=test_seq_times_in_bin,
						end_hr_bins=test_times_bin_end)

	train_data_in_bin, train_mean_bin, train_std_bin = normalize_data(train_data_in_bin)
	train_data_out_bin = normalize_data_given_param(train_data_out_bin, train_mean_bin, train_std_bin)
	test_data_in_bin, test_mean_bin, test_std_bin = normalize_data(test_data_in_bin)

	train_data_in_gaps, train_data_out_gaps, _, _, _ = \
	make_seq_from_data(train_times_gaps, enc_len, in_bin_sz, out_bin_sz, False)
	test_data_in_gaps, test_data_out_gaps, _, _, _ = \
	make_seq_from_data(test_times_gaps, enc_len, in_bin_sz, out_bin_sz, False)
	train_data_in_timestamps, train_data_out_timestamps, _, _, _ = \
	make_seq_from_data(train_times_timestamps, enc_len, in_bin_sz, out_bin_sz, False)
	test_data_in_timestamps, test_data_out_timestamps, _, _, _ = \
	make_seq_from_data(test_times_timestamps, enc_len, in_bin_sz, out_bin_sz, False)

	train_data_in_gaps, train_norm_a_gaps, train_norm_d_gaps = normalize_avg(train_data_in_gaps)
	train_data_out_gaps = normalize_avg_given_param(train_data_out_gaps, train_norm_a_gaps, train_norm_d_gaps)
	test_data_in_gaps, test_norm_a_gaps, test_norm_d_gaps = normalize_avg(test_data_in_gaps)

	[train_data_in_gaps, train_data_out_gaps, dev_data_in_gaps, dev_data_out_gaps,
	train_data_in_timestamps, train_data_out_timestamps, dev_data_in_timestamps,
	dev_data_out_timestamps] = generate_dev_data([train_data_in_gaps, train_data_out_gaps, \
	train_data_in_timestamps, train_data_out_timestamps, train_norm_a_gaps, train_norm_d_gaps])

	[train_data_in_gaps, train_data_out_gaps, train_data_in_timestamps, train_data_out_timestamps] = \
	stabalize_data( train_data_in_gaps, train_data_out_gaps, train_data_in_timestamps, 
					train_data_out_timestamps, batch_size)

	[dev_data_in_gaps, dev_data_out_gaps, dev_data_in_timestamps, dev_data_out_timestamps] = \
	stabalize_data( dev_data_in_gaps, dev_data_out_gaps, 
					dev_data_in_timestamps, dev_data_out_timestamps, batch_size)

	[test_data_in_gaps, test_data_out_gaps, test_data_in_timestamps, test_data_out_timestamps] = \
	stabalize_data( test_data_in_gaps, test_data_out_gaps, 
					test_data_in_timestamps, test_data_out_timestamps, batch_size)

	print('')
	print('Dataset size')
	print('Train in', train_data_in_gaps.shape)
	print('Train out', train_data_out_gaps.shape)
	print('Dev in', dev_data_in_gaps.shape)
	print('Dev out', dev_data_out_gaps.shape)
	print('Test in', test_data_in_gaps.shape)
	print('Test out', test_data_out_gaps.shape)
	print('')

	[test_data_in_gaps_bin, test_data_in_time_end_bin, test_end_hr_bins,
	test_gap_in_bin_norm_a, test_gap_in_bin_norm_d] = \
	get_end_time_from_bins(test_inp_times_in_bin, test_end_hr_bins, enc_len)

	[test_time_out_tb_plus, test_time_out_te_plus, 
	 test_out_event_count_true, test_out_all_event_true] = \
	get_rand_interval_count(test_out_times_in_bin)

	interval_size = args.interval_size
	[interval_range_count_less, interval_range_count_more, less_threshold, more_threshold] = \
	get_interval_count_with_threshold(test_out_times_in_bin, interval_size, dataset_name)

	test_end_hr_bins = np.expand_dims(test_end_hr_bins, axis=-1)
	test_data_in_time_end_bin = np.expand_dims(test_data_in_time_end_bin, axis=-1)
	test_data_in_gaps_bin = np.expand_dims(test_data_in_gaps_bin, axis=-1)

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
	}

	return dataset
