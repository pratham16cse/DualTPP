import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hawkes model is from https://omitakahiro.github.io/Hawkes/index.html
from modules import Hawkes as hk

para = {'mu':0.1, 'alpha':0.3, 'beta':0.6}
mu_t = lambda x: (1.0 + 0.8*np.sin(2*np.pi*x/100)) * 0.2 # baseline function for overlay
itv = [0,360000]
demo_itv = [0,360]

np.random.seed(42)
downsampling = {'taxi': 100, 'Trump': 20}

def downsampling_dataset(timestamps, dataset_name):
	print('Down-sampling', dataset_name, 'dataset by', downsampling[dataset_name])
	return timestamps[::downsampling[dataset_name]]

def hawkes_demo():
	hk_model = hk.simulator().set_kernel('exp').set_baseline('const').set_parameter(para)
	T = hk_model.simulate(demo_itv)
	hk_model.plot_l()
	plt.savefig('hawkes_intensity.png')
	plt.close()
	hk_model.plot_N()
	plt.savefig('hawkes_event_counts.png')
	plt.close()

def sin_hawkes_overlay_demo():
	hk_model = hk.simulator().set_kernel('exp').set_baseline('custom',l_custom=mu_t).set_parameter(para)
	T = hk_model.simulate(demo_itv)
	hk_model.plot_l()
	plt.savefig('sin_hawkes_overlay_intensity.png')
	plt.close()
	hk_model.plot_N()
	plt.savefig('sin_hawkes_overlay_event_counts.png')
	plt.close()

def create_sin_data():
	omega = 1.0
	points = 10000
	num_marks = 7
	x = np.linspace(0, points, 3*points)
	y = 10*np.sin(omega*x)+11
	gaps=y
	timestamp = np.cumsum(gaps)-1
	
	plt.plot(x[:25], y[:25], 'o', color='black');
	plt.savefig('sin.png')
	plt.close()
	return gaps, timestamp

def create_hawkes_data():
	hawkes_demo()
	hk_model = hk.simulator().set_kernel('exp').set_baseline('const').set_parameter(para)
	timestamp = hk_model.simulate([0, 360000])
	gaps = timestamp[1:] - timestamp[:-1]
	return gaps, timestamp

def create_sin_hawkes_overlay_data():
	sin_hawkes_overlay_demo()
	hk_model = hk.simulator().set_kernel('exp').set_baseline('custom',l_custom=mu_t).set_parameter(para)
	timestamp = hk_model.simulate([0, 360000])
	gaps = timestamp[1:] - timestamp[:-1]
	return gaps, timestamp

def create_taxi_data():
	taxi_df = pd.read_csv('../yellow_tripdata_2019-01.csv', usecols=["tpep_pickup_datetime"])
	taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'], errors='coerce')
	taxi_df = taxi_df[(taxi_df['tpep_pickup_datetime'].dt.year == 2019)]
	taxi_df = taxi_df[(taxi_df['tpep_pickup_datetime'].dt.month == 1)]
	taxi_df['tpep_pickup_datetime'] = pd.DatetimeIndex(taxi_df['tpep_pickup_datetime']).astype(np.int64)/1000000000
	taxi_df = taxi_df.sort_values('tpep_pickup_datetime').astype(np.int64)
	taxi_timestamps = taxi_df['tpep_pickup_datetime']
	taxi_timestamps = np.array(taxi_timestamps)
	taxi_timestamps -= taxi_timestamps[0]
	taxi_timestamps = taxi_timestamps
	if 'taxi' in downsampling:
		taxi_timestamps = downsampling_dataset(taxi_timestamps, 'taxi')
	taxi_gaps = taxi_timestamps[1:] - taxi_timestamps[:-1]
	plt.plot(taxi_gaps[:100])
	plt.ylabel('Gaps')
	plt.savefig('taxi_gaps.png')
	plt.close()
	return taxi_gaps, taxi_timestamps

def generate_dataset():
	os.makedirs('./data', exist_ok=True)
	os.chdir('./data')
	if not os.path.isfile("sin.txt"):
		print('Generating sin data')
		gaps, timestamps = create_sin_data()
		np.savetxt('sin.txt', timestamps)
	if not os.path.isfile("hawkes.txt"):
		print('Generating hawkes data')
		gaps, timestamps = create_hawkes_data()
		np.savetxt('hawkes.txt', timestamps)
	if not os.path.isfile("sin_hawkes_overlay.txt"):
		print('Generating sin_hawkes_overlay data')
		gaps, timestamps = create_sin_hawkes_overlay_data()
		np.savetxt('sin_hawkes_overlay.txt', timestamps)
	if not os.path.isfile("taxi.txt"):
		print('Generating taxi data')
		gaps, timestamps = create_taxi_data()
		np.savetxt('taxi.txt', timestamps)
	os.chdir('../')

def create_twitter_data(dataset_name):
	delimiter=' '
	if dataset_name in ['Movie', 'Delhi', 'Verdict', 'Fight']:
		delimiter='\t'
	twitter_df = pd.read_csv('../TwitterDataset/'+dataset_name+'.txt', delimiter=delimiter, header=None)
	twitter_df = twitter_df[1]
	timestamps = np.sort(np.array(twitter_df))
	timestamps -= timestamps[0]
	gaps = timestamps[1:] - timestamps[:-1]
	if dataset_name in downsampling:
		plt.plot(gaps)
		plt.ylabel('all_Gaps_before_downsample')
		plt.savefig(dataset_name+'_all_gaps_before_downsample.png')
		plt.close()
		timestamps = downsampling_dataset(timestamps, dataset_name)
	gaps = timestamps[1:] - timestamps[:-1]

	plt.plot(gaps[:100])
	plt.ylabel('Gaps')
	plt.savefig(dataset_name+'_gaps.png')
	plt.close()
	plt.plot(gaps)
	plt.ylabel('all_Gaps')
	plt.savefig(dataset_name+'_all_gaps.png')
	plt.close()
	return gaps, timestamps

def generate_twitter_dataset(twitter_dataset_names):
	os.makedirs('./data', exist_ok=True)
	os.chdir('./data')
	for dataset_name in twitter_dataset_names:
		if not os.path.isfile(dataset_name+'.txt'):
			print('Generating', dataset_name, 'data')
			gaps, timestamps = create_twitter_data(dataset_name)
			np.savetxt(dataset_name+'.txt', timestamps)
	os.chdir('../')

