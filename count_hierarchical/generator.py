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
downsampling = {'Trump': 10}
#downsampling = {'taxi': 20, 'Trump': 20}

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
	y_ = 10*np.sin(omega*x)
	y = y_ + 11
	gaps=y
	timestamp = np.cumsum(gaps)
	types = []
	if y_[0]<y_[1]:
		types.append(1)
	for i in range(len(y_[1:])):
		#if y_[i]>=0.:
		#	types.append(0)
		#else:
		#	types.append(1)
		if y_[i]>=0. and y_[i]>y_[i-1]:
			types.append(1)
		if y_[i]>=0. and y_[i]<y_[i-1]:
			types.append(2)
		if y_[i]<0. and y_[i]<y_[i-1]:
			types.append(3)
		if y_[i]<0. and y_[i]>y_[i-1]:
			types.append(4)

	types = np.array(types)
	
	plt.plot(x[:25], y[:25], 'o', color='black');
	plt.savefig('sin.png')
	plt.close()
	return gaps, timestamp, types

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
	# https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-01.csv
	# https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2019-02.csv
	taxi_df_jan = pd.read_csv('../yellow_tripdata_2019-01.csv', usecols=["tpep_pickup_datetime", "PULocationID"])
	taxi_df_feb = pd.read_csv('../yellow_tripdata_2019-02.csv', usecols=["tpep_pickup_datetime", "PULocationID"])
	taxi_df = taxi_df_jan.append(taxi_df_feb)
	taxi_df = taxi_df[taxi_df.PULocationID == 237]
	taxi_df = pd.to_datetime(taxi_df['tpep_pickup_datetime'], errors='coerce')
	taxi_df = taxi_df[(taxi_df.dt.year == 2019)]
	taxi_df = taxi_df[(taxi_df.dt.month < 3)]
	taxi_df = pd.DatetimeIndex(taxi_df).astype(np.int64)/1000000000
	taxi_df = taxi_df.sort_values().astype(np.int64)
	taxi_timestamps = taxi_df
	taxi_timestamps = np.array(taxi_timestamps)
	taxi_timestamps -= taxi_timestamps[0]
	taxi_timestamps = taxi_timestamps[:-1]
	dataset_name = 'taxi'
	if dataset_name in downsampling:
		taxi_timestamps = downsampling_dataset(taxi_timestamps, dataset_name)
	taxi_gaps = taxi_timestamps[1:] - taxi_timestamps[:-1]
	plt.plot(taxi_gaps[:100])
	plt.ylabel('Gaps')
	plt.savefig('taxi_gaps.png')
	plt.close()
	return taxi_gaps, taxi_timestamps

def create_911_traffic_data():
	call_df = pd.read_csv('../911.csv', usecols=["title", "timeStamp"])
	print('Types of Emergencies')
	print(call_df.title.apply(lambda x: x.split(':')[0]).value_counts())
	call_df['type'] = call_df.title.apply(lambda x: x.split(':')[0])
	print('Subtypes')
	for each in call_df.type.unique():
		subtype_count = call_df[call_df.title.apply(lambda x: x.split(':')[0]==each)].title.value_counts()
		print('For', each, 'type of Emergency, we have ', subtype_count.count(), 'subtypes')
		print(subtype_count[subtype_count>100])
	print('Out of 3 types taking Traffic type considering only Traffic')
	call_data = call_df[call_df['type']=='Traffic'].timeStamp
	call_data = pd.to_datetime(call_data, errors='coerce')
	print("We have timeline from", call_data.min(), "to", call_data.max())

	call_data = pd.DatetimeIndex(call_data).astype(np.int64)/1000000000
	call_timestamps = call_data.sort_values().astype(np.int64)
	call_timestamps = np.array(call_timestamps)
	call_timestamps -= call_timestamps[0]
	dataset_name = 'call'
	if dataset_name in downsampling:
		call_timestamps = downsampling_dataset(call_timestamps, dataset_name)
	call_gaps = call_timestamps[1:] - call_timestamps[:-1]
	plt.plot(call_gaps[:100])
	plt.ylabel('Gaps')
	plt.xlabel('timeline')
	plt.savefig('call_traffic_gaps.png')
	plt.close()
	return call_gaps, call_timestamps	

def create_911_ems_data():
	call_df = pd.read_csv('../911.csv', usecols=["title", "timeStamp"])
	call_df['type'] = call_df.title.apply(lambda x: x.split(':')[0])
	print('Out of 3 types taking EMS type considering only EMS')
	call_data = call_df[call_df['type']=='EMS'].timeStamp
	call_data = pd.to_datetime(call_data, errors='coerce')
	print("We have timeline from", call_data.min(), "to", call_data.max())

	call_data = pd.DatetimeIndex(call_data).astype(np.int64)/1000000000
	call_timestamps = call_data.sort_values().astype(np.int64)
	call_timestamps = np.array(call_timestamps)
	call_timestamps -= call_timestamps[0]
	dataset_name = 'call'
	if dataset_name in downsampling:
		call_timestamps = downsampling_dataset(call_timestamps, dataset_name)
	call_gaps = call_timestamps[1:] - call_timestamps[:-1]
	plt.plot(call_gaps[:100])
	plt.ylabel('Gaps')
	plt.xlabel('timeline')
	plt.savefig('call_ems_gaps.png')
	plt.close()
	return call_gaps, call_timestamps	

def generate_dataset():
	os.makedirs('./data', exist_ok=True)
	os.chdir('./data')
	if not os.path.isfile("sin.txt"):
		print('Generating sin data')
		gaps, timestamps, types = create_sin_data()
		np.savetxt('sin.txt', timestamps)
		np.savetxt('sin_types.txt', types)
	if not os.path.isfile("hawkes.txt"):
		print('Generating hawkes data')
		gaps, timestamps = create_hawkes_data()
		np.savetxt('hawkes.txt', timestamps)
	if not os.path.isfile("sin_hawkes_overlay.txt"):
		print('Generating sin_hawkes_overlay data')
		gaps, timestamps = create_sin_hawkes_overlay_data()
		np.savetxt('sin_hawkes_overlay.txt', timestamps)
	if not os.path.isfile("911_traffic.txt"):
		print('Generating 911 data')
		gaps, timestamps = create_911_traffic_data()
		np.savetxt('911_traffic.txt', timestamps)
	if not os.path.isfile("911_ems.txt"):
		print('Generating 911 data')
		gaps, timestamps = create_911_ems_data()
		np.savetxt('911_ems.txt', timestamps)
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
