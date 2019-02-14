import sys
import numpy as np
import time
import datetime
import pandas as pd
from collections import OrderedDict

from bitmap_model import reader
from bitmap_model.feature import getDatetime, timestampTotime


def createPerDayChunks(latlngList, data):
    new_data = list()
    for s_no, loc in enumerate(data):
        print(s_no)
        instances = OrderedDict()
        #print('Location Started -----')
        keys = [tuple(timestampTotime(ts)[:2]) for _, ts in loc]
        for key, ts in zip(keys, loc):
            if instances.get(key, None) is not None:
                instances[key].append(ts)
            else:
                instances[key] = [ts]
        #for k, v in instances.items():
        #    print(k, v)
        #print('Location Finished ------')
        for congList in instances.values():
            new_data.append(congList)
    #print(new_data)
    return new_data

def createChunks(latlngList, data):
    '''
    Creates weekly partitions of the event sequence.
    This function also returns the train-test split of the data,
    by making a chronological split of each sequence, last chunk being test chunk.
    '''
    train_data = list()
    test_data = list()
    for s_no, loc in enumerate(data):
        print(s_no)
        num_chunks = 0
        for i in range(len(loc)):
            diff = loc[i][1] - ts_start if i>0 else loc[i][1]
            if diff > 60*60*24*7: # Check if next chunk starts at next week.
                seq = []
                durn, ts_start = loc[i]
                seq.append(ts_start)
                seq.append(ts_start+durn)
                for j in range(i+1, len(loc)): # Creating chunks of one-week sequences.
                    durn, ts = loc[j]
                    if (ts - ts_start) <= 60*60*24*7:
                        seq.append(ts)
                        seq.append(ts+durn)
                    else:
                        break
                train_data.append(seq)
                num_chunks += 1
        if num_chunks > 1:
            test_data.append(train_data.pop(-1)) # Last chunk of the sequence moved from train to test.
    return train_data, test_data

def round_timestamps(data):
    
    round_to_min = lambda ts: ts // 60 * 60

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j][1] = round_to_min(data[i][j][1])
    return data

def sample(data):
    new_data = list()
    for loc in data:
        samples = list()
        for i in range(len(loc)):
            durn, ts = loc[i]
            samples_ = np.arange(ts, ts+durn, step=300).tolist()
            samples += samples_
        new_data.append(samples)
        #print(len(loc), loc, samples)

    return new_data

def normalize(data):
    norm_data = list()
    for loc in data:
        #min_ts = min(loc)
        #max_ts = max(loc)
        norm_loc = list()
        for ts in loc:
            _, _, hr, mi, sec = timestampTotime(ts)
            ts_norm = hr * 3600.0 + mi * 60.0 + sec
            norm_loc.append(ts_norm/86400.0)
        assert len(loc) == len(norm_loc)
        norm_data.append(norm_loc)
    return norm_data

def normalize_weekly(data):
    norm_data = list()
    for s_no, loc in enumerate(data):
        print(s_no, len(data))
        norm_loc = list()
        #offset = pd.to_datetime(loc[0], unit='s').replace(hour=0, minute=0, second=0)
        #offset = pd.DatetimeIndex([offset]).astype(np.int64)[0] // 10**9
        day = time.strftime("%Y %m %d", time.localtime(loc[0]))
        offset = time.mktime(datetime.datetime.strptime(day, "%Y %m %d").timetuple())
        for ts in loc:
            #print(s_no, loc[0], offset, ts, (ts-offset)/60, 60*24*7)
            # Normalize weekly timestamps as number of hours passed since beginning of the sequence.
            norm_loc.append((ts-offset)*1.0/(60*60))
        norm_data.append(norm_loc)
    return norm_data

def prependZeros(data):
    new_data = list()
    for loc in data:
        new_data.append([0.0] + loc)
    return new_data

def split_train_test(data):
    train_length = int(0.8 * len(data))
    return data[:train_length], data[train_length:]

def writeToFile(filePath, data, typ=''):
    cityName = filePath.split('/')[-1].split('-')[0]
    with open('../datasets/traffic_tpp/'+cityName+'/time-'+typ+'.txt', 'w') as time_f, \
            open('../datasets/traffic_tpp/'+cityName+'/event-'+typ+'.txt', 'w') as event_f:
        for loc in data:
            for ts in loc[:-1]:
                time_f.write(str(ts) + ' ')
                event_f.write(str(1) + ' ')
            time_f.write(str(loc[-1]) + '\n') 
            event_f.write(str(1) + '\n') 

def main():
    filePath = sys.argv[1]
    latlngList, data = reader.read(filePath)
    #data = round_timestamps(data)
    #data = createPerDayChunks(latlngList, data)
    train_data, test_data = createChunks(latlngList, data)
    #data = sample(data)
    #data = normalize(data)
    train_data, test_data = normalize_weekly(train_data), normalize_weekly(test_data)
    train_data, test_data = prependZeros(train_data), prependZeros(test_data)
    #train_data, test_data = split_train_test(data)
    #writeToFile(filePath, data)
    st = time.time()
    writeToFile(filePath, train_data, typ='train')
    writeToFile(filePath, test_data, typ='test')
    et = time.time()
    print('Time req to write train and test data to files: ', et-st, ' seconds')

if __name__ == '__main__':
    main()
