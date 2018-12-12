import sys
import numpy as np
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
        min_ts = min(loc)
        max_ts = max(loc)
        norm_loc = [(ts - min_ts)*1.0/(max(max_ts - min_ts, 1.0)) for ts in loc]
        norm_data.append(norm_loc)
        assert len(loc) == len(norm_loc)
    return norm_data

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
    data = createPerDayChunks(latlngList, data)
    data = sample(data)
    data = normalize(data)
    train_data, test_data = split_train_test(data)
    writeToFile(filePath, data)
    writeToFile(filePath, train_data, typ='train')
    writeToFile(filePath, test_data, typ='test')

if __name__ == '__main__':
    main()
