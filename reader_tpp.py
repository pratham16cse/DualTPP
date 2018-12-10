import sys
import numpy as np
from collections import OrderedDict

import reader
from feature import getDatetime, timestampTotime


def createPerDayChunks(latlngList, data):
    new_data = list()
    for loc in data:
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


def writeToFile(filePath, data):
    cityName = filePath.split('/')[-1].split('-')[0]
    with open('../datasets/traffic_tpp/'+cityName+'-congestions.txt', 'w') as f:
        for loc in data:
            for ts in loc[:-1]:
                f.write(str(ts) + ' ')
            f.write(str(loc[-1]) + '\n')


def main():
    filePath = sys.argv[1]
    latlngList, data = reader.read(filePath)
    data = createPerDayChunks(latlngList, data)
    data = sample(data)
    writeToFile(filePath, data)

if __name__ == '__main__':
    main()
