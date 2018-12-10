import sys
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


def main():
    filePath = sys.argv[1]
    latlngList, data = reader.read(filePath)
    data = createPerDayChunks(latlngList, data)

if __name__ == '__main__':
    main()
