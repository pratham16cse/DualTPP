from collections import OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
from reader import read
from feature import getDatetime
import sys

def getMaxGap(data):
    max_gap = 0
    ts1,ts2 = 0,0
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            gap = data[i][j][1] - data[i][j-1][1]
            if gap > max_gap:
                max_gap = gap
                ts1 = data[i][j-1][1]
                ts2 = data[i][j][1]
    return max_gap, ts1, ts2

def getMinGap(data):
    min_gap = data[0][0][1]
    ts1,ts2 = 0,0
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            gap = data[i][j][1] - data[i][j-1][1]
            if gap < min_gap:
                min_gap = gap
                ts1 = data[i][j-1][1]
                ts2 = data[i][j][1]
    return min_gap, ts1, ts2

def getMaxTimeStamp(data):
    max_ts = data[0][0][1]
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][1] > max_ts:
                max_ts = data[i][j][1]
    return max_ts

def getMinTimeStamp(data):
    min_ts = data[0][0][1]
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][1] < min_ts:
                min_ts = data[i][j][1]
    return min_ts

def record_duration(data):
    return getMaxTimeStamp(data) - getMinTimeStamp(data) #returns duration in seconds.

def getNumCong2NumLocs(latlngList, data):
    numCong2numLoc = OrderedDict()
    for latlng, cngList in zip(latlngList, data):
        key = len(cngList)
        if numCong2numLoc.get(key, -1) == -1:
            numCong2numLoc[key] = 0
        numCong2numLoc[key] += 1

    numCong2numLocSortedList = sorted(numCong2numLoc.items(), key=itemgetter(0))
    numCong2numLoc = OrderedDict()
    for key, val in numCong2numLocSortedList:
        numCong2numLoc[key] = val
    return numCong2numLoc

def getDurn2GapRatio(latlngList, data):
    latlng2Ratio = OrderedDict()
    for locId, congList in enumerate(data):
        if len(congList) > 1:
            totCongDurn = sum([dur for dur, _ in congList])
            startTs = [ts for _, ts in congList]
            endTs = [ts+dur for dur, ts in congList]
            gaps = [sts-ets for sts, ets in zip(startTs[1:], endTs[:-1])]
            totGap = sum(gaps)
            latlng2Ratio[locId] = totCongDurn*1.0/totGap
            #print(locId, len(congList), latlng2Ratio[locId])

    latlng2RatioSortedList = sorted(latlng2Ratio.items(), key=itemgetter(1))
    latlng2Ratio = OrderedDict()
    latlng2NumCng = OrderedDict()
    for newId, (oldId, ratio) in enumerate(latlng2RatioSortedList):
        latlng2Ratio[newId] = ratio
        latlng2NumCng[newId] = len(data[oldId])

    assert latlng2Ratio.keys() == latlng2NumCng.keys()

    return latlng2Ratio, latlng2NumCng


def generateStats(cityName, latlngList, data):
    print('Duration of data record: '+str(record_duration(data)*1.0/(60*60*24))+' days.')
    print('Number of Locations: '+str(len(data)))

    numCong2numLoc = getNumCong2NumLocs(latlngList, data)
    plt.semilogy(numCong2numLoc.keys(), numCong2numLoc.values(), 'b*-', linewidth=2)
    plt.xlabel('No. of Congestions')
    plt.ylabel('No. of Locations')
    plt.savefig('data_stats/'+cityName+'_numCong2NumLocs.eps', format='eps', dpi=1000)
    #plt.show()
    plt.close()

    maxCongLocIdx, _ = max(enumerate([len(i) for i in data]), key=itemgetter(1))
    maxCongLoc = data[maxCongLocIdx]
    hour2NumCong = OrderedDict()
    for hr in range(24):
        hour2NumCong[hr] = 0
    for _, ts in maxCongLoc:
        hr = getDatetime(ts).hour
        hour2NumCong[hr] += 1
    plt.bar(hour2NumCong.keys(), hour2NumCong.values(), align='center')
    plt.xticks(hour2NumCong.keys(), [str(i) for i in hour2NumCong.keys()])
    plt.xlabel('Hour of the Day')
    plt.ylabel('No. of Congestions')
    plt.title('Plot for a location with Maximum Congestions')
    plt.savefig('data_stats/'+cityName+'_hourOfDay2NumCongs.eps', format='eps', dpi=1000)
    #plt.show()
    plt.close()

    latlng2Ratio, latlng2NumCng = getDurn2GapRatio(latlngList, data)
    plt.semilogy(latlng2Ratio.keys(), latlng2Ratio.values(), 'r.-')
    plt.xlabel('Location ID')
    plt.ylabel('Total Cong. Duration / Total Gap Duration')
    plt.title('Congestion Duration to Gap Duration Ratio for each Location')
    #plt.semilogy(latlng2NumCng.keys(), latlng2NumCng.values(), 'b+')
    #plt.xticks(latlng2Ratio.keys(), [str(i) for i in latlng2Ratio.keys()])
    plt.savefig('data_stats/'+cityName+'_congDurn2GapDurnRatio.eps', format='eps', dpi=1000)
    #plt.show()
    plt.close()


def main():
    filePath = sys.argv[1]
    latlngList, data = read(filePath)
    print('Mininum Timestamp:', getMinTimeStamp(data))
    print('Maximum Timestamp:', getMaxTimeStamp(data))
    print('Minimum Gap:', getMinGap(data))
    print('Maximum Gap:', getMaxGap(data))
    cityName = filePath.split('/')[-1].split('-')[0]
    generateStats(cityName, latlngList, data)

if __name__ == '__main__':
    main()
