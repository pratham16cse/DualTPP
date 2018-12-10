from feature import *
import numpy as np
from math import ceil

NUM_POS_SAMPLES = 10
NUM_NEG_SAMPLES = 30
NUM_LABELS = 48
MAX_STEPS = 22400
INTERVAL = 300



def add_features(ts):
    #features = secondOfFeatures(ts)
    features = timestampTotime(ts)
    return features

def generateDecoderData(eventData, data, labels):
    for i in range(len(eventData)):
        for j in range(len(eventData[i])):
            eventData[i][j] = [eventData[i][j][1], eventData[i][j][1] + eventData[i][j][0]]
        eventData[i] = [item for sublist in eventData[i] for item in sublist]

    decoder_ts = list()
    decoder_labels = list()
    for i in range(len(data)):
        eventItr1 = 0
        loc = data[i]
        ts_loc = list()
        labels_loc = list()
        for j in range(len(loc)):
#            print('generateDecoderData',i,j)
            ts = loc[j]
            if ts >= eventData[i][eventItr1+1]:
                eventItr1 += 1
            #ts_labels = np.zeros(NUM_LABELS)
            fut_ts_list = list()
            ts_labels = list()
            eventItr2 = eventItr1
            for fut_ts in range(ts+300, ts+300+NUM_LABELS*300, 300):
                fut_ts_list.append(fut_ts)
                if eventItr2+1 >= len(eventData[i]):
                    ts_labels.append(0)
                elif fut_ts < eventData[i][eventItr2+1]:
                    ts_labels.append(abs(1-eventItr2%2))
                else:
                    eventItr2 += 1
                    ts_labels.append(abs(1-eventItr2%2))
            ts_loc.append(fut_ts_list)
            labels_loc.append(ts_labels)
        decoder_ts.append(ts_loc)
        decoder_labels.append(labels_loc)

    return decoder_ts, decoder_labels


# def sample(data):
#     #print(data)
#     sampled_data = list()
#     data_labels = list()
#     for loc in data:
#         sampled_loc = list()
#         loc_labels = list()
#         for i in range(len(loc)):
#             durn, ts = loc[i]
#             #pos_samples = sorted(np.random.randint(ts, ts+durn, size=NUM_POS_SAMPLES).tolist())
#             pos_samples = sorted(np.arange(ts, ts+durn, \
#                     step=ceil(durn*1.0/NUM_POS_SAMPLES)).astype(int).tolist())
#             sampled_loc += pos_samples
#             loc_labels += [1]*NUM_POS_SAMPLES
#             if i<len(loc)-1:
#                 _, next_ts = loc[i+1]
#                 #neg_samples = sorted(np.random.randint(ts+durn, next_ts, size=NUM_NEG_SAMPLES).tolist())
#                 neg_samples = sorted(np.arange(ts+durn, next_ts, \
#                         step=ceil((next_ts-ts-durn)*1.0/NUM_NEG_SAMPLES)).astype(int).tolist())
#                 sampled_loc += neg_samples
#                 loc_labels += [0]*NUM_NEG_SAMPLES
#         sampled_data.append(sampled_loc[:MAX_STEPS])
#         data_labels.append(loc_labels[:MAX_STEPS])
#
#     return sampled_data, data_labels

def sample(data):
    #print(data)
    sampled_data = list()
    data_labels = list()
    end_ts = 0
    end_dur = 0
    start_ts = 0
    for loc in data:
        sampled_loc = list()
        loc_labels = list()
        residual = 0
        if(len(loc) == 1):
            durn, ts = loc[0]
            end_ts = ts
            start_ts = ts
            end_dur = durn
        for i in range(len(loc)-1):
            durn, ts = loc[i]

            next_cong_durn, next_cong = loc[i+1]
            ts = ts - residual
            next_ts = ts
            while(next_ts < ts + durn and next_ts < next_cong):
                sampled_loc.append(next_ts)
                loc_labels.append(1)
                next_ts = next_ts + INTERVAL
            if(next_ts < next_cong):
                while(next_ts < next_cong):
                    sampled_loc.append(next_ts)
                    loc_labels.append(0)
                    next_ts = next_ts + INTERVAL
            residual = next_cong - (next_ts - INTERVAL)
            if(residual == INTERVAL):
                residual = 0
            #print(residual, (next_cong-ts)%INTERVAL)
            assert(residual == (next_cong - ts)%INTERVAL)
            end_dur = next_cong_durn
            end_ts = next_cong
            start_ts = next_ts
        while(start_ts <= end_ts + end_dur):
            sampled_loc.append(start_ts)
            loc_labels.append(1)
            start_ts = start_ts + INTERVAL
        sampled_data.append(sampled_loc[:MAX_STEPS])
        data_labels.append(loc_labels[:MAX_STEPS])
        print(len(sampled_loc))
    return sampled_data, data_labels

def read(filePath):
    fp=open(filePath,'r')
    data=list()
    latlngList = list()
    for line in fp:
        lineArray=line.rstrip().split('\t')
        latlng=lineArray[1].replace('[','').replace(']','').split(',')
        timeStampList=lineArray[2].replace('[','').replace(']','').split(',')
        durationList=lineArray[3].replace('[','').replace(']','').split(',')
        #print(timeStampList)
        #print(durationList)
        localData=list()
        for j in range(len(timeStampList)):
            eventFeed=list()
            eventFeed.append(int(durationList[j]))
            #eventFeed.append(0)
            eventFeed.append(int(timeStampList[j]))
            localData.append(eventFeed)
        data.append(localData)
        latlngList.append(latlng)

    assert len(data) == len(latlngList)

    return latlngList, data
