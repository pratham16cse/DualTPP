from feature import *
import numpy as np
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot


NUM_POS_SAMPLES = 2
NUM_NEG_SAMPLES = 5
NUM_LABELS = 48
MAX_STEPS = 1000
INTERVAL = 300

def get_date(epoch):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch))

def get_date_list(epoch_list):
    li=list()
    for epoch in epoch_list:
        li.append(get_date(int(epoch)))
    return li

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
#     pos_sam_list = list()
#     cong_sam_list = list()
#     for loc in data:
#         sampled_loc = list()
#         loc_labels = list()
#         for i in range(len(loc)):
#             durn, ts = loc[i]
#             cong=list()
#             #pos_samples = sorted(np.random.randint(ts, ts+durn, size=NUM_POS_SAMPLES).tolist())
#             pos_samples = sorted(np.arange(ts, ts+durn, \
#                     step=ceil(durn*1.0/NUM_POS_SAMPLES)).astype(int).tolist())
#             for i in range(len(pos_samples)-1):
#                 cong.append(pos_samples[i+1] - pos_samples[i])
#             cong.append(ts + durn- pos_samples[len(pos_samples)-1])
#             pos_sam_list += pos_samples
#             cong_sam_list += cong
#             sampled_loc += pos_samples
#             loc_labels += [1]*NUM_POS_SAMPLES
#             if i<len(loc)-1:
#                 _, next_ts = loc[i+1]
#                 #neg_samples = sorted(np.random.randint(ts+durn, next_ts, size=NUM_NEG_SAMPLES).tolist())
#                 neg_samples = sorted(np.arange(ts+durn, next_ts, \
#                         step=ceil((next_ts-ts-durn)*1.0/NUM_NEG_SAMPLES)).astype(int).tolist())
#                 sampled_loc += neg_samples
#                 loc_labels += [0]*NUM_NEG_SAMPLES
#                 cong_sam_list += [0]*NUM_NEG_SAMPLES
#         print(len(sampled_loc))
#         sampled_data.append(sampled_loc[:MAX_STEPS])
#         data_labels.append(loc_labels[:MAX_STEPS])
#         pos_sam_list=pos_sam_list[:MAX_STEPS]
#         cong_sam_list=cong_sam_list[:MAX_STEPS]
#     print(sampled_data)
#     return sampled_data, data_labels, pos_sam_list, cong_sam_list


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
        #if(len(sampled_loc)==MAX_STEPS): break

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
        #print(len(sampled_loc))
    return sampled_data, data_labels

def read(filePath):
    fp=open(filePath,'r')
    data=[]
    latlngList = []
    for line in fp:
        lineArray=line.rstrip().split('\t')
        latlng=lineArray[1].replace('[','').replace(']','').split(',')
        timeStampList=lineArray[2].replace('[','').replace(']','').split(',')
        durationList=lineArray[3].replace('[','').replace(']','').split(',')
        #print(timeStampList)
        #print(durationList)
        localData=[]
        for j in range(len(timeStampList)):
            eventFeed=[]
            eventFeed.append(int(durationList[j]))
            #eventFeed.append(0)
            eventFeed.append(int(timeStampList[j]))
            localData.append(eventFeed)
        data.append(localData)
        latlngList.append(latlng)
    #print(np.array(data).shape)
    #data=np.array(data)
    return latlngList, data


def get_first_congestion(model_predictions, encoderInput, pos_sam_list, cong_sam_list):
    encoderInput=np.array(encoderInput)
    print(encoderInput.shape)
    # For one sample
    timeStamp = encoderInput[0]
    model_pred = model_predictions[0]
    pred_list=list()
    cong_list=list()
    print(len(cong_sam_list))
    print(len(timeStamp))
    for i in range(len(model_pred)):
        ts=timeStamp[i]
        binary_pred=model_pred[i]
        index=-1
        for i in range(len(binary_pred)):
            if(binary_pred[i] == 1):
                index=i
                break
        if(index == -1):
            #cong_list.append(0)
            continue
        else:
            pred_list.append(ts + index*INTERVAL)
            cong = 0
            while index < len(binary_pred) and binary_pred[index] == 1:
                index = index + 1
                cong += INTERVAL
            cong_list.append(cong)
    print(len(pred_list))
    print(len(cong_list))
    # Merge ground truth and predictions
    merged_list = list()
    merged_prediction_list = list()
    merged_actual_list = list()
    index1=0
    index2=0
    while index1 < len(timeStamp) and index2 < len(pred_list):
        if(timeStamp[index1] == pred_list[index2]):
            merged_list.append(timeStamp[index1])
            merged_actual_list.append(cong_sam_list[index1])
            merged_prediction_list.append(cong_list[index2])
            index1 += 1
            index2 += 1
        elif(timeStamp[index1] < pos_sam_list[index2]):
            merged_list.append(timeStamp[index1])
            merged_prediction_list.append(0)
            merged_actual_list.append(cong_sam_list[index1])
            index1 += 1
        else:
            merged_list.append(pred_list[index2])
            merged_prediction_list.append(cong_list[index2])
            merged_actual_list.append(0)
            index2 += 1
    while(index1 < len(timeStamp)):
        merged_list.append(timeStamp[index1])
        merged_actual_list.append(cong_sam_list[index1])
        merged_prediction_list.append(0)
        index1 += 1
    while(index2 < len(pred_list)):
        merged_list.append(pred_list[index2])
        merged_prediction_list.append(cong_list[index2])
        merged_actual_list.append(0)
        index2 += 1

    series = pd.DataFrame({
        'Sampled Timestamp': get_date_list(merged_list[:100]),
        'Actual': merged_actual_list[:100],
        'Predicted': merged_prediction_list[:100]
    })
    series.set_index('Sampled Timestamp', inplace=True)
    filepath = os.getcwd() + '/experiments/images/rnn1.pdf'
    series.plot()
    plt.savefig(filepath)
    print(series)

def get_plot(model_predictions, decoder_output):
    print(model_predictions.shape)
    decoder_output=np.array(decoder_output)
    print(decoder_output.shape)
    last_actual = decoder_output[:,-1,:].reshape((1,NUM_LABELS))
    last_pred = model_predictions[:, -1, :].reshape((1,NUM_LABELS))
    x = list(range(1,NUM_LABELS+1,1))
    x1=list()
    y1=list()
    x2=list()
    y2=list()
    countx=0
    for i in range(len(last_actual)-1):
        cur = last_actual[i]
        next = last_actual[i+1]
        if(cur == 0 and next == 0):
            #countx += 1
            x1.append(countx)
            y1.append(0)
        elif(cur == 0 and next == 1):
            #countx += 1
            x1.append(countx)
            y1.append(0)
            countx += 1
            x1.append(countx)
            y1.append(0)
        elif(cur == 1 and next == 0):
            #countx += 1
            x1.append(countx)
            y1.append(0)
            x1.append(countx)
            y1.append(1)
            countx += 1
            x1.append(countx)
            y1.append(1)
        else:
            #countx += 1
            if(y1[:-1] != 1):
                x1.append(countx)
                y1.append(0)
            x1.append(countx)
            y1.append(1)
            countx += 1
            x1.append(countx)
            y1.append(1)
    countx=0
    for i in range(len(last_pred)-1):
        cur = last_pred[i]
        next = last_pred[i+1]
        if(cur == 0 and next == 0):
            #countx += 1
            if(y2[:-1] != 1):
                x2.append(countx)
                y2.append(0)
        elif(cur == 0 and next == 1):
            #countx += 1
            if(y2[:-1] != 1):
                x2.append(countx)
                y2.append(0)
            countx += 1
            if(y2[:-1] != 1):
                x2.append(countx)
                y2.append(0)
        elif(cur == 1 and next == 0):
            #countx += 1
            if(y2[:-1] != 1):
                x2.append(countx)
                y2.append(0)
            x2.append(countx)
            y2.append(1)
            countx += 1
            x2.append(countx)
            y2.append(1)
        else:
            #countx += 1
            if(y2[:-1] != 1):
                x2.append(countx)
                y2.append(0)
            x2.append(countx)
            y2.append(1)
            countx += 1
            x2.append(countx)
            y2.append(1)


    #print(x1.shape)
    print(last_pred[:])
    print(last_actual[:])
    #plt.plot(x2[:30], y2[:30], label='actual')
    #plt.plot(x2, y2)
    # plt.scatter(x,last_actual)
    # plt.scatter(x,last_pred)
    xt = list(range(1,len(last_actual),2))
    # plt.xticks(xt)
    # plt.matshow(last_actual, interpolation=None, aspect='auto', fignum=10, cmap=plt.cm.Blues)
    # plt.matshow(last_pred, interpolation=None, aspect='auto', cmap=plt.cm.Reds)
    plt.pcolor(last_actual-last_pred, cmap='Blues')
    #plt.show()
    filepath=os.getcwd()+'/experiments/images/heatmap_4_200.pdf'
    plt.savefig(filepath,bbox_inches='tight')