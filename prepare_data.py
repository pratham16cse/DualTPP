import sys
import numpy as np
from math import ceil
from feature import *
from time import time

NUM_POS_SAMPLES = 3
NUM_NEG_SAMPLES = 10
ENCODER_LEN = 100
NUM_LABELS = 12

def add_features(ts):
    features = secondOfFeatures(ts)
    return features

def generateFeatures(encoderInput, decoderInput):
    encoderFeatures = list()
    decoderFeatures = list()
    for i in range(len(encoderInput)):
        ef = list()
        for j in range(len(encoderInput[i])):
#            print('generateFeatures',i,j)
            ef.append(add_features(encoderInput[i][j]))
        encoderFeatures.append(ef)
    for i in range(len(decoderInput)):
        df = list()
        for j in range(len(decoderInput[i])):
#            print('generateFeatures',i,j)
            df.append(add_features(decoderInput[i][j]))
        decoderFeatures.append(df)

    return encoderFeatures, decoderFeatures

def createWindowedData(encoderInput, encoderOutput, decoderInput, decoderOutput):
    wEncoderInput, wEncoderOutput, wDecoderInput, wDecoderOutput = list(), list(), list(), list()
    for i in range(len(encoderInput)):
        for j in range(0, len(encoderInput[i])-ENCODER_LEN+1, ENCODER_LEN):
#            print('createWindowedData',i,j)
            #print(encoderInput[i][j:j+ENCODER_LEN], encoderOutput[i][j:j+ENCODER_LEN])
            #print(decoderInput[i][j+ENCODER_LEN-1], decoderOutput[i][j+ENCODER_LEN-1])
            wEncoderInput.append(encoderInput[i][j:j+ENCODER_LEN])
            wEncoderOutput.append(encoderOutput[i][j:j+ENCODER_LEN])
            wDecoderInput.append(decoderInput[i][j+ENCODER_LEN-1])
            wDecoderOutput.append(decoderOutput[i][j+ENCODER_LEN-1])

    return wEncoderInput, wEncoderOutput, wDecoderInput, wDecoderOutput

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


def sample(data):
    sampled_data = list()
    data_labels = list()
    for loc in data:
        sampled_loc = list()
        loc_labels = list()
        for i in range(len(loc)-1):
            durn, ts = loc[i]
            _, next_ts = loc[i+1]
            #pos_samples = sorted(np.random.randint(ts, ts+durn, size=NUM_POS_SAMPLES).tolist())
            #neg_samples = sorted(np.random.randint(ts+durn, next_ts, size=NUM_NEG_SAMPLES).tolist())
            pos_samples = sorted(np.arange(ts, ts+durn, step=ceil(durn*1.0/NUM_POS_SAMPLES)).astype(int).tolist())
            neg_samples = sorted(np.arange(ts+durn, next_ts, 
                step=ceil((next_ts-ts-durn)*1.0/NUM_NEG_SAMPLES)).astype(int).tolist())
            sampled_loc += pos_samples
            sampled_loc += neg_samples
            loc_labels += [1]*NUM_POS_SAMPLES
            loc_labels += [0]*NUM_NEG_SAMPLES
        sampled_data.append(sampled_loc)
        data_labels.append(loc_labels)

    return sampled_data, data_labels

def read(fileName):
    fp=open(fileName,'r')
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

    return latlngList, data

def getEncoderDecoderData(fileName):
    print('Reading data from file . . .')
    latlngList, data = read(fileName)
    print('Sampling timestamps . . .')
    encoderInput, encoderOutput = sample(data)
    print('Generating Decoder Data . . .')
    generateDecoderDataStart = time()
    decoderInput, decoderOutput = generateDecoderData(data, encoderInput, encoderOutput)
    generateDecoderDataEnd = time()
    print('Creating Windowed Data . . .')
    createWindowedDataStart = time()
    encoderInput, encoderOutput, decoderInput, decoderOutput =\
            createWindowedData(encoderInput, encoderOutput, decoderInput, decoderOutput)
    createWindowedDataEnd = time()
    print('Generating Features . . .')
    generateFeaturesStart = time()
    encoderInput, decoderInput = generateFeatures(encoderInput, decoderInput)
    generateFeaturesEnd = time()
    print(np.array(encoderInput).shape)
    print(np.array(decoderInput).shape)
    print('-------------------------------------')
    print('generateDecoderData():', generateDecoderDataEnd - generateDecoderDataStart, 'seconds')
    print('createWindowedData():', createWindowedDataEnd - createWindowedDataStart, 'seconds')
    print('generateFeatures():', generateFeaturesEnd - generateFeaturesStart, 'seconds')
    print('-------------------------------------')
    return encoderInput, encoderOutput, decoderInput, decoderOutput

def main():
    fileName = sys.argv[1]
    latlngList, data = read(fileName)
    #print(latlngList)
    #print(data)
    encoderInput, encoderOutput = sample(data)
    #print(data[-7], sampled_data[-7], sampled_labels[-7])
    generateDecoderDataStart = time()
    decoderInput, decoderOutput = generateDecoderData(data, encoderInput, encoderOutput)
    generateDecoderDataEnd = time()
    #for ts, d in zip(decoderInput[-7], decoderOutput[-7]):
        #print(zip(ts,d))
    print(np.array(encoderInput[-7]).shape)
    print(np.array(encoderOutput[-7]).shape)
    print(np.array(decoderInput[-7]).shape)
    print(np.array(decoderOutput[-7]).shape)
    print('-------------------------------------')
    createWindowedDataStart = time()
    encoderInput, encoderOutput, decoderInput, decoderOutput =\
            createWindowedData(encoderInput, encoderOutput, decoderInput, decoderOutput)
    createWindowedDataEnd = time()
    print(np.array(encoderInput).shape)
    print(np.array(encoderOutput).shape)
    print(np.array(decoderInput).shape)
    print(np.array(decoderOutput).shape)
    print('-------------------------------------')
    generateFeaturesStart = time()
    encoderInput, decoderInput = generateFeatures(encoderInput, decoderInput)
    generateFeaturesEnd = time()
    print(np.array(encoderInput).shape)
    print(np.array(decoderInput).shape)
    print('-------------------------------------')
    print('generateDecoderData:', generateDecoderDataEnd - generateDecoderDataStart)
    print('createWindowedData:', createWindowedDataEnd - createWindowedDataStart)
    print('generateFeatures:', generateFeaturesEnd - generateFeaturesStart)


if __name__ == '__main__':
    main()
