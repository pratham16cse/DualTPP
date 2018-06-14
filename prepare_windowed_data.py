import sys
import numpy as np
#from math import ceil
from reader import *
from feature import *
from time import time
import pickle

#NUM_POS_SAMPLES = 3
#NUM_NEG_SAMPLES = 10
#NUM_LABELS = 12
ENCODER_LEN = 100

def add_features(ts):
    #features = secondOfFeatures(ts)
    features = timestampTotime(ts)
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
        #for j in range(-ENCODER_LEN+1, len(encoderInput[i])-ENCODER_LEN+1, ENCODER_LEN):
        for j in range(-ENCODER_LEN+1, len(encoderInput[i])-ENCODER_LEN+1, 1):
#            print('createWindowedData',i,j)
            if j < 0:
                eI = [0]*(-j) + encoderInput[i][0:ENCODER_LEN+j]
                eO = [0]*(-j) + encoderOutput[i][0:ENCODER_LEN+j]
                #print(eI)
                #print(eO)
            else:
                eI = encoderInput[i][j:j+ENCODER_LEN]
                eO = encoderOutput[i][j:j+ENCODER_LEN]
            #wEncoderInput.append(encoderInput[i][j:j+ENCODER_LEN])
            #wEncoderOutput.append(encoderOutput[i][j:j+ENCODER_LEN])
            wEncoderInput.append(eI)
            wEncoderOutput.append(eO)
            wDecoderInput.append(decoderInput[i][j+ENCODER_LEN-1])
            wDecoderOutput.append(decoderOutput[i][j+ENCODER_LEN-1])

    return wEncoderInput, wEncoderOutput, wDecoderInput, wDecoderOutput


def getEncoderDecoderData(filePath):
    print('Reading data from file . . .')
    latlngList, data = read(filePath)
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
    filePath = sys.argv[1]
    latlngList, data = read(filePath)
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

    fileName = filePath.split('/')[-1].split('.')[0]
    pickle.dump([encoderInput, encoderOutput, decoderInput, decoderOutput], open('datasets/'+fileName, 'w'))


if __name__ == '__main__':
    main()
