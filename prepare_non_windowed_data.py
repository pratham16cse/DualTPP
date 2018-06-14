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
        df_list = list()
        for j in range(len(decoderInput[i])):
            df = list()
            for k in range(len(decoderInput[i][j])):
#                print('generateFeatures',i,j)
                df.append(add_features(decoderInput[i][j][k]))
            df_list.append(df)
        decoderFeatures.append(df_list)

    return encoderFeatures, decoderFeatures


def getEncoderDecoderData(filePath):
    print('Reading data from file . . .')
    latlngList, data = read(filePath)
    print('Sampling timestamps . . .')
    encoderInput, encoderOutput = sample(data)
    print('Generating Decoder Data . . .')
    generateDecoderDataStart = time()
    decoderInput, decoderOutput = generateDecoderData(data, encoderInput, encoderOutput)
    generateDecoderDataEnd = time()
    print('Generating Features . . .')
    generateFeaturesStart = time()
    encoderInput, decoderInput = generateFeatures(encoderInput, decoderInput)
    generateFeaturesEnd = time()
    for (i,j) in zip(encoderInput, decoderInput):
        print(np.array(i).shape, np.array(j).shape)
    print('-------------------------------------')
    print('generateDecoderData():', generateDecoderDataEnd - generateDecoderDataStart, 'seconds')
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
    generateFeaturesStart = time()
    encoderInput, decoderInput = generateFeatures(encoderInput, decoderInput)
    generateFeaturesEnd = time()
    for (i,j) in zip(encoderInput, decoderInput):
        print(np.array(i).shape, np.array(j).shape)
    print('-------------------------------------')
    print('generateDecoderData:', generateDecoderDataEnd - generateDecoderDataStart)
    print('generateFeatures:', generateFeaturesEnd - generateFeaturesStart)

    fileName = filePath.split('/')[-1].split('.')[0]
    pickle.dump([encoderInput, encoderOutput, decoderInput, decoderOutput], open('datasets/'+fileName, 'w'))


if __name__ == '__main__':
    main()
