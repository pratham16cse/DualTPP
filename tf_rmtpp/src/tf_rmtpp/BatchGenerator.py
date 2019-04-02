from collections import OrderedDict
from itertools import islice
import numpy as np
import random

class BatchGenerator:
    def __init__(self, iterable, batchSeqLen=16, shuffle_batches=False, seed=None, batch_pad=True):
        self.iterable = iterable
        #if self.iterable!=None:
        self.batchSeqLen = batchSeqLen
        # self.stepLen = stepLen
        keys = range(len(self.iterable))
        self.seed = seed
        self.batch_pad = batch_pad
        self.shuffle_batches = shuffle_batches
        if self.shuffle_batches:
            random.Random(self.seed).shuffle(keys)
            self.seed = self.seed + 1
        self.seqLengths = OrderedDict()
        self.cursorDict = OrderedDict()
        for i in keys:
            self.seqLengths[i] = len(self.iterable[i])
            self.cursorDict[i] = 0
        self.iterFinished = 0

    def reset(self):
        keys = range(len(self.iterable))
        if self.shuffle_batches:
            random.Random(self.seed).shuffle(keys)
            self.seed = self.seed + 1
        self.seqLengths = OrderedDict()
        self.cursorDict = OrderedDict()
        for i in keys:
            self.seqLengths[i] = len(self.iterable[i])
            self.cursorDict[i] = 0

        #self.iterFinished += 1

    def nextBatch(self, batchSize=1, stepLen=1):

        tsIndices = list(islice(self.cursorDict.keys(), batchSize))
        startIndices = [self.cursorDict[i] for i in tsIndices]
        seqLens = [self.seqLengths[i] for i in tsIndices]
        batch = list()
        startingTs = list()
        finishedSequences = list()
        for tsInd, startInd, seqLen in zip(tsIndices, startIndices, seqLens):

            batch.append(self.iterable[tsInd][startInd:startInd+self.batchSeqLen])

            if startInd == 0:
                startingTs.append(tsInd)
    
            if startInd+stepLen+self.batchSeqLen > seqLen:
                finishedSequences.append(tsInd)
            else:
                self.cursorDict[tsInd] += stepLen

        endingTs = finishedSequences

        for index in finishedSequences:
            del self.cursorDict[index]
            del self.seqLengths[index]

        batch = np.array(batch)
        mask = np.ones((batchSize, 1))
        if batch.shape[0] < batchSize:
            mask[batch.shape[0]:] = 0
            tsIndices = tsIndices + [0] * (batchSize-batch.shape[0])
            startingTs = startingTs + [0] * (batchSize-batch.shape[0])
            endingTs = endingTs + [0] * (batchSize-batch.shape[0])
            if self.batch_pad:
                batch = np.concatenate([batch, 
                                        np.zeros(([batchSize-batch.shape[0]] \
                                                   + list(batch.shape[1:])))],
                                        axis=0)

        if not self.cursorDict:
            self.iterFinished += 1
            #self.reset()

        return batch, tsIndices, startingTs, endingTs, np.array(mask, dtype=np.bool)

if __name__ == "__main__":
    iterable = [[[11],[12],[13],[14],[15],[16]],
                [[21],[22],[23],[24]],
                [[31],[32],[33]],
                [[41],[42],[43],[44],[45],[46],[47],[48],[49],[50]]]

    seed = 6
    itr = BatchGenerator(iterable, batchSeqLen=3, shuffle_batches=True, seed=seed)
    print(itr.cursorDict.items())
    print(itr.seqLengths.items())
    print('----------')
    print(itr.iterable)
    print('----------')
    while itr.iterFinished < 5:
        currItr = itr.iterFinished
        while currItr == itr.iterFinished:
            batch, tsIndices, startingTs, endingTs, mask = itr.nextBatch(batchSize=2, stepLen=1)
            print('Batch:', batch)
            print('tsIndices:', tsIndices)
            print('StartingTs:', startingTs)
            print('EndingTs:', endingTs)
            print(currItr,itr.iterFinished)
            print('-------')
        itr.reset()
        print('----Batch {} finished'.format(currItr))
