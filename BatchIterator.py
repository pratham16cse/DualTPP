import numpy as np
import random 

class EncoderInputIterator():
    def __init__(self, iterable):
        self.iterable = iterable
        self.size = len(iterable)
        self.NUM_FEATURES = len(iterable[0][0])
        self.cursor = 0
        self.lens = [len(i) for i in iterable]
    
    def reset(self):
        #random.shuffle(self.iterable)
        self.lens = [len(i) for i in self.iterable]
        self.cursor = 0

    def batch(self, n=1):
        b_list = self.iterable[self.cursor:self.cursor+n]
        b_lens = self.lens[self.cursor:self.cursor+n]
        max_len = max(b_lens)
        b_list_padded = np.zeros((len(b_list), max_len, self.NUM_FEATURES))
        for i,b in enumerate(b_list_padded):
            b[:b_lens[i],:] = b_list[i]
        #b = self.createTensor(b)
        if self.cursor+n >= self.size:
            self.reset()
        else:
            self.cursor += n
        return b_list_padded, b_lens

class EncoderOutputIterator():
    def __init__(self, iterable):
        self.iterable = iterable
        self.size = len(iterable)
        self.cursor = 0
        self.lens = [len(i) for i in iterable]
    
    def reset(self):
        #random.shuffle(self.iterable)
        self.lens = [len(i) for i in self.iterable]
        self.cursor = 0

    def batch(self, n=1):
        b_list = self.iterable[self.cursor:self.cursor+n]
        b_lens = self.lens[self.cursor:self.cursor+n]
        max_len = max(b_lens)
        b_list_padded = np.zeros((len(b_list), max_len))
        for i,b in enumerate(b_list_padded):
            b[:b_lens[i]] = b_list[i]
        #b = self.createTensor(b)
        if self.cursor+n >= self.size:
            self.reset()
        else:
            self.cursor += n
        return b_list_padded, b_lens

class DecoderInputIterator():
    def __init__(self, iterable):
        self.iterable = iterable
        self.size = len(iterable)
        self.NUM_LABELS = len(iterable[0][0])
        self.NUM_FEATURES = len(iterable[0][0][0])
        self.cursor = 0
        self.lens = [len(i) for i in iterable]
    
    def reset(self):
        #random.shuffle(self.iterable)
        self.lens = [len(i) for i in self.iterable]
        self.cursor = 0

    def batch(self, n=1):
        #print(n)
        b_list = self.iterable[self.cursor:self.cursor+n]
        b_lens = self.lens[self.cursor:self.cursor+n]
        max_len = max(b_lens)
        b_list_padded = np.zeros((len(b_list), max_len, self.NUM_LABELS, self.NUM_FEATURES))
        for i,b in enumerate(b_list_padded):
            b[:b_lens[i],:,:] = b_list[i]
        #b = self.createTensor(b)
        if self.cursor+n >= self.size:
            self.reset()
        else:
            self.cursor += n
        return b_list_padded, b_lens

class DecoderOutputIterator():
    def __init__(self, iterable):
        self.iterable = iterable
        self.size = len(iterable)
        self.NUM_LABELS = len(iterable[0][0])
        self.cursor = 0
        self.lens = [len(i) for i in iterable]
    
    def reset(self):
        #random.shuffle(self.iterable)
        self.lens = [len(i) for i in self.iterable]
        self.cursor = 0

    def batch(self, n=1):
        #print(n)
        b_list = self.iterable[self.cursor:self.cursor+n]
        b_lens = self.lens[self.cursor:self.cursor+n]
        max_len = max(b_lens)
        b_list_padded = np.zeros((len(b_list), max_len, self.NUM_LABELS))
        for i,b in enumerate(b_list_padded):
            b[:b_lens[i],:] = b_list[i]
        #b = self.createTensor(b)
        if self.cursor+n >= self.size:
            self.reset()
        else:
            self.cursor += n
        return b_list_padded, b_lens
