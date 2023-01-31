import random
import torch
import numpy as np
import re
import pickle

def batchify(data):
    truth, inp, dates = [], [], []
    for x, y, d in data:
        truth.append(y)
        inp.append(x)
        dates.append(d)

    truth = torch.FloatTensor(np.array(truth))
    inp = torch.FloatTensor(np.array(inp)).squeeze(3)
    truth = truth.transpose(0, 1).contiguous()
    inp = inp.transpose(0, 1).contiguous()
    return inp, truth, dates


class DataLoader(object):
    def __init__(self, filename, batch_size):
        self.batch_size = batch_size
        self.filename = filename
        with open(filename, 'rb') as f:
            _, self.data_list, self.f_max, self.f_min = pickle.load(f)
        self.epoch_id = 0

    def __iter__(self):

        random.shuffle(self.data_list)
        idx = 0
        while idx < len(self.data_list):
            yield batchify(self.data_list[idx:idx+self.batch_size])
            idx += self.batch_size

        self.epoch_id += 1
