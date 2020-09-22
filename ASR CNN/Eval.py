
#loads dependencies
import torch
import numpy as np

from Model import Model

import DataLoader as dl

#constants from training
MAX_BIN = 2812
MAX_TARGET = 365
VOCAB_SIZE = 211

#loads data
data, word2Idx, phenomeDict, maxBin, maxTarget = dl.loadData('./filePaths.txt',
                                                             verbose = True)

#creates conversion back from indicies to words
idx2Word = dict()
for key in word2Idx:
    idx2Word[word2Idx[key]] = key

# loads a single data file
data = dl.loadSingleAudio('./data/84/121123/84-121123-0000.flac')

# creates model
model = Model(dl.N_MFCC, 701, VOCAB_SIZE)

model.load_state_dict(torch.load('./States/state100'))

model.eval()

# true value is "go do you hear"
inputV  = torch.tensor(data)
size1, size2 = inputV.shape
inputV = inputV.view(1, 1, size1, size2)
pad = torch.zeros((1, 1, size1, MAX_BIN - size2))
inputV = torch.cat((inputV, pad), 3)

pred = model(inputV, 1)

for guess in pred[0]:
    yhat = torch.argmax(guess[:-1])

    print(idx2Word[int(yhat)])
