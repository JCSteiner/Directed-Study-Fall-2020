###############################################################################
#                                                                             #
#                                    Main.py                                  #
#                                   J. Steiner                                #
#                                                                             #
###############################################################################

#%%########################## IMPORT DEPENDENCIES #############################

#Imports helper functions from another file
from Helpers import *

#Imports the model structure
from Model import *

#Imports the dataloader
from DataLoader import *

#imports random functionality
import random

#%%############################ MODEL TRAINING ################################

#the number of words we want to generate
numWords = 200

#the number of epochs we want to start training from
epochStart = 0

#how much of our dataset we want to load
numRowsLoad = 5

#word2Idx dictionary
word2Idx = dict()
#idx2Word dictionary
idx2Word = dict()

#loads in our dataset
data = load('./Data/stories.csv', numRowsLoad, word2Idx, idx2Word)

#creates an instance of the model
model = Model(EMBEDDING_DIMS, HIDDEN_DIMS, len(word2Idx))

#trains the model
train(model, data, NUM_EPOCHS, LEARNING_RATE, epochStart, word2Idx)

#tries to load the state dict we specified
model.load_state_dict(torch.load("./States/state100"))

#prints the model output
word = random.choice(list(word2Idx.keys()))
line = runModel(model, word, numWords, word2Idx, idx2Word)
outFile = open('output.txt', 'w')
outFile.write('------------------------------------------------------------\n')
for i in range(len(line)):
    outFile.write(line[i] + " ")
    if i % 20 == 0:
        outFile.write(" \n ")
    i += 1
outFile.write('------------------------------------------------------------\n')
outFile.close()