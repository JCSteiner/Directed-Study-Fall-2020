###############################################################################
#                                                                             #
#                                  Model.py                                   #
#                                 J. Steiner                                  #
#                                                                             #
###############################################################################

#%%########################## IMPORT DEPENDENCIES #############################

#Imports the core torch module
import torch
#Import the torch neural network module
import torch.nn            as nn
#Import torch functional modules
import torch.nn.functional as f
#Imports torch optimizer
import torch.optim         as optim

#Imports the ability to work with matricies easily
import numpy               as np

#%%######################### CREATION OF THE MODEL ############################

#The Model class, defines model structure
class Model(nn.Module):

    ###########################################################################
    # Name:  __init__
    # Param: embeddingDims
    #        hiddenDims
    #        vocabSize
    # Notes: class constructor, defines model instance variables and runes
    #        super class constructor
    def __init__(self, embeddingDims, hiddenDims, vocabSize):

        #Runs superclass constructor
        super(Model, self).__init__()

        #Stores the model word embeddings as an instance variable
        self.embeddings = nn.Embedding(vocabSize, embeddingDims)

        #the lstm layer
        self.lstm = nn.LSTM(embeddingDims, hiddenDims)

        #the lstm layer feeds into a single linear output layer so we
        #can run log softmax on the output
        self.linear = nn.Linear(hiddenDims, vocabSize)

    ###########################################################################
    # Name:   forward
    # Param:  inputSequence - the list of inputs to the model
    # Return: ouput         - the output of the model
    # Notes:  a forward pass through the model
    def forward(self, inputSequence):

        #Gets the word embeddings from the input sequence
        embeds = self.embeddings(inputSequence)

        #gets the ouput from the lstm layer, discards the hidden states
        lstmOut, hidden = self.lstm(embeds.view(len(inputSequence), 1, -1))


        #the output from the linear layer to reshape our data
        linear = self.linear(lstmOut.view(len(inputSequence), -1))

        #puts the ouput from the linear layer through a log softmax activation
        #function
        output = f.log_softmax(linear, dim = 1)

        #gets the output from the last linear layer
        return output, hidden