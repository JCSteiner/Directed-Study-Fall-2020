

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

#%%############################### CONSTANTS ##################################

EMBEDDING_DIMS = 256
HIDDEN_DIMS    = 256

NUM_EPOCHS     = 100
LEARNING_RATE  = 0.01

VERBOSE        = True

#%%############################ HELPER FUNCTIONS ##############################

#Notes: prepares an input sequence of words that is already split over spaces
#       so it can be passed through the model
def words2Idx(wordList, word2Idx):

    ###########################################################################
    # Name:   words2Idx
    # Param:  wordList - the list of words to embed
    #         word2Idx - the dictionary to use to label the words with ints
    # Return: a torch tensor of the enumerated words
    # Notes:  prepares an input sentence, split over spaces, to be passed into the
    #         model

    #embeds each word in the word list
    words = [ word2Idx[word] for word in wordList ]

    #converts the embedding to a tensor and returns it
    return torch.tensor(words, dtype = torch.long)


###############################################################################
# Name:  train
# Param: model        - the model we are training
#        trainingSet  - the data set we are using to train
#        numEpochs    - the number of epochs we are going to train for
#        learningRate - the learning rate we are going to train with
def train(model, trainingSet, numEpochs, learningRate, e, word2Idx):

    #chooses the NLL loss function
    cost = nn.NLLLoss()
    #creates an instance of the optimizer
    optimizer = optim.Adam(model.parameters(), lr = learningRate)

    #tries to load the state dict we specified
    try:
        model.load_state_dict(torch.load("./States/state" + str(e)))
    except:
        print('No state is saved for epoch', e)

    #trains model for the number of epochs to train for
    for epoch in range(e+1, numEpochs+1):

        #gets the inputs and labels in the training data
        for ins, outs in trainingSet:

            #Resets the gradients
            model.zero_grad()

            #embeds the input vector
            inputVec  = words2Idx(ins, word2Idx)
            #embeds the target vector
            targetVec = words2Idx(outs, word2Idx)

            #runs a forward pass through the model, stores the prediction
            pred, _ = model(inputVec)

            #computes loss for this pass through the network
            loss = cost(pred, targetVec)

            #Backpropagates error in the network
            loss.backward()
            optimizer.step()

        #Prints the epoch and the loss every 10 epochs
        if VERBOSE:
            print('EPOCH {}/{}||LOSS = {}'.format(epoch,numEpochs,loss.item()))

            if epoch % 10 == 0:
                torch.save(model.state_dict(), "./States/state" + str(epoch))
                file = open("epoch.txt", "w")
                file.write(str(epoch))
                file.close


###############################################################################
# Name:   runModel
# Param:  model    - the model that we are running through
#         testWord - the word we are going to run through
#         numWords - how long our sequence will be
# Return: sentence - the sequence we will output
# Notes:  runs a forward pass through the network for a test word
def runModel(model, testWord, numWords, word2Idx, idx2Word):

    #Prepares to run the trained model
    with torch.no_grad():

        #the word we want to test
        newWord = [testWord]

        #outputs the length of the sequence
        for _ in range(numWords):

            #pre processes the input vector
            inputVec = words2Idx(newWord, word2Idx)

            #makes a model prediction
            pred, _ = model(inputVec)

            #gets what the predicted word is
            newWord.append(idx2Word[int(torch.argmax(pred[-1]))])

    #returns the predicted sentence
    return newWord