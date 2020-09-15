###############################################################################
#                                                                             #  
#                                  Trainer.py                                 # 
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%########################## IMPORT DEPENDENCIES #############################

# imports the core torch module, this is for basic functionality and data 
# structures
import torch
# import the torch neural network module, this is for the layers in the model
import torch.nn            as nn
# import torch functional module, this is for things like activation functions
import torch.nn.functional as f
# imports torch optimizer, this is for the cost function and optimizers for the
# model
import torch.optim         as optim

# imports the ability to work with matricies
import numpy as np

# imports our data loading and preprocessing functions
import DataLoader as dl

# imports our model class
import Model

#%%############################### CONSTANTS ##################################

# The dimensionality of our model
HIDDEN_DIMS   = 128

# the learning rate during training
LEARNING_RATE = 1e-2

# the number of epochs we want to train for
NUM_EPOCHS    = 100

# we want to see status updates
VERBOSE       = True

#%%########################### TRAINING FUNCTION ##############################

# training function
def train(model, trainingSet, token2Idx, numEpochs, 
          learningRate = 0.001, verbose = True):

    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    Param: 
        
        model        - the model we are training
        trainingSet  - the data set we are using to train
        numEpochs    - the number of epochs we are going to train for
        learningRate - the learning rate we are going to train with
        
    Notes:
        
        trains the model for the specified number of epochs
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    # chooses the NLL loss function 
    cost = nn.CrossEntropyLoss()
    # creates an instance of the optimizer
    optimizer = optim.SGD(model.parameters(), lr = learningRate)

    # trains model for the number of epochs to train for
    for epoch in range(1, numEpochs+1):

        # gets the inputs and labels in the training data
        for inputSeq, out in trainingSet:

            # Resets the gradients
            model.zero_grad()

            # embeds the input vector
            inputVec  = torch.tensor(inputSeq)
            # embeds the target vector
            target    = torch.tensor([token2Idx[out]], dtype = torch.long)

            # runs a forward pass through the model, stores the prediction
            pred = model(inputVec)

            # computes loss for this pass through the network
            loss = cost(pred[-1].view(1, len(word2Idx)), target)

            # Backpropagates error in the network
            loss.backward()
            optimizer.step()


        # status update
        if verbose:
        
            # initializes success rate
            successRate = 0.0
            
            # every 2 epochs
            if epoch % 2 == 0:
                
                # initializes our counter for whether or not the model got
                # something right
                successCtr = 0
                
                # loops through our training set
                for inputSeq, out in trainingSet:
                    
                    # runs our model in eval mode
                    with torch.no_grad():
                        
                        # prepares data
                        inputVec  = torch.tensor(inputSeq)
                        target    = torch.tensor([token2Idx[out]],
                                                 dtype = torch.long)
                        
                        # gets the prediction
                        pred = model(inputVec)
                        
                        # if we predicted our guess correctly
                        if np.argmax(pred[-1]) == target:
                            # incrments our counter
                            successCtr += 1
                # calculates our success rate
                successRate = successCtr / len(trainingSet)
            
            # prints a status update
            print('EPOCH {}/{}||LOSS = {}||SUCCESS RATE = {}'.format(epoch,
            numEpochs,round(loss.item(),4),round(successRate,4)))
            
            # saves the model state
            torch.save(model.state_dict(), "./States/state" + str(epoch))
    
# loads data
data, word2Idx, phenomeDict = dl.loadData('./filePaths.txt', verbose = VERBOSE)

# prints update
if VERBOSE:
    print('Creating Model...')

# creates our model
model = Model.Model(dl.N_MFCC, HIDDEN_DIMS, len(word2Idx))

# prints update
if VERBOSE:
    print('Training...')

# trains our model
train(model, data, word2Idx, NUM_EPOCHS, LEARNING_RATE)