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
LEARNING_RATE = 1e-4

# the number of epochs we want to train for
NUM_EPOCHS    = 100

BATCH_SIZE = 50

# we want to see status updates
VERBOSE       = True

#%%########################### TRAINING FUNCTION ##############################

# training function
def train(model, trainingSet, token2Idx, numEpochs, maxBin, maxTarget,
          learningRate = 0.001, verbose = True):
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    Param: 
        
        model        - the model we are training
        trainingSet  - the data set we are using to train
        token2Idx    - a dictionary that converts our tokens to indicies
        numEpochs    - the number of epochs we are going to train for
        maxBin       - the maximum length of our inputs so we can pad them
        maxTarget    - the maximum length of our targets so we can pad it
        learningRate - the learning rate we are going to train with
        verbose      - if we should print status updates
        
    Notes:
        
        trains the model for the specified number of epochs
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    # chooses the NLL loss function 
    cost = nn.CTCLoss(blank = len(token2Idx)-1)
    
    # creates an instance of the optimizer
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4)
    
    #creates an instance of a learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
	max_lr=LEARNING_RATE,
	steps_per_epoch=len(trainingSet) // BATCH_SIZE,
	epochs=NUM_EPOCHS,
	anneal_strategy='linear')

    # trains model for the number of epochs to train for
    for epoch in range(1, numEpochs+1):
        
        #initializes our error as 0
        err = 0.0
        
        # shuffles our training sset
        np.random.shuffle(trainingSet)
        
        #initializes defualt values for our backpropagation
        batchCtr = 0
        inputVec = torch.tensor([])
        target = torch.tensor([])
        input_lengths = torch.full((BATCH_SIZE,), 701, dtype=torch.long)
        target_lengths = torch.tensor([], dtype = torch.long)
        
        # gets the inputs and labels in the training data
        for inputSeq, out in trainingSet:

            # constructs the batch            
            if batchCtr < BATCH_SIZE:
                
                # pads the input vector
                inputV  = torch.tensor(inputSeq)
                size1, size2 = inputV.shape
                inputV = inputV.view(1, 1, size1, size2)
                pad = torch.zeros((1, 1, size1, maxBin - size2))
                inputV = torch.cat((inputV, pad), 3)
                
                # creates and pads the target vector
                target1    = torch.tensor([token2Idx[t] for t in out], 
                                          dtype = torch.long)
                target_lengths = torch.cat((target_lengths, 
                            torch.tensor([len(target1)], dtype=torch.long)), 0)
                target1 = target1.view(1, len(target1))
                pad = -1 * torch.ones((1, maxTarget - len(target1[0])), 
                                      dtype = torch.long)
                target1 = torch.cat((target1, pad), 1)
                
                # constructs batch
                if batchCtr > 0:
                    
                    inputVec = torch.cat((inputVec, inputV), 0)
                    target = torch.cat((target, target1), 0)
                else:
                    inputVec = inputV
                    target = target1
                    
                batchCtr += 1
                    
                
            else:
                
                # runs a forward pass through the model, stores the prediction
                pred = model(inputVec, BATCH_SIZE)
                pred = pred.transpose(0, 1)
                    
                # Resets the gradients
                model.zero_grad()
                
                # computes loss for this pass through the network
                loss = cost(pred, target, input_lengths, target_lengths)
                err += loss.item()
    
                # Backpropagates error in the network
                loss.backward()
                optimizer.step()
                scheduler.step()
                    
                # resets batch
                batchCtr = 0
                inputVec = torch.tensor([])
                target = []
                target_lengths = torch.tensor([], dtype = torch.long)


        # status update
        if verbose:
        
            # initializes success rate
            successRate = 0.0
            
            # # every 2 epochs
            # if epoch % 1 == 0:
                
            #     # initializes our counter for whether or not the model got
            #     # something right
            #     successCtr = 0
                
            #     # loops through our training set
            #     for inputSeq, target in trainingSet:
                    
            #         # prepares data
            #         # embeds the input vector
            #         inputVec  = torch.tensor(inputSeq)
            #         size1, size2 = inputVec.shape
            #         inputVec = inputVec.view(1, 1, size1, size2)
            #         pad = torch.zeros((1, 1, size1, maxBin - size2))
            #         inputVec = torch.cat((inputVec, pad), 3)
                    
            #         # runs our model in eval mode
            #         with torch.no_grad():
                        
                        
            #             # gets the prediction
            #             predtst = model(inputVec)
                        
            #             # if we predicted our guess correctly
            #             if np.argmax(predtst) == word2Idx[target]:
            #                 # incrments our counter
            #                 successCtr += 1
            #     # calculates our success rate
            #     successRate = successCtr / len(trainingSet)
            
            # prints a status update
            print('EPOCH {}/{}||LOSS = {}'.format(epoch,
            numEpochs,round(err/len(data),4)))#,round(successRate,4)))
            
            # saves the model state
            torch.save(model.state_dict(), "./States/state" + str(epoch))
    
# loads data
data, word2Idx, phenomeDict, maxBin, maxTarget = dl.loadData('./filePaths.txt', verbose = VERBOSE)

# prints update
if VERBOSE:
    print('Creating Model...')

# creates our model
model = Model.Model(dl.N_MFCC, maxBin, len(word2Idx))

# prints update
if VERBOSE:
    print('Training...')

# trains our model
train(model, data, word2Idx, NUM_EPOCHS, maxBin, maxTarget, LEARNING_RATE)