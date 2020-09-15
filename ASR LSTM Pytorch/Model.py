###############################################################################
#                                                                             #  
#                                   Model.py                                  #
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

#%%############################## MODEL CLASS #################################

#The Model class, defines model structure
class Model(nn.Module):

    # initializes model
    def __init__(self, inputDims, hiddenDims, vocabSize):

        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        Param:
         
            inputDims  - the dimensionality of our input
            hiddenDims - the dimensionality of our model
            vocabSize  - the dimensionality of our output
             
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        # runs the superclass constructor
        super(Model, self).__init__()

        # initializes the lstm layer
        self.lstm = nn.LSTM(inputDims, hiddenDims)

        # the lstm layer feeds into a single linear output layer so we
        # can run log softmax on the output
        self.linear = nn.Linear(hiddenDims, vocabSize)

    # forward pass
    def forward(self, inputSequence):
        
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        Param:
        
            inputSequence - a tensor of our inputs
    
        Return:
            
            output        - the model output with respect to each input
        
        Notes:
            
        forward pass of the model, makes a prediction for each input given
             
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        
        # gets the ouput from the lstm layer, discards the hidden states
        lstmOut, hidden = self.lstm(inputSequence.view(inputSequence.shape[1], 
                                                       1, -1))

        # the output from the linear layer to reshape our data
        linear = self.linear(lstmOut.view(inputSequence.shape[1], -1))

        # puts the ouput from the linear layer through a log softmax activation
        # function
        output = f.log_softmax(linear, dim = 1)

        # gets the output from the last linear layer
        return output
    