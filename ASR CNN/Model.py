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
        
        self.conv1 = nn.Sequential( 
            nn.Conv2d(1, 32, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2) )
        
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        
        # initializes the lstm layer
        self.lstm = nn.LSTM(64*5, 256)

        self.layerNorm = nn.LayerNorm(256)

        self.fc   = nn.Sequential(nn.Linear(256, 512), nn.ReLU())
        

        # the lstm layer feeds into a single linear output layer so we
        # can run log softmax on the output
        self.linear = nn.Linear(512, vocabSize)

    # forward pass
    def forward(self, inputSequence, batchSize):
        
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        Param:
        
            inputSequence - a tensor of our inputs
    
        Return:
            
            output        - the model output with respect to each input
        
        Notes:
            
        forward pass of the model, makes a prediction for each input given
             
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        
        #convoltional block 1
        convOut = self.conv1(inputSequence)
        
        #dropout
        self.dropout1(convOut)
        
        #convoloutional block 2
        convOut = self.conv2(convOut)
        
        #output
        convOut = convOut.reshape(batchSize, 701, 64*5)
        
        # gets the ouput from the lstm layer, discards the hidden states
        lstmOut, hidden = self.lstm(convOut)

        self.layerNorm(lstmOut)

        # the output from the linear layer to reshape our data
        linear = self.fc(lstmOut)
        
        linear = self.linear(linear)

        # puts the ouput from the linear layer through a log softmax activation
        # function
        output = f.log_softmax(linear, dim = 2)

        # gets the output from the last linear layer
        return output
    