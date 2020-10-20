###############################################################################
#                                                                             #
#                                  Trainer.py                                 #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%########################### LOADS DEPENDENCIES #############################

# imports the ability to easily work with matricies
import numpy      as np

# imports the ability to load our data in using our data loading function
import Dataloader as dl

# imports the ability to use our model class that we created in a separate file
import Model      as m

#%%################################ CONSTANTS #################################

# how many dimensions we embed our tokens into
EMBEDDING_DIMS = 64

# how many dimensions the internals of our model has
MODEL_DIMS     = 64

# how much we need to scale down our gradients by
LEARNING_RATE  = 0.1

# how long we need to train our data
NUM_EPOCHS     = 100

#%%############################ ININTIALIZATION ###############################

# loads in our data
data, token2Idx, idx2Token = dl.loadData('./suess.txt')

# creates our model
model = m.Model(len(token2Idx), EMBEDDING_DIMS, MODEL_DIMS)

#%%########################### TRAINING FUNCTION ##############################
def train():

    # loops through each epoch we need to train
    for epoch in range(1, NUM_EPOCHS+1):

        # sets the cost increment to be zero
        cost = 0.0
        # zeros out model gradients
        model.zeroGrad()

        # loops through input and target in the dataset
        for inputs, targets in data:

            # converts our inputs and targets to one hot vectors
            x = dl.oneHotVec(inputs, token2Idx)
            y = dl.oneHotVec(targets, token2Idx)

            # forward pass through the model
            model.forward(x)

            # cross entropy loss
            cost += model.backward(y)

        # steps the model
        model.step(LEARNING_RATE, len(data))

        print('Epoch:', epoch, 'Cost:', cost / len(data))

# trains the model
train()