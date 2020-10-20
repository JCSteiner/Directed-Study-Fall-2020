###############################################################################
#                                                                             #
#                                   Model.py                                  #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%########################### LOADS DEPENDENCIES #############################

# imports the ability to work with matricies and advanced math functions
import numpy as np

# import self attention layer class
from SelfAttention import SelfAttention

# import dense layer class
from Dense         import Dense

# imports layer normalization class
from LayerNorm     import LayerNorm

# imports activation functions and their derivatives
import Functional as f

#%%##################### ENCODER BLOCK CLASS DEFINITION #######################
class Enocder_Block():

    # class constructor
    def __init__(self, vocabSize, inputDims, modelDims):

        #######################################################################
        # Param: vocabSize - the number of possible inputs and outputs
        #        inputDims - the dimensionality of our input into the block
        #        modelDims - the inner dimensionality of our model

        # initilaizes our attention layer
        self.attentionLayer  = SelfAttention(inputDims, modelDims)

        # layer normalization layer
        self.norm1           = LayerNorm(modelDims)
        # feed forward network
        self.ff1             = Dense(modelDims, modelDims)
        # feed forward network
        self.ff2             = Dense(modelDims, modelDims)

        # initializes and zeros out model gradients
        self.zeroGrad()

    # zeros out model gradients
    def zeroGrad(self):

        # zeros out gradients of each layer
        self.norm1.zeroGrad()
        self.ff1.zeroGrad()
        self.ff2.zeroGrad()
        self.attentionLayer.zeroGrad()

    # forward pass through the model
    def forward(self, x):

        #######################################################################
        # Param: x - the input into the model

        # gets the attention layer
        attention   = self.attentionLayer.forward(x)

        # first feed forward
        self.ff1Out = f.tanh(self.ff1.forward(attention))
        # second feed forward
        self.ff2Out = f.tanh(self.ff2.forward(self.ff1Out))
        # layer normalization
        norm1       = self.norm1.forward(self.ff2Out)

        # returns model output
        return norm1

    # backward pass through the model
    def backward(self, dL_dBlock):

        #######################################################################
        # Param: y - the target output of the model

        # backwards pass thorugh layer normalizations and feed forward networks
        dL_dFF2  = self.norm1.backward(dL_dBlock)
        dL_dFF1  = f.tanh_prime(self.ff2Out) * self.ff2.backward(dL_dFF2)
        dL_dAttn = f.tanh_prime(self.ff1Out) * self.ff1.backward(dL_dFF1)

        # finds the gradient with respect to the embedding's loss and updates
        # the attention layer's gradients weights and bias gradients
        dL_dEmbeds = self.attentionLayer.backward(dL_dAttn)

        # returns the loss of the given batch
        return dL_dEmbeds

    # steps model parameters
    def step(self, learningRate, batchSize):

        # steps the layer normalizations
        self.norm1.step(learningRate, batchSize)

        # steps the feed forward networks
        self.ff1.step(learningRate, batchSize)
        self.ff2.step(learningRate, batchSize)

        # steps attention layer's parameters
        self.attentionLayer.step(learningRate, batchSize)

#%%######################### MODEL CLASS DEFINITION ###########################

class Model():

    def __init__(self, vocabSize, embeddingDims, modelDims):

        # initializes our embedding layer
        self.embeddingLayer = Dense(vocabSize, embeddingDims)

        # initializes our encoder style blocks
        self.block1 = Enocder_Block(vocabSize, embeddingDims, modelDims)
        self.block2 = Enocder_Block(vocabSize, modelDims, modelDims)
        self.block3 = Enocder_Block(vocabSize, modelDims, modelDims)

        # initializes our fully connected classifier layer
        self.classifier      = Dense(modelDims, vocabSize)

        # initializes model gradients
        self.zeroGrad()

    # zeros out model gradients
    def zeroGrad(self):

        # zeros out the embedding gradient
        self.embeddingLayer.zeroGrad()

        # zeros out the encoder block gradients
        self.block1.zeroGrad()
        self.block2.zeroGrad()
        self.block3.zeroGrad()

        # zeros out the classifier gradients
        self.classifier.zeroGrad()

    # forward pass through the model
    def forward(self, x):

        #######################################################################
        # Param: x - the input sequence into the model, of shape (feature,time)

        # gets word embeddings
        embeddings     = self.embeddingLayer.forward(x)

        # forward encoder style blocks
        self.blockOut1 = self.block1.forward(embeddings)
        self.blockOut2 = self.block2.forward(self.blockOut1)
        self.blockOut3 = self.block3.forward(self.blockOut2)

        # classifies based on our attention
        classified      = self.classifier.forward(self.blockOut3)
        #stores model output
        self.modelOut   = f.softmax(classified)

        return self.modelOut

    # backwards pass through the model
    def backward(self, y):

        #######################################################################
        # Param: y - the target output of the model

        # calculates the loss with respect to the current batch by averaging
        # error over time
        loss = -np.sum( y * np.log(self.modelOut)) / self.modelOut.shape[1]

        # finds the gradient with respect to cross entropy loss
        dL_dZ      = self.modelOut - y

        # finds the gradient with resepct to the attention's loss and upadates
        # the classifiers weight and bias gradients
        dL_dBlock3   = self.classifier.backward(dL_dZ)

        # backwards pass through the encoder style blocks, each block
        # backward pass is followed by the application of the chain rule of
        # calculus
        dL_dBlock2   = self.block2.backward(dL_dBlock3)
        dL_dBlock1   = self.block2.backward(dL_dBlock2)
        dL_dEmbeds   = self.block1.backward(dL_dBlock1)

        # updates the embedding layer's gradients
        self.embeddingLayer.backward(dL_dEmbeds)

        # returns the loss for a given pass
        return loss

    # steps model parameters
    def step(self, learningRate, batchSize):

        # steps the embedding layer's parameters
        self.embeddingLayer.step(learningRate, batchSize)

        # steps the parameters for each encoder style block
        self.block1.step(learningRate, batchSize)
        self.block2.step(learningRate, batchSize)
        self.block3.step(learningRate, batchSize)

        # steps the parameters for the classifier layer
        self.classifier.step(learningRate, batchSize)