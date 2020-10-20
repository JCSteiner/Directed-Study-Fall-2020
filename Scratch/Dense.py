###############################################################################
#                                                                             #
#                                   Dense.py                                  #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%########################### LOAD DEPENDENCIES ##############################

# imports the ability to work
import numpy as np

#%%######################### DENSE LAYER DEFINITION ###########################
class Dense():

    # class constructor
    def __init__(self, inputDims, outputDims):

        #######################################################################
        # Param: inputDims  - the dimensionality of our input into the dense
        #                     layer
        #        outputDims - the dimensionality of our output into the dense
        #                     layer

        # initializes our weights from a random normal distribution with a mean
        # of 0 and a standard deviation of 1/sqrt(outputDims)
        self.weights = np.random.normal(scale = np.power(outputDims, -0.5),
                                        size = (outputDims, inputDims))

        # initializes our bias vectors as a 0 vector
        self.bias    = np.zeros((outputDims, 1))

        # initializes and zeros out our model gradients
        self.zeroGrad()

    # zeros out our model gradients
    def zeroGrad(self):

        # zeros out model gradients
        self.dL_dWeights = np.zeros_like(self.weights)
        self.dL_dBias    = np.zeros_like(self.bias)

    # runs a forward pass through our model gradient
    def forward(self, x):

        # stores the input into the model for backpropagtion
        self.x = x

        # calculates the dense output
        output = np.matmul(self.weights, x) + self.bias

        # returns the pre-activation function dense output
        return output

    # backwards pass through the dense layer
    def backward(self, dL_dDense):

        #######################################################################
        # Param: dL_dDense - the gradient of loss with respect to the dense
        #                    layer with the chain rule (activation function
        #                    prime(denseOutput) * dL_dDense) already performed

        # calculates the gradient of loss with respect to the model output
        self.dL_dWeights += np.matmul(dL_dDense, self.x.T)

        # calculates the gradient of the bias with respect to the model output
        self.dL_dBias    += np.sum(dL_dDense, axis = 1, keepdims = True)

        # calculates the gradient of loss with respect to the input into the
        # dense layer
        dL_dX = np.matmul(self.weights.T, dL_dDense)

        # returns the gradient of loss with respect to the input into the dense
        # layer
        return dL_dX

    # steps model parameters
    def step(self, learningRate, batchSize):

        # calculates the factor we scale down the input by
        scaleFactor = learningRate / batchSize

        # steps model parameters after scaling them down
        self.weights -= scaleFactor * self.dL_dWeights
        self.bias    -= scaleFactor * self.dL_dBias
