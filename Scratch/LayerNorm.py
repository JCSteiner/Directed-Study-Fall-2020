###############################################################################
#                                                                             #
#                                   LayerNorm.py                              #
#                                    J. Steiner                               #
#                                                                             #
###############################################################################

#%%########################### LOAD DEPENDENCIES ##############################

# imports the ability to work with matrix functions easily
import numpy      as np

# imports the ability to work with our activation functions
import Functional as f

#%%################### LAYER NORMALIZATION IMPLEMENTATION #####################

class LayerNorm():

    def __init__(self, dims, normAxis = 1, epsilon = 1e-8):

        #######################################################################
        # Param: dims - the dimensionality of our features into the layer norm,
        #               this tells us the dimensions we need for our model
        #               parameters
        #

        # stores the axis we normalize over
        self.normAixs = normAxis

        # stores the dimensionality of our input
        self.dims = dims

        # initializes our model parameters of weights (gamma AKA gain) and bias
        # (beta)
        self.gamma = np.random.normal(np.power(dims, -0.5), size = (dims, 1))
        self.beta  = np.zeros((dims, 1))

        # stores our epilon, the constant we add to our variance just in case
        # we get a variance that rounds to 0
        self.epsilon = epsilon

        # initializes and zeros out the gradients for our model parameters
        self.zeroGrad()

    # zeros out model gradients
    def zeroGrad(self):

        # zeros out model gradients
        self.dL_dGamma = np.zeros_like(self.gamma)
        self.dL_dBeta  = np.zeros_like(self.beta)

    # forward pass through the layer normalization
    def forward(self, inputMatrix):

        #######################################################################
        # Param: inputMatrix - the input into the layer normalization

        # calculates the mean over the feature axis
        xbar = np.mean(inputMatrix, axis = 1,keepdims=True)

        # calculates the numerator of our standardization, xi-xbar
        self.stdNumerator = inputMatrix - xbar

        # calculates the variance of over our feature axis, we add epsilon just
        # in case we get a sample variance that rounds to 0
        self.variance  = np.sum(self.stdNumerator ** 2, axis = 1,keepdims=True)
        self.variance += self.epsilon

        # calculates our standard deviation as the square root of our variance
        self.stdev = np.sqrt(self.variance)

        # calculates our standardized inputs
        self.xhat = self.stdNumerator / self.stdev

        # weights and shifts our normalized output
        output = self.gamma * self.xhat + self.beta

        # returns our layer normalization output
        return output

    # runs a backward pass through our layer normalization
    def backward(self, dL_dNorm):

        # calculates the gradient of our bias (beta) and weights (gamma AKA
        # gain)
        self.dL_dBeta  += np.sum(dL_dNorm, axis = 1, keepdims = True)
        self.dL_dGamma += np.sum(dL_dNorm * self.xhat, axis = 1, keepdims=True)

        # calculates the gradient of our loss with respect to our normalized x
        dL_dXhat = dL_dNorm * self.xhat

        # calculates the gradients of our loss with respect to our variance
        dL_dVariance = dL_dXhat * np.sum(self.stdNumerator * (-0.5) *         \
                       (self.stdev ** -3), axis = 1, keepdims=True)

        # calculates the gradients of our loss with respect to our mean
        dL_dMu       = dL_dXhat * (-1. / self.stdev) + dL_dVariance *         \
                       np.sum(-self.stdNumerator,axis = 1, keepdims=True)

        # calculates the gradients of our loss with respect to our pre
        # normalized input
        dL_dX        = (dL_dXhat * ( 1. / self.stdev)) + (dL_dVariance *      \
                       ((2 * self.stdNumerator) / self.dims)) +               \
                       (dL_dMu / self.dims)

        # returns the gradient of our loss with respect to our pre normalized
        # input
        return dL_dX

    # steps model parameters
    def step(self, learningRate, batchSize):

        #######################################################################
        # Param: learningRate - how big we want our steps to be
        #        batchSize    - over how many gradients we are averaging

        # calculates the factor we scale down our gradients by
        scaleFactor = learningRate / batchSize

        # steps model parameters
        self.gamma -= scaleFactor * self.dL_dGamma
        self.beta  -= scaleFactor * self.dL_dBeta