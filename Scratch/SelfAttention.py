###############################################################################
#                                                                             #
#                               SelfAttention.py                              #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%########################### LOAD DEPENDENCIES ##############################

# imports the ability to work with matricies and advanced math functions
import numpy      as np

# imports our helper activation functions
import Functional as f

#%%#################### SELF ATTENTION LAYER DEFINITION #######################
class SelfAttention():

    # class constructor
    def __init__(self, inputDims, modelDims):

        #######################################################################
        # Param: inputDims - the dimensionality of our input into the self
        #                    attention layer
        #        modelDims - the dimensionality of our model parameters, the
        #                    dimensionality within the model

        # calculates and stores the scaling factor to scale down our queries
        # and keys since we multiply them together
        self.scaleFactor = (1. / np.sqrt(modelDims))

        # initializing weights with random numbers
        self.queryWeights = np.random.normal(scale = np.power(modelDims, -0.5),
                                             size=(modelDims, inputDims))
        self.keyWeights   = np.random.normal(scale = np.power(modelDims, -0.5),
                                             size=(modelDims, inputDims))
        self.valueWeights = np.random.normal(scale = np.power(modelDims, -0.5),
                                             size=(modelDims, inputDims))

        # initializes our query, key, and value bias vectors as 0 vectors
        self.queryBias = np.zeros((modelDims, 1))
        self.keyBias   = np.zeros((modelDims, 1))
        self.valueBias = np.zeros((modelDims, 1))

        # initializes and zeros out all model gradients
        self.zeroGrad()

    # zeros out model gradients
    def zeroGrad(self):

        # zeros out model weight gradients
        self.dL_dQueryWeights = np.zeros_like(self.queryWeights)
        self.dL_dKeyWeights   = np.zeros_like(self.keyWeights)
        self.dL_dValueWeights = np.zeros_like(self.valueWeights)

        # zeros out model bias gradients
        self.dL_dQueryBias = np.zeros_like(self.queryBias)
        self.dL_dKeyBias   = np.zeros_like(self.keyBias)
        self.dL_dValueBias = np.zeros_like(self.valueBias)

    # forward pass through self attention
    def forward(self, inputMatrix):

        #######################################################################
        # Param: inputMatrix - the input sequence to our self attention layer

        # stores our input matrix for our backpropagation
        self.inputMatrix = inputMatrix

        # passes the input matrix through our query, key, and value weights
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
         the way that these are set up through the forward pass
         queries - we expect our queries at time step t to be how we weight
                   our input at time step t in the model
         keys    - we expect our keys at time step t to be how we measure our
                   input at time step t in it's interatction with the queries
         values  - we expect our values at time step t to be how we weight
                   our input at time step t in relation to every other value
                   knowing it will be given an attention score later
        '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        self.queries = np.matmul(self.queryWeights, inputMatrix) + self.queryBias
        self.keys    = np.matmul(self.keyWeights,   inputMatrix) + self.keyBias
        self.values  = np.matmul(self.valueWeights, inputMatrix) + self.valueBias

        # computes the weight matrix (transposed) prior to softmaxing
        weights_prime  = np.matmul(self.keys.T, self.queries) * self.scaleFactor

        # runs a column-wise softmax, instead of a row-wise like normally done,
        # this is because our weights_prime matrix is transposed from normal,
        # this is because we think of our input vectors as column vectors not
        # row vectors
        self.weights = f.softmax(weights_prime)

        # weights our values given the attention
        attn = np.matmul(self.values, self.weights)

        # reutrns our self attention output
        return f.relu(attn)

    # backwards pass through the
    def backward(self, dL_dAttn):

        dL_dAttn = f.relu_prime(dL_dAttn)

        # the gradient of loss function with respect to our values, we mutiply
        # our gradient with respect to the attention output by our transposed
        # weights because our attention output is our values times the weights
        dL_dValues  = np.matmul(dL_dAttn, self.weights.T)

        # calculates the gradient of our loss function with respect to our
        # value weights by backpropagating further through our values
        self.dL_dValueWeights += np.matmul(dL_dValues, self.inputMatrix.T)
        self.dL_dValueBias    += np.mean(dL_dValues, axis = 1, keepdims = True)

        # calculates the gradient of our loss function with respect to our
        # weights, this will help us find the gradient for our queries and keys
        dL_dWeights = np.matmul(dL_dAttn.T, self.values)

        # reshapes our weights as a column vector
        weights = dL_dWeights.reshape((-1, 1))

        # calculates the jacobian with respect to our softmax and stores the
        # gradient with respect to our weights as the diagonal of that jacobian
        jacobian = np.diagflat(weights) - np.matmul(weights, weights.T)
        dL_dWeights = np.diagonal(jacobian).reshape(self.weights.shape).T

        # using the gradient of our loss with respect to our weights
        # (including the chain rule with softmax), we calculate the gradient
        # of loss with respect to our queries and keys
        dL_dQueries = np.matmul(self.keys,    dL_dWeights)  * self.scaleFactor
        dL_dKeys    = np.matmul(self.queries, dL_dWeights)  * self.scaleFactor

        # increments our query and key weights and biases
        self.dL_dQueryWeights += np.matmul(dL_dQueries, self.inputMatrix.T)
        self.dL_dKeyWeights   += np.matmul(dL_dKeys, self.inputMatrix.T)
        self.dL_dQueryBias    += np.mean(dL_dQueries, axis = 1, keepdims=True)
        self.dL_dKeyBias      += np.mean(dL_dKeys, axis = 1, keepdims = True)

        # initializes the gradient of our loss with respect to our input into
        # the self attention layer
        dL_dX  = np.zeros_like(self.inputMatrix, dtype = np.float64)

        # increments the gradient of our loss with respect to our input into
        # the self attention layer by the loss with respect to the queries,
        # keys, and values
        dL_dX += np.matmul(self.queryWeights.T, dL_dQueries)
        dL_dX += np.matmul(self.keyWeights.T,   dL_dKeys)
        dL_dX += np.matmul(self.valueWeights.T, dL_dValues)

        # returns the loss with respect to our queries keys and values
        return dL_dX

    # steps model parameters
    def step(self, learningRate, batchSize):

        # determines how we should scale down our model gradients
        scaleFactor = learningRate / batchSize

        # decrements model weights
        self.queryWeights -= scaleFactor * self.dL_dQueryWeights
        self.keyWeights   -= scaleFactor * self.dL_dKeyWeights
        self.valueWeights -= scaleFactor * self.dL_dValueWeights

        # decrements model biases
        self.queryBias    -= scaleFactor * self.dL_dQueryBias
        self.keyBias      -= scaleFactor * self.dL_dKeyBias
        self.valueBias    -= scaleFactor * self.dL_dValueBias
