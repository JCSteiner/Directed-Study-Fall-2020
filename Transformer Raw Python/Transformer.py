
#%%########################## IMPORT DEPENDENCIES #############################

# Numpy is a library for python that will give us the ability to work with
# matricies and do things like matrix multiplication, the hadamard
# (element-wise) product, and transposing. This will also allow us to do
# inutitive things with matricies, like add and subtract them, without the need
# for nested for loops
import numpy as np


#%%########################### TRANSFORMER MODEL ##############################

class Transformer():

    # CLASS CONSTRUCTOR
    def __init__(self, dims, vocabSize, numHeads, numBlocks):

        # Param: dims      - the dimensionality we transform our input vectors
        #                    to this is also the dimensionality that a
        #                    transformer block outputs to
        #        vocabSize - the size of our vocab we draw input vectors from
        #                    this is the number of possible input types we
        #                    could have, as well as the number of inputs we
        #                    will cast the final output to
        #        numBlocks - the number of transformer blocks we have in our
        #                    model. Having many blocks can help calculate more
        #                    complex relationships between inputs

        #Stores the parameters of our network as instance variables
        self.dims      = dims
        self.vocabSize = vocabSize
        self.numBlocks = numBlocks
        self.numHeads  = numHeads

        # Initializes a matrix for our initial input embeddings, this will
        # transform from the dimensionality of the vocab to the dimensionality
        # of our transformer
        self.embeddingMat = np.random.uniform( 0, 1, (vocabSize, dims) )

        # Self Attention:
        # Self-attention mechanisms have 3 ways to weight the input.
        # They are called queries, keys, and values. Queries, keys, and values
        # are all learned ways to weight your input vector. The queries and
        # keys are then multiplied together to weight the output of the value
        # matrix and the input vector. This is the self attention mechanism.
        # Instead of having a query, key, and value matrix for each attention
        # head and then concatenating all the attention heads and transforming
        # them back into the same dimensions, we have matricies that are
        # numHeads times as tall as they need to be. In doing this, we
        # pre-concatenate our values so we can be more memory efficient in
        # transforming them back to the correct dimensionality.

        # Defines our query, key, value, and unifyHeads matricies lists. We need
        # a separate matrix for each block
        self.queryMats = []
        self.keyMats   = []
        self.valueMats = []
        self.unifyMats = []

        # Defines the height of our pre-concatenated matricies
        matHeight = dims * numHeads

        # We perform the following operations, appending and intiailizing,
        # numBlocks number of times
        for _ in range(numBlocks):

            # Defines our query, key, and value matricies
            self.queryMats.append( np.random.uniform( -1, 1,              \
                                                    ( matHeight, dims ) ) )
            self.keyMats.append(   np.random.uniform( -1, 1,              \
                                                    ( matHeight, dims ) ) )
            self.valueMats.append( np.random.uniform( -1, 1,              \
                                                    ( matHeight, dims ) ) )

            # Defines the matrix that will unify our heads, transforming our
            # output into the correct dimensionality
            self.unifyMats.append( np.random.uniform( -1, 1,              \
                                                    ( dims, matHeight ) ) )

        self.outputWeights = np.random.uniform(-1, 1, ( vocabSize, dims ) )

        #Initializes and zeros out our gradients
        self.zeroGrad()

    def zeroGrad(self):

        self.loss = 0.0

        # Initializes our gradients
        self.dL_dZ              = np.zeros(      self.vocabSize     )
        self.dL_dOutputWeights  = np.zeros_like( self.outputWeights )
        self.dL_dUnifyMats      = np.zeros_like( self.unifyMats[-1] )

        # Initializes lists to store the values of the gradient of many weight
        #  matricies within the transformer block
        self.dL_dUnifyMatsLst = [ np.zeros_like(self.unifyMats[-1]) ] * self.numBlocks
        self.dL_dQueriesLst   = [ np.zeros_like(self.queryMats[-1]) ] * self.numBlocks
        self.dL_dKeysLst      = [ np.zeros_like(self.keyMats[-1]) ] * self.numBlocks
        self.dL_dValuesLst    = [ np.zeros_like(self.valueMats[-1]) ] * self.numBlocks

    #Here we define public lists that we will append to. They are needed for the learning algorithm we employ.
    #If this was in a class structure, they would be instance variables

    def fwdPass(self, inSeq):
        # Param: inSeq - the sequence of labels we will pass into the transformer

        # stores the values of queries, keys, and values for each block
        self.blockQueries = []
        self.blockKeys    = []
        self.blockValues  = []

        # stores the weighted values of the transformer blocks
        self.weightedValues = []

        # stores the weighted output values once they are passed through the
        # unify heads layer
        self.blockOutputs = []

        #First extracts the embedded input from the x sequence
        embeddings = [ self.embeddingMat[x] for x in inSeq ]

        #initializes the block output list
        self.blockOutputs.append(embeddings)

        #Loops through each block of the transformer
        for block in range(self.numBlocks):

            #Initializes lists to store the queries, keys, and value vectors for each vector in the sequence x
            queries = []
            keys    = []
            values  = []

            #Loops through each x label in the input sequence passed into the transformer
            for x in self.blockOutputs[ -1 ]:

                #Calculates this embedded vectors query, key, and value
                q = np.dot(self.queryMats[block], x)
                k = np.dot(self.keyMats[block],   x)
                v = np.dot(self.valueMats[block], x)

                #Our scale factor is the square root of the query and key dimensionality
                scaleFactor = self.dims ** (1/2)

                #Before we save these vectors, we need to scale down the query and key vector by their scale factor
                q /= scaleFactor
                k /= scaleFactor

                #Appends the calculated query, key and value vectors
                queries.append( q )
                keys.append(    k )
                values.append(  v )

            self.blockQueries.append(queries)
            self.blockKeys.append(keys)
            self.blockValues.append(values)

            #Initializes the weight matrix for our weighted value vectors, we want to initialize it here so we can index
            #into it without error below. We use the mathematical notation and loops for this to make an easier to understand
            #implementation
            weights = np.zeros( ( len( self.blockOutputs[ -1 ] ), len( self.blockOutputs[ -1 ] ) ) )

            #Stores the weighted value vectors
            y = []

            #We want to compare each query to every other key and take their dot product, this will give us raw self attention
            for i, q in enumerate(queries):
                for j, k in enumerate(keys):
                    #compares each query q to ever other key j
                    weights[i][j] = np.dot(q, k)

            #We already calculated our raw weights, now we need to normalize those weights by passing them through the softmax function
            self.weights_prime = weights
            self.weights = np.exp(weights) / np.sum(np.exp(weights), axis = 1)
            weightedValuesBlock = []
            #For each value vector
            for i in range(len(inSeq)):

                #initializes the weighted values as an array of zeros
                weightedValue = np.zeros_like(values[i])

                #sums the weights of values over j
                for j in range(len(inSeq)):
                    for h in range(self.numHeads):
                        idx = h * self.dims
                        #We first calculate the weighted value vector
                        weightedValue[idx:idx+self.dims] += weights[i][j] * values[i][idx:idx+self.dims]

                #Appends to the weighted values list
                weightedValuesBlock.append( weightedValue )

                #Append the weighted value vector to the y vector
                y.append( np.dot( self.unifyMats[block], weightedValue ) )


            #TODO normalize layer

            self.weightedValues.append(weightedValuesBlock)
            #at the end of the block, append the block output to the block outputs list
            self.blockOutputs.append(y)

        #stores the output vectors of the entire transformer
        modelOutputs = []

        self.yFinal = np.zeros_like(self.blockOutputs[-1][-1])

        #Loops through each y vector in the block outputs we have just appended to
        for y in self.blockOutputs[ -1 ]:

            self.yFinal += y

        #The intermediate activation of casting this y vector to the
        activation = np.dot( self.outputWeights, self.yFinal )

        #Appends to the output vector after applying the softmax activation function to it
        modelOutputs = np.exp( activation ) / np.sum( np.exp( activation ) )

        #TODO possible layer normalization

        #Return the model output
        return modelOutputs

    def bkwdPass(self, predicted, targets):

        #Converts our target into a one hot vector
        t = np.zeros( self.vocabSize )
        t[ targets ] = 1

        #Increments loss
        self.loss = -np.sum( t * np.log(predicted) )

        #Increments the loss with respect to the outputs
        self.dL_dZ = predicted - t

        #Loops through the length of the final output weights
        for i in range(self.vocabSize):
            for j in range(self.dims):
                self.dL_dOutputWeights[i][j] += self.dL_dZ[i] * self.yFinal[j]

        dL_dY = np.dot(self.outputWeights.transpose(), self.dL_dZ)

        for block in range(1, self.numBlocks + 1):

            for i in range(self.dims):
                for j in range(self.dims * self.numHeads):
                    for t in range(len(self.weightedValues[-block])):
                        self.dL_dUnifyMats[i][j] += dL_dY[i] * self.weightedValues[-block][t][j]

            self.dL_dUnifyMats /= len(self.weightedValues[-block])

            self.dL_dUnifyMatsLst.append( self.dL_dUnifyMats )

            dL_dValueOut = np.dot(self.unifyMats[-block].transpose(), dL_dY)

            self.dL_dQueries = np.zeros_like(self.blockQueries[-block][-1])
            self.dL_dKeys    = np.zeros_like(self.blockKeys[-block][-1])
            self.dL_dValues  = dL_dValueOut

            for i in range(len(self.weights)):
                q = self.blockQueries[-block]
                k = self.blockKeys[-block]
                v = self.blockValues[-block]
                sumK = np.zeros_like(k[i])
                sumQ = np.zeros_like(q[i])
                for t in range(len(k)):
                    sumK += k[t] * self.weights[i][t]
                    sumQ += q[t] * self.weights[t][i]

                self.dL_dQueries += v[i] * sumK
                self.dL_dKeys += v[i] * sumQ

            self.dL_dQueries *= dL_dValueOut
            self.dL_dKeys *= dL_dValueOut

            self.dL_dQueryWeights = np.zeros_like(self.queryMats[-block])
            self.dL_dKeyWeights   = np.zeros_like(self.keyMats[-block])
            self.dL_dValueWeights = np.zeros_like(self.valueMats[-block])

            for i in range(self.numHeads * self.dims):
                for j in range(self.dims):
                    for t in self.blockOutputs[-block - 1]:
                        self.dL_dQueryWeights[i][j] += self.dL_dQueries[i] * t[j]
                        self.dL_dKeyWeights[i][j] += self.dL_dQueries[i] * t[j]
                        self.dL_dValueWeights[i][j] += self.dL_dQueries[i] * t[j]

            self.dL_dUnifyMatsLst[-block] += self.dL_dUnifyMats
            self.dL_dQueriesLst[-block] += self.dL_dQueryWeights
            self.dL_dKeysLst[-block] += self.dL_dKeyWeights
            self.dL_dValuesLst[-block] += self.dL_dValueWeights

        return self.loss

    def step(self, learningRate, batchSize):

        for block in range(self.numBlocks):

            self.queryMats[block] -= (learningRate * self.dL_dQueriesLst[block]) / batchSize
            self.keyMats[block] -= (learningRate * self.dL_dKeysLst[block]) / batchSize
            self.valueMats[block] -= (learningRate * self.dL_dValuesLst[block]) / batchSize

            self.unifyMats[block] -= (learningRate * self.dL_dUnifyMatsLst[block]) / batchSize

        self.outputWeights -= (learningRate * self.dL_dOutputWeights) / batchSize