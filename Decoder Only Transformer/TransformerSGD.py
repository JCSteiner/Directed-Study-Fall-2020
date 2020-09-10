###############################################################################
#                                                                             #
#                                Transformer.py                               #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%########################## IMPORT DEPENDENCIES #############################

# Imports the ability to work with advanced matrix functions
import numpy as np

# Imports advanced mathematics functions
import math

#%%############################ DEFINE CONSTANTS ##############################

# Our first constant is MODEL_DIMS this is how many dimensions we transform
# our inputs into
MODEL_DIMS = 4

# Our fourth constant is NUM_BLOCKS, this is how many transformer blocks we
# will have. This will help us calculate more complex relationships between
# sets of sequences just like multiple layers would in any other type of
# neural network.
NUM_BLOCKS = 1

# The size of our batches we step gradients for
BATCH_SIZE = 20

# The number of iterations through our entire dataset
NUM_EPOCHS = 10

#%%############################### LOAD DATA ##################################

# Loads our data and stores all unique words in a word2Idx dictionary
def loadData(filePath):

    # Initializes a list for the training data
    data = []

    # Initializes a dictionary for storing word indicies
    word2Idx = dict()

    # Loads in the data file
    dataFile = open(filePath)

    # Loops thorugh each line in the data file
    for line in dataFile:

        # Strips off the new line chracter
        line = line.strip()

        # Replaces punctuation with a space on either side of the punctuation
        # this way we can consider the punctuation as its own word
        line = line.replace('.', ' . ')
        line = line.replace('!', ' ! ')
        line = line.replace('?', ' ? ')
        line = line.replace(',', ' , ')

        # Removes quotation marks from the model
        line = line.replace('"', '')

        # Splits the line of text over spaces
        line = line.split()

        # adds a token for the start of the sequence
        line = ['<SOS>'] + line[:]

        # adds a token for the end of a sequence
        line += ['<EOS>']

        # Loops through each word in the sentence
        for word in line:

            # If the word to index entry does not exist
            if word not in word2Idx:
                # Adds it to the dictionary
                word2Idx[word] = len(word2Idx)

        # Creates the input sequence as all words in the sentence except the
        # last word
        inputSequence  = line[:-1]

        # creates the target sequence as all the words in the sentence except
        # the first word
        targetSequence = line[1:]

        # If there is data in the input and target sequence
        if len(inputSequence) > 1 and len(targetSequence) > 1:

            # Appends the input and target to the data list
            data.append((inputSequence, targetSequence))

    # Closes the data file
    dataFile.close()

    # Initializes a dictionary to turn from embeddings to words
    idx2Word = dict()

    # Loops through each word in the vocab set
    for key in word2Idx:

        # Reverses the word embedding dict
        idx2Word[word2Idx[key]] = key

    # Returns our data list
    return data, word2Idx, idx2Word

# converts a sequence of labels into a sequence of one-hot vectors
def seqToVecs(seq, word2Idx):

    # empty list to append to
    seqOfVecs = []

    # for each token in the label list
    for token in seq:
        # intializes as zeros
        vec = np.zeros( (len(word2Idx), 1) )
        # changes the correct label to 1
        vec[word2Idx[token]][0] = 1

        # appends new vector
        seqOfVecs.append(vec)

    # returns sequence of one-hot vectors
    return seqOfVecs

# position encoding of sequence of vectors
def posEncoding(pos, dimModel):

    # initializes the encoding vector
    encoding = np.zeros((dimModel,1))

    # encodes based on sine and cosine waves of different frequencies
    for i in range(dimModel):
        if i % 2 == 0:
            encoding[i][0] = math.sin(pos / math.pow(10000, i / dimModel))
        else:
            encoding[i][0] = math.cos(pos / math.pow(10000, (i-1) / dimModel))

    # returns the encoding vectors
    return encoding

#%%############################ HELPER FUNCTIONS ##############################

# Trains the model
def train(model, word2Idx, data, batchSize, numEpochs, learningRate):

    ###########################################################################
    # model        - the model we are training
    # word2Idx     - a dictionary of word indicies that we will use to embed
    #                words later
    # batchSize    - the number of items per batch when training
    # numEpochs    - the number of iterations through the whole dataset we want
    #                to train for
    # learningRate - the step size we take when stepping parameters

    # Loops through a given number of epochs through the dataset
    for epoch in range(1, numEpochs + 1):

        # randomly shuffles the dataset, so all our batches will be random samples
        # with replacement
        np.random.shuffle(data)

        # intializes counters for our training loop
        batchCtr = 0
        cost  = 0
        numBatchesCtr = 1

        # For each input sequence and target sequence in the dataset
        for inputs, targets in data:

            # Converts our input sequence into a sequence of one-hot vectors
            inpSeq = seqToVecs(inputs, word2Idx)
            # converts our target sequence into a sequence of one-hot vectors
            targSeq = seqToVecs(targets, word2Idx)

            # calculates our predicted output as given by the model's forward pass
            pred = model.forward(inpSeq)

            # using what the model was given, what it outputed, and what it would
            # ideally output, we perform a backwards pass with the model
            cost += model.backward(pred, inpSeq, targSeq)

            # After beforming a backwards pass, increment the number of batches
            # we've performed
            batchCtr += 1

            # If our batch counter indicates it's time to step the parameters
            if batchCtr == batchSize:

                # steps the barametrs
                model.step(learningRate, batchSize)

                # prints that we've completed a batch
                print("Batch:", numBatchesCtr, "of", len(data) // batchSize,
                      end = '\r')

                # zeroes out model gradients
                model.zeroGrad()

                # resets our batch counter
                batchCtr = 0

                # incrments the nubmer of total batches we've performed
                numBatchesCtr += 1

        # prints a new line
        print()
        # prints that we've completed an epoch and what the cost was
        print('Epoch:', epoch, "of", numEpochs, "Cost:", cost / len(data))

# Runs a single sequence of the model
def runModel(model, idx2Word, sequence, stopAfter = 20):

    ###########################################################################
    # model     - the model we are evaluating
    # idx2Word  - a dictionary converting indicies back into words
    # sequence  - the sequence we start predicting off of
    # stopAfter - a hard stop for how many elements of a sequence we want to
    #             predict if the end of sequence token is not produced

    # initializes a sequence for running the model with the sequence start token
    seq = list(sequence)

    # we want 10 words
    for _ in range(stopAfter):

        # converts our input sequence to one-hot vectors
        inpSeq  = seqToVecs(seq, word2Idx)

        # calculates the probability score for each word
        pred    = model.forward(inpSeq)

        # calculates hte predicted next word
        newWord = np.argmax(pred, axis = 0)[-1]

        # if the new word is the <EOS> token
        if idx2Word[ newWord ] == '<EOS>':
            # breaks out of the for loop
            break

        # appends to the input sequence
        seq.append(idx2Word[ newWord ])

    # prints the total sequence
    return ' '.join(seq)


#%%################################# MODEL ####################################

class Decoder:

    # initializes important instance variables
    def __init__(self, modelDims, numBlocks, vocabSize):

        #######################################################################
        # modelDims - the dimensionality within our transformer blocks
        # numBlocks - the number of decoder blocks we run through each pass
        # vocabSize - the number of tokens we embed from and cast to each pass

        # stores our model parameters as instance variables
        self.modelDims = modelDims
        self.numBlocks = numBlocks
        self.vocabSize = vocabSize

        # initializes a matrix to embed our one hot input vectors
        self.embeddingMat = np.random.uniform(-1, 1, (modelDims, vocabSize))

        # initializes the lists that store our query, key, and value weights
        # for each block
        self.toQueries = []
        self.toKeys    = []
        self.toValues  = []
        self.toOut     = []

        # initializes the gammas and betas used for our layer normalizations
        # for each block
        self.gammas = []
        self.betas  = []

        # initializes a list of FFN weights and biases
        self.ffnW = []
        self.ffnB = []

        # Initializes our final output weights that cast our transformer
        # output to our original vocab size
        self.outputWeights = np.random.uniform(-1, 1, (vocabSize, modelDims))

        # creates a query, key, and value weight matrix for each block
        for _ in range(numBlocks):

            # Appends a matrix with values drawn from a random uniform
            # distribution from -1 to 1 for the query, keys, values, and output
            # matricies
            self.toQueries.append( np.random.uniform(-1, 1,
                                                    (modelDims, modelDims)))
            self.toKeys.append(    np.random.uniform(-1, 1,
                                                    (modelDims, modelDims)))
            self.toValues.append(  np.random.uniform(-1, 1,
                                                    (modelDims, modelDims)))
            self.toOut.append(     np.random.uniform(-1, 1,
                                                    (modelDims, modelDims)))

            # Appends scalars with a random values drawn from a uniform
            # distribution on the interval [-1, 1) for our gammas and betas
            # for the layer normalization in each block
            self.gammas.append([ np.random.uniform(-1, 1, 1),
                                 np.random.uniform(-1, 1, 1)])
            self.betas.append([  np.random.uniform(-1, 1, 1),
                                 np.random.uniform(-1, 1, 1)])

            # Appends a vector of random values for our point wise FFN in each
            # block as well as a scalar for our FFN bias in each block
            self.ffnW.append([ np.random.uniform(-1, 1, (modelDims, 1)),
                               np.random.uniform(-1, 1, (modelDims, 1))])
            self.ffnB.append([ np.random.uniform(-1, 1, (modelDims, 1)),
                               np.random.uniform(-1, 1, (modelDims, 1))])

        # zeroes out all gradients
        self.zeroGrad()

    # forward pass of an input sequence of one-hot vectors through the network
    def forward(self, inpSeq):

        #######################################################################
        # inpSeq - the sequence of inputs we are passing through our network

        # initializes a list that stores our inputs after they have been
        # embedded and encoded
        embeddedInps = []

        # For each position in all of the input sequence's positions
        for pos in range(len(inpSeq)):

            # Calculates the input embedding by multiplying the input one-hot
            # vector with the embedding matrix
            x  = np.dot(self.embeddingMat, inpSeq[pos])

            # Calculates the position encoding using our custom position
            # encoding function and adds it to the embedded input
            x += posEncoding(pos, self.modelDims)

            # Appends our calculated input vector to our sequence of input
            # vectors
            embeddedInps.append(x)

        # transforms our input vector to a matrix where time is the horizontal
        # axis and model dimensions are along the vertical axis
        embeddedInps = np.array(embeddedInps)
        embeddedInps = embeddedInps.reshape(len(inpSeq), self.modelDims)
        embeddedInps = embeddedInps.transpose()

        # Unpacks the query weights, key weights, value weights, output weights
        # the gammas, betas, ffn weights and ffn biases from our block params
        for wQ, wK, wV, wO, gamma, beta, ffnW, ffnB in \
            zip(self.toQueries, self.toKeys, self.toValues, self.toOut,
                self.gammas, self.betas, self.ffnW, self.ffnB):

            ####################### ATTENTION MECHANISM #######################

            # calculates the scale factor we use to control the size of our
            # query and key values
            scaleFactor = math.sqrt(self.modelDims)

            # calculates our queries, keys, and values by weighting our inputs
            self.Q = np.dot(wQ, embeddedInps) / scaleFactor
            self.K = np.dot(wK, embeddedInps) / scaleFactor
            self.V = np.dot(wV, embeddedInps)

            # calculates the raw weights for our values by multiplying each
            # qi by each kj
            weights_prime  = np.dot(self.Q.transpose(), self.K)

            # to normalize our weights we perform a row-wise softmax on the raw
            # weights
            stable         = weights_prime - np.max(weights_prime)
            self.weights_exp    = np.exp(stable)
            softmaxbottom  = np.sum(self.weights_exp,axis = 1, keepdims = True)
            weights        = self.weights_exp / softmaxbottom

            # calculates our weighted value vectors by considering our value
            # vectors and our weights
            self.weightedV = np.dot(weights, self.V.transpose())

            # calculates the output of our scaled dot-product attention by
            # unifying multiple attention heads if they exist
            attnOut        = np.dot(wO, self.weightedV.transpose())

            ####################### LAYER NORMALIZATION #######################

            # adds a residual connection between our attention output and our
            # embedded input by summing the two vectors
            norm = attnOut + embeddedInps

            # calculates the sample mean of our residually connected values
            mean = np.mean(norm, axis = 0)
            # calculates the sample variance of our residually connected values
            self.variance1  = np.power(np.sum(norm - mean, axis = 0), 2) + 1
            self.variance1 /= self.modelDims

            # caclualates our first predicted x values by standardizing the x
            # values, we add the value of 1 to our standard deviation to ensure
            # we are not massively increasing our xhat values since the purpose
            # of layer normalization is to scale things down
            self.xhat1 = (norm - mean) / (np.sqrt(self.variance1))

            # calculates the output from the first layer normalization by
            # scaling up xhat by gamma and shifting it by beta
            self.norm1 = (gamma[0] * self.xhat1) + beta[0]

            ################# POINT-WISE FEED FORWARD NETWORK #################

            # The inner value to our point wise FFN network, multiply by
            # weights and add a bias
            ffnInner      = ( self.norm1 * ffnW[0] ) + ffnB[0]

            # pass the inner value through a relu activation function
            self.ffnInner = ffnInner * (ffnInner > 0)

            # output for the outer value of our point wise FFN, multiply by
            # weights (different from the first) and add a bias (also different)
            ffnOut        = ( self.ffnInner * ffnW[1] ) + ffnB[1]

            ####################### LAYER NORMALIZATION #######################

            # adds a residual connection between our FFN output and our
            # pre FFN normalization output by summing the two vectors
            norm = ffnOut + self.norm1

            # calculates the sample mean of our residually connected values
            mean = np.mean(norm, axis = 0)

            # calculates the sample variance of our residually connected values
            self.variance2  = np.power(np.sum(norm - mean, axis = 0), 2) + 1
            self.variance2 /= self.modelDims

            # caclualates our first predicted x values by standardizing the x
            # values, we add the value of 1 to our standard deviation to ensure
            # we are not massively increasing our xhat values since the purpose
            # of layer normalization is to scale things down
            self.xhat2 = (norm - mean) / (np.sqrt(self.variance2))

            # calculates the output from the first layer normalization by
            # scaling up xhat by gamma and shifting it by beta
            self.norm2 = (gamma[1] * self.xhat2) + beta[1]

        ########################### MODEL OUTPUT ##############################

        # Calculates the raw ouput of our model
        output = np.dot(self.outputWeights, self.norm2)

        # performs softmax activation on our model output
        stable     = output - np.max(output)
        output_exp = np.exp(stable)
        bottom     = np.sum(np.exp(output_exp), axis = 0)
        softmax    = output_exp / bottom

        # returns our softmax output
        return softmax

    # zeroes out our locally stored gradients for the model
    def zeroGrad(self):

        # final output weights
        self.dL_dOutputWeights = np.zeros_like(self.outputWeights)

        # param for second layer normalization
        self.dL_dGamma2 = np.zeros_like(self.gammas[0][1])
        self.dL_dBeta2 = np.zeros_like(self.betas[0][1])

        # param for outer point-wise FFN
        self.dL_dFFNW2 = np.zeros_like(self.ffnW[0][1])
        self.dL_dFFNB2 = np.zeros_like(self.ffnB[0][1])

        # param for inner point-wise FFN
        self.dL_dFFNW1 = np.zeros_like(self.ffnW[0][0])
        self.dL_dFFNB1 = np.zeros_like(self.ffnB[0][0])

        # param for first layer normalization
        self.dL_dGamma1 = np.zeros_like(self.gammas[0][0])
        self.dL_dBeta1 = np.zeros_like(self.betas[0][0])

        # param for attention output
        self.dL_dWO = np.zeros_like(self.toOut[0])

        # param for self attention calculation
        self.dL_dQueryWeights = np.zeros_like(self.toQueries[0])
        self.dL_dKeyWeights   = np.zeros_like(self.toKeys[0])
        self.dL_dValueWeights = np.zeros_like(self.toValues[0])

        # param for input embeddings
        self.dL_dEmbeddings = np.zeros_like(self.embeddingMat)

    # calculates the gradient of our error with respect to every parameter
    def backward(self, predicted, inputs, targSeq):

        # reshapes our target sequence into an array so we can do matrix math
        # with it
        targetSeq = np.array(targSeq).reshape(len(targSeq), self.vocabSize)
        targetSeq = targetSeq.transpose()

        # calculates our error
        cost = -np.sum( targetSeq * np.log(predicted)) / len(inputs)

        ############################# OUTPUT LAYER ############################

        # calculates the gradeitn of error with respect to the output layer
        dL_dZ = predicted - targetSeq

        # Loops through the length of the final output weights, calculating
        # error with respect to our output layer
        for i in range(self.vocabSize):
            for j in range(self.modelDims):
                self.dL_dOutputWeights[i][j] = np.dot(dL_dZ[i], self.norm2[j])

        ###################### SECOND LAYER NORMALIZATION #####################

        # the derivative of loss with respect to the output of the second layer
        # normalization
        dL_dNorm2 = np.dot(self.outputWeights.transpose(), dL_dZ)

        # the derivaitve of loss with respect to the second gamma value for
        # layer normalization
        self.dL_dGamma2 += np.mean(dL_dNorm2 * self.xhat2)

        # the derivative of loss with respect to the second beta value for
        # layer normalization
        self.dL_dBeta2 += np.mean(dL_dNorm2)

        # the derivative of loss with respect to the second xhat value found
        dL_dXHat2 = dL_dNorm2 * self.gammas[0][1]

        # Calculates the derivative with loss with respect to the x value prior
        # to being standardized using our mean and variance
        dL_dX2    = (-1/(self.modelDims * np.sqrt(self.variance2)))           \
                  * ((self.modelDims*dL_dXHat2) - np.sum(dL_dXHat2, axis=0)   \
		          - (dL_dXHat2 * np.sum( dL_dXHat2 * self.xhat2, axis=0)))

        ######################### FEED FORWARD NETWORK ########################

        # Finds the derivative of loss with respect to our outer weights in our
        # feed forward network
        self.dL_dFFNW2 += np.mean(dL_dX2 * self.ffnInner, axis = 1,
                                  keepdims = True)

        # Calculates the deriative of loss with respect to the outer bias in
        # our feed forward network
        self.dL_dFFNB2 += np.mean(dL_dX2, axis = 1, keepdims = True)

        # Calculates the derivative of our loss with respect to the inner value
        # of our feed forward network
        dL_dFFNInner = dL_dX2 * self.ffnW[0][1]

        # Calculates the derivative of loss with respect to our inner weights
        # and bais of our feed forward network
        self.dL_dFFNW1 += np.mean(self.norm1 * dL_dFFNInner, axis = 1,
                                  keepdims = True)
        self.dL_dFFNB1 += np.mean(dL_dFFNInner, axis = 1, keepdims = True)

        ###################### FIRST LAYER NORMALIZATION ######################

        # the derivaitve of loss with respect to the 1st gamma value for layer normalization
        self.dL_dGamma1 += np.mean(dL_dFFNInner * self.xhat1)

        # the derivative of loss with respect to the 1st beta value for layer normalization
        self.dL_dBeta1 +=  np.mean(dL_dFFNInner)

        # the derivative off loss with respect to what goes into the first normalization
        dL_dXHat1 = dL_dFFNInner * self.gammas[0][0]

        # calculates the derivative of loss with respect to our x value before
        # it has been standardized with its mean and variance
        dL_dX1 = (-1/(self.modelDims * np.sqrt(self.variance1)))               \
                  * ((self.modelDims*dL_dXHat1) - np.sum(dL_dXHat1, axis=0)    \
		          - (dL_dXHat1 * np.sum( dL_dXHat1 * self.xhat1, axis=0)))

        ####################### SELF ATTENTION MECHANISM ######################

        # Loops through each model dimension and each time value to calculate
        # the derivative of the ouptut weights
        for i in range(self.modelDims):
            for j in range(self.modelDims):
                for t in range(len(inputs)):
                    self.dL_dWO[i][j] += dL_dX1[i][t] * self.weightedV[t][j]

        # Calculates the derivative of the loss with respect to the weighted
        # values
        dL_dWeightedV = np.dot(self.toOut[0].transpose(), dL_dX1)

        # Calculates the derivatives of the loss with respect to the weights
        dL_dWeights = np.dot(self.V,self.weights_exp.transpose())*dL_dWeightedV

        # Calculates the derivatives of the loss with respect to the queries
        # keys and weights
        self.dL_dQueries = dL_dWeights * self.K
        self.dL_dKeys    = dL_dWeights * self.Q
        self.dL_dValues  = dL_dWeights

        # Loops through each index in the input sequence
        for idx, label in enumerate(inputs):

            # calculates which row of our word embeddings we are changing
            embedCol = np.argmax(label)

            # sums the derivative of the embedding with respect to the queries,
            # keys, and values
            change = np.dot( self.toQueries[0].transpose(),
                             self.dL_dQueries.transpose()[idx])
            change += np.dot(self.toKeys[0].transpose(),
                             self.dL_dKeys.transpose()[idx])
            change += np.dot(self.toValues[0].transpose(),
                             self.dL_dValues.transpose()[idx])

            # Adds the change vector to the embedding derivative in the correct column
            for i in range(self.modelDims):
                self.dL_dEmbeddings[i][embedCol] += change[i]

        # Loops through the model dims and the input time sequence
        for i in range(self.modelDims):
            for j in range(self.modelDims):
                for x, t in enumerate(inputs):
                    # Calculates the derivative of cost with respect to the
                    # query, key, and value weights
                    self.dL_dQueryWeights[i][j] += self.dL_dQueries[i][x] *   \
                                                np.dot(self.embeddingMat, t)[j]
                    self.dL_dKeyWeights[i][j]   += self.dL_dKeys[i][x] *      \
                                                np.dot(self.embeddingMat, t)[j]
                    self.dL_dValueWeights[i][j] += self.dL_dValues[i][x] *    \
                                                np.dot(self.embeddingMat, t)[j]

        # Returns the calulcated cost
        return cost

    # increments our model parameters by their gradients, scaled down by the
    # learning rate and batch size
    def step(self, learningRate, batchSize):

        # steps the output weights
        self.outputWeights -= learningRate*(self.dL_dOutputWeights / batchSize)

        # steps single block second normalization
        self.gammas[0][1]  -= learningRate * (self.dL_dGamma2 / batchSize)
        self.betas[0][1]   -= learningRate * (self.dL_dBeta2 / batchSize)

        # steps the outter FFN parameters
        self.ffnW[0][1]    -= learningRate * (self.dL_dFFNW2 / batchSize)
        self.ffnB[0][1]    -= learningRate * (self.dL_dFFNB2 / batchSize)

        # steps the inner FFN parameters
        self.ffnW[0][0]    -= learningRate * (self.dL_dFFNW1 / batchSize)
        self.ffnB[0][0]    -= learningRate * (self.dL_dFFNB1 / batchSize)

        # steps single block 1st normalization
        self.gammas[0][0]  -= learningRate * (self.dL_dGamma1 / batchSize)
        self.betas[0][0]   -= learningRate * (self.dL_dBeta1 / batchSize)

        # steps the ouput weights that unify many heads
        self.toOut[0]     -= learningRate * (self.dL_dWO / batchSize)

        # steps the query, key, and value weights
        self.toQueries[0]    -= learningRate * (self.dL_dQueryWeights / batchSize)
        self.toKeys[0]       -= learningRate * (self.dL_dKeyWeights / batchSize)
        self.toValues[0]     -= learningRate * (self.dL_dValueWeights / batchSize)

        # steps our input embedding matrix
        self.embeddingMat -= learningRate * (self.dL_dEmbeddings / batchSize)

#%%########################## MODEL IMPLEMENTATION ############################

# initializes a word2Idx dictionary
word2Idx = dict()

# intializes an idx2Word dictionary
idx2Word = dict()

# loads data
data, word2Idx, idx2Word = loadData('./suess.txt')
# cuts off data
data = data[:100]

# Initializes the model with our constants and calculated parameters
model = Decoder(MODEL_DIMS, NUM_BLOCKS, len(word2Idx))

# Trains the model
train(model, word2Idx, data, BATCH_SIZE, NUM_EPOCHS, 0.001)

# Prints an output of the model
print(runModel(model, idx2Word, ['<SOS>']))