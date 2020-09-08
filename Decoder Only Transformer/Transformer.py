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
MODEL_DIMS = 8

# Our fourth constant is NUM_BLOCKS, this is how many transformer blocks we
# will have. This will help us calculate more complex relationships between
# sets of sequences just like multiple layers would in any other type of
# neural network.
NUM_BLOCKS = 1

# The size of our batches we step gradients for
BATCH_SIZE = 10

# The number of iterations through our entire dataset
NUM_EPOCHS = 10

#%%############################### LOAD DATA ##################################

# Initializes a list for the training data
data = []

# Initializes a word embedding dictionary
word2Idx = dict()

# Loads in the data file
dataFile = open('./suess.txt')

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
    if len(inputSequence) > 0 and len(targetSequence) > 0:

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
            self.gammas.append([ np.random.uniform(-1, 1),
                                 np.random.uniform(-1, 1)])
            self.betas.append([  np.random.uniform(-1, 1),
                                 np.random.uniform(-1, 1)])

            # Appends a vector of random values for our point wise FFN in each
            # block as well as a scalar for our FFN bias in each block
            self.ffnW.append([ np.random.uniform(-1, 1, (modelDims, 1)),
                               np.random.uniform(-1, 1, (modelDims, 1))])
            self.ffnB.append([ np.random.uniform(-1, 1),
                               np.random.uniform(-1, 1)])

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
            Q = np.dot(wQ, embeddedInps) / scaleFactor
            K = np.dot(wK, embeddedInps) / scaleFactor
            V = np.dot(wV, embeddedInps)

            # calculates the raw weights for our values by multiplying each
            # qi by each kj
            weights_prime  = np.dot(Q.transpose(), K)

            # to normalize our weights we perform a row-wise softmax on the raw
            # weights
            weights_exp    = np.exp(weights_prime)
            softmaxbottom  = np.sum(weights_exp,axis = 1).reshape(len(inpSeq),1)
            weights        = weights_exp / softmaxbottom

            # calculates our weighted value vectors by considering our value
            # vectors and our weights
            self.weightedV = np.dot(weights, V.transpose())

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
            variance  = np.power(np.sum(norm - mean, axis = 0), 2)
            variance /= self.modelDims

            # caclualates our first predicted x values by standardizing the x
            # values, we add the value of 1 to our standard deviation to ensure
            # we are not massively increasing our xhat values since the purpose
            # of layer normalization is to scale things down
            self.xhat1 = (norm - mean) / (np.sqrt(variance) + 1)

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
            variance  = np.power(np.sum(norm - mean, axis = 0), 2)
            variance /= self.modelDims

            # caclualates our first predicted x values by standardizing the x
            # values, we add the value of 1 to our standard deviation to ensure
            # we are not massively increasing our xhat values since the purpose
            # of layer normalization is to scale things down
            self.xhat2 = (norm - mean) / (np.sqrt(variance) + 1)

            # calculates the output from the first layer normalization by
            # scaling up xhat by gamma and shifting it by beta
            self.norm2 = (gamma[1] * self.xhat2) + beta[1]

        ########################### MODEL OUTPUT ##############################

        # Calculates the raw ouput of our model
        output = np.dot(self.outputWeights, self.norm2)

        # performs softmax activation on our model output
        output_exp = np.exp(output)
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

        # param for attention
        self.dL_dWO = np.zeros_like(self.toOut[0])

        self.dL_dQueryWeights = np.zeros_like(self.toQueries[0])
        self.dL_dKeyWeights   = np.zeros_like(self.toKeys[0])
        self.dL_dValueWeights = np.zeros_like(self.toValues[0])

        # param for input embeddings
        self.dL_dEmbeddings = np.zeros_like(self.embeddingMat)

    # calculates the gradient of our error with respect to every parameter
    def backward(self, predicted, inputs, tarSeq):

        # reshapes our target sequence into an array so we can do matrix math
        # with it
        targetSeq = np.array(targSeq).reshape(len(targets), self.vocabSize)
        targetSeq = targetSeq.transpose()

        # calculates our error
        cost = -np.sum( targetSeq * np.log(predicted)) / len(inputs)

        # calculates the gradeitn of error with respect to the output layer
        dL_dZ = predicted - targetSeq

        # Loops through the length of the final output weights, calculating
        # error with respect to our output layer
        for i in range(self.vocabSize):
            for j in range(self.modelDims):
                for t in range(len(inputs)):
                    self.dL_dOutputWeights[i][j] += dL_dZ[i][t] * self.norm2[j][t]

        # scales down our output weight gradient
        self.dL_dOutputWeights /= len(inputs)

        # the derivative of loss with respect to the output of the second layer normalization
        dL_dNorm2 = np.dot(self.outputWeights.transpose(), dL_dZ)

        # the derivaitve of loss with respect to the second gamma value for layer normalization
        self.dL_dGamma2 += np.mean(dL_dNorm2 * self.xhat1)

        # the derivative of loss with respect to the second beta value for layer normalization
        self.dL_dBeta2 += np.mean(dL_dNorm2)

        # the derivative of loss with respect to the second xhat value found
        dL_dX = dL_dNorm2 * self.gammas[0][1]

        # derivative of loss with respect to the second weights
        dL_dFFNW2 = dL_dX * (self.ffnInner > 0) * np.mean(self.ffnInner,
                            axis = 1).reshape(self.modelDims, 1)

        self.dL_dFFNW2 += np.mean(dL_dFFNW2, axis = 1).reshape(self.modelDims, 1)

        self.dL_dFFNB2 += np.mean(dL_dX)

        dL_dFFNInner = dL_dX * self.ffnW[0][1]

        dL_dFFNW1 = dL_dFFNInner * np.mean(self.norm1,
                                   axis = 1).reshape(self.modelDims, 1)
        self.dL_dFFNW1 += np.mean(dL_dFFNW1,
                                    axis = 1).reshape(self.modelDims, 1)
        self.dL_dFFNB1 += np.mean(dL_dFFNInner)

        # the derivaitve of loss with respect to the 1st gamma value for layer normalization
        self.dL_dGamma1 += np.mean(dL_dFFNInner * self.xhat2)

        # the derivative of loss with respect to the 1st beta value for layer normalization
        self.dL_dBeta1 +=  np.mean(dL_dFFNInner)

        # the derivative off loss with respect to what goes into the first normalization
        dL_dAttn = dL_dFFNInner * self.gammas[0][0]

        self.dL_dWO = np.zeros_like(self.toOut[0])

        dL_dWeightedV = np.dot(self.toOut[0].transpose(), dL_dAttn)

        for i in range(self.modelDims):
            for j in range(self.modelDims):
                for t in range(len(inputs)):
                    self.dL_dWO[i][j] += dL_dAttn[i][t] * self.weightedV[t][j]

                self.dL_dWO[i][j] /= len(inputs)

        self.dL_dQueries = np.zeros_like(dL_dWeightedV)
        self.dL_dKeys    = np.zeros_like(dL_dWeightedV)
        self.dL_dValues  = dL_dWeightedV

        for idx, label in enumerate(inpSeq):
            embedRow = np.argmax(label)
            change = np.dot(self.toQueries[0].transpose(), self.dL_dQueries.transpose()[idx])
            change += np.dot(self.toKeys[0].transpose(), self.dL_dKeys.transpose()[idx])
            change += np.dot(self.toValues[0].transpose(), self.dL_dValues.transpose()[idx])

            for i in range(self.modelDims):
                self.dL_dEmbeddings[i][embedRow] += change[i]


        for i in range(self.modelDims):
            for j in range(self.modelDims):
                for x, t in enumerate(inputs):
                    self.dL_dQueryWeights[i][j] += self.dL_dQueries[i][x] * np.dot(self.embeddingMat, t)[j]
                    self.dL_dKeyWeights[i][j] += self.dL_dQueries[i][x] * np.dot(self.embeddingMat, t)[j]
                    self.dL_dValueWeights[i][j] += self.dL_dQueries[i][x] * np.dot(self.embeddingMat, t)[j]
                self.dL_dQueryWeights /= len(inputs)
                self.dL_dKeyWeights /= len(inputs)
                self.dL_dValueWeights /= len(inputs)

        return cost

    # increments our model parameters by their gradients, scaled down by the
    # learning rate and batch size
    def step(self, learningRate, batchSize):

        #steps
        self.outputWeights -= learningRate*(self.dL_dOutputWeights / batchSize)

        #steps single block second normalization
        self.gammas[0][1]  -= learningRate * (self.dL_dGamma2 / batchSize)
        self.betas[0][1]   -= learningRate * (self.dL_dBeta2 / batchSize)

        #steps the outter FFN parameteers
        self.ffnW[0][0]    -= learningRate * (self.dL_dFFNW2 / batchSize)
        self.ffnB[0][1]    -= learningRate * (self.dL_dFFNB2 / batchSize)

        #steps the inner FFN parameters
        self.ffnW[0][1]    -= learningRate * (self.dL_dFFNW1 / batchSize)
        self.ffnB[0][1]    -= learningRate * (self.dL_dFFNB1 / batchSize)

        #steps single block 1st normalization
        self.gammas[0][0]  -= learningRate * (self.dL_dGamma1 / batchSize)
        self.betas[0][0]   -= learningRate * (self.dL_dBeta1 / batchSize)

        #steps the ouput weights that unify many heads
        self.toOut[0]     -= learningRate * (self.dL_dWO / batchSize)

        self.toQueries    -= learningRate * (self.dL_dQueryWeights / batchSize)
        self.toKeys       -= learningRate * (self.dL_dKeyWeights / batchSize)
        self.toValues     -= learningRate * (self.dL_dValueWeights / batchSize)

        self.embeddingMat -= learningRate * (self.dL_dEmbeddings / batchSize)

#%%########################## MODEL IMPLEMENTATION ############################
data = data[:100]

# Changes the numpy seed
np.random.seed(1)

# Initializes the model with our constants and calculated parameters
model = Decoder(MODEL_DIMS, NUM_BLOCKS, len(word2Idx))

# Loops through a given number of epochs through the dataset
for epoch in range(1, NUM_EPOCHS + 1):

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
        if batchCtr == BATCH_SIZE:

            # steps the barametrs
            model.step(0.01, BATCH_SIZE)

            # prints that we've completed a batch
            print("Batch:", numBatchesCtr, "of", len(data) // BATCH_SIZE,
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
    print('Epoch:', epoch, "of", NUM_EPOCHS, "Cost:", cost / len(data))

# initializes a sequence for running the model with the sequence start token
sequence = [ '<SOS>' ]

# we want 10 words
for i in range(10):

    # converts our input sequence to one-hot vectors
    inpSeq = seqToVecs(sequence, word2Idx)

    # calculates the probability score for each word
    pred = model.forward(inpSeq)

    # calculates hte predicted next word
    newWord = np.argmax(pred, axis = 0)[-1]

    # appends to the input sequence
    sequence.append(idx2Word[ newWord ])

# prints the total sequence
print(' '.join(sequence))