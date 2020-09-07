###############################################################################
#                                                                             #
#                                Transformer.py                               #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

#%%########################## IMPORT DEPENDENCIES #############################

#Imports the ability to work with advanced matrix functions
import numpy as np

#Imports advanced mathematics functions
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

#%%############################### LOAD DATA ##################################

#Initializes a list for the training data
data = []

#Initializes a word embedding dictionary
word2Idx = dict()

#Loads in the data file
dataFile = open('./suess.txt')

#Loops thorugh each line in the data file
for line in dataFile:

    #Strips off the new line chracter
    line = line.strip()

    #Replaces punctuation with a space on either side of the punctuation
    #this way we can consider the punctuation as its own word
    line = line.replace('.', ' . ')
    line = line.replace('!', ' ! ')
    line = line.replace('?', ' ? ')
    line = line.replace(',', ' , ')

    #Removes quotation marks from the model
    line = line.replace('"', '')

    #Splits the line of text over spaces
    line = line.split()

    #adds a token for the start of the sequence
    line = ['<SOS>'] + line[:]

    #adds a token for the end of a sequence
    line += ['<EOS>']

    #Loops through each word in the sentence
    for word in line:

        #If the word to index entry does not exist
        if word not in word2Idx:
            #Adds it to the dictionary
            word2Idx[word] = len(word2Idx)

    #Creates the input sequence as all words in the sentence except the
    #last word
    inputSequence  = line[:-1]

    #creates the target sequence as all the words in the sentence except
    #the first word
    targetSequence = line[1:]

    #If there is data in the input and target sequence
    if len(inputSequence) > 0 and len(targetSequence) > 0:

        #Appends the input and target to the data list
        data.append((inputSequence, targetSequence))

#Closes the data file
dataFile.close()

#Initializes a dictionary to turn from embeddings to words
idx2Word = dict()

#Loops through each word in the vocab set
for key in word2Idx:

    #Reverses the word embedding dict
    idx2Word[word2Idx[key]] = key

def seqToVecs(seq, word2Idx):

    seqOfVecs = []

    for token in seq:
        vec = np.zeros( (len(word2Idx), 1) )
        vec[word2Idx[token]][0] = 1
        seqOfVecs.append(vec)

    return seqOfVecs

def posEncoding(pos, dimModel):

    encoding = np.zeros((dimModel,1))

    for i in range(dimModel):
        if i % 2 == 0:
            encoding[i][0] = math.sin(pos / math.pow(10000, i / dimModel))
        else:
            encoding[i][0] = math.cos(pos / math.pow(10000, (i-1) / dimModel))

    return encoding

#%%################################# MODEL ####################################

class Decoder:

    def __init__(self, modelDims, numBlocks, vocabSize):

        #######################################################################
        # modelDims - the dimensionality within our transformer blocks
        # numBlocks - the number of decoder blocks we run through each pass
        # vocabSize - the number of tokens we embed from and cast to each pass

        #initializes a matrix to embed our one hot input vectors
        self.embeddingMat = np.random.uniform(-1, 1, (MODEL_DIMS, len(word2Idx)))

        #initializes the lists that store our query, key, and value weights for each block
        self.toQueries = []
        self.toKeys = []
        self.toValues = []
        self.toOut = []

        #initializes the gammas and betas used for our layer normalization
        self.gammas = []
        self.betas  = []

        #initializes a list of FFN
        self.ffnW = []
        self.ffnB = []

        self.outputWeights = np.random.uniform(-1, 1, (len(word2Idx), MODEL_DIMS))

        #creates a query, key, and value weight matrix for each block
        for _ in range(NUM_BLOCKS):
            self.toQueries.append(np.random.uniform(-1, 1, (MODEL_DIMS, MODEL_DIMS)))
            self.toKeys.append(np.random.uniform(-1, 1, (MODEL_DIMS, MODEL_DIMS)))
            self.toValues.append(np.random.uniform(-1, 1, (MODEL_DIMS, MODEL_DIMS)))
            self.toOut.append(np.random.uniform(-1, 1, (MODEL_DIMS, MODEL_DIMS)))

            self.gammas.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            self.betas.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

            self.ffnW.append([np.random.uniform(-1, 1, (MODEL_DIMS, 1)),
                        np.random.uniform(-1, 1, (MODEL_DIMS, 1))])
            self.ffnB.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

        self.blockOutputs = []

    def decoderFWD(self, inpSeq):

        embeddedInps = []

        for pos in range(len(inpSeq)):
            x = np.dot(self.embeddingMat, inpSeq[pos])
            x += posEncoding(pos, MODEL_DIMS)

            embeddedInps.append(x)

        embeddedInps = np.array(embeddedInps).reshape(len(inpSeq), MODEL_DIMS).transpose()

        for wQ, wK, wV, wO, gamma, beta, ffW, ffB in \
            zip(self.toQueries, self.toKeys, self.toValues, self.toOut, self.gammas, self.betas, self.ffnW, self.ffnB):

            scaleFactor = math.sqrt(MODEL_DIMS)
            Q = np.dot(wQ, embeddedInps) / scaleFactor
            K = np.dot(wK, embeddedInps) / scaleFactor
            V = np.dot(wV, embeddedInps)

            #calculates weights
            weights_prime = np.dot(Q.transpose(), K)

            #row-wise softmax
            weights_exp = np.exp(weights_prime)
            softmaxbottom = np.sum(weights_exp, axis = 1).reshape(len(inpSeq), 1)
            weights = weights_exp / softmaxbottom

            self.weightedV = np.dot(weights, V.transpose())

            #the output from our scaled dot product self attention
            attnOut = np.dot(wO, self.weightedV.transpose())

            #here we would concatenate the output from all our heads

            #the following is layer normalization

            #residual connection before normalization
            norm = attnOut + embeddedInps

            mean = np.mean(norm, axis = 0)
            variance = np.power(np.sum(norm - mean, axis = 0), 2) / MODEL_DIMS

            self.xhat1 = (norm - mean) / (np.sqrt(variance) + 1)

            self.norm1 = (gamma[0] * self.xhat1) + beta[0]

            #element wise FFN
            ffnInner = ( self.norm1 * ffW[0] ) + ffB[0]

            #relu
            self.ffnInner = ffnInner * (ffnInner > 0)

            ffnOut = ( self.ffnInner * ffW[1] ) + ffB[1]

            #the following is layer normalization

            #residual connection before layer normalization
            norm = ffnOut + self.norm1

            mean = np.mean(norm, axis = 0)
            variance = np.power(np.sum(norm - mean, axis = 0), 2) / MODEL_DIMS

            self.norm2bottom = (np.sqrt(variance) + 1)

            self.xhat2 = (norm - mean) / self.norm2bottom

            self.norm2 = (gamma[1] * self.xhat2) + beta[1]

            self.blockOutputs.append(self.norm2)

        output = np.dot(self.outputWeights, self.norm2)
        output_exp = np.exp(output)
        bottom = np.sum(np.exp(output_exp), axis = 0)

        return output_exp / bottom

    def zeroGrad(self):

        self.dL_dOutputWeights = np.zeros_like(self.outputWeights)

        self.dL_dGamma2 = np.zeros_like(self.gammas[0][1])
        self.dL_dBeta2 = np.zeros_like(self.betas[0][1])

        self.dL_dFFNW2 = np.zeros_like(self.ffnW[0][1])
        self.dL_dFFNB2 = np.zeros_like(self.ffnB[0][1])

        self.dL_dFFNW1 = np.zeros_like(self.ffnW[0][0])
        self.dL_dFFNB1 = np.zeros_like(self.ffnB[0][0])

        self.dL_dGamma1 = np.zeros_like(self.gammas[0][0])
        self.dL_dBeta1 = np.zeros_like(self.betas[0][0])

        self.dL_dWO = np.zeros_like(self.toOut[0])

        self.dL_dQueryWeights = np.zeros_like(self.toQueries[0])
        self.dL_dKeyWeights   = np.zeros_like(self.toKeys[0])
        self.dL_dValueWeights = np.zeros_like(self.toValues[0])

        self.dL_dEmbeddings = np.zeros_like(self.embeddingMat)

    def decoderBKWD(self, predicted, inputs, tarSeq):

        targetSeq = np.array(targSeq).reshape(len(targets), len(word2Idx)).transpose()

        cost = -np.sum( targetSeq * np.log(predicted)) / len(inputs)

        #Increments the loss with respect to the outputs
        dL_dZ = predicted - targetSeq

        #Loops through the length of the final output weights
        for i in range(len(word2Idx)):
            for j in range(MODEL_DIMS):
                for t in range(len(inputs)):
                    self.dL_dOutputWeights[i][j] += dL_dZ[i][t] * self.norm2[j][t]

        self.dL_dOutputWeights /= len(inputs)

        #the derivative of loss with respect to the output of the second layer normalization
        dL_dNorm2 = np.dot(self.outputWeights.transpose(), dL_dZ)

        #the derivaitve of loss with respect to the second gamma value for layer normalization
        self.dL_dGamma2 += np.mean(dL_dNorm2 * self.xhat1)

        #the derivative of loss with respect to the second beta value for layer normalization
        self.dL_dBeta2 += np.mean(dL_dNorm2)

        #the derivative of loss with respect to the second xhat value found
        dL_dX = dL_dNorm2 * self.gammas[0][1]

        dL_dFFNW2 = dL_dX * (self.ffnInner > 0) * np.mean(self.ffnInner, axis = 1).reshape(MODEL_DIMS, 1)
        self.dL_dFFNW2 += np.mean(dL_dFFNW2, axis = 1).reshape(MODEL_DIMS, 1)

        self.dL_dFFNB2 += np.mean(dL_dX)

        dL_dFFNInner = dL_dX * self.ffnW[0][1]

        dL_dFFNW1 = dL_dFFNInner * np.mean(self.norm1, axis = 1).reshape(MODEL_DIMS, 1)
        self.dL_dFFNW1 += np.mean(dL_dFFNW1, axis = 1).reshape(MODEL_DIMS, 1)
        self.dL_dFFNB1 += np.mean(dL_dFFNInner)

        #the derivaitve of loss with respect to the 1st gamma value for layer normalization
        self.dL_dGamma1 += np.mean(dL_dFFNInner * self.xhat2)

        #the derivative of loss with respect to the 1st beta value for layer normalization
        self.dL_dBeta1 +=  np.mean(dL_dFFNInner)

        #the derivative off loss with respect to what goes into the first normalization
        dL_dAttn = dL_dFFNInner * self.gammas[0][0]

        self.dL_dWO = np.zeros_like(self.toOut[0])

        dL_dWeightedV = np.dot(self.toOut[0].transpose(), dL_dAttn)

        for i in range(MODEL_DIMS):
            for j in range(MODEL_DIMS):
                for t in range(len(inputs)):
                    self.dL_dWO[i][j] += dL_dAttn[i][t] * self.weightedV[t][j]

                self.dL_dWO[i][j] /= len(inputs)

        self.dL_dQueries = dL_dWeightedV
        self.dL_dKeys    = dL_dWeightedV
        self.dL_dValues  = dL_dWeightedV

        for idx, label in enumerate(inpSeq):
            embedRow = np.argmax(label)
            change = np.dot(self.toQueries[0].transpose(), self.dL_dQueries.transpose()[idx])
            change += np.dot(self.toKeys[0].transpose(), self.dL_dKeys.transpose()[idx])
            change += np.dot(self.toValues[0].transpose(), self.dL_dValues.transpose()[idx])

            for i in range(MODEL_DIMS):
                self.dL_dEmbeddings[i][embedRow] += change[i]


        for i in range(MODEL_DIMS):
            for j in range(MODEL_DIMS):
                for x, t in enumerate(inputs):
                    self.dL_dQueryWeights[i][j] += self.dL_dQueries[i][x] * np.dot(self.embeddingMat, t)[j]
                    self.dL_dKeyWeights[i][j] += self.dL_dQueries[i][x] * np.dot(self.embeddingMat, t)[j]
                    self.dL_dValueWeights[i][j] += self.dL_dQueries[i][x] * np.dot(self.embeddingMat, t)[j]
                self.dL_dQueryWeights /= len(inputs)
                self.dL_dKeyWeights /= len(inputs)
                self.dL_dValueWeights /= len(inputs)

        return cost

    def step(self, learningRate, batchSize):

        #steps
        self.outputWeights -= (learningRate * self.dL_dOutputWeights) / batchSize

        #steps single block second normalization
        self.gammas[0][1] -= (learningRate * self.dL_dGamma2) / batchSize
        self.betas[0][1] -= (learningRate * np.mean(self.dL_dBeta2)) / batchSize

        #steps the second FFN weight
        self.ffnW[0][0] -= (learningRate * self.dL_dFFNW2) / batchSize
        #steps the second FFN bias
        self.ffnB[0][1] -= (learningRate * self.dL_dFFNB2) / batchSize

        #steps the first FFN weight
        self.ffnW[0][1] -= (learningRate * self.dL_dFFNW1) / batchSize
        self.ffnB[0][1] -= (learningRate * self.dL_dFFNB1) / batchSize

        #steps single block 1st normalization
        self.gammas[0][0] -= (learningRate * np.mean(self.dL_dGamma1)) / batchSize
        self.betas[0][0] -= (learningRate * np.mean(self.dL_dBeta1)) / batchSize

        #steps the ouput weights that unify many heads
        self.toOut[0] -= (learningRate * self.dL_dWO) / batchSize

        self.toQueries -= (learningRate * self.dL_dQueryWeights) / batchSize
        self.toKeys -= (learningRate * self.dL_dKeyWeights) / batchSize
        self.toValues -= (learningRate * self.dL_dValueWeights) / batchSize

        self.embeddingMat -= (learningRate * self.dL_dEmbeddings) / batchSize

batchSize = len(data)
numEpochs = 30

model = Decoder(MODEL_DIMS, NUM_BLOCKS, len(word2Idx))
model.zeroGrad()

for epoch in range(1, numEpochs + 1):

    np.random.shuffle(data)

    batchCtr = 0
    cost  = 0
    numBatchesCtr = 1

    for inputs, targets in data:

        inpSeq = seqToVecs(inputs, word2Idx)
        targSeq = seqToVecs(targets, word2Idx)
        pred = model.decoderFWD(inpSeq)
        cost += model.decoderBKWD(pred, inpSeq, targSeq)

        batchCtr += 1

        if batchCtr == batchSize:
            model.step(0.1, batchSize)
            print("Batch:", numBatchesCtr, "of", len(data) // batchSize, "Cost:", cost / batchSize)
            batchCtr = 0
            cost = 0
            model.zeroGrad()
            numBatchesCtr += 1

    print('Epoch:', epoch, "of", numEpochs)

sequence = [ '<SOS>' ]

for i in range(20):

    inpSeq = seqToVecs(sequence, word2Idx)

    pred = model.decoderFWD(inpSeq)

    newWord = np.argmax(pred, axis = 0)[-1]

    sequence += [ idx2Word[ newWord ] ]

print(' '.join(sequence))