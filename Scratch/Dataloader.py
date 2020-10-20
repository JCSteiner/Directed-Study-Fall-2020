###############################################################################
#                                                                             #
#                                Dataloader.py                                #
#                                 J. Steiner                                  #
#                                                                             #
###############################################################################

#%%########################### LOADS DEPENDENCIES #############################

# imports the ability to work with matricies easily
import numpy      as np

# imports activation functions and their derivatives
import Functional as f

#%%######################### DATA LOADING FUNCTION ############################
def loadData(filePath):

    ###########################################################################
    # Param: filePath - the path to the text file we are loading

    # initializes a list for the training data
    data = []

    # initializes a token embedding dictionary
    token2Idx = dict()

    # loads in the data file
    dataFile = open(filePath)

    # loops thorugh each line in the data file
    for line in dataFile:

        # strips off the new line chracter
        line = line.strip()

        # only lowercase words
        line = line.lower()

        # replaces punctuation with an empty string,
        # effectively removing it from consideration in the model
        line = line.replace('.', '')
        line = line.replace('!', '')
        line = line.replace('?', '')
        line = line.replace(',', '')
        line = line.replace('"', '')

        # adds a start of sequence and end of sequence token to the line
        line = line.split()
        line = ['<SOS>'] + line + ['<EOS>']

        # loops through each word in the sentence
        for word in line:
            # if the word embedding does not exist
            if word not in token2Idx:
                # adds it to the dictionary
                token2Idx[word] = len(token2Idx)

        # creates the input sequence as all words in the sentence except the
        # last word
        inputSequence  = line[:-1]

        # creates the target sequence as all the words in the sentence except
        # the first word
        targetSequence = line[1:]

        # if there is data in the input and target sequence
        if len(inputSequence) > 1 and len(targetSequence) > 1:

            # appends the input and target to the data list
            data.append((inputSequence, targetSequence))

    # closes the data file
    dataFile.close()

    # initializes a dictionary to turn from embeddings to words
    idx2Token = dict()

    # loops through each word in the vocab set
    for key in token2Idx:

        # reverses the word embedding dict
        idx2Token[token2Idx[key]] = key

    # returns the dataset, the token to idx dictionary and the idx to token
    # dictionary
    return data, token2Idx, idx2Token

#%%######################### INPUT VECTOR CREATION ############################
def oneHotVec(inputSequence, token2Idx):

    # creates a matrix of zero vectors
    oneHot = np.zeros((len(token2Idx), len(inputSequence)))

    # loops through our input sequence
    for idx in range(len(inputSequence)):
        # changes the token's index to a 1
        oneHot[token2Idx[inputSequence[idx]]][idx] = 1

    # returns our one hot vector
    return oneHot