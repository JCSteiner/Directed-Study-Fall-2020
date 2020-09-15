###############################################################################
#                                                                             #
#                                DataLoader.py                                #
#                                 J. Steiner                                  #
#                                                                             #
###############################################################################

#%%########################## IMPORT DEPENDENCIES #############################

# imports librosa
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 Loading audio files is not simple to do in python like loading in a text file
 is, because of this, we need to import a library called librosa. this allows
 the library to handle the technical aspects of loading in the audio file 
 loading. it also contains useful functions for audio pre-processing
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import librosa as lb

# imports numpy
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 once we have loaded in our audio, we will need to treat it numerically. this
 will be greater simplified if we treat the pre-processed audio as a matrix of
 column feature vectors, to work with matricies, we need to import numpy
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy   as np

#%%############################### CONSTANTS ##################################

# description of audio pre-processing
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
 To pre-process our data we will need to get the Mel-frequency cepstrum 
 coefficients or MFCCs of each signal. This is a logarithmic transform to 
 place greater emphasis on changes in lower frequencies vs changes in higher
 frequencies. The standard number of MFCCs extracted is 13, so that's how many
 we'll create for this. The MFCCs are created by applying a fourier transform
 to the data in n_fft sample intervals, the hop length is the amount of samples
 we move the left bound of the interval (a hop length < n_fft means the window)
 overlaps. so we define all these constants below
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# the number times we record frequency, this is in Hz
SAMPLE_RATE = 44100

# the number of MFCCs we get from each audio signal
N_MFCC      = 13

# the size of our MFCC window (in samples)
N_FFT       = 2048

# the size of our steps between windows (in samples)
HOP_LENGTH  = 512

#%%############################## DATA LOADING ################################

# loads our phenomes into the cache
def loadPhenomes(filePath):
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    Param:

        filePath    - the directory of the file that contains all our phenomes
        
    Return:
        
        phenomeDict - the dictionary of our pheonmes indexed by the word the 
                      phenomes apply to
        
    Notes:
        
    loads all our phenomes into a dictionary indexed by the word the phenomes
    apply to
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    # initilializes a dictionary that will store our phenomes
    phenomeDict = dict()
    
    # opens the file that stores the phenomes
    infile = open(filePath)
    
    # for each line in the phenome file
    for line in infile:
        
        # takes of the '\n' character
        line = line.strip()
        
        # converts the line to be all lowercase
        line = line.lower()
        
        # splits the line over spaces
        line = line.split()
        
        # the first item is the word, the rest are the phenomes
        phenomeDict[line[0]] = line[1:]
        
    # deallocates space for the file
    infile.close()
    
    # returns the dict we loaded in
    return phenomeDict

# data loading function
def loadData(filePathsSrc, dataFolderName = 'data', phePath = 'phenomes.txt',
              transExt = '.trans.txt', audioExt = '.flac', verbose = True):
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    Param:
        
        filePathsSrc   - the file path for the text file that stores all of our 
                         folder directories            
        dataFolderName - the name of the folder that stores all of the data
        phePath        - the filepath for the text file storing the phenomes
        transExt       - the file extension for the text file that contains all
                         the list of audio file names and what words they 
                         contain    
        audioExt       - the file extension for the audio files
        verbose        - whether or not we will print status updates along the
                         way
    Returns:
        
        data           - a list of tuples, in each tuple, the first element is
                         an array of MFCC column vectors and the second element
                         is the word/phenome that particular sound segment
                         applies to
        token2Idx      - a dictionary that maps each token to a unique number
        phenomeDict    - a dictionary that maps words to a list of phennomes
    
    '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
    # initializes a dictionary that will store indexes for words we take in
    token2Idx = dict()
    
    # loads the text file that stores the file paths
    filePaths = open(filePathsSrc)
    
    # initializes an empty list to store all of our data 
    data = []
    
    # status update
    if verbose:
        print('Loading Phenomes...')
    
    # loads the phenome dict
    phenomeDict = loadPhenomes(phePath)
    
    # status update
    if verbose:
        print('Loading Audio...')
    
        # defines a file counter for status updates
        fileCtr = 0
        
    # for each folder directory in the file path text file
    for path in filePaths:
        
        # strips the '\n' character off of the line
        path = path.strip()
        
        # if the file path is our stopping token, we stop loading paths
        if path == 'STOP':
            break
        
        # creates the file path using the data folder name and the current path
        filePath = './' + dataFolderName + path
        
        # loads in the file that contains each file name and a transcript of
        # what it says
        transcript = open(filePath + path[1:-1].replace('/', '-') + transExt)
        
        # loops through each line in the transcript file
        for line in transcript:
            
            # strips off the '\n' character from the line
            line = line.strip()
            
            # converts the line to be all lowercase
            line = line.lower()
            
            # splits the line over spaces
            wordList = line.split()
            
            # loads in the audio file
            audio, _ = lb.load(filePath + wordList[0] + audioExt, SAMPLE_RATE)
            
            # normalizes audio to have amplitudes between -5 and 5
            audio = (audio / np.max(np.abs(audio))) * 5
            
            # creates the mfccs for the audio file
            mfcc = lb.feature.mfcc(audio, SAMPLE_RATE, n_mfcc=N_MFCC, 
                                   hop_length = HOP_LENGTH)
            
            # initilizes an empty list to words and phenomes
            tokenList = []
            
            # loops through each word in the word list
            for word in wordList[1:]:
                
                  # if the word has phenomes listed for it
                  if word in phenomeDict:
                    
                      # concatenates the list of phenomes associated with that
                      # word
                      tokenList += phenomeDict[word]
                
                  # otherwise, appends the word to the list
                  else:
                    
                    # appends the word
                    tokenList.append(word)
            
            # how many mfcc column vectors we have
            audioLength = mfcc.shape[1]
            
            # the number of tokens associated with this audio signal
            numWords = len(tokenList)
            
            # how many 'bins' we can segment the mfcc's into by evenly 
            # splitting among tokens, note: this makes a big assumption that
            # every token takes an equal amount of time to say, it also assumes
            # we start and stop speaking instantly with no fade in or fade out
            binSize = audioLength // numWords
            
            # for each token in the token to index dictionary
            for idx, token in enumerate(tokenList):
                
                # splits up the input signal
                inputs = mfcc[:, binSize * idx : (idx+1) * binSize]
                
                # if the token is not in the token2Idx dict
                if not token in token2Idx:
                    # adds it
                    token2Idx[token] = len(token2Idx)
                
                # appends this given input with its target
                data.append((inputs, token))
            
        # closes the audio transcript file
        transcript.close()
    
    # closes the file paths dict
    filePaths.close()
    
    # returns the data, token2Idx dictionary, and phenome dictionary
    return data, token2Idx, phenomeDict
