###############################################################################
#                                                                             #
#                                DataLoader.py                                #
#                                 J. Steiner                                  #
#                                                                             #
###############################################################################

#Imports the ability to work with csv files
import csv

#imports the ability to work with system level attributes
import sys

#initializes the value of our maximum csv field size
maxInt = sys.maxsize

#while we have not broken out of the loop (we recieved an error)
while True:

    #we try to do the following
    try:
        #resize the field size limit for a csv file
        #this is because we have very large fields
        csv.field_size_limit(maxInt)

        #break out of the while loop
        break

    #if we recieve an overflow error, we execute this block
    except OverflowError:
        #floor divides our "maximum integer"
        maxInt = maxInt // 10

#the loop above will run until our "maximum integer" is within the bounds of
#a python integer that our system can handle

#%%########################### PRE-PROCESSES DATA #############################

#Loads data from a csv of specified file path, assumes word data is in column
#B of every row
def load(filePath, numRows, word2Idx, idx2Word):

    #a counter of row numbers so we can display progress
    rowNum = 0

    #creates an empty list for the data
    data = []

    #opens the file "stories.csv", assuming that it is in the local folder
    #"Data" we need to change the encoding from the defualt because some of the
    #characters could not be decoded using the default text encoding
    #we store the file we opened to the object "csvFile"
    with open(filePath, encoding='utf-8-sig') as csvFile:

        #creates a csv reader object from our file we opened, this will allow
        #python to iterate across rows
        reader = csv.reader(csvFile)

        #Loops through each row in the reader, after the first row, which is
        #a header
        for row in reader:

            #If we've exceeded the maximum number of rows we want to load
            if rowNum >= numRows:
                break

            #increments our counter for which row we are loading
            rowNum = rowNum + 1

            #prints a progress update
            print('Loading row', rowNum)

            #create a variable called "line" which stores the entire sequence
            line = row[1]

            #replaces punctuation with that punctuation with a space on either,
            #side, this way when we split over spaces, punctuation is counted
            line = line.replace('.', ' . ')
            line = line.replace('!', ' ! ')
            line = line.replace('?', ' ? ')
            line = line.replace(',', ' , ')
            line = line.replace('"', '')

            #replaces the newline character with an empty string, we do not
            #want to consider the newline character in our model
            line = line.replace('\n', '')

            #splits the line over spaces
            line = line.split()

            #Loops through each word in the sequence we have edited
            for word in line:

                #If the word embedding does not exist
                if word not in word2Idx:
                    #Adds it to the dictionary
                    word2Idx[word] = len(word2Idx)

            #Creates the input sequence as all words in the sentence except the
            #last word
            inputSequence  = line[:-1]

            #creates the target sequence as all the words in the sentence
            #except the first word
            targetSequence = line[1:]

            #If there is data in the input and target sequence
            if len(inputSequence) != 0 and len(targetSequence) != 0:

                #Appends the input and target to the data list
                data.append((inputSequence, targetSequence))

        #Closes the data file
        csvFile.close()

    #Loops through each word in the vocab set
    for key in word2Idx:

        #Reverses the word embedding dict
        idx2Word[word2Idx[key]] = key

    #returns the dataset
    return data
