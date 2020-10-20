###############################################################################
#                                                                             #
#                                 LSTMCell.py                                 #
#                                 J. Steiner                                  #
#                                                                             #
###############################################################################

#%%########################### LOAD DEPENDENCIES ##############################

# imports the ability to work with advanced matrix and mathematical functions
import numpy      as np

# imports the file that stores our activation functions
import Functional as f

#%%########################## LSTM CELL DEFINITION ############################
class LSTMCell():

    # class constructor
    def __init__(self, inputDims, hiddenDims, batchSize):

        #######################################################################
        # Param: inputDims  - the dimensionality of our input vector
        #        hiddenDims - the dimensionality of our hidden states
        #        batchSize  - the size of the current batch to be passed

        # stores our hidden dimensions for the model
        self.hiddenDims = hiddenDims

        # stores the dimensionality of our total input into the LSTM, this
        # means our input vector concatenated with our hidden state column-wise
        self.xDims = inputDims + hiddenDims

        # defines the weights for our forget, input, gate, and output gates
        # with elements drawn from a random uniform distribution on the
        # interval [-1, 1)
        self.forgetWeights = np.random.uniform(-1, 1, (hiddenDims, self.xDims))
        self.inputWeights  = np.random.uniform(-1, 1, (hiddenDims, self.xDims))
        self.gateWeights   = np.random.uniform(-1, 1, (hiddenDims, self.xDims))
        self.outputWeights = np.random.uniform(-1, 1, (hiddenDims, self.xDims))

        # defines the biases for our forget, input, gate, and output gates
        # with elements drawn from a random uniform distribution on the
        # interval [-1, 1)
        self.forgetBias    = np.random.uniform(-1, 1, (hiddenDims, 1))
        self.inputBias     = np.random.uniform(-1, 1, (hiddenDims, 1))
        self.gateBias      = np.random.uniform(-1, 1, (hiddenDims, 1))
        self.outputBias    = np.random.uniform(-1, 1, (hiddenDims, 1))

        # zeros out the gradients and storage of the forward pass data, we call
        # this in the constructor so we initialize all the instance variables
        # within the zero grad function
        self.zeroGrad(batchSize)

    # zeros out the cell's gradients
    def zeroGrad(self, batchSize):

        # defaults the batch size to 1
        self.batchSize = batchSize

        # defaults the lists for our input sequences
        self.inputSequences = []

        # defaults the lists for our forget, input, gate, and output states
        # with respect to each input vector
        self.forgetStates   = []
        self.inputStates    = []
        self.gateStates     = []
        self.outputSates    = []

        # defaults our hiddenState and cellState lists, the first hidden state
        # and first cell state are defaulted to 0 vectors of the appropriate
        # size
        self.hiddenStates = [ np.zeros((self.hiddenDims, batchSize)) ]
        self.cellStates   = [ np.zeros((self.hiddenDims, 1)) ]

        # initializes the gradient of loss with respect to the input vector
        self.dL_dX = np.zeros((self.xDims, 1))

        # initializes the gradient of loss with respect to our forget, input,
        # gate, and output weight matricies
        self.dL_dForgetWeights = np.zeros_like(self.forgetWeights)
        self.dL_dInputWeights  = np.zeros_like(self.inputWeights)
        self.dL_dGateWeights   = np.zeros_like(self.gateWeights)
        self.dL_dOutputWeights = np.zeros_like(self.outputWeights)

        # initializes the gradient of loss with respect to our forget, input,
        # gate, and output, bias vectors
        self.dL_dForgetBias    = np.zeros_like(self.forgetBias)
        self.dL_dInputBias     = np.zeros_like(self.inputBias)
        self.dL_dGateBias      = np.zeros_like(self.gateBias)
        self.dL_dOutputBias    = np.zeros_like(self.outputBias)

    # forward pass through the network
    def forward(self, inputSequence):

        #######################################################################
        # Param: inputSequence - a sequence of input vectors in the shape
        #                        (time, feature, batch) so we can iterate over
        #                        time, then multiply feature column vectors and
        #                        maintain a (feature, batch) shape for any
        #                        given time

        # Loops through each feature vector in the input sequence of feature
        # vectors, each feature vector will have the shape (feature, batch)
        for featureVectors in inputSequence:

            # calculates x, our input vector to the LSTM, x is created by
            # concatenating the hidden state to the bottom of the feature
            # vector over each vector in the batch
            x   = np.vstack((featureVectors, self.hiddenStates[-1]))

            # calculates the value of the forget, input, and gate states
            forgetState = f.sigmoid(np.dot(self.forgetWeights, x) + self.forgetBias)
            inputState  = f.sigmoid(np.dot(self.inputWeights,  x) + self.inputBias)
            gateState   = np.tanh(  np.dot(self.gateWeights,   x) + self.gateBias)

            # calculates the current cell state based off the forget, input,
            # gate, and previous cell states
            cellState = gateState * inputState + forgetState * self.cellStates[-1]

            # calculates the value of the output states
            outputState = f.sigmoid(np.dot(self.outputWeights, x) + self.outputBias)

            # calculates the value of the next hidden state based on the output
            # state and the hyperbolic tangent of the cell state
            hiddenState = outputState * np.tanh(cellState)

            # appends the values of the forget, input, gate, cell, output, and
            # hidden states to lists to store them for backpropagation
            self.forgetStates.append( forgetState )
            self.inputStates.append(  inputState  )
            self.gateStates.append(   gateState   )
            self.cellStates.append(   cellState   )
            self.outputSates.append(  outputState )
            self.hiddenStates.append( hiddenState )

            # appends the value of our current input vector to the lstm to a
            # list, it will be used for backpropagation
            self.inputSequences.append(x)

        # returns all cell states expect for the first dummy cell state, this
        # allows us to backpropagate over a sequence to sequence operation all
        # in one pass
        return self.cellStates[1:]

    # backward pass through the network
    def backward(self, dL_dCell, dL_dHid):

        #######################################################################
        # Param: dL_dCell - the derivative with respect to the cell output
        #        dL_dHid  - the derivative with respect to the hidden state

        # initializes the gradient of loss with respect to the forget, input,
        # gate, and output states
        dL_dForgetState = np.zeros((self.hiddenDims, self.batchSize))
        dL_dInputState  = np.zeros((self.hiddenDims, self.batchSize))
        dL_dGateState   = np.zeros((self.hiddenDims, self.batchSize))
        dL_dOutputState = np.zeros((self.hiddenDims, self.batchSize))

        # loop through each index of time t, we loop from 1 to the length of
        # our stored input sequences + 1 because we are going to index at -t,
        # this allows us to index backwards while iterating forward
        for t in range(1, len(self.inputSequences) + 1):

            # calculates the gradient of the cell state with respect to loss
            dL_dCellState = self.outputSates[-t]  * dL_dHid + dL_dCell

            # calculates the gradient of our input, forget, output, and gate
            # gates, this does not include the chain rule so we are not taking
            # the derivative of the whole state, just the inner derivaitive. I
            # split up the two halves of the chain rule primarility to make it
            # easier to read
            dL_dInput     = self.gateStates[  -t   ] * dL_dCellState
            dL_dForget    = self.cellStates[  -t-1 ] * dL_dCellState
            dL_dOutput    = self.cellStates[  -t   ] * dL_dHid
            dL_dGate      = self.outputSates[ -t   ] * dL_dCellState

            # finishes calculating the gradient of the lstm cell with respect
            # to the input, forget, output, and gate states, this calculation
            # considers the chain rule
            dL_dInputState  += f.sigmoid_prime(self.inputStates[-t])  * dL_dInput
            dL_dForgetState += f.sigmoid_prime(self.forgetStates[-t]) * dL_dForget
            dL_dOutputState += f.sigmoid_prime(self.outputSates[-t])  * dL_dOutput
            dL_dGateState   += f.tanh_prime(self.gateStates[-t])      * dL_dGate

            # our gradient with respect to bias is just our gradient with
            # respect to the state multiplied by 1, by the identity property of
            # multiplication (1 * number = number), we just increment our bias
            # gradients by our state gradients
            self.dL_dForgetBias += np.mean(dL_dForgetState, axis=1, keepdims=True)
            self.dL_dInputBias  += np.mean(dL_dInputState,  axis=1, keepdims=True)
            self.dL_dOutputBias += np.mean(dL_dOutputState, axis=1, keepdims=True)
            self.dL_dGateBias   += np.mean(dL_dGateState,   axis=1, keepdims=True)

            # loops through each index in our hidden dimensionality
            for i in range(self.hiddenDims):
                # loops through each index in our input vector dimensionality
                for j in range(self.xDims):

                    # xt is our input vector at given time step -t at index j
                    xtj = self.inputSequences[-t][j]

                    # calculates the gradient with respect to our weights at
                    # index [i][j]
                    self.dL_dForgetWeights[i][j] += np.mean(dL_dForgetState[i] * xtj)
                    self.dL_dInputWeights[i][j]  += np.mean(dL_dInputState[i]  * xtj)
                    self.dL_dGateWeights[i][j]   += np.mean(dL_dGateState[i]   * xtj)
                    self.dL_dOutputWeights[i][j] += np.mean(dL_dOutputState[i] * xtj)

            # increments the derivative of loss with respect to our input
            # vectors x
            self.dL_dX += np.dot(self.inputWeights.T,
                          np.mean(dL_dInputState,  axis = 1, keepdims = True))
            self.dL_dX += np.dot(self.forgetWeights.T,
                          np.mean(dL_dForgetState, axis = 1, keepdims = True))
            self.dL_dX += np.dot(self.outputWeights.T,
                          np.mean(dL_dOutputState, axis = 1, keepdims = True))
            self.dL_dX += np.dot(self.gateWeights.T,
                          np.mean(dL_dGateState,   axis = 1, keepdims = True))

            # backpropagates the derivaitve with respect to the cell so we can
            # calculate gradients for another time step
            dL_dCell *= self.forgetStates[-t]
            # backpropagates the derivative with respect to the hidden state so
            # we can calculate gradients for another time step back
            dL_dHid   = self.dL_dX[self.xDims-self.hiddenDims:]

        # returns the deriviative of our loss with respect to the model inputs,
        # this allows us to backpropagate a layer prior to the lstm layer
        return self.dL_dX

    # decrements cell parameters by their gradients
    def step(self, stepSize):

        # Param: stepSize - the constant we multiply our gradient by to
        #                   scale it down to a reasonable step size

        # calculates the scaling factor for the gradient based off batch size
        # and the length of the input sequence
        scaleFactor = stepSize / (self.batchSize * len(self.inputSequences))

        # decrements the forget, input, gate, and output gate weights
        self.forgetWeights -= scaleFactor * self.dL_dForgetWeights
        self.inputWeights  -= scaleFactor * self.dL_dInputWeights
        self.gateWeights   -= scaleFactor * self.dL_dGateWeights
        self.outputWeights -= scaleFactor * self.dL_dOutputWeights

        # decrements the forget, input, gate, and output gate biases
        self.forgetBias    -= scaleFactor * self.dL_dForgetBias
        self.gateBias      -= scaleFactor * self.dL_dGateBias
        self.outputBias    -= scaleFactor * self.dL_dOutputBias
        self.gateBias      -= scaleFactor * self.dL_dGateBias