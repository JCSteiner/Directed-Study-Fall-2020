###############################################################################
#                                                                             #
#                                 ConvLayer.py                                #
#                                  J. Steiner                                 #
#                                                                             #
###############################################################################

# imports the ability to work with matricies and used advanced mathematics
import numpy as np

# imports the file that stores our activation functions
import Functional as f

#%%################### THE CONVOLUTIONAL LAYER DEFINITION #####################
class ConvLayer():

    # class constructor
    def __init__(self, numFilters, channels, kernel, stride):

        #######################################################################
        # Param: numFilters - the number of filters we want our convolutional
        #                     layer to have
        #        channels   - the number of channels deep we expect our input
        #                     to be
        #        kernel     - the size of our square filters we create
        #        stride     - how many units we shift over our filter after
        #                     each convolution

        # stores the hyper parameters for our model as instance variables to be
        # used later
        self.numFilters = numFilters
        self.channels   = channels
        self.kernel     = kernel
        self.stride     = stride

        # generates a matrix of filters that is numFilters long, channels
        # deep, with a width and height of kernel
        self.filters = np.random.normal(size = (numFilters, channels,
                                                kernel, kernel))

        # generate a matrix of biases that is numFilters long, and 1 value
        self.bias    = np.zeros((numFilters, 1))

        # zeroes out all gradients in the model
        self.zeroGrad()

    # zeroes out model gradients
    def zeroGrad(self):

        # zeroes out model paramter gradients
        self.dL_dFilters = np.zeros_like(self.filters)
        self.dL_dBias    = np.zeros_like(self.bias)

    # forward pass through the convolutional layer
    def forward(self, image):

        #######################################################################
        # Param: image - the input into our convolutional layer

        # unpacks the input dimensions of the image we are passing through the
        # given filters
        self.image           = image
        _, self.inputDims, _ = image.shape

        # calculates how many output dimensions we will have based on our
        # kernel, stride, and input dimensions
        self.outputDims = int((self.inputDims - self.kernel) / self.stride) + 1

        # initializes our convolution output, we will add to it as we compute
        # during the forward pass
        self.convOut = np.zeros((len(self.filters), self.outputDims,
                                                    self.outputDims))

        # for each filter n in all of our filters
        for n in range(self.filters.shape[0]):

            # intializes the y index for our convolution input
            inY = 0

            # incrmentes vertically in the range of our known output dims
            for outY in range(self.outputDims):

                # initializes the x index counter for our convolution input
                inX = 0

                # increments horizontally in the range of our known output dims
                for outX in range(self.outputDims):

                    # performs the convolution
                    self.convOut[n, outY, outX] = np.sum(self.filters[n] *    \
                        image[:, inY:inY+self.kernel, inX:inX+self.kernel])   \
                        + self.bias[n]

                    # increments the output x index
                    inX += self.stride

                # increments the output y index
                inY += self.stride

        # passes our output through relu
        self.convOut = f.relu(self.convOut)

        # returns the convolution output
        return self.convOut

    # backward pass through the convolutional layer
    def backward(self, dL_dConv):

        #######################################################################
        # Param: dL_dConv  - the gradient of our loss with respect to our
        #                    convolution forward pass output

        # initializes the gradient of loss with respect to the input to our
        # convolutional block

        # backpropagates through relu
        dL_dConv = f.relu_prime(dL_dConv)

        # defaults our gradient with respect to our convolution input
        dConvIn = np.zeros(self.image.shape)


        # loops through each filter in our range of filters
        for n in range(self.numFilters):

            # starts a counter for our input y
            inY = 0

            # loops in the range of our known output dimensions
            for outY in range(self.outputDims):

                # starts a counter for our input x index
                inX = 0

                # loops in the range of our known output width
                for outX in range(self.outputDims):

                    # calculates the gradient of loss with respect to our
                    # filter
                    self.dL_dFilters[n] += dL_dConv[n, outY, outX] *          \
                        self.image[:, inY:inY+self.kernel, inX:inX+self.kernel]

                    # calculates the gradient of loss function with respect to
                    # our convolution input with
                    dConvIn[:, inY:inY+self.kernel, inX:inX+self.kernel] +=   \
                        dL_dConv[n, outY, outX] * self.filters[n]

                    # increments our x counter by the filter's stride
                    inX += self.stride

                # increments the y counter by the filter's stride
                inY += self.stride

            # calculates the gradient of loss with respect to the filter's bias
            self.dL_dBias[n] = np.sum(dL_dConv[n])

        # returns the gradient of loss with respect to the convolution input so
        # we can keep back propagating
        return dConvIn

    # steps parameters by their gradients
    def step(self, learningRate, batchSize):

        # calculates the scaling factor for the gradient
        scaleFactor = learningRate / batchSize

        # steps the model parameters by their scaled down gradients
        self.filters -= scaleFactor * self.dL_dFilters
        self.bias    -= scaleFactor * self.dL_dBias

#%%#################### MAXPOOL DIMENSIONALITY REDUCTION ######################
class Maxpool():

    # class constructor
    def __init__(self, kernel, stride):

        #######################################################################
        # Param: kernel - the size of the window we consider when finding the
        #                 maxes of an input
        #        stride - the number of units we shitf the window by each step

        # stores the model hyper parameters as instance variables
        self.poolKernel = kernel
        self.poolStride = stride

    # forward pass
    def forward(self, conv):

        #######################################################################
        # Param: conv - the output of the convolutional layer we are running
        #               maxpool over

        # stores the input to the forward pass to be stored for the backwards
        # pass
        self.convOut = conv

        # unpacks the dimensions of the parameter
        numChannels, size, _ = conv.shape

        # calculates the size of our pool output dimensions
        self.poolSize = int((size - self.poolKernel) / self.poolStride) + 1

        # initializes a matrix of zeroes to store the pooled convolution
        self.pooled = np.zeros((numChannels, self.poolSize, self.poolSize))

        # loops through each channel in the range of channels
        for n in range(numChannels):

            # defaults our counter for the vertical input indicies
            inY = 0

            # loops through the range of values in the output matrix's height
            for outY in range(self.poolSize):

                # defaults our counter for the vertical pooled output
                inX = 0

                # loops through the range of values in the output matrix's
                # width
                for outX in range(self.poolSize):

                    # maxpools across the kernel size
                    self.pooled[n, outY, outX] += np.max(conv[n,              \
                        inY:inY+self.poolKernel, inX:inX+self.poolKernel])

                    # increments the input x value by the pool stride
                    inX += self.poolStride

                # increments the input y value by the pool stride
                inY += self.poolStride

        # returns the maxpooled array
        return self.pooled

    # backwards pass through the max pool layer
    def backward(self, dL_dPool):

        # initializes the gradient of the maxpool
        dL_dPool = dL_dPool.reshape(self.pooled.shape)

        # unpacks the shape of the output of the convoloutional block ( the
        # input to our maxpool layer
        numChannels, _, _ = self.convOut.shape

        # the gradient of loss with respect to the input to our maxpool layer
        dPoolIn = np.zeros(self.convOut.shape)

        # loops through each channel
        for n in range(numChannels):

            # initializes a counter for our maxpool input vertically
            inY = 0

            # loops over the vertical indicies of our gradient array
            for outY in range(self.poolSize):

                # initializes a counter for our maxpool input horizontally
                inX = 0

                # loops over the horizontal indicies of our gradient array
                for outX in range(self.poolSize):

                    # calculates the elements of the convolution output our
                    # kernel would normally go over
                    pool = self.convOut[n, inY:inY+self.poolKernel,            \
                                        inX:inX+self.poolKernel]

                    # calculates the indicies of the max arguments
                    a, b = np.unravel_index(np.nanargmax(pool), pool.shape)

                    # pushes loss through the maxpool
                    dPoolIn[n, inY + a, inX+b] = dL_dPool[n, outY, outX]

                    # increments horizontally by our pool stride
                    inX += self.poolStride

                # increments vertically by our pool stride
                inY += self.poolStride

        # returns the gradient of loss with input to the maxpool
        return dPoolIn