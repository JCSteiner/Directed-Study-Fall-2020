###############################################################################
#                                                                             #
#                                Functional.py                                #
#                                 J. Steiner                                  #
#                                                                             #
###############################################################################

#%%########################### LOAD DEPENDENCIES ##############################

# imports the ability to work with advanced matrix and mathematical functions
import numpy as np

#%%########################## ACTIVATION FUNCTIONS ############################

# sigmoid activation function
def sigmoid(x):
    # x - a numpy array of values to be passed through the sigmoid curve

    # passes each element of our array x through the sigmoid function to cast
    # it between 0 and 1
    return 1. / ( 1 + np.exp(-x))

# derivative of sigmoid activation function
def sigmoid_prime(x):
    # x - a numpy array of values that have already been through the sigmoid
    #     curve, this makes our derivative less compelex because we don't need
    #     to include any exponentials in the calculation

    # calculates the derivative of the sigmoid function with respect to each
    # element in the array x
    return x * (1 - x)

# the hyperbolic tangent function
def tanh(x):

    # x - a numpy array that needs to be passed through our hyperbolic tangent
    #     function

    return np.tanh(x)

# derivative of tanh activation function
def tanh_prime(x):
    # x - a numpy array of values that have been previously passed through the
    #     tanh activation function, since we already have the tanh values, it
    #     makes our derivative less complex since we don't need an exponential

    # returns the derivaitve of the tanh function with respect to each element
    # in our numpy array x
    return 1. - x ** 2

# relu activation function
def relu(x):

    # returns the maximum of x and 0
    return x * (x > 0)

def relu6(x):

    x = relu(x)
    x[x>=6] = 6
    return x

# derivative of relu activation function
def relu_prime(x):

    # derivative is x is 1 if x is positive, 0 if x is negative
    x[x<=0] = 0
    return x

def relu6_prime(x):

    x = relu_prime(x)
    x[x>=6] = 0
    return x

# softmax activation function
def softmax(x, featureAxis = 0):

    # x           - a numpy array that we will take the softmax of, allowing us
    #               to interpret the values of our array as probabilities
    # featureAxis - the axis we want to sum over, this is the axis our features
    #               lie in, we default it to 0

    exps = np.exp( x - np.max(x, axis = featureAxis, keepdims=True))

    # returns the softmax of the array x over the specified feature axis
    return exps / np.sum(exps, axis = featureAxis, keepdims=True)

# dropout function
def dropout(x, p):

    # x - the input into the dropout
    # p - the probability we dropout with
    # zeros out neurons with probability p, this regulates overfitting by
    # focusing the model to put focus on a few neurons at a time instead of all
    # neurons all at once

    # creates the dropout matrix
    drop    = np.random.uniform(size = x.shape)

    # zeros out neurons with probability p
    dropped = x * (drop > p)

    # returns dropped out layer activation
    return dropped