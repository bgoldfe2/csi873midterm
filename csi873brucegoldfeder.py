# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:29:16 2017

@author:    Bruce Goldfeder
            CSI 873
            Fall 2017
            Midterm
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sig_funcs import sigmoid, sigmoid_prime, tanh_prime,tanh


# **Paramter Defaults**
input = 7
hidden = 4
output = 10
iterations = 50
epochs = 2
lrn_rate = 0.5
momentum = 0.5
rate_decay = 0.01

def print_params():
    
    global iterations
    global epochs
    global lrn_rate
    global momentum
    global rate_decay
    global input
    global hidden
    global output
    
    print ('%-10s ==> %10d' % ('input', input))
    print ('%-10s ==> %10d' % ('hidden', hidden))
    print ('%-10s ==> %10d' % ('output', output))
    print ('%-10s ==> %10d' % ('iterations', iterations))
    print ('%-10s ==> %10d' % ('epochs', epochs))
    print ('%-10s ==> %10.2f' % ('learn_rate', lrn_rate))
    print ('%-10s ==> %10.2f' % ('momentum', momentum))
    print ('%-10s ==> %10.2f' % ('rate_decay', rate_decay))
    
def initialize(ins, hids, outs, its, eps, lrn, mom, rd):
    
    """
    **Network Parameters**
    input: number of input units
    hidden: number of hidden units
    output: number of output units
    
    **Training Parameters**
    iterations: number of iterations to run
    epochs: number of full training set run throughs
    
    **Hyperparameters**
    lrn_rate: this is the factor for learning rate
    momentum: this is the momentum term added to weight update
    rate_decay: used for annealing the learning rate using a 1/t decay using
                the mathematical form α=α0/a0+(a0+a0*rate_decay)
    """
    
    global iterations
    global epochs
    global lrn_rate
    global momentum
    global rate_decay
    global input
    global hidden
    global output

    # Training Parameters    
    iterations = its
    epochs = eps
    
    # Hyperparameters
    lrn_rate = lrn
    momentum = mom
    rate_decay = rd
    
    # initialize arrays
    input = ins + 1 # add 1 for bias or intercept or threshold node
    hidden = hids # ? doesnt this need a threshold node?
    output = outs


    
    
def train(patterns,ai,ah,ao,wi,wo,ci,co):
    
    global iterations
    global lrn_rate
    global rate_decay
   
    # N: learning rate
    for i in range(iterations):
        error = 0.0
        #random.shuffle(patterns)
        for p in patterns:
            inputs = p[0]
            targets = p[1]
            ai,ah,ao,wi,wo,ci,co = feedForward(inputs,ai,ah,ao,wi,wo,ci,co)
            errorNew,ai,ah,ao,wi,wo,ci,co = backPropagate(targets,ai,ah,ao,wi,wo,ci,co)
            error += errorNew
        with open('error.txt', 'a') as errorfile:
            errorfile.write(str(error) + '\n')
            errorfile.close()
        if i % 10 == 0:
            print('error %-.5f' % error)
        # learning rate decay
        lrn_rate = lrn_rate * (lrn_rate / (lrn_rate + (lrn_rate * rate_decay)))
        
    return ai,ah,ao,wi,wo,ci,co
 
def feedForward(inputs,ai,ah,ao,wi,wo,ci,co):
    
    global input
    print (input,len(inputs))
    """
    The feedforward algorithm loops over all the nodes in the hidden layer and
    adds together all the outputs from the input layer * their weights
    the output of each node is the sigmoid function of the sum of all inputs
    which is then passed on to the next layer.
    :param inputs: input data
    :return: updated activation output vector
    """
    if len(inputs) != input-1:
        raise ValueError('input length incorrect')

    # input activations
    for i in range(input -1): # -1 is to avoid the bias
        ai[i] = inputs[i]

    # hidden activations
    for j in range(hidden):
        sum = 0.0
        for i in range(input):
            sum += ai[i] * wi[i][j]
        ah[j] = tanh(sum)

    # output activations
    for k in range(output):
        sum = 0.0
        for j in range(hidden):
            sum += ah[j] * wo[j][k]
        ao[k] = sigmoid(sum)

    return ai,ah,ao,wi,wo,ci,co   
    
def backPropagate(targets,ai,ah,ao,wi,wo,ci,co):
    
    global output,hidden,momentum,lrn_rate
    """
    For the output layer
    1. Calculates the difference between output value and target value
    2. Get the derivative (slope) of the sigmoid function in order to determine how much the weights need to change
    3. update the weights for every node based on the learning rate and sig derivative
    For the hidden layer
    1. calculate the sum of the strength of each output link multiplied by how much the target node has to change
    2. get derivative to determine how much weights need to change
    3. change the weights based on learning rate and derivative
    :param targets: y values
    :param N: learning rate
    :return: updated weights
    """
    if len(targets) != output:
        raise ValueError('Wrong number of targets you silly goose!')

    # calculate error terms for output
    # the delta tell you which direction to change the weights
    output_deltas = [0.0] * output
    for k in range(output):
        error = -(targets[k] - ao[k])
        output_deltas[k] = sigmoid_prime(ao[k]) * error

    # calculate error terms for hidden
    # delta tells you which direction to change the weights
    hidden_deltas = [0.0] * hidden
    for j in range(hidden):
        error = 0.0
        for k in range(output):
            error += output_deltas[k] * wo[j][k]
        hidden_deltas[j] = tanh_prime(ah[j]) * error

    # update the weights connecting hidden to output
    for j in range(hidden):
        for k in range(output):
            change = output_deltas[k] * ah[j]
            wo[j][k] -= lrn_rate * change + co[j][k] * momentum
            co[j][k] = change

    # update the weights connecting input to hidden
    for i in range(input):
        for j in range(hidden):
            change = hidden_deltas[j] * ai[i]
            wi[i][j] -= lrn_rate * change + ci[i][j] * momentum
            ci[i][j] = change

    # calculate error
    error = 0.0
    for k in range(len(targets)):
        error += 0.5 * (targets[k] - ao[k]) ** 2
    
    return error,ai,ah,ao,wi,wo,ci,co

def ReadInFiles(path,trnORtst):
    # This reads in all the files from a directory filtering on what the file
    # starts with
    fullData = []
    fnames = os.listdir(path)
    for fname in fnames:
        if fname.startswith(trnORtst):
            print (fname)
            data = np.loadtxt(path + "\\" + fname)
            fullData.append(data)
    numFiles = len (fullData)
    print(numFiles)
   
    return fullData

def ReadInOneList(fullData,maxRows):
    # This function combines all of the data into one array for ease of use
    # It contains a capping ability to configure how many results to use
    allData = []
    numFiles = len (fullData)
    for j in range (numFiles):
        # allows for smaller data set sizes
        numRows = len (fullData[j])
        #print('numrows,maxrows ',numRows,maxRows)
        if (maxRows < numRows):
            numRows = maxRows
    
        for k in range(numRows):
            allData.append(fullData[j][k])
    return np.asarray(allData)


def Output(resultList,index):
    # converts the test results into counts per number (10 counts for 0-9)
    outList = np.zeros(10)
    start = 0
    end = index - 1
    # to iterate over the numbers 0-9 make range 0,9
    # this does a vertical count of each column to find
    # the frequency that a cell is '1' or written in
    for x in range(0,10):
        outList[x] = resultList[start:end,2].sum(axis=0)
        start = start + index
        end = end + index
        
    return outList
        
def HeatMap(numberIn):
    #heat map to show numbers
    plt.matshow(numberIn.reshape(28,28))
    plt.colorbar()
    plt.show()

def main():
    
    global iterations
    global epochs
    global lrn_rate
    global momentum
    global rate_decay
    global input
    global hidden
    global output
    
    # Theses are the number counts for training and test data sets
    trnNum = 5000
    tstNum = 890
    
    dpath = os.getcwd()+'\data'
    ####### SWAP THIS OUT AFTER PROVING ALGORITHM #####
    """
    run NN demo on the digit recognition dataset from sklearn
    """
    def load_data():
        data = np.loadtxt('datasmall/sklearn_csv.txt', delimiter = ',')

        # first ten values are the one hot encoded y (target) values
        y = data[:,0:10]
        #y[y == 0] = -1 # if you are using a tanh transfer function make the 0 into -1
        #y[y == 1] = .90 # try values that won't saturate tanh
        
        data = data[:,10:] # x data
        #data = data - data.mean(axis = 1)
        data -= data.min() # scale the data so values are between 0 and 1
        data /= data.max() # scale
        
        out = []
        print (data.shape)

        # populate the tuple list with the data
        for i in range(data.shape[0]):
            fart = list((data[i,:].tolist(), y[i].tolist())) # don't mind this variable name
            out.append(fart)

        return out

    X = load_data()

    print (X[9]) # make sure the data looks right
    
######## SWAP THIS OUT ########
    
    initialize(64, 100, 10, its=50, eps = 3,lrn = 0.5, mom = 0.5, rd = 0.01)

    print_params()
    
    # set up array of 1s for activations
    ai = [1.0] * input
    ah = [1.0] * hidden
    ao = [1.0] * output

    # create random weights between -0.05 and 0.05 as per text on pg. 98
    wi = np.random.uniform(-0.05, 0.05, size = (input, hidden))
    wo = np.random.normal(-0.05, 0.05, size = (hidden, output))
    
    # temporary arrays to hold the numbers to be updated each iteration
    ci = np.zeros((input, hidden))
    co = np.zeros((hidden, output))
    
    ai,ah,ao,wi,wo,ci,co = train(X,ai,ah,ao,wi,wo,ci,co)
    
    asdf
    
    
    # Read in the Training data first
    dataset = ReadInFiles(dpath,'train')
    my_data = ReadInOneList(dataset,trnNum)
    #np.savetxt('fooout.txt',my_data,fmt='%1i')
    
    # Read in the test data
    dataset2 = ReadInFiles(dpath,'test')
    my_test = ReadInOneList(dataset2,tstNum)
    
###### Neural Net Structure and Behavior Variables ######
    
    # array of the layers - in the midterm this is three layers
    # The layers go in order from left to right - [input,hidden,output]
    layers = np.array(())
    
    
    outputs = Output(results,tstNum)
    print(outputs)
    print(outputs.sum(axis=0))
    
    # print the percentages per number tested
    print(outputs/tstNum)
    
    # print the percentage correct over all the numbers tested
    print (outputs.sum(axis=0)/(tstNum*10))
    
main()