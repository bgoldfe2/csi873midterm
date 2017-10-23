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
from sig_funcs import sigmoid, sigmoid_prime


# **Paramter Defaults**
input = 784
hidden = 4
output = 10
iterations = 50
epochs = 2
learning_rate = 0.5
momentum = 0.5
rate_decay = 0.01


def initialize(input, hidden, output, iterations, epochs, lrn_rate, momentum, rate_decay):
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
    # Training Parameters
    iterations = iterations
    epochs = epochs
    
    # Hyperparameters
    lrn_rate = lrn_rate
    momentum = momentum
    rate_decay = rate_decay
    
    # initialize arrays
    input = input + 1 # add 1 for bias or intercept or threshold node
    hidden = hidden # ? doesnt this need a threshold node?
    output = output

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
    # Theses are the number counts for training and test data sets
    trnNum = 5000
    tstNum = 890
    
    dpath = os.getcwd()+'\data'
    
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