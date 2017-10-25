# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:39:24 2017

@author: Bruce
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt


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
    #numFiles = len (fullData)
    #print(numFiles)
   
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

def HeatMap(numberIn):
    #heat map to show numbers
    plt.matshow(numberIn.reshape(28,28))
    plt.colorbar()
    plt.show()
    
def makeHotList(yin):
    rows = len(yin)
    outputY = []    
    for x in range(rows):
        tk = int(yin[x])
        #print('tk is ', tk)
        tkArray = np.zeros(10)
        tkArray[tk]=1
        outputY.append(tkArray)
    return np.asarray(outputY)

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

def main():
    
    # Read in the test data set
        # Theses are the number counts for training and test data sets
    trnNum = 5000
    #tstNum = 890
    
    dpath = os.getcwd()+'\data'
    
    # Read in the Training data first
    dataset = ReadInFiles(dpath,'train')
    my_data = ReadInOneList(dataset,trnNum)
    #np.savetxt('fooout.txt',my_data,fmt='%1i')
    
    # Convert the 0-255 to 0 or 1 values in data
    my_data[:,1:] /= 255.0
    HeatMap(my_data[40,1:])
    # This module converts the 0-255 to 0 or 1 binomial
    #my_data[:,1:][my_data[:,1:] > 0] = 1
    
    X = my_data[:,1:]
    
    
    print(X[0,])
    
    print(X.shape)
    
    y = makeHotList(my_data[:,0])
    imgIter = len(y)
   
   
    #Variable initialization
    epoch=100 #Setting training iterations
    lr=0.1 #Setting learning rate
    inputlayer_neurons = X.shape[1] #number of features in data set
    hiddenlayer_neurons = 3 #number of hidden layers neurons
    output_neurons = 10 #number of neurons at output layer
    
    #weight and bias initialization
    wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
    bh=np.random.uniform(size=(1,hiddenlayer_neurons))
    wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
    bout=np.random.uniform(size=(1,output_neurons))
    
    for i in range(epoch):
    
        #Forward Propogation
        hidden_layer_input1=np.dot(X,wh)
        hidden_layer_input=hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,wout)
        output_layer_input= output_layer_input1+ bout
        output = sigmoid(output_layer_input)
        
        #Backpropagation
        E = y-output
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += hiddenlayer_activations.T.dot(d_output) *lr
        bout += np.sum(d_output, axis=0,keepdims=True) *lr
        wh += X.T.dot(d_hiddenlayer) *lr
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    
    print (output.shape)
    print(output[:10,])
    print (np.argmax(output, axis=1))

    
main()