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