# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:29:16 2017

@author:    Bruce Goldfeder
            CSI 873
            Fall 2017
            Midterm
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt



# **Paramter Defaults**
input = 7
hidden = 4
output = 10
iterations = 50
epochs = 2
lrn_rate = 0.5
momentum = 0.5
rate_decay = 0.01

def sigmoid(z):
    try:
        if (z < 0):
            sig = 1.0 - 1.0/(1.0 + math.exp(z))
        else:
            sig = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        if (z > 0):
            sig = 0.0000001
        else:
            sig = 1000000
    return sig

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
    input = ins # add 1 for bias or intercept or threshold node
    hidden = hids # add 1 for bias or intercept or threshold node
    output = outs


 
def feedForward(cnt,inputs,ai,ah,ao,wi,wo,ci,co):
    
    #print (input,len(inputs))
    """
    The feedforward algorithm propogates the input forward through the network
    It calculates the net by summing from one to n the weights (including bias 
    w0) times the value sigma term for each node in the hidden and output
    layers.  This is then used as the negative exponent value in the sigmoid
    function
        net = Sigma[i=0,n]wi*xi
        sigma(net) = 1/(1 + e**-net)
    """
    
    # input activations
    # Get the training answer and replace it with 1 as the bias

    ai[1:] = inputs[cnt,1:] # remember to account for actual value in 0th index
    answer = inputs[cnt,0]
    ai[0] = 1  # set the bias/threshold to always be 1
    #print('answer is',answer)
    # hidden activations
    
    #ah = np.multiply(ai,wi).sum(axis=0)
    
    for j in range(hidden+1):
        sum = 0.0
        for i in range(input+1):
            prodAW = ai[i] * wi[i][j]
            sum += prodAW
        #print ('sum is ',sum)
        ah[j] = sigmoid(sum) #what about using tanh here?
        
    # output activations
    ah[0] = 1 # set the bias/threshold to always be 1
    for k in range(output):
        sum = 0.0
        for j in range(hidden+1):
            prodAWO = ah[j] * wo[j][k]
            sum += prodAWO
        ao[k] = sigmoid(sum)

    return answer,ai,ah,ao,wi,wo,ci,co  

def calcDeltaKO(answer,ao):
    """
    Calculates the delta_k for each output unit using the formula from the text
    delta_k = ouput_k * (1-ouput_k)*(target_k - output_k)
    """
    
    tk = int(answer)
    #print('tk is ', tk)
    tkArray = np.zeros(10)
    tkArray[tk]=1
    
    delta_k = np.zeros(10)
        
    for y in range(10):
        delta_k[y] = ao[y] * (1-ao[y]) * (tkArray[y] - ao[y])
            
    #print('deltaKO is: ',delta_k,' shape is ',delta_k.shape)
    
    return delta_k

def calcDeltaKH(o_h,w_kh,d_k):
    """
    Calculates the delta_h for each hidden unit using the forumula from the text
    delta_h = output_h * (1-ouput_h) * Sum(weight_kh * delta_ko)
    o_h is the output of hidden units or the array in main 'ao'
    w_kh is the weight of hidden units or the array in main 'wi'
    d_k is the delta_k returned from the function 'calcDeltaKO' for output nodes
    """
    #print('o_h ',o_h)
    numH = len(o_h)
    rows,cols = w_kh.shape
    #print ('num rows,cols of d_k ',rows, '-',cols,' shape is ',w_kh.shape)
    delta_h1 = np.zeros(numH)
    for z in range(numH):
        delta_h1[z] = o_h[z] * (1 - o_h[z])
    #print (' delta_h1 shape is ',delta_h1.shape)
    
    delta_h = np.zeros(numH)
    
    
    for y in range(rows):
        sum = 0.0
        for x in range(cols):
            #print('w_kh ',w_kh[y,x],' d_k ',d_k[x])
            delta_h2 = w_kh[y,x] * d_k[x]
            sum += delta_h2
            #print(sum)
        #print('full sum ',sum)
        #print('dh1 ',delta_h1[y])
        delta_h[y] = delta_h1[y] * sum
    
    #print('deltaKH is: ',delta_h,' shape is ',delta_h.shape)    
    actual_nodes_delta_h = delta_h[1:]
    return actual_nodes_delta_h

def updateWeights(answer,d_ko,d_kh,ai,ah,ao,wi,wo,ci,co,lrn_rate,hidden,output,momentum):
    """
    The weights are updated using the forumula from the text
    w_ji <-- w_ji + Delta(w_ji)
    
    where
    Delta(w_ji) = n*delta_j*x_ji
    """
    
    # create tk the target/answer array of what output should be
    tk = int(answer)
    #print('tk is ', tk)
    tkArray = np.zeros(10)
    tkArray[tk]=1
    
    # update the weights connecting hidden to output
    # the co array represents the n-1 or prior iteration delta value
    for j in range(hidden+1):
        for k in range(output):
            delta = lrn_rate * d_ko[k] * ah[j] + (momentum * co[j][k])
            #print('delta is ',delta)
            wo[j][k] += delta
            co[j][k] = delta

    # update the weights connecting input to hidden
    for i in range(input+1):      # add in w0 threshold term
        for j in range(hidden): # add in w0 threshold term
            delta = lrn_rate * d_kh[j] * ai[i] + momentum * ci[i][j]
            wi[i][j] += delta
            ci[i][j] = delta
         

    # calculate error
    error = 0.0
    for k in range(output):
        #print('tk is ',tkArray[k],' ouput is ',ao[k])
        error += (tkArray[k] - ao[k]) ** 2 
        
    #print('the summed square error is ',error)
    
    return answer,d_ko,d_kh,ai,ah,ao,wi,wo,ci,co,lrn_rate,hidden,output,momentum
    
def ReadInFiles(path,trnORtst):
    # This reads in all the files from a directory filtering on what the file
    # starts with
    fullData = []
    fnames = os.listdir(path)
    for fname in fnames:
        if fname.startswith(trnORtst):
            #print (fname)
            data = np.loadtxt(path + "\\" + fname)
            fullData.append(data)
    numFiles = len (fullData)
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
    trnNum = 1000
    tstNum = 890
    
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
    
    HeatMap(my_data[40,1:])
    
    # randomize the rows for better training?
    np.random.shuffle(my_data)
    
    inNum,cols = my_data.shape
    print('num rows ',inNum)
    
    initialize(784, 4, 10, its=50, eps = 3,lrn = 0.7, mom = 0.7, rd = 0.01)

    print_params()
    
    # set up array of 1s for activations
    ai = np.ones(input+1) # plus one for the bias/threshold unit
    ah = np.ones(hidden+1) # plus one for the bias/threshold unit
    ao = np.ones(output)
    
    #print ('input array inited',ai)


    # create random weights between -0.05 and 0.05 as per text on pg. 98
    # plus one for the bias/threshold unit
    wi = np.random.uniform(-0.05, 0.05, size = (input+1, hidden+1))
    wo = np.random.normal(-0.05, 0.05, size = (hidden+1, output))
    
    #print('wi array random inited',wi)
    
    # temporary arrays to hold the numbers to be updated each iteration
    ci = np.zeros((input+1, hidden+1))
    co = np.zeros((hidden+1, output))
    
    # Iterate over the range of total images for training
    for imgNum in range(inNum):
        answer,ai,ah,ao,wi,wo,ci,co = feedForward(imgNum,my_data,ai,ah,ao,wi,wo,ci,co)
        
        #print ('answer is ',answer)
        
        # Calculate the error term deltaKO for each output unit
        deltaKO = calcDeltaKO(answer,ao)
        
        # Calculate the error term deltaKH for each hidden unit
        deltaKH = calcDeltaKH(ah,wo,deltaKO)
        
        #print ('delta h is ',deltaKH)
        
        answer,d_ko,d_kh,ai,ah,ao,wi,wo,ci,co,lrn_rate,hidden,output,momentum = updateWeights(answer,deltaKO,deltaKH,ai,ah,ao,wi,wo,ci,co,lrn_rate,hidden,output,momentum)
    
    # Read in the test data
    dpath2 = os.getcwd()+'\data3'
    dataset2 = ReadInFiles(dpath2,'test')
    my_test = ReadInOneList(dataset2,tstNum) 
    
    print(' shape ',my_test.shape)
    
    tstNum,cols = my_test.shape
    print('num rows ',tstNum)
    
    # Convert the 0-255 to 0 or 1 values in data
    my_test[:,1:] /= 255.0
    
    print (my_test)
    
    
    
    
    accuracyList = []
    
    for imgNum in range(tstNum):
    
        answer,ai,ah,ao,wi,wo,ci,co = feedForward(imgNum,my_test,ai,ah,ao,wi,wo,ci,co)
        print(ao)
        testAnswer = ao.argmax(axis=0)
        print('Test Answer is ',testAnswer)
        if (testAnswer - answer == 0):
            accuracyList.append(1)
        else:
            accuracyList.append(0)
    
    print(accuracyList)
    right = sum(accuracyList)
    total = len(accuracyList)
    print('Results of ',right,' out of ',total,' accuracy is ',right/total)
        

    
main()