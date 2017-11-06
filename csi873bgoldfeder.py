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

def sigmoid2(z):
    try:
        if (z < 0):
            sig = 1.0 - 1.0/(1.0 + math.exp(z))
        else:
            sig = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        if (z > 0):
            sig = 0.01
        else:
            sig = 100
    return sig

def sigmoid(z):
    sig = 1.0 / (1.0 + math.exp(-z))
    return sig

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

class NeuralNet(object):
    
    def __init__(self, input, hidden, output, inNum, epochs, lrn_rate, momentum):
        """
        **Network Parameters**
        input: number of input units
        hidden: number of hidden units
        output: number of output units
        
        **Training Parameters**
        inNum: number of images to run
        epochs: number of full training set run throughs
        
        **Hyperparameters**
        lrn_rate: this is the factor for learning rate
        momentum: this is the momentum term added to weight update
        rate_decay: used for annealing the learning rate using a 1/t decay using
                    the mathematical form alpha=alpha0/a0+(a0+a0*rate_decay)
    
        """
        # Training Parameters 
        self.inNum = inNum
        self.epochs = epochs
        
        # Hyperparameters
        self.lrn_rate = lrn_rate
        self.momentum = momentum
        
        # initialize arrays
        self.input = input # add 1 for bias node
        self.hidden = hidden   # This bias node add causes trouble
        self.output = output
    
        # set up arrays for the outputs of the nodes
        self.ai = np.ones(self.input)  # removes the threshold input 
        print('ai shape ',self.ai.shape)
        self.ah = np.ones(self.hidden) # removes the threshold input
        self.ao = np.ones(self.output)
        
        # create random weights between -0.05 and 0.05 as per text on pg. 98
        # plus one for the bias/threshold unit
        self.wHidThresh = np.random.uniform(-0.05, 0.05, size = (self.hidden))
        self.wOutThresh = np.random.uniform(-0.05, 0.05, size = (self.output))

        self.wi = np.random.uniform(-0.05, 0.05, size = (self.input, self.hidden))
        self.wo = np.random.normal(-0.05, 0.05, size = (self.hidden, self.output))
        
        # temporary arrays to hold the numbers to be updated each iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))
        
        # Error array to capture the output array for each output unit
        self.outUnitErr = np.zeros((self.inNum,self.output))

    def print_params(self):
        
        print ('%-10s ==> %10d' % ('input', self.input))
        print ('%-10s ==> %10d' % ('hidden', self.hidden))
        print ('%-10s ==> %10d' % ('output', self.output))
        print ('%-10s ==> %10d' % ('number of images', self.inNum))
        print ('%-10s ==> %10d' % ('epochs', self.epochs))
        print ('%-10s ==> %10.2f' % ('learn_rate', self.lrn_rate))
        print ('%-10s ==> %10.2f' % ('momentum', self.momentum))
        
    def makeTargetArray(self,answer):
        tk = int(answer)
        #print('tk is ', tk)
        # Make the trained output 0.1 for 0 and 0.9 for 1 as per the text
        # from the first paragraph on the top of page 115
        tkArray = np.add(np.zeros(10),0.1)
        tkArray[tk]= 0.9
        #print("answer array is ",tkArray)
        
        return tkArray
        
    def feedForward(self,image,answer):
        
        #print (input,len(inputs))
        """
        The feedforward algorithm propogates the input forward through the network
        It calculates the net by summing from one to n the weights (including bias 
        w0) times the value sigma term for each node in the hidden and output
        layers.  This is then used as the negative exponent value in the sigmoid
        function
            net = Sigma[i=0,n]wi*xi
            ah = sigmoid(net) = 1/(1 + e**-net)
        """
        #self.ai[0] = 1.0  # This is the threshold for every hidden unit
        self.ai = image # remember to account for actual value in 0th index
        
        # hidden activations
        
        for j in range(self.hidden):
            sum = self.wHidThresh[j]
            for i in range(self.input):    
                prodAW = self.ai[i] * self.wi[i][j]
                sum += prodAW
            #print ('sum is ',sum)
            self.ah[j] = sigmoid(sum) #what about using tanh here?
            
        # output activations
        for k in range(self.output):
            sum = self.wOutThresh[k]
            for j in range(self.hidden):
                prodAWO = self.ah[j] * self.wo[j][k]
                sum += prodAWO
            self.ao[k] = sigmoid(sum)
    
        #self.calculateError(answer)
    
    def calcDeltaKO(self,answer):
        """
        Calculates the delta_k for each output unit using the formula from the text
        delta_k = ouput_k * (1-ouput_k)*(target_k - output_k)
        """
        tkArray= self.makeTargetArray(answer)
        
        delta_k = np.zeros(self.output)
        
        # Use the derivative of sigmoid times actual - observed
        for y in range(self.output):
            delta_k[y] = self.ao[y] * (1-self.ao[y]) * (tkArray[y] - self.ao[y])
            #print ('tkarray[y] ',tkArray[y],' ao[y] ',self.ao[y],' sub ',tkArray[y] - self.ao[y])    
        
        
        return delta_k
    
    def calcDeltaKH(self,d_k):
        """
        Calculates the delta_h for each hidden unit using the forumula from the text
        delta_h = output_h * (1-ouput_h) * Sum(weight_kh * delta_ko)
        ah is the output of hidden units 
        w_kh is the weight of hidden units or the array in main 'wi'
        d_k is the delta_k returned from the function 'calcDeltaKO' for output nodes
        """
       
        # Use derivative term for sigmoid applied to the hidden outputs
        delta_h1 = np.zeros(self.hidden) # first part of equation             
        delta_h = np.zeros(self.hidden)  # output of equation
        # corrected - switched rows and columns
        for hid in range(self.hidden):
            sum = 0.0   # question should this be the threshold value?
            delta_h1[hid] = self.ah[hid] * (1 - self.ah[hid])
            #print('y is ',y)
            # This calculates the second part of equation summing over this hidden
            # node the product of the weight_kh*delta_k of the output node
            for out in range(self.output):
                #print('x is ',x)
                
                # the product of the weight_kh*delta_k of the output node
                delta_h2 = self.wo[hid,out] * d_k[out]
                # Sum up all weight*dk for all output this hidden node touches
                sum += delta_h2
                #print(sum)
                       
            # Calculate the derivative times the sum of the w_kh * delta_k
            dh = delta_h1[hid] * sum 
            #print('delta h is ',dh)
            delta_h[hid] = dh
        
        #print('delta_h is: ',delta_h,' shape is ',delta_h.shape)    
        return delta_h
    
    def updateWeights(self,answer,d_ko,d_kh):
        """
        The weights are updated using the forumula from the text
        w_ji <-- w_ji + Delta(w_ji)
        
        where
        Delta(w_ji) = n*delta_j*x_ji
        """
        # update the weights connecting hidden to output
        # the co array represents the n-1 or prior iteration delta value
        for j in range(self.hidden):
            for k in range(self.output):
                delta = self.lrn_rate * d_ko[k] * self.ah[j] + (self.momentum * self.co[j][k])
                #print('delta is ',delta)
                self.wo[j][k] += delta
                self.co[j][k] = delta
    
        # update the weights connecting input to hidden
        for i in range(self.input):      # add in w0 threshold term
            for j in range(self.hidden): # add in w0 threshold term
                delta = self.lrn_rate * d_kh[j] * self.ai[i] + (self.momentum * self.ci[i][j])
                self.wi[i][j] += delta
                self.ci[i][j] = delta
                
    def calculateError(self,num,answer):
        
        tkArray = self.makeTargetArray(answer)
        for out in range(self.output):
            self.outUnitErr[num][out] = (tkArray[out] - self.ao[out])**2
            
    def plotError(self):
        
        plt.imshow(self.outUnitErr[:,:])
        plt.xlabel('iterations')
        plt.ylabel('error')
        plt.title('Sum of squared errors for each output unite')
        plt.grid(True)
        plt.savefig("test.png")
        plt.show()
                 
    
def main():
    
    # Theses are the number counts for training and test data sets
    trnNum = 100
    tstNum = 500
    
    epochs = 10
    
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
    
    #HeatMap(my_data[40,1:])
    
    # Recommended to mix the data to avoid overfitting last trained
    # randomize the rows for better training?
    np.random.shuffle(my_data)
    
    inNum,cols = my_data.shape
    #print('num rows ',inNum)
    
    myNet = NeuralNet(784, 4, 10, inNum, epochs,lrn_rate=0.3, momentum = 0.3)

    myNet.print_params()
    
    just_img_data = my_data[:,1:]
    answer = my_data[:,0]
    
  
    for eps in range(myNet.epochs):
        print("Training epoch ",eps)
        # Iterate over the range of total images for training
        for imgNum in range(inNum):
            myNet.feedForward(just_img_data[imgNum,:],answer[imgNum])
            
            #print ('answer is ',answer[imgNum])
            
            # Calculate the error term deltaKO for each output unit
            deltaKO = myNet.calcDeltaKO(answer[imgNum])
            #print('deltaKO is: ',deltaKO,' shape is ',deltaKO.shape)
            
            # Calculate the error term deltaKH for each hidden unit
            deltaKH = myNet.calcDeltaKH(deltaKO)
            #print ('delta h is ',deltaKH)
            
            # Update the weights for output and hidden weight sets
            myNet.updateWeights(answer[imgNum],deltaKO,deltaKH)
            
            # Calculate the error per image per output unit
            myNet.calculateError(imgNum,answer[imgNum])
        
        errPerEpoch = np.sum(myNet.outUnitErr,dtype='float')
        print("Total Error for epoch ",eps," is ",errPerEpoch)
    #np.savetxt('output\\error.csv', myNet.outUnitErr, delimiter=',')
    #myNet.plotError()   
    #print('The last image trained was ',answer[-1])    
        
        
    # Read in the test data
    dpath2 = os.getcwd()+'\data3'
    dataset2 = ReadInFiles(dpath2,'test')
    my_test = ReadInOneList(dataset2,tstNum) 
    
    #print(' shape ',my_test.shape)
    
    tstNum,cols = my_test.shape
    #print('num rows ',tstNum)
    
    # Convert the 0-255 to 0 or 1 values in data
    my_test[:,1:] /= 255.0
    
    #print (my_test)
    
    just_test_data = my_test[:,1:]
    answerImg = my_test[:,0]
    
    #print('test image size array ',just_test_data.shape)
    accuracyList = []
    
    for imgNum in range(tstNum):
    
        myNet.feedForward(just_test_data[imgNum,:],answerImg[imgNum])
        #HeatMap(just_test_data[imgNum])
        #print(myNet.ao)
        testAnswer = myNet.ao.argmax(axis=0)
        #print('Test Answer is ',testAnswer, ' image answer is ',answerImg[imgNum])
        if (testAnswer - answerImg[imgNum] == 0):
            accuracyList.append(1)
        else:
            accuracyList.append(0)
    
    print(accuracyList)
    right = sum(accuracyList)
    total = len(accuracyList)
    print('Results of ',right,' out of ',total,' accuracy is ',right/total)
    
    print ('wi ', myNet.wi)
    print ('wo ', myNet.wo)
    np.savetxt('output\\wi.csv', myNet.wi, delimiter=',')
    np.savetxt('output\\wo.csv', myNet.wo, delimiter=',')
    np.savetxt('output\\accuracyList.csv', accuracyList, delimiter=',')

    
    
    
main()