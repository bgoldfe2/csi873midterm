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
    
    def __init__(self, input, hidden, output, iterations, epochs, lrn_rate, momentum):
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
        self.iterations = iterations
        self.epochs = epochs
        
        # Hyperparameters
        self.lrn_rate = lrn_rate
        self.momentum = momentum
        
        # initialize arrays
        self.input = input + 1 # add 1 for bias node
        self.hidden = hidden + 1   # This bias node add causes trouble
        self.output = output
    
        # set up arrays for the outputs of the nodes
        self.ai = np.ones(self.input) 
        print('ai shape ',self.ai.shape)
        self.ah = np.ones(self.hidden) 
        self.ao = np.ones(self.output)
        
        # create random weights between -0.05 and 0.05 as per text on pg. 98
        # plus one for the bias/threshold unit
        self.wi = np.random.uniform(-0.05, 0.05, size = (self.input, self.hidden))
        self.wo = np.random.normal(-0.05, 0.05, size = (self.hidden, self.output))
        
        # temporary arrays to hold the numbers to be updated each iteration
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def print_params(self):
        
        print ('%-10s ==> %10d' % ('input', self.input))
        print ('%-10s ==> %10d' % ('hidden', self.hidden))
        print ('%-10s ==> %10d' % ('output', self.output))
        print ('%-10s ==> %10d' % ('iterations', self.iterations))
        print ('%-10s ==> %10d' % ('epochs', self.epochs))
        print ('%-10s ==> %10.2f' % ('learn_rate', self.lrn_rate))
        print ('%-10s ==> %10.2f' % ('momentum', self.momentum))
        
    def feedForward(self,image):
        
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
        
        self.ai[0:self.input-1] = image # remember to account for actual value in 0th index
        
        self.ai[self.input-1] = 1  # set the bias/threshold to always be 1
        #print('answer is',answer)
        # hidden activations
        
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):    # removes the bias
                prodAW = self.ai[i] * self.wi[i][j]
                sum += prodAW
            #print ('sum is ',sum)
            self.ah[j] = sigmoid(sum) #what about using tanh here?
            
        # output activations
        self.ah[self.hidden-1] = 1 # set the bias/threshold to always be 1
        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                prodAWO = self.ah[j] * self.wo[j][k]
                sum += prodAWO
            self.ao[k] = sigmoid(sum)
    

    
    def calcDeltaKO(self,answer):
        """
        Calculates the delta_k for each output unit using the formula from the text
        delta_k = ouput_k * (1-ouput_k)*(target_k - output_k)
        """
        
        tk = int(answer)
        #print('tk is ', tk)
        tkArray = np.zeros(10)
        tkArray[tk]=1
        
        delta_k = np.zeros(10)
        
        # Use the derivative of sigmoid times actual - observed
        for y in range(10):
            delta_k[y] = self.ao[y] * (1-self.ao[y]) * (tkArray[y] - self.ao[y])
                
        #print('deltaKO is: ',delta_k,' shape is ',delta_k.shape)
        
        return delta_k
    
    def calcDeltaKH(self,d_k):
        """
        Calculates the delta_h for each hidden unit using the forumula from the text
        delta_h = output_h * (1-ouput_h) * Sum(weight_kh * delta_ko)
        ah is the output of hidden units 
        w_kh is the weight of hidden units or the array in main 'wi'
        d_k is the delta_k returned from the function 'calcDeltaKO' for output nodes
        """
        #print('ah ',self.ah)
        numH = len(self.ah)
        rows,cols = self.wo.shape
        #print ('num rows,cols of wo ',rows, '-',cols,' shape is ',self.wo.shape)
        #print('shape of d_k ',d_k.shape)
        # Use derivative term for sigmoid applied to the hidden outputs
        delta_h1 = np.zeros(numH)
        for z in range(numH):
            delta_h1[z] = self.ah[z] * (1 - self.ah[z])
        
        
        delta_h = np.zeros(numH)
        # corrected - switched rows and columns
        for y in range(rows):
            sum = 0.0
            #print('y is ',y)
            for x in range(cols):
                #print('x is ',x)
                # Sum up all of the outputs this hidden node touches
                delta_h2 = self.wo[y,x] * d_k[x]
                sum += delta_h2
                #print(sum)
                       
            # Calculate the derivative times the sum of the wo * d_k
            dh = delta_h1[y] * sum # simulate the threshold I took out
            #print('delta h is ',dh)
            delta_h[y] = dh
        
        #print('deltaKH is: ',delta_h,' shape is ',delta_h.shape)    
        return delta_h
    
    def updateWeights(self,answer,d_ko,d_kh):
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
        for j in range(self.hidden):
            for k in range(self.output):
                delta = self.lrn_rate * d_ko[k] * self.ah[j] + (self.momentum * self.co[j][k])
                #print('delta is ',delta)
                self.wo[j][k] += delta
                self.co[j][k] = delta
    
        # update the weights connecting input to hidden
        for i in range(self.input-1):      # add in w0 threshold term
            for j in range(self.hidden): # add in w0 threshold term
                delta = self.lrn_rate * d_kh[j] * self.ai[i] + self.momentum * self.ci[i][j]
                self.wi[i][j] += delta
                self.ci[i][j] = delta/self.lrn_rate  #w_ij does not contain lrn rate
             
    
        # calculate error
        error = 0.0
        for k in range(self.output):
            #print('tk is ',tkArray[k],' ouput is ',self.ao[k])
            error += (tkArray[k] - self.ao[k]) ** 2 
            
        #print('the summed square error is ',error)
        
        return error
        


def main():
    
    # Theses are the number counts for training and test data sets
    trnNum = 1000
    tstNum = 890
    
    dpath = os.getcwd()+'\data3'
    
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
    
    # Recommended to mix the data to avoid overfitting last trained
    # randomize the rows for better training?
    np.random.shuffle(my_data)
    
    inNum,cols = my_data.shape
    print('num rows ',inNum)
    
    myNet = NeuralNet(784, 2, 10, 50, 3,lrn_rate=0.3, momentum = 0.01)

    myNet.print_params()
    
    just_img_data = my_data[:,1:]
    answer = my_data[:,0]
    # Iterate over the range of total images for training
    for imgNum in range(inNum):
        myNet.feedForward(just_img_data[imgNum,:])
        
        #print ('answer is ',answer)
        
        # Calculate the error term deltaKO for each output unit
        deltaKO = myNet.calcDeltaKO(answer[imgNum])
        
        # Calculate the error term deltaKH for each hidden unit
        deltaKH = myNet.calcDeltaKH(deltaKO)
        
        #print ('delta h is ',deltaKH)
        
        error = myNet.updateWeights(answer[imgNum],deltaKO,deltaKH)
    
    # Read in the test data
    dpath2 = os.getcwd()+'\data4'
    dataset2 = ReadInFiles(dpath2,'test')
    my_test = ReadInOneList(dataset2,tstNum) 
    
    print(' shape ',my_test.shape)
    
    tstNum,cols = my_test.shape
    print('num rows ',tstNum)
    
    # Convert the 0-255 to 0 or 1 values in data
    my_test[:,1:] /= 255.0
    
    print (my_test)
    
    just_test_data = my_test[:,1:]
    answerImg = my_test[:,0]
    
    print('test image size array ',just_test_data.shape)
    accuracyList = []
    
    for imgNum in range(tstNum):
    
        myNet.feedForward(just_test_data[imgNum,:])
        HeatMap(just_test_data[imgNum])
        print(myNet.ao)
        testAnswer = myNet.ao.argmax(axis=0)
        print('Test Answer is ',testAnswer, ' image answer is ',answerImg[imgNum])
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
    np.savetxt('output\\wo.csv', accuracyList, delimiter=',')    
    
    
main()