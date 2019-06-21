import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image

class build_model ():
    def __init__ (self):
        self.numOfTrainingexamples = 0
        self.x = 0
        self.w = 0
        self.b = 0
        self.y = 0
        self.learning_rate = 0
        self.numOfepochs = 0
    #------------------------------------------------
    def set_y (self , y):
        self._y = y
    def get_y (self):
        return self._y
    #------------------------------------------------
    def set_x (self , x):
        self._x = x
    def get_x (self):
        return self._x
    #------------------------------------------------
    def set_numOfTrainingexamples (self , numOfTrainingexamples):
        self._numOfTrainingexamples = numOfTrainingexamples
    def get_numOfTrainingexamples (self):
        return self._numOfTrainingexamples
     #------------------------------------------------
    def set_weight (self , w):
        self._w = w
    def get_weight (self):
        return self._w
    #------------------------------------------------
    def set_bias (self ,b):
        self._b = b
    def get_bais (self):
        return self._b
    #------------------------------------------------
    def set_learningrate (self ,learning_rate):
        self._learning_rate =learning_rate
    def get_learning_rate (self):
        return self._learning_rate
    #------------------------------------------------
    def set_numOfepochs (self ,numOfepochs):
        self._numOfepochs =numOfepochs
    def get_numOfepochs (self):
        return self._numOfepochs
    #------------------------------------------------  
    def sigmoid (self , z):
        return 1. / ( 1 + np.exp(-z))
    #------------------------------------------------
    def forword_back_probagation (self ,weight , bais , training_x , training_y , numOfTrainingexample):
        A = self.sigmoid (np.dot (weight.T ,training_x) + bais)
        cost = -1. /numOfTrainingexample * np.sum ((training_y*np.log(A) + (1 -training_y)*np.log(1-A)), axis=1)
        dw = 1 /numOfTrainingexample * np.dot (training_x , (A - training_y).T)
        db = 1 /numOfTrainingexample * np.sum (A - training_y  ,axis=1)
        gradient = {"dw": dw  , "db": db}
        return gradient , cost 
    #------------------------------------------------
    def Training_model (self ,numberOfepochs, learningRate ,weight , bais , training_x , training_y , numOfTrainingexample , print_cost = False):
        costs =[]
        for i in range (numberOfepochs):
            gradient , cost = self.forword_back_probagation (weight , bais , training_x , training_y , numOfTrainingexample)
            dw = gradient["dw"]
            db = gradient ["db"]
            weight = weight - learningRate * dw
            bais = bais - learningRate *db
            if i % 100 == 0:
                costs.append(cost)
            if print_cost and i % 100 == 0:
                print ("iteration " + str(i) + " cost = " + str (cost) )
        
        params = {"w": weight,"b": bais}
        grads = {"dw": dw,"db": db}

        return params, grads, costs
    #------------------------------------------------
    #Testing model using test data set and predict the label
    def Testing_model (self , params , grads ,testing_X , testing_Y):
        numOftestingexamples = testing_Y.shape[1]
        predeiction = np.zeros (shape = (1,numOftestingexamples))
        weight = params ["w"]
        bais = params ["b"]
        A = self.sigmoid (np.dot (weight.T ,testing_X) + bais) #1*m
        for i in range (numOftestingexamples):
            if A[0][i] >= 0.5:
                predeiction[0][i] = 1
            else:
                predeiction[0][i] = 0
        return predeiction










