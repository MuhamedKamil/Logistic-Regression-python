import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from read_data import loadingData
from model import build_model
def main ():

    train_path ='train_catvnoncat.h5'
    test_path = 'test_catvnoncat.h5'
    ReadTestandTrain_data = loadingData (train_path , test_path)
    TrainingDataX, TrainingDataY, TestingDataX, TestingDataY, Classes = ReadTestandTrain_data.Data_transformation()
    #-----------------------------------------------------------------------------------------------------------------------
    #building_model 
    model_initailize = build_model()
    model_initailize.set_bias(0)
    model_initailize.set_numOfTrainingexamples(TrainingDataX.shape[1])
    model_initailize.set_weight(np.zeros(shape = (TrainingDataX.shape[0] , 1)))
    model_initailize.set_y(TrainingDataY)
    model_initailize.set_x(TrainingDataX)
    modelParameters, gradients, costs = model_initailize.Training_model (2000 , 0.005 ,model_initailize.get_weight() , model_initailize.get_bais(),model_initailize.get_x(),
    model_initailize.get_y(),model_initailize.get_numOfTrainingexamples(),print_cost = True)
    Predictions = model_initailize.Testing_model (modelParameters , gradients,TestingDataX , TestingDataY) 
    #------------------------------------------------------------------------------------------------------------------------
    #check single picture
    test_dataset  = h5py.File('test_catvnoncat.h5', "r")
    TestingDX = np.array(test_dataset["test_set_x"][:]) 
    indexOftest =int(input("please enter index Of test number : "))
    print (Predictions[0][indexOftest])
    plt.imshow(TestingDX [indexOftest])
    plt.show()
   # ---------------------------------------------------------------------------------------------------------------------------
main()
