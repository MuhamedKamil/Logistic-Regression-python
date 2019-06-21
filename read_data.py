import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
class loadingData :

    def __init__ (self , train_path , test_path):
        self.train_path = train_path
        self.test_path = test_path
    

    def loadFromfile (self):
        # TrainingDataX, TrainingDataY, TestingDataX, TestingDataY, Classes

            train_dataset =h5py.File(self.train_path, "r")
            test_dataset  = h5py.File(self.test_path, "r")
            #---------------------------------------------------------
            TrainingDataX = np.array (train_dataset["train_set_x"][:])
            TrainingDataY = np.array (train_dataset["train_set_y"][:])
            #-------------------------------------------------------------
            TestingDataX = np.array(test_dataset["test_set_x"][:]) 
            TestingDataY = np.array(test_dataset["test_set_y"][:])
            #-------------------------------------------------------------- 
            Classes = np.array(test_dataset["list_classes"][:])
            #--------------------------------------------------------------
            return TrainingDataX, TrainingDataY, TestingDataX, TestingDataY, Classes

    def Data_transformation(self):
        TrainingDataX, TrainingDataY, TestingDataX, TestingDataY, classes = self.loadFromfile()
        # TrainingDataX, TrainingDataY, TestingDataX, TestingDataY, Classes
        #----------------------------------------------------------------------------------------------------
        TrainingDataY = TrainingDataY.reshape(1, TrainingDataY.shape[0])
        TestingDataY = TestingDataY.reshape (1, TestingDataY.shape[0])
        TrainingDataX = TrainingDataX.reshape(TrainingDataX.shape[0], -1).T /255
        TestingDataX = TestingDataX.reshape(TestingDataX.shape[0], -1).T /255
       
        return TrainingDataX, TrainingDataY, TestingDataX, TestingDataY, classes


