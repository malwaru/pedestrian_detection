## Essential imports
import glob
import os
from datetime import datetime
import numpy as np
import joblib
import cv2
import json
from matplotlib import pyplot as plt
import copy 
from skimage.feature import hog
from sklearn.svm import LinearSVC
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression


class SvmTrainer():
    def __init__(self,
                file_path,
                model_save_path,                
                data_type="*.png"):
        ## Default values
        print(f"Staring the Classifier")
        self.train_data=[]
        self.train_labels=[]
        self.classifier=LinearSVC()
        self.file_path=file_path        
        self.data_type=data_type
        self.model_save_path=model_save_path
    
    def load_train_data(self,category_path,data_class):
        '''
        Load the images from the filepath and get the HoG featues
        If dataclass=1 positive images 
        If dataclass=0 negative images
        '''
        files_found=False
        file_path=os.path.join(self.file_path,category_path)          
        data_type=self.data_type
        for filename in glob.glob(os.path.join(file_path,data_type)):
            files_found=True
            current_image=cv2.imread(filename)
            current_image=cv2.resize(current_image,(64,128))
            # cv2.imshow("test",current_image)
            current_image=cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY)
            hog_features=hog(image=current_image,orientations=9,pixels_per_cell=(6,6),visualize=False,cells_per_block=(2,2))
            self.train_data.append(hog_features)
            self.train_labels.append(data_class)
            
        if files_found:
            print(f" HoG Feature extractor complete for data type {data_class}")
        else:
            raise Exception("No such file or folder")
            
    def train_svm(self):
        '''
        This function calls for training the data set
        '''
        model_save_path=self.model_save_path
        ## Asserts if there are training data
        assert len(self.train_data)>0,f"No training data available"
        # Convert to numpy arrays
        self.train_data = np.float32(self.train_data)
        self.train_labels= np.array(self.train_labels)
        #Fitting the data
        self.classifier.fit(self.train_data,self.train_labels)
        print(f"Training complete saving model at {model_save_path}")
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        model_name=model_save_path+"svm_model_"+dt_string+".dat"
        joblib.dump(self.classifier,model_name)
        print(f"Model saved as at {model_name}")
        
    
        