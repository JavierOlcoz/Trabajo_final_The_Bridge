### Created by Javier Olcoz 04/03/2021
### Last evaluation: 04/03/2021 

### Imports
import numpy as np 
import pandas as pd 
import os
import glob as gb 
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model



### Class


class Folders:

    def training_datagen(self, rescale, featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, rotation_range, width_shift_range, height_shift_range, horizontal_flip, vertical_flip):
        """ This function creates a datagen that is gonna be used ater to create a generator
            Args: rescale [float]
                featurewise_center [bool]
                samplewise_center [bool]
                featurewise_std_normalization [bool]
                samplewise_std_normalization [bool]
                rotation_range [float]
                width_shift_range [float]
                height_shift_range [float]
                horizontal_flip [bool]
                vertical_flip [bool]
        """
        training_datagen = ImageDataGenerator(rescale=rescale,
        featurewise_center=featurewise_center, 
        samplewise_center=samplewise_center, 
        featurewise_std_normalization=featurewise_std_normalization, 
        samplewise_std_normalization=samplewise_std_normalization, 
        rotation_range=rotation_range, 
        width_shift_range=width_shift_range, 
        height_shift_range=height_shift_range, 
        horizontal_flip=horizontal_flip, 
        vertical_flip=vertical_flip)

        return training_datagen


    def train_generator(self,training_datagen, TRAINING_DIR,target_size, class_mode, batch_size, shuffle):
        """
        Creates an image generator
        Args: TRAINING_DIR
              target_size
              class_mode 
              batch_size
              shuffle
        """
        train_generator = training_datagen.flow_from_directory(TRAINING_DIR,target_size=target_size, class_mode=class_mode, batch_size=batch_size, shuffle=shuffle)

        return train_generator

    def class_indices(self, train_generator):
        """ Returns a dictionary with the classes
        """
        class_map = train_generator.class_indices
        inv_class_map = {v: k for k, v in class_map.items()}
        
        return inv_class_map

    def test_images(self, testpath, s):
        """ Returns an array with the test images normalized
        """
        acum = 0
        for files in  os.listdir(testpath):
            acum += 1
        print(f'For testing data , found {acum} files')

        s = s
        
        X_test = []
        files = gb.glob(str(testpath + '/*.jpg'))
        for file in files: 
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image , (s,s))
            X_test.append(list(image_array))
        
        X_test = np.array(X_test, dtype = 'float32')
        print(f'X_test shape : {X_test.shape}')
        X_test = X_test/255.0

        return X_test


    def load_model(self, path):
        """ Load the model required
            Path [string]: path were the model is saved
        """
        model = load_model(path)

        return model
