### CREATED BY JAVIER OLCOZ MAVAYO
### LAST EVALUATION 04/03/2021


### imports
import cv2
import numpy as np
import glob as gb
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, MaxPooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


### class

class Modelo:

    def vgg16_base_1(self, include_top, input_tensor, input_shape):
        """Create a basic model vgg16
        """
        vgg16_base = VGG16(include_top=include_top,
                   input_tensor=input_tensor, input_shape=input_shape)

        return vgg16_base

    def vgg16_model_1(self, vgg16_base, classes):
        """Create our vgg16 model
        """
        print('Adding new layers...')
        output = vgg16_base.get_layer(index = -1).output
        output = Flatten()(output)
        output = Dense(classes, activation='softmax')(output)
        print('New layers added!')
        vgg16_model = Model(vgg16_base.input, output)
        for layer in vgg16_model.layers[:-7]:
            layer.trainable = False
        
        return vgg16_model


    def compile_and_fit(self, vgg16_model, train_generator, epochs, path):
        """ Fit and save the model
        """
        vgg16_model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics =['accuracy'])
        history = vgg16_model.fit_generator(train_generator,
                              epochs=epochs,
                              verbose=1
                              )
        vgg16_model.save(path)

        return history

    def predict_x_test(self, vgg16_model, X_test):
        """ Predicts the X_test
        """
        predictions = vgg16_model.predict(X_test)
        predictions_1 = np.argmax(predictions, axis=-1)

        return predictions_1


    def predict_image(self, path, model):
        """ Recognise the family Simpsons character
            Args: path [string]: path of the image to predict
                model [.h5]: model used to predict
        """

        s = 224

        X_test = []
        files = gb.glob(str(path + '/*.jpg'))
        for file in files: 
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_array = cv2.resize(image , (s,s))
            X_test.append(list(image_array))
        
        X_test = np.array(X_test, dtype = 'float32')
        print(f'X_test shape : {X_test.shape}')
        X_test = X_test/255.0

        predictions = model.predict(X_test)
        predictions_1 = np.argmax(predictions, axis=-1)

        return predictions_1