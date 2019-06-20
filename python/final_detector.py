# Initialize session
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.layers import Dense, Input, Dropout, Conv2D, Conv1D, Flatten, MaxPooling2D, BatchNormalization, Convolution2D, Activation, Reshape, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adadelta, Adagrad
import numpy as np 
from PIL import Image

  
class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        self.RESOLUTION = 224

        inp = Input(shape=(self.RESOLUTION, self.RESOLUTION, 3))
        base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(self.RESOLUTION,self.RESOLUTION,3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=None, pooling='avg', classes=1000)

        features = base_model.output
        # Bounding box prediction model
        x = Dense(400,activation='sigmoid')(features)
        x = Dropout(0.2)(x)
        x = Dense(4,activation='sigmoid')(x)
        model1 =  Model(inputs=base_model.input, outputs=x)

        # Classification model
        features_before_pooling = base_model.layers[-4].output
        y = Conv2D(30, kernel_size=1, activation='relu')(features_before_pooling)
        y = BatchNormalization()(y)
        y = GlobalAveragePooling2D()(y)
        y = Dense(1)(y)
        model2 = Model(inputs=base_model.input, outputs=y)
        #model2.summary()

        # Complete model
        self.model_tot = Model(inputs=base_model.input, outputs=[x,y])
        self.model_tot.summary()

        # Load weights for complete model
        self.model_tot.load_weights('5th_model.h5')

    def forward(self, img, thresh = 0.2): # default threshold to 0.2
        im = np.array(img)/255
        im_expanded = np.expand_dims(im,0)
        output = self.model_tot.predict(im_expanded)
        bbox = output[0][0]
        if output[1] > thresh:
                binary_two = True
        else:
                binary_two = False
        
        center_x = output[0][0][0]
        center_y = output[0][0][1]
        width = output[0][0][2]
        height = output[0][0][3]

        new_center_x = np.ceil(center_x*self.RESOLUTION)
        new_center_y = np.ceil(center_y*self.RESOLUTION)
        new_width = np.ceil(width*self.RESOLUTION)
        new_height = np.ceil(height*self.RESOLUTION)

        bbox = [new_center_x,new_center_y,new_width,new_height]

        return  bbox,binary_two

