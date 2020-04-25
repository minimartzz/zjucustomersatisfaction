# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:13:39 2019

@author: Micaela
"""

import cv2 
import os
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.models import Model
import numpy as np 

# Already have the processed data in the form of a numpy array 

# 1st step: introduce the channel dimension in input dataset
""" This step may be unnecessary for the dataset we are dealing with
    because it is already in a numpy array form.""" 
    
#xtrain = np.ndarray((x_train.shape[0],4096,3))
#xtest = np.ndarray((x_test.shape[0],4096,3))
df_angle1_bgra = np.load('D:/Martin Ho/SUTD/DATE 2019/Data/angle_2-bgra-500-130x130.npy', allow_pickle=True)

def to_model(dataset, split=4):
    new_dict = {"data": [],"label": []}
    for dataset in dataset:
        new_dict["data"].append(dataset[0])
        new_dict["label"].append(dataset[1])
    
    train_x = np.array(new_dict['data'][:-split])
    test_x = np.array(new_dict['data'][-split:])
    train_y = np.array(new_dict['label'][:-split])
    test_y = np.array(new_dict['label'][-split:])
    
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = to_model(df_angle1_bgra, 4)

print(train_x.shape)


def generator(data):
    n_images = 50
    
    for imageset in data:
        index = 0
        while index <= 500:
            batch_x = imageset[0][index*n_images:(index+1)*n_images]
            batch_y = imageset[1]
            
            yield (batch_x, batch_y)

# 2nd step: Iterate in train and test, add RGB dimension 
#""" In this step, a class ScalarMappable under Matplotlib is introduced
#    to normalize data before return RGBA colors from the given color map"""

#def add_rgb_dimension(array):
#    scaler_map = cm.ScalarMappable(cmap="Greys")
#    array = scaler_map.to_rgba(array)[:, : -1]
#    return array
#    
#for i in range(x_train.shape[0]):
#    x_train[i] = add_rgb_dimension(x_train[i])
#    
#for i in range(x_test.shape[0]):
#    x_test[i] = add_rgb_dimension(x_test[i])
    
    
# 3rd step: convert to 1 + 4D space (1st argument represents number of rows in the dataset)
#xtrain = x_train.reshape(.shape[0], 16, 16, 16, 3)
#xtest = x_test.reshape(xtest.shape[0], 16, 16, 16, 3)
#
### convert target variable into one-hot
#y_train = keras.utils.to_categorical(y_train, 10)
#y_test = keras.utils.to_categorical(y_test, 10)

## input layer
input_layer = Input((500, 130, 130, 4))

## convolutional layers
conv_layer1 = Conv3D(filters=64, kernel_size=(3, 3, 4), activation='relu')(input_layer)
conv_layer2 = Conv3D(filters=128, kernel_size=(3, 3, 4), activation='relu')(conv_layer1)
conv_layer3 = Conv3D(filters=256, kernel_size=(3, 3, 4), activation='relu')(conv_layer2)

## add max pooling to obtain the most imformatic features
pooling_layer1 = MaxPool3D(pool_size=(2, 2, 4))(conv_layer3)

conv_layer3 = Conv3D(filters=512, kernel_size=(3, 3, 4), activation='relu')(pooling_layer1)
conv_layer4 = Conv3D(filters=1024, kernel_size=(3, 3, 4), activation='relu')(conv_layer3)
conv_layer5 = Conv3D(filters=2048, kernel_size=(3, 3, 4), activation='relu')(conv_layer4)
pooling_layer2 = MaxPool3D(pool_size=(2, 2, 4))(conv_layer5)

## perform batch normalization on the convolution outputs before feeding it to MLP architecture
pooling_layer2 = BatchNormalization()(pooling_layer2)
flatten_layer = Flatten()(pooling_layer2)

## create an MLP architecture with dense layers : 4096 -> 512 -> 10
## add dropouts to avoid overfitting / perform regularization
dense_layer1 = Dense(units=4096, activation='relu')(flatten_layer)
dense_layer1 = Dropout(0.4)(dense_layer1)
dense_layer2 = Dense(units=512, activation='relu')(dense_layer1)
dense_layer2 = Dropout(0.4)(dense_layer2)
output_layer = Dense(units=2, activation='softmax')(dense_layer2)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), verbose=2, epochs=1, shuffle=True)

print(model.summary()) # see the summary of results of the model 

#%%
''' c3d model '''
import numpy as np
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization, ZeroPadding3D
from tensorflow.keras.models import Model



df_angle1_bgra = np.load('D:/Martin Ho/SUTD/DATE 2019/Data/angle_1-bgra-500-130x130.npy', allow_pickle=True)

def to_model(dataset, split=4):
    new_dict = {"data": [],"label": []}
    for dataset in dataset:
        new_dict["data"].append(dataset[0])
        new_dict["label"].append(dataset[1])
    
    train_x = np.array(new_dict['data'][:-split])
    test_x = np.array(new_dict['data'][-split:])
    train_y = np.array(new_dict['label'][:-split])
    test_y = np.array(new_dict['label'][-split:])
    
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = to_model(df_angle1_bgra, 4)

print(train_x.shape)

''' bgra_angle_1 model '''
# Input Layer
bgra_1_input = Input((500, 130, 130, 4))

# Conv Layer 1
conv_layer_1 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(bgra_1_input)
pooling_layer_1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), data_format='channels_last')(conv_layer_1)
# Conv Layer 2
conv_layer_2 = Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(pooling_layer_1)
pooling_layer_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(conv_layer_2)
# Conv Layer 3
conv_layer_3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(pooling_layer_2)
conv_layer_3 = Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(conv_layer_3)
pooling_layer_3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(conv_layer_3)
# Conv Layer 4
conv_layer_4 = Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(pooling_layer_3)
conv_layer_4 = Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(conv_layer_4)
pooling_layer_4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(conv_layer_4)
# Conv Layer 5
conv_layer_5 = Conv3D(filters=512, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(pooling_layer_4)
conv_layer_5 = Conv3D(filters=512, kernel_size=(2, 2, 2), activation='relu', data_format='channels_last')(conv_layer_5)
padding_layer_5 = ZeroPadding3D(padding=(0, 1, 1))(conv_layer_5)
pooling_layer_5 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(padding_layer_5)
flatten_layer = Flatten()(pooling_layer_5)
# Fully Connected Layers
dense_layer_1 = Dense(units=4096, activation='relu')(flatten_layer)
dense_layer_1 = Dropout(0.5)(dense_layer_1)
dense_layer_2 = Dense(units=4096, activation='relu')(dense_layer_1)
dense_layer_2 = Dropout(0.5)(dense_layer_2)
output_layer_1 = Dense(units=2, activation='softmax')(dense_layer_2)

model = Model(inputs=bgra_1_input, outputs=output_layer_1)

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), verbose=1, epochs=3, shuffle=True)

print(model.summary())


















