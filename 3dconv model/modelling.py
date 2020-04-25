# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 12:01:19 2019

@author: Martin Ho
"""

from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoints, EarlyStopping
from keras.layers.convolutional import MaxPooling3D, Conv3D
from collections import deque
import sys
import os
import time
    
''' Modelling '''
def conv_3d(nb_classes, input_shape):
    model = Sequential()
    model.add(Conv3D(
            32, (3,3,3), activation='relu', input_shape=input_shape
        ))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(Conv3D(64, (3,3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(Conv3D(128, (3,3,3), activation='relu'))
    model.add(Conv3D(128, (3,3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    model.add(Conv3D(256, (2,2,2), activation='relu'))
    model.add(Conv3D(256, (2,2,2), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def c3d(nb_classes, input_shape):
    """
    Build a 3D convolutional network, aka C3D.
        https://arxiv.org/pdf/1412.0767.pdf
    With thanks:
        https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
    """
    nb_classes = 2
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv1',
                     subsample=(1, 1, 1),
                     input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv3a',
                     subsample=(1, 1, 1)))
    model.add(Conv3D(256, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv3b',
                     subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv4a',
                     subsample=(1, 1, 1)))
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv4b',
                     subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool4'))

     # 5th layer group
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv5a',
                     subsample=(1, 1, 1)))
    model.add(Conv3D(512, 3, 3, 3, activation='relu',
                     border_mode='same', name='conv5b',
                     subsample=(1, 1, 1)))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool5'))
    model.add(Flatten())

    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def train(data, seq_length, image_shape, batch_size=3, nb_epochs=100):
    # Model Checkpoints - Saves the weights of the model between epochs
    checkpoints = ModelCheckpoints('./weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
    # TensorBoard - visualization of the changing cost function and optimizaation process
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))
    # EarlyStopper - Stop the training of the model when the cost function doesnt change much anymore
    early_stopper = EarlyStopping(patience=5)
    # CSVLogger - 
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))
    
    steps_per_epoch = (len(data) * 0.7) // batch_size
    
    3dconv = c3d(nb_classes=2, input_shape=?????)
    3dconv.model.fit(X, y, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1,
                     callbacks=[tb, early_stopper, checkpoints, csv_logger], epochs=nb_epoch)

def main():
    '''
    1. data collection
    2. data processing
    3. push through training the model
    4. hopefully output works
    '''

if __name__ == '__main__':
    main()