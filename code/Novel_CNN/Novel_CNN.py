import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

def Novel_CNN(input_size = ( 1024, 1)):
    inputs = Input(input_size)
    conv1 = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = AveragePooling1D(pool_size= 2)(conv1)

    conv2 = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = AveragePooling1D(pool_size= 2)(conv2)

    conv3 = layers.Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = AveragePooling1D(pool_size= 2)(conv3) #9

    conv4 = layers.Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = AveragePooling1D(pool_size = 2)(drop4)  #13

    conv5 = layers.Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)  #14
    conv5 = layers.Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)
    ###
    pool5 = AveragePooling1D(pool_size = 2)(drop5)

    conv6 = layers.Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)  #18
    conv6 = layers.Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    drop6 = layers.Dropout(0.5)(conv6)

    pool6 = AveragePooling1D(pool_size = 2)(drop6)

    conv7 = layers.Conv1D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)  #22
    conv7 = layers.Conv1D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    drop7 = layers.Dropout(0.5)(conv7)
    #######
    flatten1 = layers.Flatten()(drop7)
    #output1 = layers.Dense(2048)(flatten1)
    output1 = layers.Dense(1024)(flatten1)
    model = Model(inputs = inputs, outputs = output1)

    model.summary()
    return model

