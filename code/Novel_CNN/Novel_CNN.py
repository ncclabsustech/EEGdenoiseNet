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

def unet5(input_size = ( 1024, 1)):
    inputs = Input(input_size)
    conv1 = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling1D(pool_size= 2)(conv1)

    conv2 = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling1D(pool_size= 2)(conv2)

    conv3 = layers.Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling1D(pool_size= 2)(conv3)

    conv4 = layers.Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling1D(pool_size = 2)(conv4)

    conv5 = layers.Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    pool5 = MaxPooling1D(pool_size = 2)(conv5)

    conv6 = layers.Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv6 = layers.Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    pool6 = MaxPooling1D(pool_size = 2)(conv6)

    conv7 = layers.Conv1D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool6)
    conv7 = layers.Conv1D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    #######
    flatten1 = layers.Flatten()(conv7)
    output1 = layers.Dense(1024)(flatten1)
    model = Model(inputs = inputs, outputs = output1)

    model.summary()

    return model

