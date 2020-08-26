#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def Model(input_shape):
    
    # inputs.shape = (batch, length, h, w, c)
    inputs = tf.keras.Input(input_shape[-4:]);
    # layer 1
    results = tf.keras.layers.Conv3D(64, kernel_size = (3,3,3), padding = 'same')(inputs);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.MaxPool3D(pool_size = (1,2,2), strides = (1,2,2), padding = 'same')(results);
    # layer 2
    results = tf.keras.layers.Conv3D(128, kernel_size = (3,3,3), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(results);
    # layer 3
    results = tf.keras.layers.Conv3D(256, kernel_size = (3,3,3), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.Conv3D(256, kernel_size = (3,3,3), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.ReLU()(results);
    results = tf.keras.layers.MaxPool3D(pool_size = (2,2,2), strides = (2,2,2), padding = 'same')(results);
    # layer 4
    
