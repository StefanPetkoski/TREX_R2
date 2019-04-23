# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:21:39 2018

@author: stefan
"""
##neural network with 10 layers :
## 4 convolutional layers, 4 pooling layers and 2 fully connected layers

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def neural_network(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    
    network = conv_2d(network, 86, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    
    network = conv_2d(network, 128, 5, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    

    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
       
    network = conv_2d(network, 350, 2, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    
    network = local_response_normalization(network)
    
    network = fully_connected(network, 1024, activation='tanh', weight_decay=0.001)
    network = dropout(network, 0.7)
    
    network = fully_connected(network, 1024, activation='tanh', weight_decay=0.001)
    network = dropout(network, 0.7)

    network = fully_connected(network, 2, activation='softmax')
    
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_neural_net',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model


 #%tensorboard% --logdir=foo:G:\R2\T_REX_AI\log --port=8088 --host=localhost 