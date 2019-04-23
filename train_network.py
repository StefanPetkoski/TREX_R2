# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:27:36 2018

@author: stefan
"""

import numpy as np
from neural_network import neural_network

##width and height of the window that we will use
WIDTH = 70   #window's width for the data file
HEIGHT = 48   #window's height for the data file
LR = 1e-3
##how many times does the algorithm will go back and forth through the data
EPOCHS = 15
BatchSize = 300
MODEL_NAME = 'trex-18kTRY-Epochs {}-LearningRate {}-{}x{}_new_data.model'.format(EPOCHS, LR, WIDTH, HEIGHT)
train_data = np.load('new_data_processed.npy')


model = neural_network(WIDTH, HEIGHT, LR)    
    
train = train_data[:-600]
test = train_data[-600:]
    
X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
Y = [i[1] for i in train]
    
test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
test_y = [i[1] for i in test]
    
model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
              snapshot_step=100, show_metric=True, run_id=MODEL_NAME, batch_size = BatchSize)

model.save(MODEL_NAME)
## run it in command prompt for live data of accuracy and loss of the network
## cd C:\Users\stefa\Anaconda3\Scripts
##   %tensorboard% --logdir=foo:G:\R2\T_REX_AI_SMALL\log --port=8088 --host=localhost 
