# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:43:34 2018

@author: stefan
"""

 import cv2
import numpy as np
import os
import time

from neural_network import neural_network
from directKeys import UP, SPACE, PressKey, ReleaseKey
from getKeys import key_check
from grabScreen import grab_screen


WIDTH = 70   #window's width for the data file
HEIGHT = 48   #window's height for the data file
LR = 1e-3
##how many times does the algorithm will go back and forth through the data
EPOCHS = 15

# function that returns 1 if we have the wanted object in the picture
def match_template(img, template):
    match_temp = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) #create a table with matched pixels
    threshold = 0.8
    w, h = template.shape[::-1]
    loc = np.where(match_temp >= threshold) #find the object
    ### draw a rectangle and return x-coordinate
    try:
        cv2.rectangle(img, (loc[1][0], loc[0][0]), (loc[1][0] + w, loc[0][0]+h), (0, 255, 255), 2)
        flag = 1
    except:
        flag = 0
        
    return flag
    
MODEL_NAME = 'trex-18kTRY-Epochs {}-LearningRate {}-{}x{}_new_data.model'.format(EPOCHS, LR, WIDTH, HEIGHT)

model = neural_network(WIDTH, HEIGHT, LR)                                                      
model.load(MODEL_NAME)


## function for jumping
## time delay to complete the jump
def jump():
    PressKey(SPACE)
    time.sleep(0.12)
    ReleaseKey(SPACE)

def restart1():
    PressKey(SPACE)
    ReleaseKey(SPACE)

def main():

    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)
    jump()
    restart = cv2.imread('restart.png', 0)
    #last_time = time.time()   
    flag_restart = 0     
    paused = False
    predict = []
    thresh = 0.95 #threshold for the rex to jump
    while(True):
        if not paused:
            screen = grab_screen(region=(175, 90, 350,210))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            flag_restart = match_template(screen, restart)
            #dino_flag = match_template(screen, trex)
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            # resize to something a bit more acceptable for a CNN
            
#             print('Time {} sec'.format(time.time() - last_time))
#            last_time = time.time() 
            
            #make a prediction about current frame
            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]

            if (prediction[0] > thresh):
                jump()
                print('JUMP! Prediction is %.5f' % prediction[0])
#            
            if(flag_restart == 1):
                restart1()
                    
        keys = key_check()
                    
        if 'P' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Paused!')
                paused = True
                time.sleep(1)
 
main()
