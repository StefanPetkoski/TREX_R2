# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:13:27 2018

@author: stefan
"""

##code for self collecting data

from getKeys import key_check
from grabScreen import grab_screen
import cv2
import numpy as np
import os
import time
from directKeys import SPACE, PressKey, ReleaseKey
from neural_network import neural_network


#width, height, lr, epochs needed for the model (check the train_network.py for that info)
WIDTH = 70   #window's width for the data file
HEIGHT = 48   #window's height for the data file
LR = 1e-3
##how many times does the algorithm will go back and forth through the data
EPOCHS = 15

MODEL_NAME = 'trex-9kTRY-Epocsh {}-LearningRate {}-{}x{}.model'.format(EPOCHS, LR, WIDTH, HEIGHT)



# function that returns 1 if we have the wanted object in the picture
def match_template(img, template):
    match_temp = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) #create a table with matched pixels
    threshold = 0.8
    w, h = template.shape[::-1]
    loc = np.where(match_temp >= threshold) #find the object
    try:
        cv2.rectangle(img, (loc[1][0], loc[0][0]), (loc[1][0] + w, loc[0][0]+h), (0, 255, 255), 2)
        flag = 1
    except:
        flag = 0
        
    return flag
    

model = neural_network(WIDTH, HEIGHT, LR)                                                      
model.load(MODEL_NAME)

#file to save the new training data
file_name = 'new_data.npy'

if os.path.isfile(file_name):
    print('file exist')
    training_data = list(np.load(file_name))
else:
    print('file does not exist')
    training_data = []
    
    
#timer 5 sec
for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)



def jump():
    PressKey(SPACE)
    time.sleep(0.12)
    ReleaseKey(SPACE)
    
def restart1():
    PressKey(SPACE)
    ReleaseKey(SPACE)

    
    
def decision_to_make(prediction, threshold):
    output = [0, 0]
    if prediction[0] > threshold:
        output[0] = 1
        jump()
        print(prediction)
    else:
        output[1] = 1
        
    return output


def main(training_data):
    restart = cv2.imread('restart.png', 0)      #read the image for restart
    flag_restart = False
    prediction = []     
    paused = False
    last_time = time.time()
    threshold = 0.96    #threshold for jumping 
    
    while(True):
        if not paused:
            screen = grab_screen(region = (175, 90, 350,210))   #take part of the screen
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)   #convert it to grayscale
            screen_for_input = cv2.resize(screen, (WIDTH, HEIGHT))  #resize the screen for the prediction
            
            flag_restart = match_template(screen, restart)      #see if the game ended and it needs to restart itself
            
            
            ##check the no of frames per second(optional)
#            print('Time for frame is : {}'.format(time.time() - last_time))
#            last_time = time.time()
            
            if not flag_restart:        #the game is still on
                prediction = model.predict([screen_for_input.reshape(WIDTH, HEIGHT, 1)])[0]     #take the prediction array
                
                output = decision_to_make(prediction, threshold)    #make the decision to jump or not
                
                training_data.append([screen_for_input, output])    #append the screen and the output 
                
                if len(training_data) % 25000 == 0:
                    print(len(training_data))
                    np.save(file_name, training_data)   #store the training data
                    training_data = training_data[-50:] + training_data
             
                
            else:      #the game has ended (it needs to restart)
                 restart1()
                 print(len(training_data))
                 training_data = training_data[:-20] #take off last 20 frames in case of losing (no bad data)
                 print(len(training_data))
                 flag_restart = False
                 time.sleep(2)
                 
        key_pause = key_check()
        
        #check if you need to pause
        if 'P' in key_pause:
            if paused:
                paused = False
                print('Game unpaused')
                time.sleep(1)
            else:
                paused = True
                print('Game paused')
                time.sleep(1)
                
main(training_data)

    



    
    