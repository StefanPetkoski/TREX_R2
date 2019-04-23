# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 18:42:43 2018

@author: stefan
"""
from getKeys import key_check
from grabScreen import grab_screen
import cv2
import numpy as np
import os
import time

WIDTH = 70   #window's width for the data file
HEIGHT = 48   #window's height for the data file

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
    

# function that outputs if space key is pressed or not
def keys_to_output(keys):
    #[ ,nothing]
    output = [0, 0]
    
    if ' ' in keys:
        output[0] = 1
    else:
        output[1] = 1
        
    return output
     
file_name = 'training_data_without_trex_full_size.npy'  #file with saved data

if os.path.isfile(file_name):
    print('file exist')
    training_data = list(np.load(file_name))
else:
    print('file does not exist')
    training_data = []

def main(training_data):

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    restart = cv2.imread('restart.png', 0)  #template picture restart
    #trex = cv2.imread('t-rex.png', 0)
    last_time = time.time()   
    flag_restart = 0     
    paused = False
    while(True):
        if not paused:
            screen = grab_screen(region=(175, 90, 350,210)) #dimensions of the screen that we will you
                                                             
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)   #convert it to gray
#            cv2.imshow('image', screen)    #show the screen optional(loss of frames)
#            if cv2.waitKey(25) & 0xFF == ord('q'):
#                cv2.destroyAllWindows()
#                break

            flag_restart = match_template(screen, restart)
            screen = cv2.resize(screen, (WIDTH,HEIGHT))    
                                                                                                                         
                             
            # resize to something a bit more acceptable for a CNN
            
            if (flag_restart == 0) :
                keys = key_check()
                output = keys_to_output(keys)
                #print(output)
                training_data.append([screen,output])
        
              
                if len(training_data) % 5000 == 0:
                    print(len(training_data))
                    np.save(file_name,training_data)
                    training_data = training_data[-50:] + training_data
                    print('save done')
            else:
                training_data = training_data[:-50]    #delete last 50 samples in case of restart
                print(len(training_data))
                time.sleep(2)
#        
#                    
        keys = key_check()
                    
        if 'P' in keys: #press P for pause
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

main(training_data)
