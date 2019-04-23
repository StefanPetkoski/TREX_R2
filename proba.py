# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 22:32:10 2018

@author: stefan
"""

from grabScreen import grab_screen
import cv2
import numpy as np
import os
import time

while(True):
    screen = grab_screen(region=(175, 90, 350,210)) #dimensions of the screen that we will you
                                                             
    #screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)   #convert it to gray
    cv2.imshow('pic', screen)