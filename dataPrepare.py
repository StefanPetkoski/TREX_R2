# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:54:24 2018

@author: stefan
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle

train_data = np.load('new_data.npy')

df = pd.DataFrame(train_data)
#print(df.head())
shuffle(train_data)

print(Counter(df[1].apply(str)))

##The goal here is to take equal number of cases that t-rex jumped or not
jumps = []
no_jumps = []



for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0]:
        jumps.append([img,choice])
    elif choice == [0,1]:
        no_jumps.append([img,choice])


no_jumps = no_jumps[:len(jumps)]
jumps = jumps[:len(no_jumps)]


final_data = jumps + no_jumps
print(len(final_data))
shuffle(final_data)

np.save('new_data_processed.npy', final_data)   #new data that we will use for the cnn