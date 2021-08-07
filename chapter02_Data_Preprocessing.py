# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 18:31:12 2021

@author: 82108
"""

"""
Chapter02 : Data Preprocessing
- train & test split
- scaling
"""

# Data
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14


# train_test_split by numpy
import numpy as np
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr.shape) # (49, 2)


np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)

print(index)
'''
[13 45 47 44 17 27 26 25 31 19 12  4 34  8  3  6 40 41 46 15  9 16 24 33
 30  0 43 32  5 29 11 36  1 21  2 37 35 23 39 10 22 18 48 20  7 42 14 28
 38]
'''

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]


# Visualization
import matplotlib.pyplot as plt

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()


# K-Nearest Neighbors Test
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_input, train_target)

knn.score(test_input, test_target) # 1.0
knn.predict(test_input) # [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
test_target # [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]

##############################################################################

fish_data = np.column_stack((fish_length, fish_weight))
fish_trgt = np.concatenate((np.ones(35), np.zeros(14)))


# train_test_split by sklearn
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

print(knn.predict([[25, 150]])) # [0]

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker="^", c="r")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
"""
According to the image, the fish(length=25, weight=150) is included in the bream species.
But, the result of the K-Nearest Neighbors analysis is that the fish is included in smelt species.
"""

distances, indexes = knn.kneighbors([[25, 150]])

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker="^", c="r")
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker="D", c="g")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
"""
four of the most closest five smaples are smelts.

This problem originates from difference in scale.
"""


# PreProcessing
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

train_scaled = (train_input - mean) / std
new = ([25, 150] - mean) / std

plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker="^", c="r")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()


knn.fit(train_scaled, train_target)
knn.predict([new]) # array([1])
"""
The results changes (0:smelt -> 1:bream)
"""

test_scaled = (test_input - mean) / std
knn.score(test_scaled, test_target) # 1.0


distances, indexes = knn.kneighbors([new])

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker="^", c="r")
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

"""
Everyting is fine!
"""





