# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 09:18:48 2021

@author: 82108
"""

"""
chapter04 : vairous Algorithms
- Logistic Regression
- Stochastic Gredient Descent
"""

# Data load
import pandas as pd
fish = pd.read_csv("https://bit.ly/fish_csv_data")
fish.head()
'''
  Species  Weight  Length  Diagonal   Height   Width
0   Bream   242.0    25.4      30.0  11.5200  4.0200
1   Bream   290.0    26.3      31.2  12.4800  4.3056
2   Bream   340.0    26.5      31.1  12.3778  4.6961
3   Bream   363.0    29.0      33.5  12.7300  4.4555
4   Bream   430.0    29.0      34.0  12.4440  5.1340
'''

# Data Preprocessing

## input, target split
fish_input = fish.iloc[:, 1:].to_numpy()
fish_target = fish['Species'].to_numpy()
print(fish_input.shape, fish_target.shape) # (159, 5) (159,)

## train, test split
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

## standardization
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


##########################################
#####      Logistic Regression      #####
##########################################

## Sigmoid function (= Logistic function)
import numpy as np
import matplotlib.pyplot as plt
z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel("z")
plt.ylabel("phi")
plt.show()


# 1. Binary Classification

## Boolean Indexing (choosing "Bream", and "Smelt")
bream_smelt_indexes = (train_target == "Bream") | (train_target == "Smelt")
scaled_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(scaled_bream_smelt, target_bream_smelt)
print(lr.predict(scaled_bream_smelt[:5])) 
# ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
print(lr.classes_)
# ['Bream' 'Smelt']
print(lr.predict_proba(scaled_bream_smelt[:5]))
'''
[[0.99759855 0.00240145] => 'Bream'
 [0.02735183 0.97264817] => 'Smelt'
 [0.99486072 0.00513928] => 'Bream'
 [0.98584202 0.01415798] => 'Bream'
 [0.99767269 0.00232731]] => 'Bream'
'''

## check by scipy
decisions = lr.decision_function(scaled_bream_smelt[:5])
print(decisions)
'''
z = [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]
'''

from scipy.special import expit 
print(expit(decisions))
# [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]


# 1. multiclass Classification

## Hyperparameters
# 1) C
# The higher the value of C, the more relaxed the regulation. (Default = 1)
# 2) max_iter
# The number of repetitions. (Default = 100)

lr = LogisticRegression(max_iter = 1000, C = 20)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target)) # 0.9327731092436975
print(lr.score(test_scaled, test_target)) # 0.925

print(lr.predict(test_scaled[:5]))
# ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']

print(lr.classes_)
# ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

print(lr.predict_proba(test_scaled[:5]))
'''
[[7.24999757e-06 1.35120672e-02 8.41274357e-01 3.14319571e-04
  1.35668813e-01 6.67127434e-03 2.55191903e-03] => 'Perch'(84%)
 [7.15103975e-09 2.55598054e-03 4.39087387e-02 3.37995722e-05
  7.31066123e-03 9.46185557e-01 5.25560284e-06] => 'Smelt'(95%)
 [1.86557893e-05 2.79657553e-06 3.40599548e-02 9.34804631e-01
  1.50477198e-02 1.60365798e-02 2.96619937e-05] => 'Pike'(94%)
 [1.09325322e-02 3.40497146e-02 3.05543068e-01 6.60899677e-03
  5.66578217e-01 6.87250775e-05 7.62187465e-02] => 'Roach'(57%)
 [4.48988929e-06 3.67289269e-04 9.04002854e-01 2.41270652e-03
  8.94741521e-02 2.40965219e-03 1.32885575e-03]] => 'Perch'(90%)
'''

'''
binary classification => Sigmoid
multiclass classification => Softmax
'''

## check by scipy
decisions = lr.decision_function(test_scaled[:5])
print(decisions)
'''

[[ -6.49807915   1.03225815   5.16359283  -2.72867007   3.33889172
    0.32648589  -0.63447937]
 [-10.85943868   1.92725003   4.77092747  -2.39849302   2.97814784
    7.84125281  -4.25964645]
 [ -4.33527185  -6.23303272   3.17444926   6.48666447   2.35755339
    2.42119926  -3.87156181]
 [ -0.68335257   0.45272612   2.64699522  -1.18666366   3.26451961
   -5.75273664   1.25851192]
 [ -6.39704142  -1.99271973   5.81571833  -0.11036503   3.50283559
   -0.11163177  -0.70679596]]
'''

from scipy.special import softmax
proba = softmax(decisions, axis = 1)
print(np.round(proba, decimals=3))
'''
[[0.    0.014 0.841 0.    0.136 0.007 0.003]
 [0.    0.003 0.044 0.    0.007 0.946 0.   ]
 [0.    0.    0.034 0.935 0.015 0.016 0.   ]
 [0.011 0.034 0.306 0.007 0.567 0.    0.076]
 [0.    0.    0.904 0.002 0.089 0.002 0.001]]
'''


##########################################
#####  Stochastic Gredient Descent  #####
##########################################

'''
In this chapter, I do not deal with the concept of "Deep Learning,"
which means that only the SGD algorithm in terms of "Machine Learning" is covered.
'''

from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
'''
loss = 'log' => logistic loss function  /  default = 'hinge'
max_iter = 10 => epochs = 10  /  default = '1000'
'''
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) # 0.773109243697479
print(sc.score(test_scaled, test_target)) # 0.775

# plus 1 epoch
sc.partial_fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) # 0.8151260504201681
print(sc.score(test_scaled, test_target)) # 0.85
# train_score < test_score => "Underfitting"


## Finding the optimal max_iter value
sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)

for i in range(0, 300) :
    sc.partial_fit(train_scaled, train_target, classes = classes)
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

'''
According to the image, 100 is the best value of max_iter.
'''

sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) # 0.957983193277311
print(sc.score(test_scaled, test_target)) # 0.925