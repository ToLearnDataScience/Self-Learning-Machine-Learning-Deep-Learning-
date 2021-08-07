# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 17:29:13 2021

@author: 82108
"""

"""
chapter03 : Regression
- K-Nearest Neighbors Regression
- Linear Regression
- Logistic Regression
- Regularization
"""

# Data
import numpy as np

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

## Data Visualization
import matplotlib.pyplot as plt

plt.scatter(perch_length, perch_weight)
plt.xlabel("lengt")
plt.ylabel("weight")
plt.show()

## Data split
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

## Reshape
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape) # (42, 1) (14, 1)

##########################################
##### K-Nearest Neighbors Regression #####
##########################################
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print(knr.score(test_input, test_target)) 
# R^2(Coefficient of Determination) = 0.992809406101064
'''
R^2 (Coefficient of Determination)

R^2 = 1 - sum((target-expectation)^2) / sum((target - mean)^2) 
'''

from sklearn.metrics import mean_absolute_error

test_prediction = knr.predict(test_input)

mae = mean_absolute_error(test_target, test_prediction)
print(mae) # 19.157142857142862

print(knr.score(train_input, train_target)) # 0.9698823289099254

'''
Overfitting vs Underfitting
1) Overfitting 
-> train_score >>> test_score
=> The reason of Overfitting is that "lack of diversity in samples' patterns"


2) Underfitting
-> train_score < test_score
   or "both train_score and test_score are too low"
=> The reason of Underfitting is that "lack of data"


Solution : adding more samples or using "regularization"
'''

## Regularization in K-Nearest Neighbors
knr.n_neighbors = 3 # default = 5

knr.fit(train_input, train_target)
train_score = knr.score(train_input, train_target)
test_score = knr.score(test_input, test_target)

print(train_score, "vs", test_score)
# 0.9804899950518966 vs 0.9746459963987609
# train_score > test_score

"""
There are a huge disadvantage of KNeighborsRegressor.
This regressor works fine only in predicting data included in the range of train_set.
This is supported by the example follwing.
"""

print(knr.predict([[50]])) # [1033.33333333]

distances, indexs = knr.kneighbors([[50]])

plt.scatter(train_input, train_target)
plt.scatter(50, 1033, marker="^", c="r")
plt.scatter(train_input[indexs], train_target[indexs], marker="D")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

print(np.mean(train_target[indexs])) # 1033.3333333333333


##########################################
#####       Linear Regression       #####
##########################################

from sklearn.linear_model import LinearRegression

# Simple Linear Model
lr = LinearRegression()
lr.fit(train_input,train_target)

print(lr.predict([[50]])) # [1241.83860323]

## Simple Linear Model Visualization

plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*lr.coef_ + lr.intercept_, 50*lr.coef_ + lr.intercept_], c="g")
plt.scatter(50, 1241.83860323, marker="^", c="r")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()


# Polynomial Linear Regression
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

print(train_poly.shape, test_poly.shape) # (42, 2) (14, 2)

lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]])) # [1573.98423528]
'''
Simple Linear Regression Prediction = 1241.84
Polynomial Linear Regression Prediction = 1573.98
'''

## Polynomial Linear Model Visualization

point = np.arange(15, 50)
print(lr.coef_, lr.intercept_)
# [  1.01433211 -21.55792498] 116.05021078278259

plt.scatter(train_input, train_target)
plt.plot(point, lr.coef_[0]*point ** 2 + lr.coef_[1]*point + lr.intercept_, c="y")
plt.scatter(50, 1573.98, marker="^", c="r")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# 0.9706807451768623 vs 0.9775935108325121
# train_score < test_score => Underfitting

'''
To solve the underfitting problem, let's use more features.
I will use two more features, height & width.
'''

##########################################
#####      Multiple Regression      #####
##########################################

import pandas as pd
df = pd.read_csv("https://bit.ly/perch_csv_data")
perch_full = df.to_numpy()
print(perch_full)

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

# Polynomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape, test_poly.shape) # (42, 9) (14, 9)

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# 0.9903183436982125 vs 0.9714559911594145 => great!


'''
There are several ways to improve R^2.
- Standardization
- Regularization
'''

# Standardization
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)


# Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
# 0.9857915060511934 vs 0.9835057194929057

## Finding Optimal alpha value
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    ridge = Ridge(alpha = alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel("alpha")
plt.ylabel("R^2")
plt.show()

# -1 = alpha(0.1)

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
# 0.9889354346720892 vs 0.9856564835209132
# -> Both scores increases slightly.


# Lasso Regression
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
# 0.986591255464559 vs 0.9846056618190413

## Finding Optimal alpha value
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    lasso = Lasso(alpha = alpha)
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel("alpha")
plt.ylabel("R^2")
plt.show()

# According to the graph, alpha values under 10 are all fine.

lasso = Lasso(alpha=0.1)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
# 0.9889737410220835 vs 0.9845845445966905


'''
Usually, people prefer "Ridge" to "Lasso."
'''

