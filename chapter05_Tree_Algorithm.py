# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:12:53 2021

@author: 82108
"""

"""
chapter05 : Tree Algorithm
- Decision Tree
- Cross Validation
- Grid Search + Random Search
- Ensemble Learning (Random Forest, Extra Forest, 
                     Gradient Boosting, Histogram-based Gradient Boosting)
"""

# Data load
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
wine.head()
'''
   alcohol  sugar    pH  class
0      9.4    1.9  3.51    0.0
1      9.8    2.6  3.20    0.0
2      9.8    2.3  3.26    0.0
3      9.8    1.9  3.16    0.0
4      9.4    1.9  3.51    0.0
'''

'''
0 : Red wine
1 : White wine
'''


# check
wine.describe()
'''
           alcohol        sugar           pH        class
count  6497.000000  6497.000000  6497.000000  6497.000000
mean     10.491801     5.443235     3.218501     0.753886
std       1.192712     4.757804     0.160787     0.430779
min       8.000000     0.600000     2.720000     0.000000
25%       9.500000     1.800000     3.110000     1.000000
50%      10.300000     3.000000     3.210000     1.000000
75%      11.300000     8.100000     3.320000     1.000000
max      14.900000    65.800000     4.010000     1.000000
'''

red_wine = wine[wine['class'] <= 0.5]
white_wine = wine[wine['class'] >= 0.5]


red_wine.describe()
white_wine.describe()
'''
By average,
alcohol : rea <= white
sugar : red <<< white
pH : red >= white
'''



# Data Preprocessing
wine_input = wine.iloc[:, :3].to_numpy()
wine_target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(wine_input, wine_target, test_size=0.2, random_state=42)
print(train_input.shape, test_input.shape) # (5197, 3) (1300, 3)


##########################################
#####         Decision Tree          #####
##########################################

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target)) # 0.996921300750433
print(dt.score(test_input, test_target)) # 0.8584615384615385
'''
There are huge difference between train_score and test_score. (Overfitting)
To solve this problem, I will change the value of the max_depth.
'''


dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target)) # 0.8454877814123533
print(dt.score(test_input, test_target)) # 0.8415384615384616
'''
Even though the accuracy scores decreases, the overfitting problem is solved.
'''

print(dt.feature_importances_)
'''
[0.12345626 0.86862934 0.0079144 ]
Sugar is the most importance factor.
'''


# Visualization
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=3, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()


##########################################
##### Cross Validation & Grid Search #####
##########################################

# 1. Cross Validation
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
'''
{'fit_time': array([0.00451851, 0.00199914, 0.00251675, 0.00199914, 0.00199938]), 
 'score_time': array([0.00099945, 0.00099921, 0.00100088, 0.00100017, 0.00100136]), 
 'test_score': array([0.84230769, 0.83365385, 0.84504331, 0.8373436 , 0.8479307 ])}
'''

import numpy as np
print(np.mean(scores['test_score'])) # 0.8412558303102096

# splitter
from sklearn.model_selection import StratifiedKFold
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score'])) # 0.8335549132947977


# 2. Grid Search
'''
Grid Search = finding the optimal Hyperparameter + Cross Validation
'''
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target)) # 0.9615162593804117
print(gs.best_params_) # {'min_impurity_decrease': 0.0001}
print(gs.cv_results_['mean_test_score']) # [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]


params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
print(gs.best_params_)
# {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
print(np.max(gs.cv_results_['mean_test_score'])) # 0.8683865773302731


# 3. Random Search
from scipy.stats import uniform, randint
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25)}

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
print(gs.best_params_)
# {'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}
print(np.max(gs.cv_results_['mean_test_score'])) # 0.8695428296438884

dt = gs.best_estimator_
print(dt.score(test_input, test_target)) # 0.86


##########################################
#####       Ensemble Learning        #####
##########################################

# 1. Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9973541965122431 0.8905151032797809

rf.fit(train_input, train_target)
print(rf.feature_importances_) # [0.23167441 0.50039841 0.26792718]

# OOB (out of bag)
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_) # 0.8934000384837406


# 2. Extra Forest
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9974503966084433 0.8887848893166506

et.fit(train_input, train_target)
print(et.feature_importances_) # [0.20183568 0.52242907 0.27573525]


# 3. Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.8881086892152563 0.8720430147331015
'''
The advantage of the "Gradient Boosting" is that this Algorithm is good at preventing Overfitting.
This can be prove by the example following.
'''

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
# the default of "n_estimator" is 100.
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9464595437171814 0.8780082549788999

gb.fit(train_input, train_target)
print(gb.feature_importances_)
# [0.15872278 0.68010884 0.16116839]


# 4. Histogram-based Gradient Boosting
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9321723946453317 0.8801241948619236

from sklearn.inspection import permutation_importance
hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
# [0.08876275 0.23438522 0.08027708]

result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
# [0.05969231 0.20238462 0.049]
hgb.score(test_input, test_target) # 0.8723076923076923

## XGBoost
'''
tree_method = 'hist' == Histogram_based Gradient Boosting 
'''
from xgboost import XGBClassifier
xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) # 0.9555033709953124 0.8799326275264677

## LightGBM
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score'])) # 0.935828414851749 0.8801251203079884








