# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:44:17 2021

@author: 82108
"""

"""
chapter06 : Unsupervised Learning
- Clustering
- K-Means (centroid)
- Principal Component Analysis (Dimensionality Reduction)
"""

# Data load
import numpy as np
path = "C:/Users/82108/Desktop/Study/Educational Background/Programming/개인 공부/7_혼공머신/dataset/"
fruits = np.load(path + 'fruits_300.npy')

# check
print(fruits.shape) # (300, 100, 100)
print(fruits[0,0,:])
'''
[  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   2   1
   2   2   2   2   2   2   1   1   1   1   1   1   1   1   2   3   2   1
   2   1   1   1   1   2   1   3   2   1   3   1   4   1   2   5   5   5
  19 148 192 117  28   1   1   2   1   4   1   1   3   1   1   1   1   1
   2   2   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1]
0 ~ 255
'''

import matplotlib.pyplot as plt
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()


# analyzing
apple = fruits[:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape) # (100, 10000)

fig, axs = plt.subplots(1,3, figsize=(20,5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()


apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fix, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1,2)) # 0 : height, 1 : row, 2 : column
print(abs_mean.shape) # (300,)


##########################################
#####           Clustering           #####
##########################################
apple_index = np.argsort(abs_mean[:100])

fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10+j]], cmap='gray_r')
        axs[i, j].axis("off")
plt.show()


##########################################
#####            K-Means             #####
##########################################

fruits_2d = fruits.reshape(-1, 100*100)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)
print(km.labels_)
print(np.unique(km.labels_, return_counts=True))
# (array([0, 1, 2]), array([111,  98,  91], dtype=int64))


def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)

    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n :
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    
    plt.show()

draw_fruits(fruits[km.labels_==0])

draw_fruits(fruits[km.labels_==1])

draw_fruits(fruits[km.labels_==2])

# cluster center
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

print(km.transform(fruits_2d[100:101]))
# [[3393.8136117  8837.37750892 5267.70439881]]
# Fruits_2d[100] => 0

draw_fruits(fruits[100:101])
# Pineapple

print(km.n_iter_) # 4

## Ellbow

inertia = []

for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2,7), inertia)
plt.xlabel("k")
plt.ylabel("inertia")
plt.show()


##########################################
#####  Principal Component Analysis  #####
##########################################

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)

print(pca.components_.shape) # (50, 10000)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape) # (300, 50)

# restore
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape) # (300, 10000)

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)

for i in [0, 100, 200]:
    draw_fruits(fruits[i:i+100])
    print("\n")

# explained variance
print(np.sum(pca.explained_variance_ratio_)) # 0.9215111031475837






