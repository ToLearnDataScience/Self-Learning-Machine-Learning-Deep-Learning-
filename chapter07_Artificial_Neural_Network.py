# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:52:32 2021

@author: 82108
"""

"""
chapter07 : Artificial Neural Network
- Artificial Neural Network (input layer, output layer)
- Deep Neural Network (hidden layer, relu, optimizer)
- Training (dropout, callback)
"""

# Data load
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()


# check
print(train_input.shape, test_input.shape) # (60000, 28, 28) (10000, 28, 28)


import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis("off")
plt.show()

'''
0 : T-shirt
1 : Pants
2 : Sweater
3 : Dress
4 : Coat
5 : Sandals
6 : Shirts
7 : Sneakers
8 : Bag
8 : Ankle Boots
'''

import numpy as np
print(np.unique(train_target, return_counts=True))
'''
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
 array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))
'''

# Data scaling + split
train_scaled = train_input / 255
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape) # (60000, 784)

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(train_scaled.shape, train_target.shape) # (48000, 784) (48000,)
print(val_scaled.shape, val_target.shape) # (12000, 784) (12000,)


##########################################
#####   Artificial Neural Network    #####
##########################################

# Layer
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(dense)

# Excute
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
'''
Binary Classification => loss = 'binary_crossentropy'
Multiple Classification => loss = 'categorical_crossentropy'
One-Hot Encoding + Multiple Classification => loss = 'sparse_categorical_crossentropy'
'''

model.fit(train_scaled, train_target, epochs=5)
'''
Epoch 1/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.6132 - accuracy: 0.7912
Epoch 2/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.4784 - accuracy: 0.8389
Epoch 3/5
1500/1500 [==============================] - 1s 668us/step - loss: 0.4556 - accuracy: 0.8483
Epoch 4/5
1500/1500 [==============================] - 1s 495us/step - loss: 0.4435 - accuracy: 0.8534
Epoch 5/5
1500/1500 [==============================] - 1s 502us/step - loss: 0.4362 - accuracy: 0.8550
'''

model.evaluate(val_scaled, val_target)
'''
375/375 [==============================] - 0s 616us/step - loss: 0.4418 - accuracy: 0.8509
Out[20]: [0.44181475043296814, 0.8509166836738586]
'''


##########################################
#####      Deep Neural Network       #####
##########################################

# Layer(method01)
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')

model = keras.Sequential([dense1, dense2])

model.summary()
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_5 (Dense)              (None, 100)               78500     
_________________________________________________________________
dense_6 (Dense)              (None, 10)                1010      
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________

784 x 100 + 100 = 78,500
100 x 10 + 10 = 1,010
78,500 + 1,010 = 79,510
'''

# Layer(method02)
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')], name='Fashion_MNIST_model')

model.summary()
'''
Model: "Fashion_MNIST_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
hidden (Dense)               (None, 100)               78500     
_________________________________________________________________
output (Dense)               (None, 10)                1010      
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
'''


# Layer(method03)
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

# Execute
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
'''
Epoch 1/5
1500/1500 [==============================] - 2s 886us/step - loss: 0.5626 - accuracy: 0.8080
Epoch 2/5
1500/1500 [==============================] - 1s 856us/step - loss: 0.4099 - accuracy: 0.8532
Epoch 3/5
1500/1500 [==============================] - 1s 946us/step - loss: 0.3753 - accuracy: 0.8641
Epoch 4/5
1500/1500 [==============================] - 1s 960us/step - loss: 0.3516 - accuracy: 0.8730
Epoch 5/5
1500/1500 [==============================] - 1s 998us/step - loss: 0.3351 - accuracy: 0.8786
'''

model.evaluate(val_scaled, val_target)
'''
375/375 [==============================] - 0s 759us/step - loss: 0.3607 - accuracy: 0.8708
Out[37]: [0.3607034981250763, 0.8707500100135803]
'''


## Relu

# Layer + Flatten()
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
'''
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense_9 (Dense)              (None, 100)               78500     
_________________________________________________________________
dense_10 (Dense)             (None, 10)                1010      
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
'''

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(train_scaled.shape, train_target.shape) # (48000, 28, 28) (48000,)
print(val_scaled.shape, val_target.shape) # (12000, 28, 28) (12000,)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
'''
Epoch 1/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.5263 - accuracy: 0.8140
Epoch 2/5
1500/1500 [==============================] - 2s 2ms/step - loss: 0.3873 - accuracy: 0.8600
Epoch 3/5
1500/1500 [==============================] - 1s 878us/step - loss: 0.3513 - accuracy: 0.8733
Epoch 4/5
1500/1500 [==============================] - 1s 890us/step - loss: 0.3313 - accuracy: 0.8807
Epoch 5/5
1500/1500 [==============================] - 1s 857us/step - loss: 0.3172 - accuracy: 0.8865
'''

model.evaluate(val_scaled, val_target)
'''
375/375 [==============================] - 0s 726us/step - loss: 0.3748 - accuracy: 0.8786
Out[51]: [0.37483924627304077, 0.8785833120346069]
'''

# Optimizer

## Stochastic Gradient Descent
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')

## Adagrad
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')

## RMSprop
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics='accuracy')

## Adam
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
'''
Epoch 1/5
1500/1500 [==============================] - 1s 833us/step - loss: 0.5320 - accuracy: 0.8151
Epoch 2/5
1500/1500 [==============================] - 1s 780us/step - loss: 0.3939 - accuracy: 0.8615
Epoch 3/5
1500/1500 [==============================] - 1s 771us/step - loss: 0.3559 - accuracy: 0.8697
Epoch 4/5
1500/1500 [==============================] - 1s 783us/step - loss: 0.3263 - accuracy: 0.8807
Epoch 5/5
1500/1500 [==============================] - 1s 780us/step - loss: 0.3084 - accuracy: 0.8854
'''

model.evaluate(val_scaled, val_target)
'''
375/375 [==============================] - 0s 788us/step - loss: 0.3391 - accuracy: 0.8772
Out[72]: [0.33912578225135803, 0.8771666884422302]
'''


##########################################
#####            Training            #####
##########################################

def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

model = model_fn()
model.summary()
'''
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_4 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_16 (Dense)             (None, 100)               78500     
_________________________________________________________________
dense_17 (Dense)             (None, 10)                1010      
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)

# history - visualization
print(history.history.keys()) # dict_keys(['loss', 'accuracy'])


plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


# Dropout

model = model_fn(keras.layers.Dropout(0.3))
model.summary()
'''
Model: "sequential_8"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten_5 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_18 (Dense)             (None, 100)               78500     
_________________________________________________________________
dropout (Dropout)            (None, 100)               0         
_________________________________________________________________
dense_19 (Dense)             (None, 10)                1010      
=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

"""
According to the result(image), the best value of "epochs" is 10.
"""

# model save & load
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=10, verbose=0, validation_data=(val_scaled, val_target))

## parameters
model.save_weights('model-weights.h5')
## structure + parameters
model.save('model-whole.h5')


'''
[Test]
model-weights vs model-whole
'''

# model-weights
model = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model-weights.h5')

val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print(np.mean(val_labels==val_target))
# 0.882

# model-whole
model = keras.models.load_model('model-whole.h5')
model.evaluate(val_scaled, val_target)
'''
375/375 [==============================] - 0s 669us/step - loss: 0.3298 - accuracy: 0.8820
Out[108]: [0.3297847807407379, 0.8820000290870667]
'''


# Callback

model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
model.fit(train_scaled, train_target, epochs=10, verbose=0, 
          validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb])

model = keras.models.load_model('best-model.h5')
model.evaluate(val_scaled, val_target)
'''
375/375 [==============================] - 0s 604us/step - loss: 0.3289 - accuracy: 0.8825
Out[115]: [0.3288578987121582, 0.8824999928474426]
'''




model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, 
                    validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

print(early_stopping_cb.stopped_epoch) # 11 => the optimal epoch = 10

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

model.evaluate(val_scaled, val_target)

'''
375/375 [==============================] - 0s 642us/step - loss: 0.3234 - accuracy: 0.8798
Out[133]: [0.32340511679649353, 0.8797500133514404]
'''





