# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 14:58:12 2021

@author: 82108
"""

"""
chapter09 : Neural Network (Text)
- Recurrent Neural Network
- LSTM & GRU
"""

# Data load
from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
print(train_input.shape, test_input.shape) # (25000,) (25000,)

print(len(train_input[0])) # 218
print(train_target[:20])
# [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1] => 0 : Negative, 1 : Positive

from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
print(train_input.shape, val_input.shape) # (20000,) (5000,)

 
# Padding
import numpy as np
lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.median(lengths)) # 239.00925 178.0

import matplotlib.pyplot as plt
plt.hist(lengths, bins=50)
plt.xlabel('length')
plt.ylabel('fluency')
plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)
print(train_seq.shape) # (20000, 100)
val_seq = pad_sequences(val_input, maxlen=100)

##########################################
#####           Simple RNN           #####
##########################################
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# One-hot Encoding
train_oh = keras.utils.to_categorical(train_seq)
print(train_oh.shape) # (20000, 100, 500)
val_oh = keras.utils.to_categorical(val_seq)

model.summary()
'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
simple_rnn (SimpleRNN)       (None, 8)                 4072      
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9         
=================================================================
Total params: 4,081
Trainable params: 4,081
Non-trainable params: 0
_________________________________________________________________
'''

# training
rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-smiplernn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
'''
Epoch 31/100
313/313 [==============================] - 6s 18ms/step - loss: 0.4144 - accuracy: 0.8183 - val_loss: 0.4551 - val_accuracy: 0.7928
Epoch 32/100
313/313 [==============================] - 6s 19ms/step - loss: 0.4126 - accuracy: 0.8177 - val_loss: 0.4492 - val_accuracy: 0.7940
Epoch 33/100
313/313 [==============================] - 6s 20ms/step - loss: 0.4121 - accuracy: 0.8190 - val_loss: 0.4530 - val_accuracy: 0.7948
Epoch 34/100
313/313 [==============================] - 6s 19ms/step - loss: 0.4101 - accuracy: 0.8209 - val_loss: 0.4537 - val_accuracy: 0.7918
Epoch 35/100
313/313 [==============================] - 6s 21ms/step - loss: 0.4096 - accuracy: 0.8195 - val_loss: 0.4545 - val_accuracy: 0.7914
'''

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'])
plt.show()


# Word embedding
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()
'''
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 100, 16)           8000      
_________________________________________________________________
simple_rnn_3 (SimpleRNN)     (None, 8)                 200       
_________________________________________________________________
dense_7 (Dense)              (None, 1)                 9         
=================================================================
Total params: 8,209
Trainable params: 8,209
Non-trainable params: 0
_________________________________________________________________
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
'''
Epoch 25/100
313/313 [==============================] - 4s 12ms/step - loss: 0.4135 - accuracy: 0.8184 - val_loss: 0.4644 - val_accuracy: 0.7862
Epoch 26/100
313/313 [==============================] - 4s 12ms/step - loss: 0.4127 - accuracy: 0.8188 - val_loss: 0.4552 - val_accuracy: 0.7936
Epoch 27/100
313/313 [==============================] - 4s 12ms/step - loss: 0.4112 - accuracy: 0.8193 - val_loss: 0.4575 - val_accuracy: 0.7916
Epoch 28/100
313/313 [==============================] - 4s 12ms/step - loss: 0.4096 - accuracy: 0.8217 - val_loss: 0.4558 - val_accuracy: 0.7948
Epoch 29/100
313/313 [==============================] - 4s 12ms/step - loss: 0.4086 - accuracy: 0.8221 - val_loss: 0.4746 - val_accuracy: 0.7790
'''

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'val'])
plt.show()


##########################################
#####           LSTM & GRU           #####
##########################################
'''
LSTM = Long Short-Term Memory
GRU = Gated Recurrent Unit
'''

# LSTM
model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid'))

model3.summary()
'''
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 100, 16)           8000      
_________________________________________________________________
lstm_2 (LSTM)                (None, 100, 8)            800       
_________________________________________________________________
lstm_3 (LSTM)                (None, 8)                 544       
_________________________________________________________________
dense_10 (Dense)             (None, 1)                 9         
=================================================================
Total params: 9,353
Trainable params: 9,353
Non-trainable params: 0
_________________________________________________________________
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model3.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model3.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

'''
Epoch 48/100
313/313 [==============================] - 13s 41ms/step - loss: 0.4040 - accuracy: 0.8155 - val_loss: 0.4252 - val_accuracy: 0.8022
Epoch 49/100
313/313 [==============================] - 13s 41ms/step - loss: 0.4055 - accuracy: 0.8141 - val_loss: 0.4243 - val_accuracy: 0.8024
Epoch 50/100
313/313 [==============================] - 13s 42ms/step - loss: 0.4044 - accuracy: 0.8133 - val_loss: 0.4250 - val_accuracy: 0.8028
Epoch 51/100
313/313 [==============================] - 13s 41ms/step - loss: 0.4049 - accuracy: 0.8151 - val_loss: 0.4249 - val_accuracy: 0.8036
Epoch 52/100
313/313 [==============================] - 13s 42ms/step - loss: 0.4034 - accuracy: 0.8135 - val_loss: 0.4253 - val_accuracy: 0.8046
'''


# GRU
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))

model4.summary()
'''
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_4 (Embedding)      (None, 100, 16)           8000      
_________________________________________________________________
gru (GRU)                    (None, 8)                 624       
_________________________________________________________________
dense_11 (Dense)             (None, 1)                 9         
=================================================================
Total params: 8,633
Trainable params: 8,633
Non-trainable params: 0
_________________________________________________________________
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)
model4.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model4.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

'''

Epoch 33/100
313/313 [==============================] - 7s 23ms/step - loss: 0.4158 - accuracy: 0.8158 - val_loss: 0.4420 - val_accuracy: 0.7942
Epoch 34/100
313/313 [==============================] - 8s 25ms/step - loss: 0.4149 - accuracy: 0.8161 - val_loss: 0.4414 - val_accuracy: 0.7952
Epoch 35/100
313/313 [==============================] - 29s 92ms/step - loss: 0.4150 - accuracy: 0.8151 - val_loss: 0.4434 - val_accuracy: 0.7904
Epoch 36/100
313/313 [==============================] - 32s 104ms/step - loss: 0.4144 - accuracy: 0.8164 - val_loss: 0.4422 - val_accuracy: 0.7926
Epoch 37/100
313/313 [==============================] - 22s 70ms/step - loss: 0.4136 - accuracy: 0.8164 - val_loss: 0.4459 - val_accuracy: 0.7904
'''



























