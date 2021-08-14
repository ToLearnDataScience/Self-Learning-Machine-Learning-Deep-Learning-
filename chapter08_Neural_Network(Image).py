# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 13:21:21 2021

@author: 82108
"""

"""
chapter08 : Neural Network (Image)
- Convolution Neural Network
"""

# Data load
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_input.shape # (60000, 28, 28)
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
print(train_scaled.shape, val_scaled.shape) # (48000, 28, 28, 1) (12000, 28, 28, 1)


##########################################
#####   Convolution Neural Network   #####
##########################################

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 3136)              0         
_________________________________________________________________
dense (Dense)                (None, 100)               313700    
_________________________________________________________________
dropout (Dropout)            (None, 100)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1010      
=================================================================
Total params: 333,526
Trainable params: 333,526
Non-trainable params: 0
_________________________________________________________________
'''

keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

'''
1500/1500 [==============================] - 20s 13ms/step - loss: 0.5342 - accuracy: 0.8083 - val_loss: 0.3422 - val_accuracy: 0.8734
Epoch 2/20
1500/1500 [==============================] - 20s 13ms/step - loss: 0.3527 - accuracy: 0.8737 - val_loss: 0.2887 - val_accuracy: 0.8914
Epoch 3/20
1500/1500 [==============================] - 20s 13ms/step - loss: 0.3000 - accuracy: 0.8920 - val_loss: 0.2632 - val_accuracy: 0.9013
Epoch 4/20
1500/1500 [==============================] - 20s 13ms/step - loss: 0.2691 - accuracy: 0.9019 - val_loss: 0.2486 - val_accuracy: 0.9087
Epoch 5/20
1500/1500 [==============================] - 20s 13ms/step - loss: 0.2415 - accuracy: 0.9114 - val_loss: 0.2282 - val_accuracy: 0.9157
Epoch 6/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2233 - accuracy: 0.9172 - val_loss: 0.2272 - val_accuracy: 0.9155
Epoch 7/20
1500/1500 [==============================] - 24s 16ms/step - loss: 0.2050 - accuracy: 0.9253 - val_loss: 0.2303 - val_accuracy: 0.9162
Epoch 8/20
1500/1500 [==============================] - 25s 17ms/step - loss: 0.1927 - accuracy: 0.9284 - val_loss: 0.2210 - val_accuracy: 0.9198
Epoch 9/20
1500/1500 [==============================] - 28s 18ms/step - loss: 0.1770 - accuracy: 0.9339 - val_loss: 0.2231 - val_accuracy: 0.9181
Epoch 10/20
1500/1500 [==============================] - 23s 16ms/step - loss: 0.1638 - accuracy: 0.9386 - val_loss: 0.2384 - val_accuracy: 0.9143
'''

# visualization
import matplotlib.pyplot as plt
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

'''
the optimal value of epochs is 8.
'''

# evaluate
model.evaluate(val_scaled, val_target)
'''
375/375 [==============================] - 2s 5ms/step - loss: 0.2210 - accuracy: 0.9198
Out[28]: [0.22097016870975494, 0.9198333621025085]
'''

# predict
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)
'''
[[1.0559521e-16 1.0278653e-24 3.6755361e-18 3.5146643e-20 2.5561501e-17
  2.8999276e-15 3.3514737e-15 8.2578400e-17 1.0000000e+00 1.3246210e-18]]
'''

# final test
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255

model.evaluate(test_scaled, test_target)
'''
313/313 [==============================] - 3s 10ms/step - loss: 0.2501 - accuracy: 0.9136
Out[33]: [0.25009018182754517, 0.9136000275611877]
'''

















