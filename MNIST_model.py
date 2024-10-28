import pandas as pd
import numpy as np

import keras
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

import tensorflow as tf

datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train), np.shape(y_train))
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))

# Use Input shape for the first layer
model = Sequential()
model.add(Input(shape=(28, 28, 1)))  # Added Input layer
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

# Updated optimizer initialization
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # Updated argument
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# TRAIN NETWORKS
history = [0]
epochs = 64

# Calculate steps_per_epoch
steps_per_epoch = x_train.shape[0] // 64  # Assuming batch_size is 64

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=epochs, steps_per_epoch=steps_per_epoch,
                    validation_data=(x_test, y_test), callbacks=[annealer], verbose=1)

j = 0
print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, test accuracy={3:.5f}".format(
        j + 1, epochs, max(history.history['accuracy']), max(history.history['val_accuracy'])))

model.save('train.h5')