import numpy as np
import tensorflow as tf
from PGD_attack import pgd
import sys
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Updated import
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# Load MNIST data
mnist = tf.keras.datasets.mnist
(_, _), (x_test, _) = mnist.load_data()
x_test = np.reshape(x_test, (-1, 28, 28, 1))

# Load prepared data
npzfile = np.load('data_prepare.npz')

# Prepare images
for i in range(len(npzfile['X'])):
    v = int(npzfile['X'][i])
    img = pgd(x_test[v], v)
    img = (np.asarray(img) / 255.0).astype(np.float32)
    img = np.reshape(img, (1, 28, 28, 1))
    if i == 0:
        t = img
    else:
        t = np.vstack((t, img))

x = t
y = npzfile['Y']

# Build the model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(GlobalMaxPooling2D())

# Compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

# Fit the model
model.fit(x, y,
          batch_size=128,
          epochs=64,  # Changed from nb_epoch to epochs
          shuffle=True)

# Save the model
model.save('train_cnn.h5')