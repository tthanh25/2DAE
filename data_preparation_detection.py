import numpy as np
import skimage
import imageio
from MSCN import calculate_brisque_features
import tensorflow as tf
from PGD_attack import pgd

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(_, _), (x_test, _) = mnist.load_data()

# Reshape the dataset
x_test = np.reshape(x_test, (-1, 28, 28, 1))

# Prepare the feature matrix
for i in range(1000):
    img = x_test[i]
    img = pgd(img, i)
    parameters = calculate_brisque_features(img, kernel_size=7, sigma=7/6)
    t = np.hstack((1, parameters))  # Label for adversarial image
    if i == 0:
        v = t
    else:
        v = np.vstack((v, t))

    img = x_test[i]
    parameters = calculate_brisque_features(img, kernel_size=7, sigma=7/6)
    t = np.hstack((0, parameters))  # Label for clean image
    v = np.vstack((v, t))

print(np.shape(v))
np.savez_compressed('data_training', X=v[:, 1:], Y=v[:, 0:1])