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
v = []  # List for features
t_values = []  # List for labels

# Prepare the feature matrix
for i in range(1000):
    # Create adversarial image
    img = x_test[i]
    img = pgd(img, i)
    parameters = calculate_brisque_features(img, kernel_size=7, sigma=7/6)
    v.append(np.hstack((parameters)))  # Store features for adversarial image
    t_values.append(1)  # Label for adversarial image

    # Create clean image
    img = x_test[i]
    parameters = calculate_brisque_features(img, kernel_size=7, sigma=7/6)
    v.append(np.hstack((parameters)))  # Store features for clean image
    t_values.append(0)  # Label for clean image

# Convert lists to numpy arrays
v = np.array(v)
t_values = np.array(t_values)
print("v: ", v)
print("t_values: ", t_values)
print(np.shape(v), np.shape(t_values))  # Print shapes of features and labels
# Save the compressed data
np.savez_compressed('data_training', X=v, Y=t_values)