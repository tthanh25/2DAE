import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load tập dữ liệu MNIST từ keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chọn số lượng hình ảnh cần in ra
num_images = 5

# In ra các hình ảnh
plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()
plt.figure(figsize=(10, 5))
for i in range(num_images):
    print("x_train: ")
    print(x_train[i].shape)
    print("y_train:")
    print(y_train[i])
    print("x_test: ")
    print(x_test[i].shape)
    print("y_test:")
    print(y_test[i])

    plt.subplot(1, num_images, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Label: {y_test[i]}")
    plt.axis('off')
plt.show()