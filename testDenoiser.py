import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from bm3d import bm3d  # Ensure you have the bm3d package installed
import matplotlib.pyplot as plt
from PGD_attack import pgd  # Ensure PGD_attack is correctly implemented

def test_model(model_path, second_model_path):
    # Load the first model
    model = load_model(model_path)

    # Load MNIST data
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = np.reshape(x_test, (-1, 28, 28, 1))  # Reshape to (num_samples, 28, 28, 1)
    y_test = to_categorical(y_test, num_classes=10)  # One-hot encode the labels

    # Initialize for the first 10 images with noise
    noisy_images = np.zeros((10, 28, 28))  
    for i in range(10):  # Process only the first 10 images
        noisy_images[i] = pgd(x_test[i].squeeze(), i).squeeze()  # Apply PGD attack and remove the last dimension

    # Normalize noisy images
    noisy_images = noisy_images / 255.0

    # Make predictions with the first model on noisy images
    predictions = model.predict(noisy_images[..., np.newaxis])  # Add channel dimension
    print("predictions: ", predictions)

    # Get the noise level (sigma) from the predictions (assuming predictions are probabilities)
    sigma = np.std(predictions, axis=1)  # Example: using the std deviation as noise level

    # Denoise images using BM3D for the first 10 images
    denoised_images = np.zeros((10, 28, 28))  # Initialize for denoised images
    for i in range(10):  # Process only the first 10 images
        denoised_images[i] = bm3d(noisy_images[i].squeeze(), sigma[i])  # Apply BM3D

    # Load the second model
    second_model = load_model(second_model_path)

    # Make predictions on the denoised images
    denoised_predictions = second_model.predict(denoised_images[..., np.newaxis])  # Add channel dimension
    predicted_classes = np.argmax(denoised_predictions, axis=1)  # Get the class with the highest probability
    true_classes = np.argmax(y_test[:10], axis=1)  # Get the true class labels for the first 10 images

    # Print some predictions from the denoised images
    print("\nSample Predictions from Denoised Images:")
    for i in range(10):  # Display the first 10 predictions
        print(f"True Label: {true_classes[i]}, Predicted: {predicted_classes[i]}")

    # Visualization of original, noisy, and denoised images
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(3, 10, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(3, 10, i + 11)
        plt.imshow(noisy_images[i], cmap='gray')
        plt.title("Noisy")
        plt.axis('off')

        plt.subplot(3, 10, i + 21)
        plt.imshow(denoised_images[i], cmap='gray')
        plt.title(f"Denoised: {predicted_classes[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to test the model with denoising
test_model('train_cnn.h5', 'train.h5')