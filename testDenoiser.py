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

    # Initialize for the first 20 images
    noisy_images = np.zeros((20, 28, 28))
    
    for i in range(20):  # Process the first 20 images
        noisy_images[i] = pgd(x_test[i].squeeze(), i).squeeze()  # Apply PGD attack

    noisy_images = noisy_images / 255.0

    # Make predictions with the first model on noisy images
    predictions = model.predict(noisy_images[..., np.newaxis])  # Add channel dimension
    print("Predictions on Noisy Images: ", predictions)

    # Use the maximum predicted probability as sigma for denoising
    sigma = np.max(predictions, axis=1)  # Get the maximum probability for each prediction
    print("Sigma values for denoising: ", sigma)
    noisy_images = noisy_images * 255
    # Denoise images using BM3D for the first 20 images
    denoised_images = np.zeros((20, 28, 28))  # Initialize for denoised images
    for i in range(20):  # Process the first 20 images
        #print("sigma", i, ":", sigma[i])
        denoised_images[i] = bm3d(noisy_images[i], sigma[i])  # Apply BM3D

    # Load the second model
    second_model = load_model(second_model_path)

    # Make predictions on the denoised images
    denoised_predictions = second_model.predict(denoised_images[..., np.newaxis])  # Add channel dimension
    predicted_classes = np.argmax(denoised_predictions, axis=1)  # Get the class with the highest probability
    true_classes = np.argmax(y_test[:20], axis=1)  # Get the true class labels for the first 20 images

    # Print predictions from the denoised images
    print("\nSample Predictions from Denoised Images:")
    for i in range(20):  # Display the first 20 predictions
        print(f"True Label: {true_classes[i]}, Predicted: {predicted_classes[i]}")

    # Visualization of original, noisy, and denoised images
    plt.figure(figsize=(15, 7))
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