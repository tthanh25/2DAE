import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PGD_attack import pgd  # Ensure PGD_attack is correctly implemented

def predict_and_display(model_path):
    # Load the model
    model = load_model(model_path)

    # Load MNIST data
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = np.reshape(x_test, (-1, 28, 28, 1))  # Reshape to (num_samples, 28, 28, 1)
    y_test = to_categorical(y_test, num_classes=10)  # One-hot encode the labels

    # Initialize for the first 20 images attacked by PGD
    attacked_images = np.zeros((20, 28, 28))
    
    for i in range(20):  # Process the first 20 images
        attacked_images[i] = pgd(x_test[i].squeeze(), i).squeeze()  # Apply PGD attack

    attacked_images = attacked_images / 255.0  # Normalize images for prediction

    # Make predictions with the model on attacked images
    predictions = model.predict(attacked_images[..., np.newaxis])  # Add channel dimension
    predicted_classes = np.argmax(predictions, axis=1)  # Get the predicted class
    true_classes = np.argmax(y_test[:20], axis=1)  # Get the true class labels for the first 20 images

    # Print predictions and visualize the attacked images
    print("\nPredictions for Attacked Images:")
    for i in range(20):
        print(f"Image {i + 1}: True Class = {true_classes[i]}, Predicted Class = {predicted_classes[i]}")

    # Visualization of attacked images
    plt.figure(figsize=(15, 7))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(attacked_images[i], cmap='gray')
        plt.title(f"T: {true_classes[i]}, P: {predicted_classes[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the function to predict and display the images
predict_and_display('train.h5')