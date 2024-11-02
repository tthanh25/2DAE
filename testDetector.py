from MSCN import calculate_brisque_features
import tensorflow as tf
from joblib import dump, load
from PGD_attack import pgd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

# Load the saved SVM model
svclassifiertraining = load('svm_sigmoid_training.joblib')

# Initialize lists for predictions and actual labels
predictions = []
actual_labels = []
cleanPredict = []
# Test with 10 images
for i in range(10):
    img = x_test[i]  # Select the test image
    noisy_img = pgd(img, i)  # Apply PGD attack
    features = calculate_brisque_features(noisy_img, kernel_size=7, sigma=7/6)  # Extract features
    features = features.reshape(1, -1)  # Reshape to 2D array for prediction
    param = calculate_brisque_features(img, kernel_size=7, sigma=7/6)  # Extract features
    param = param.reshape(1, -1)  # Reshape to 2D array for prediction
    
    # Make predictions on the test data
    Y_pred = svclassifiertraining.predict(features)
    Y_predict = svclassifiertraining.predict(param)
    predictions.append(Y_pred[0])  # Append the predicted label
    actual_labels.append(y_test[i])  # Append the actual label (assuming y_test has clean/noisy labels)
    cleanPredict.append(Y_predict[0])
    # Print prediction and actual label
    print(f"Image {i}: Predicted: {Y_pred[0]}, Actual: 1")
    print(f"Image {i}: Khong nhieu Predicted: {Y_predict[0]}, Actual: 0")

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(actual_labels, predictions))

print("\nClassification Report:")
print(classification_report(actual_labels, predictions))

# Visualization of the original and noisy images
plt.figure(figsize=(15, 5))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2, 10, i + 11)
    noisy_img = pgd(x_test[i], i)  # Apply PGD attack again for visualization
    plt.imshow(noisy_img.squeeze(), cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

plt.tight_layout()
plt.show()