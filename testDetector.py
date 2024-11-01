from MSCN import calculate_brisque_features
import tensorflow as tf
from joblib import dump, load
from PGD_attack import pgd
from sklearn.metrics import classification_report, confusion_matrix



# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()



# Load the saved SVM model
svclassifiertraining = load('svm_sigmoid_training.joblib')

# Load your test data (assuming it's in a similar format as the training data)
img = x_test[5]  # Select the test image
img = pgd(img, 5)
test = calculate_brisque_features(img, kernel_size=7, sigma=7/6)  # Extract features
test = test.reshape(1, -1)  # Reshape to 2D array for prediction

# Make predictions on the test data
Y_pred = svclassifiertraining.predict(test)

# Assuming you have the actual label for the test image
# For example, if '0' indicates a clean image and '1' indicates a noisy image
# You should have the actual label for this image
actual_label = 0  # Replace with the actual label for img

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix([actual_label], Y_pred))

print("\nClassification Report:")
print(classification_report([actual_label], Y_pred))