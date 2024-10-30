import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from joblib import dump, load
import tensorflow as tf
from MSCN import calculate_brisque_features

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(_, _), (x_test, _) = mnist.load_data()


npzfile = np.load('data_training.npz')

svclassifier = SVC(kernel='sigmoid', gamma='auto')
svclassifier.fit(npzfile['X'], npzfile['Y'])
dump(svclassifier, 'svm_sigmoid_training.joblib')

# Load the saved SVM model
svclassifiertraining = load('svm_sigmoid_training.joblib')

# Load your test data (assuming it's in a similar format as the training data)
img = x_test[5]  # Select the test image
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