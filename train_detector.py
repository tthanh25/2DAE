import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from joblib import dump, load


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


img = x_test[5]
test = calculate_brisque_features(img, kernel_size=7, sigma=7/6)
print(test)
t = np.hstack((0, parameters))  # Label for clean image
v = np.vstack((v, t))
X_test=v[:, 1:]
Y_test=v[:, 0]
print(X_test)
print(Y_test)
# Make predictions on the test data
Y_pred = svclassifiertraining.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))