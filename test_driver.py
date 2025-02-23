from tensorflow.keras.datasets import mnist
import numpy as np
from weighted_knn import weighted_kNN

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Take a quarter of the dataset
X_train = X_train[:15000]  # Use first 15,000 samples
y_train = y_train[:15000]  # Corresponding labels

X_test = X_test[:2500]  # Use first 2,500 test samples
y_test = y_test[:2500]  # Corresponding labels

# normalize the data by dividing by 255 to scale the pixel values to [0, 1]
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# flatten the data for easier coding and efficient computation (from 60,000x28x28 to 60,000x784)
X_train = X_train.reshape(X_train.shape[0], -1)  # (15000, 784)
X_test = X_test.reshape(X_test.shape[0], -1)  # (2500, 784)

# call function
predicted_values = weighted_kNN(training_data=X_train, training_labels=y_train, testing_data=X_test, k=31)

# transfer test labels to numpy array
test_labels = np.array(y_test)

# count correct predictions
correct_predictions = np.sum(predicted_values == test_labels)

# find accuracy
accuracy = correct_predictions / len(test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
