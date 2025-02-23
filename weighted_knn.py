import numpy as np

'''
This program is a simple extension of k-NN, it weights neighbor's votes inversely according to distance. 
This should help the model ignore outlier neighbors, though I have not set up the tools to run this expirement repeatedly.
'''


def weighted_kNN(training_data, training_labels, testing_data, k=None) -> list:

    # Convert to NumPy arrays
    training_data = np.array(training_data)
    testing_data = np.array(testing_data)
    training_labels = np.array(training_labels)

    if not k:
        k = len(training_data)

    predicted_values = []
    sum_of_distances = 0

    num_classes = 10 # set to 'np.max(training_labels) + 1' for dynamic range, just no need to iterate 60k elements find the range of mnist

    # Iterate over test points
    for test_point in testing_data:
        # Compute distances using NumPy's vectorization
        distances = np.linalg.norm(training_data - test_point, axis=1)
        sum_of_distances += np.sum(distances)

        # Get the indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = training_labels[nearest_indices]
        nearest_distances = distances[nearest_indices] + 1e-8  # Avoid division by zero

        # Weighted voting
        weights = 1 / nearest_distances
        vote_count = np.zeros(num_classes)
        for idx, label in enumerate(nearest_labels):
            vote_count[label] += weights[idx]

        predicted_value = np.argmax(vote_count)
        predicted_values.append(predicted_value)

    total_distances = len(testing_data) * len(training_data)
    print(f"Average distance: {sum_of_distances / total_distances}")

    return predicted_values
