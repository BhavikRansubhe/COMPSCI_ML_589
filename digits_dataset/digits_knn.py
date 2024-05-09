import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# Euclidean distance function
def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# KNN algorithm implementation with precomputed distances and vectorized operations
def train_knn(training_data, training_labels, num_neighbors=3):
    distances_train = np.sqrt(((training_data[:, np.newaxis] - training_data) ** 2).sum(axis=2))

    def predict_knn(data, distances_train):
        distances = np.sqrt(((data[:, np.newaxis] - training_data) ** 2).sum(axis=2))
        nearest_neighbor_indices = np.argsort(distances)[:, :num_neighbors]
        nearest_neighbor_labels = training_labels[nearest_neighbor_indices]
        predicted_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=nearest_neighbor_labels)
        return predicted_labels

    return predict_knn, distances_train

# Function to calculate accuracy
def calculate_accuracy(predicted_labels, actual_labels):
    correct = np.sum(predicted_labels == actual_labels)
    total = len(actual_labels)
    accuracy = correct / total
    return accuracy

# Function to calculate F1 score
def calculate_f1_score(predicted_labels, actual_labels):
    unique_labels = np.unique(actual_labels)
    f1_scores = []
    for label in unique_labels:
        true_positives = np.sum((predicted_labels == label) & (actual_labels == label))
        false_positives = np.sum((predicted_labels == label) & (actual_labels != label))
        false_negatives = np.sum((predicted_labels != label) & (actual_labels == label))
        
        precision = true_positives / (true_positives + false_positives + 1e-9)  # Adding small epsilon to avoid division by zero
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)  # Adding small epsilon to avoid division by zero
        f1_scores.append(f1)
    average_f1_score = np.mean(f1_scores)
    return average_f1_score

# Load digits dataset
digits = datasets.load_digits(return_X_y=True)
features = digits[0]
target = digits[1]


# Shuffle dataset
shuffle_index = np.random.permutation(len(features))
features_shuffled = features[shuffle_index]
target_shuffled = target[shuffle_index]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

training_accuracies = []
testing_accuracies = []
training_f1_scores = []
testing_f1_scores = []

num_folds = 10
fold_size = len(features) // num_folds

for k in range(1, 32, 2):  # Odd values from 1 to 51
    k_training_accuracies = []
    k_testing_accuracies = []
    k_training_f1_scores = []
    k_testing_f1_scores = []
 
    for fold_index in range(num_folds):
        # Define the indices for the current fold
        start_index = fold_index * fold_size
        end_index = start_index + fold_size

        # Split dataset into training and testing sets for this fold
        features_train = np.concatenate((features_shuffled[:start_index], features_shuffled[end_index:]), axis=0)
        target_train = np.concatenate((target_shuffled[:start_index], target_shuffled[end_index:]), axis=0)
        features_test = features_shuffled[start_index:end_index]
        target_test = target_shuffled[start_index:end_index]

        # Train KNN model with precomputed distances
        predict_knn, distances_train = train_knn(features_train, target_train, num_neighbors=k)

        # Predictions on training set
        predicted_labels_train = predict_knn(features_train, distances_train)
        accuracy_train = calculate_accuracy(predicted_labels_train, target_train)
        k_training_accuracies.append(accuracy_train)
        f1_train = calculate_f1_score(predicted_labels_train, target_train)
        k_training_f1_scores.append(f1_train)

        # Predictions on testing set
        predicted_labels_test = predict_knn(features_test, distances_train)
        accuracy_test = calculate_accuracy(predicted_labels_test, target_test)
        k_testing_accuracies.append(accuracy_test)
        f1_test = calculate_f1_score(predicted_labels_test, target_test)
        k_testing_f1_scores.append(f1_test)

    # Calculate average accuracies and F1 scores for this value of k
    training_accuracies.append(np.mean(k_training_accuracies))
    testing_accuracies.append(np.mean(k_testing_accuracies))
    training_f1_scores.append(np.mean(k_training_f1_scores))
    testing_f1_scores.append(np.mean(k_testing_f1_scores))

# Plot training accuracy
ax1.plot(range(1, 32, 2), training_accuracies, '-o', color='b', label='Accuracy')
ax1.plot(range(1, 32, 2), training_f1_scores, '-o', color='r', label='F1 Score')
ax1.set_xlabel("Number of Neighbors (k)")
ax1.set_ylabel("Score")
ax1.set_title("Performance of KNN Model for Different k on Training Set")
ax1.legend()

# Plot testing accuracy
ax2.plot(range(1, 32, 2), testing_accuracies, '-o', color='g', label='Accuracy')
ax2.plot(range(1, 32, 2), testing_f1_scores, '-o', color='orange', label='F1 Score')
ax2.set_xlabel("Number of Neighbors (k)")
ax2.set_ylabel("Score")
ax2.set_title("Performance of KNN Model for Different k on Testing Set")
ax2.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()
