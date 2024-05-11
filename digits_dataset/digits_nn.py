import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load sklearn digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Preprocess data

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize network parameters
layer_sizes = [X_train.shape[1], 16, 10]  # 10 output classes: digits 0-9
reg_lambda = 0
# Define activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation
def forward_propagation(X, Theta):
    a = [None] * len(layer_sizes)
    z = [None] * (len(layer_sizes) - 1)
    a[0] = np.hstack((np.ones((X.shape[0], 1)), X))

    for i in range(1, len(layer_sizes)):
        z[i-1] = np.dot(a[i-1], Theta[i-1].T)
        if i == len(layer_sizes) - 1:
            a[i] = softmax(z[i-1])
        else:
            a[i] = sigmoid(z[i-1])
            a[i] = np.hstack((np.ones((a[i].shape[0], 1)), a[i]))

    hypothesis = a[-1]
    return hypothesis, a, z

# Define softmax activation function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

# Cost calculation
def compute_cost(y, hypothesis, Theta):
    m = len(X_train)
    y_one_hot = np.eye(layer_sizes[-1])[y.ravel()]
    cost_per_instance = -np.sum(y_one_hot * np.log(hypothesis), axis=1)
    regularization = sum(np.sum(theta[:, 1:] ** 2) for theta in Theta)
    J = np.sum(cost_per_instance) / m + (reg_lambda / (2 * m)) * regularization
    return J

# Backpropagation
def backpropagation(y, hypothesis, a, z, Theta):
    delta = [None] * (len(layer_sizes) - 1)
    y_one_hot = np.eye(layer_sizes[-1])[y.ravel()]
    delta[-1] = hypothesis - y_one_hot

    for i in range(len(layer_sizes) - 2, 0, -1):
        delta[i-1] = np.dot(delta[i], Theta[i][:, 1:]) * sigmoid(z[i-1]) * (1 - sigmoid(z[i-1]))

    Delta = [None] * (len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        Delta[i] = np.dot(delta[i].T, a[i])

    m = len(X_train)
    Theta_grad = []
    for i in range(len(layer_sizes) - 1):
        theta_grad = Delta[i] / m
        Theta_grad.append(theta_grad)

    for i in range(len(layer_sizes) - 1):
        Theta_grad[i][:, 1:] += (reg_lambda / m) * Theta[i][:, 1:]

    return Theta_grad

def compute_accuracy(y, hypothesis):
    predicted_labels = np.argmax(hypothesis, axis=1)
    accuracy = np.mean(predicted_labels == y.ravel())
    return accuracy

def compute_f1_score(y, hypothesis):
    predicted_labels = np.argmax(hypothesis, axis=1)
    f1_scores = []
    for class_label in range(layer_sizes[-1]):
        tp = np.sum((predicted_labels == class_label) & (y.ravel() == class_label))
        fp = np.sum((predicted_labels == class_label) & (y.ravel() != class_label))
        fn = np.sum((predicted_labels != class_label) & (y.ravel() == class_label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1_score)
    avg_f1_score = np.mean(f1_scores)
    return avg_f1_score


def stratified_k_fold_cross_validation(X, y, k, epsilon):
    fold_size = len(X) // k
    indices = np.random.permutation(len(X))

    train_costs, test_costs = [], []
    train_sizes = []

    for train_size in range(fold_size, len(X), fold_size):
        avg_accuracy, avg_f1_score, avg_train_cost, avg_test_cost = 0, 0, 0, 0
        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))[:train_size]
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            Theta = [np.random.uniform(-1, 1, (layer_sizes[i+1], layer_sizes[i] + 1)) for i in range(len(layer_sizes) - 1)]

            prev_cost = np.inf
            improvement = np.inf
            iteration = 0

            while improvement > epsilon:
                hypothesis, a, z = forward_propagation(X_train, Theta)

                train_cost = compute_cost(y_train, hypothesis, Theta)
                test_hypothesis, _, _ = forward_propagation(X_test, Theta)
                test_cost = compute_cost(y_test, test_hypothesis, Theta)

                improvement = prev_cost - train_cost
                prev_cost = train_cost

                if improvement <= epsilon:
                    break

                Theta_grad = backpropagation(y_train, hypothesis, a, z, Theta)

                for i in range(len(layer_sizes) - 1):
                    Theta[i] -= 0.01 * Theta_grad[i]

                iteration += 1

            avg_accuracy += compute_accuracy(y_test, test_hypothesis)
            avg_f1_score += compute_f1_score(y_test, test_hypothesis)
            avg_train_cost += train_cost
            avg_test_cost += test_cost

        avg_accuracy /= k
        avg_f1_score /= k
        avg_train_cost /= k
        avg_test_cost /= k

        test_costs.append(avg_test_cost)
        train_sizes.append(train_size)

    return avg_accuracy, avg_f1_score, test_costs, train_sizes

# Perform stratified k-fold cross-validation with k=10
k = 10
epsilon = 0.0001
avg_accuracy, avg_f1_score, test_costs, train_sizes = stratified_k_fold_cross_validation(X_train, y_train, k, epsilon)

print(f"Average Accuracy: {avg_accuracy}")
print(f"Average F1 Score: {avg_f1_score}")

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, test_costs)
plt.xlabel('Number of Training Samples')
plt.ylabel('Cost J')
plt.title('Learning Curve (Digits dataset)')
plt.show()
