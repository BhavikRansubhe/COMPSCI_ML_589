import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
df = pd.read_csv('titanic.csv')

# Preprocess data
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
X = df[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']].values
y = df['Survived'].values.reshape(-1, 1)

# Feature normalization
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Initialize network parameters
layer_sizes = [X.shape[1], 64, 2]  # 2 output classes: survived or not survived
reg_lambda = 0
 
# Function for sigmoid activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function for softmax activation
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

# Function for forward propagation
def forward_propagation(X, Theta):
    a = [None] * len(layer_sizes)  # activation
    z = [None] * (len(layer_sizes) - 1)  # layer input
    a[0] = np.hstack((np.ones((X.shape[0], 1)), X))  # Add bias term to input layer

    for i in range(1, len(layer_sizes)):
        z[i-1] = np.dot(a[i-1], Theta[i-1].T)
        if i == len(layer_sizes) - 1:
            a[i] = softmax(z[i-1])  # Use softmax for output layer
        else:
            a[i] = sigmoid(z[i-1])  # Use sigmoid for hidden layers
            a[i] = np.hstack((np.ones((a[i].shape[0], 1)), a[i]))  # Add bias term to hidden layers

    hypothesis = a[-1]  # Predicted outputs
    return hypothesis, a, z

# Function for cost calculation
def compute_cost(y, hypothesis, Theta):
    m = len(X)
    y_one_hot = np.eye(layer_sizes[-1])[y.ravel()]  # One-hot encoding of target labels
    cost_per_instance = -np.sum(y_one_hot * np.log(hypothesis), axis=1)
    regularization = sum(np.sum(theta[:, 1:] ** 2) for theta in Theta)
    J = np.sum(cost_per_instance) / m + (reg_lambda / (2 * m)) * regularization
    return J

# Function for backpropagation
def backpropagation(y, hypothesis, a, z, Theta):
    delta = [None] * (len(layer_sizes) - 1)
    y_one_hot = np.eye(layer_sizes[-1])[y.ravel()]  # One-hot encoding of target labels
    delta[-1] = hypothesis - y_one_hot

    for i in range(len(layer_sizes) - 2, 0, -1):
        delta[i-1] = np.dot(delta[i], Theta[i][:, 1:]) * sigmoid(z[i-1]) * (1 - sigmoid(z[i-1]))

    # Gradients for each training instance
    Delta = [None] * (len(layer_sizes) - 1)
    for i in range(len(layer_sizes) - 1):
        Delta[i] = np.dot(delta[i].T, a[i])

    # Average gradients over the training set
    m = len(X)
    Theta_grad = []
    for i in range(len(layer_sizes) - 1):
        theta_grad = Delta[i] / m
        Theta_grad.append(theta_grad)

    # Regularize gradients (excluding bias terms)
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
            # Split data into train and test sets
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))[:train_size]
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Initialize weights randomly
            Theta = [np.random.uniform(-1, 1, (layer_sizes[i+1], layer_sizes[i] + 1)) for i in range(len(layer_sizes) - 1)]

            prev_cost = np.inf
            improvement = np.inf
            iteration = 0

            while improvement > epsilon:
                # Perform forward propagation
                hypothesis, a, z = forward_propagation(X_train, Theta)

                # Calculate cost
                train_cost = compute_cost(y_train, hypothesis, Theta)
                test_hypothesis, _, _ = forward_propagation(X_test, Theta)
                test_cost = compute_cost(y_test, test_hypothesis, Theta)

                # Check for improvement
                improvement = prev_cost - train_cost
                prev_cost = train_cost

                if improvement <= epsilon:
                    break

                # Perform backpropagation
                Theta_grad = backpropagation(y_train, hypothesis, a, z, Theta)

                # Update weights
                for i in range(len(layer_sizes) - 1):
                    Theta[i] -= 0.01 * Theta_grad[i]  # Update weights using gradient descent

                iteration += 1

            # Accumulate accuracy, F1 score, and costs for averaging
            avg_accuracy += compute_accuracy(y_test, test_hypothesis)
            avg_f1_score += compute_f1_score(y_test, test_hypothesis)
            avg_train_cost += train_cost
            avg_test_cost += test_cost

        # Average accuracy, F1 score, and costs over all folds
        avg_accuracy /= k
        avg_f1_score /= k
        avg_train_cost /= k
        avg_test_cost /= k

        # Append average costs to the lists
        test_costs.append(avg_test_cost)
        train_sizes.append(train_size)

    return avg_accuracy, avg_f1_score, test_costs, train_sizes

# Perform stratified k-fold cross-validation with k=10
k = 10
epsilon = 0.0001
avg_accuracy, avg_f1_score, test_costs, train_sizes = stratified_k_fold_cross_validation(X, y, k, epsilon)

print(f"Average Accuracy: {avg_accuracy}")
print(f"Average F1 Score: {avg_f1_score}")

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, test_costs)
plt.xlabel('Number of Training Samples')
plt.ylabel('Cost J')
plt.title('Learning Curve (Titanic dataset)')
plt.show()
