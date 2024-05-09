import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define parameters
max_depth = 10  # Maximum depth of each decision tree
k_values = [1, 5, 10, 20, 40, 60, 100]  # Number of trees in the forest

# Function to calculate entropy
def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Function to calculate information gain
def calculate_information_gain(y, y_left, y_right):
    entropy_parent = calculate_entropy(y)
    entropy_left = calculate_entropy(y_left)
    entropy_right = calculate_entropy(y_right)
    weight_left = len(y_left) / len(y)
    weight_right = len(y_right) / len(y)
    information_gain = entropy_parent - (weight_left * entropy_left + weight_right * entropy_right)
    return information_gain

# Function to find best split
def find_best_split(X, y, max_features):
    num_samples, num_features = X.shape
    best_information_gain = -1
    best_feature_index = None
    best_threshold = None

    for feature_index in np.random.choice(num_features, max_features, replace=False):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            y_left = y[X[:, feature_index] <= threshold]
            y_right = y[X[:, feature_index] > threshold]
            information_gain = calculate_information_gain(y, y_left, y_right)
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

# Function to build decision tree
def build_tree(X, y, depth, max_features):
    num_samples, num_features = X.shape

    # Stopping criteria
    if depth >= max_depth or len(np.unique(y)) == 1 or num_samples <= 1:
        if len(y) == 0:
            # Handle empty y array
            return None  # or any other appropriate action
        return np.argmax(np.bincount(y))

    # Find best split
    feature_index, threshold = find_best_split(X, y, max_features)

    if feature_index is None:
        if len(y) == 0:
            # Handle empty y array
            return None  # or any other appropriate action
        return np.argmax(np.bincount(y))

    # Split data
    X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)

    # Recursively build tree
    left_subtree = build_tree(X_left, y_left, depth + 1, max_features)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_features)

    return {'feature_index': feature_index, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}

# Function to predict single sample
def predict_sample(tree, x):
    if not isinstance(tree, dict):
        return tree
    feature_index, threshold, left, right = tree['feature_index'], tree['threshold'], tree['left'], tree['right']
    if x[feature_index] <= threshold:
        if left is not None:
            return predict_sample(left, x)
    else:
        if right is not None:
            return predict_sample(right, x)
    # If both left and right subtrees are None, return a default value
    return 0  # You can choose any default value here

# Function to predict
def predict(X, tree):
    return np.array([predict_sample(tree, x) for x in X])

# Function to split data
def split_data(X, y, feature_index, threshold):
    left_indices = np.where(X[:, feature_index] <= threshold)[0]
    right_indices = np.where(X[:, feature_index] > threshold)[0]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    return X_left, y_left, X_right, y_right

# Define evaluation metrics manually
def calculate_metrics(y_true, y_pred):
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    # Calculate F1 score
    f1_score = calculate_f1_score(y_true, y_pred)
    return accuracy, f1_score

# Calculate F1 score
def calculate_f1_score(y_true, y_pred):
    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    f1_scores = []
    for class_ in unique_classes:
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        false_positives = np.sum((y_true != class_) & (y_pred == class_))
        false_negatives = np.sum((y_true == class_) & (y_pred != class_))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1)
    weighted_f1_score = np.average(f1_scores, weights=np.bincount(y_true))
    return weighted_f1_score

# Store evaluation metrics for each k value
metrics_dict = {'k': [], 'accuracy': [], 'f1_score': []}

# Perform stratified k-fold cross-validation
for n_estimators in k_values:
    print(f"Number of trees: {n_estimators}")
    np.random.seed(42)  # for reproducibility
    avg_scores = {'accuracy': [], 'f1_score': []}

    # Load Titanic dataset
    titanic_df = pd.read_csv("titanic.csv")  # Adjust the path as per your dataset location

    # Drop columns that are not needed for prediction
    columns_to_drop = ['Name']  # Example columns to drop
    titanic_df = titanic_df.drop(columns=columns_to_drop)

    # Preprocess data
    # Handle categorical variables
    label_encoders = {}
    categorical_columns = ['Pclass', 'Sex']
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        titanic_df[col] = label_encoders[col].fit_transform(titanic_df[col])

    # Handle numerical variables
    numerical_columns = ['Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
    for col in numerical_columns:
        titanic_df[col] = titanic_df[col].fillna(titanic_df[col].median())  # Fill missing values with median
        # Scale numerical features
        scaler = StandardScaler()
        titanic_df[col] = scaler.fit_transform(titanic_df[col].values.reshape(-1, 1))

    # Separate features and target variable
    X = titanic_df.drop(columns=['Survived'])  # Features
    y = titanic_df['Survived']  # Target variable

    # Stratified splitting
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)

        # Train random forest classifier
        forest = []
        for _ in range(n_estimators):
            tree = build_tree(X_train.values, y_train.values, depth=0, max_features=int(np.sqrt(X_train.shape[1])))
            forest.append(tree)

        # Evaluate the model
        y_pred = np.array([np.argmax(np.bincount([predict_sample(tree, x) for tree in forest])) for x in X_test.values])

        # Calculate evaluation metrics
        accuracy, f1_score = calculate_metrics(y_test.values, y_pred)

        # Store scores
        avg_scores['accuracy'].append(accuracy)
        avg_scores['f1_score'].append(f1_score)

    # Compute average scores
    avg_scores = {metric_name: np.mean(scores) for metric_name, scores in avg_scores.items()}
    print(f"Average accuracy: {avg_scores['accuracy']:.4f}")
    print(f"Average F1 score: {avg_scores['f1_score']:.4f}")

    # Store metrics for plotting
    metrics_dict['k'].append(n_estimators)
    metrics_dict['accuracy'].append(avg_scores['accuracy'])
    metrics_dict['f1_score'].append(avg_scores['f1_score'])

    print()


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot accuracy
axes[0].plot(metrics_dict['k'], metrics_dict['accuracy'], marker='o', color="g", label='Accuracy')
axes[0].set_title('n_tree vs Accuracy')
axes[0].set_xlabel('n_tree')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Plot F1 score
axes[1].plot(metrics_dict['k'], metrics_dict['f1_score'], marker='o', color="b", label='F1 Score')
axes[1].set_title('n_tree vs F1 Score')
axes[1].set_xlabel('n_tree')
axes[1].set_ylabel('F1 Score')
axes[1].legend()
axes[1].grid(True)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plots
plt.show()
