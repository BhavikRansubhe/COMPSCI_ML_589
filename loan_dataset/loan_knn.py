import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("loan.csv")

# Drop Loan_ID column
data = data.drop(columns=['Loan_ID'])

# Encode Gender column
gender_map = {'Male': 0, 'Female': 1}
data['Gender'] = data['Gender'].map(gender_map)

# Encode Married column
married_map = {'No': 0, 'Yes': 1}
data['Married'] = data['Married'].map(married_map)

# Encode Education column
education_map = {'Not Graduate': 0, 'Graduate': 1}
data['Education'] = data['Education'].map(education_map)

# Encode Self_Employed column
self_employed_map = {'No': 0, 'Yes': 1}
data['Self_Employed'] = data['Self_Employed'].map(self_employed_map)

# Encode Property_Area column
property_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
data['Property_Area'] = data['Property_Area'].map(property_map)

# Encode Dependents column
dependents_map = {'0': 0, '1': 1, '2': 2, '3+':3}
data['Dependents'] = data['Dependents'].map(dependents_map)

# Encode Loan_Status column
loan_status_map = {'N': 0, 'Y': 1}
data['Loan_Status'] = data['Loan_Status'].map(loan_status_map)

# Separate features and target
x = data.iloc[:, :-1].values  # All columns except the last one

y = data.iloc[:, -1].values 

# to calculate the euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# to find the indices of the k nearest neighbors in the training dataset
def nearest_neighbours(x_train_point, x_normal_train, k):
    distances = [euclidean_distance(x_train_point, each_row) for each_row in x_normal_train]
    nearest_indices = np.argsort(distances)[:k]
    return nearest_indices

# to predict the target value for a given data point
def predict(x_train_point, x_normal_train, y_train, k):
    nearest_indices = nearest_neighbours(x_train_point, x_normal_train, k)
    target_values = y_train[nearest_indices]
    return np.bincount(target_values).argmax()

def KNN_alg(x_normal_train, y_train, k):
    def predict_knn(data):
        return np.array([predict(point, x_normal_train, y_train, k) for point in data])
    return predict_knn

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return f1_score

train_accuracies = []
test_accuracies = []
train_avg_accuracy = []
test_avg_accuracy = []
train_std_accuracy = []
test_std_accuracy = []
train_avg_f1score = []
test_avg_f1score = []
train_f1_scores = []
test_f1_scores = []

no = 30
for k in range(1, no, 2):
    print("no", k)
    for _ in range(20):
        # shuffling the dataset
        x_shuffled, y_shuffled = shuffle(x, y, random_state=42)
        # splitting the data set
        x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2, random_state=42)
        # normalization
        mean_train = np.mean(x_train, axis=0)
        std_train = np.std(x_train, axis=0)

        mean_test = np.mean(x_test, axis=0)
        std_test = np.std(x_test, axis=0)

        x_normal_train = (x_train - mean_train) / std_train
        x_normal_test = (x_test - mean_test) / std_test
        # training the model
        train_prediction = KNN_alg(x_normal_train, y_train, k)

        train_data_predictions = train_prediction(x_normal_train)
        train_accuracy = np.mean(train_data_predictions == y_train)
        train_accuracies.append(train_accuracy)
        # evaluating the performance of a model based on test data
        test_data_predictions = train_prediction(x_normal_test)
        test_accuracy = np.mean(test_data_predictions == y_test)
        test_accuracies.append(test_accuracy)
        
        train_f1_score = calculate_metrics(y_train, train_data_predictions)
        train_f1_scores.append(train_f1_score)
        test_f1_score = calculate_metrics(y_test, test_data_predictions)
        test_f1_scores.append(test_f1_score)
        
    # calculate average accuracy and standard deviation
    train_avg_accuracy.append(np.mean(train_accuracies))
    train_std_accuracy.append(np.std(train_accuracies))
    test_avg_accuracy.append(np.mean(test_accuracies))
    test_std_accuracy.append(np.std(test_accuracies))
    train_avg_f1score.append(np.mean(train_f1_scores))
    test_avg_f1score.append(np.mean(test_f1_scores))

# Data for train plot
x_axis = [k for k in range(1, no, 2)]
y_axis_accuracy = train_avg_accuracy
y_axis_f1score = train_avg_f1score

# Data for test plot
x1_axis = [k for k in range(1, no, 2)]
y1_axis_accuracy = test_avg_accuracy
y1_axis_f1score = test_avg_f1score

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for train data
axs[0].plot(x_axis, y_axis_accuracy, label='Accuracy (Train)', marker='o', color='blue')
axs[0].plot(x_axis, y_axis_f1score, label='F1 Score (Train)', marker='o', color='red')
axs[0].set_xlabel('Value of k')
axs[0].set_ylabel('Performance')
axs[0].set_title('Train graph')
axs[0].legend()

# Plot for test data
axs[1].plot(x1_axis, y1_axis_accuracy, label='Accuracy (Test)', marker='o', color='blue')
axs[1].plot(x1_axis, y1_axis_f1score, label='F1 Score (Test)', marker='o', color='red')
axs[1].set_xlabel('Value of k')
axs[1].set_ylabel('Performance')
axs[1].set_title('Test graph')
axs[1].legend()

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()