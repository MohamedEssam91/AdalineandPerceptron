import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# This class represents the Adaline (Adaptive Linear Neuron) model for binary classification.
class Adaline:
    def __init__(self, learning_rate=0.01, n_iterations=100, mse_threshold=None, bias=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.mse_threshold = mse_threshold
        self.bias = bias
        self.weights = None
        self.errors = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
    
        if self.bias:
            self.weights = np.zeros(n_features + 1)
        else:
            self.weights = np.zeros(n_features)
    
        for _ in range(self.n_iterations):
            errors = []
            for xi, target in zip(X, y):
                net_input = self.net_input(xi)
                output = self.linear_activation(net_input)
                error = target - output
    
                if self.bias:
                    self.weights[1:] += self.learning_rate * error * xi
                    self.weights[0] += self.learning_rate * error
                else:
                    self.weights += self.learning_rate * error * xi
    
                errors.append(0.5 * error**2)
            self.errors.append(sum(errors) / n_samples)
    
            if self.mse_threshold is not None and self.errors[-1] < self.mse_threshold:
                break

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0] if self.bias else np.dot(X, self.weights)

    def linear_activation(self, net_input):
        return net_input

    def predict(self, X):
        raw_predictions = self.net_input(X)
        return np.where(raw_predictions >= 0.0, 1, -1) 




# Function to calculate the confusion matrix
def calculate_confusion_matrix(true_labels, predicted_labels):
    classes = np.unique(true_labels)
    num_classes = len(classes)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = np.sum((true_labels == classes[i]) & (predicted_labels == classes[j]))

    return confusion_matrix

# Function to calculate the overall accuracy given a confusion matrix.
def calculate_overall_accuracy(confusion_matrix):
    correct_predictions = np.sum(np.diag(confusion_matrix))
    total_predictions = np.sum(confusion_matrix)
    overall_accuracy = correct_predictions / total_predictions
    return overall_accuracy


def Adaline_GUI(class1, class2, feature1, feature2, eta, epochs, threshold, biass):
    # Load the dataset and preprocess it.
    df = pd.read_excel("Dry_Bean_Dataset.xlsx")
    df["MinorAxisLength"].fillna(df["MinorAxisLength"].mean(), inplace=True)
    
    df_BOMBAY = df[df['Class'] == "BOMBAY"]
    df_CALI = df[df['Class'] == "CALI"]
    df_SIRA = df[df['Class'] == "SIRA"]
    
    # Splitting data into 30 samples for training and 20 samples for testing for each class
    random_BOMBAY = df_BOMBAY.sample(n=30)
    new_df_BOMBAY = pd.DataFrame(random_BOMBAY)
    df_BOMBAY = df_BOMBAY.drop(random_BOMBAY.index)
    
    random_CALI = df_CALI.sample(n=30)
    new_df_CALI = pd.DataFrame(random_CALI)
    df_CALI = df_CALI.drop(random_CALI.index)
    
    random_SIRA = df_SIRA.sample(n=30)
    new_df_SIRA = pd.DataFrame(random_SIRA)
    df_SIRA = df_SIRA.drop(random_SIRA.index)
    
    # Combining dataframes for training and testing
    training_df = pd.concat([new_df_BOMBAY, new_df_CALI, new_df_SIRA], ignore_index=True)
    testing_df = pd.concat([df_BOMBAY, df_CALI, df_SIRA], ignore_index=True)
    
    id_to_class = {
        "C1" : "BOMBAY",
        "C2" : "CALI",
        "C3" : "SIRA"
        }
    
    id_to_feature = {
        "Feature 1": 'Area',
        "Feature 2": 'Perimeter',
        "Feature 3": 'MajorAxisLength',
        "Feature 4": 'MinorAxisLength',
        "Feature 5": 'roundnes'
        }
    
    feature_1Name = id_to_feature.get(feature1)
    feature_2Name = id_to_feature.get(feature2)
    
    class1_Name = id_to_class.get(class1)
    class2_Name = id_to_class.get(class2)
    
    training_df = training_df[training_df['Class'].isin([class1_Name, class2_Name])]
    testing_df = testing_df[testing_df['Class'].isin([class1_Name, class2_Name])]
    
    X = training_df[[feature_1Name, feature_2Name]]
    y = training_df['Class']
    
    label_encoder = LabelEncoder()
    
    y_encoded = label_encoder.fit_transform(y)  # Encodes class labels to numerical values
    
    X_test = testing_df[[feature_1Name, feature_2Name]]
    y_test = testing_df['Class']
    
    y_encoded_test = label_encoder.transform(y_test)  # Encodes class labels to numerical values
    
    binary_encoded_labels = 2 * y_encoded - 1
    binary_encoded_labels2 = 2 * y_encoded_test - 1
    # Instance of the MulticlassAdaline class for fitting the training data.
    adaline = Adaline(learning_rate=eta, n_iterations=epochs, mse_threshold=threshold, bias=biass)

    # Choose a scaler (MinMaxScaler or StandardScaler) to preprocess the input features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test = scaler.fit_transform(X_test)
    
    # Train the Adaline model on the training data.
    adaline.fit(X_scaled, binary_encoded_labels)
    
    # Predict class labels on the test data.
    predictions = adaline.predict(X_test)
    
    # Calculate the confusion matrix and overall accuracy.
    confusion_matrix = calculate_confusion_matrix(binary_encoded_labels2, predictions)
    overall_accuracy = calculate_overall_accuracy(confusion_matrix)
    overall_accuracy = overall_accuracy*100
    
    # Print the confusion matrix and overall accuracy.
    print("Confusion Matrix:")
    print(confusion_matrix)
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    
    # Extract the weights and bias
    W1i = adaline.weights[1] if len(adaline.weights) > 2 else adaline.weights[0]
    W1j = adaline.weights[2] if len(adaline.weights) > 2 else adaline.weights[1]
    b = adaline.weights[0] if len(adaline.weights) > 2 else 0
    
    # Define a range of values for Xi
    min_Xi = X_scaled[:, 0].min() - 1
    max_Xi = X_scaled[:, 0].max() + 1
    
    # Calculate Xj values for the decision boundary
    if W1j != 0:
        Xj_decision_boundary = (-W1i * np.arange(min_Xi, max_Xi) - b) / W1j
    else:
        Xj_decision_boundary = np.zeros_like(np.arange(min_Xi, max_Xi))  # No decision boundary if W1j is zero
    
    # Scatter plot of the data points
    plt.scatter(X_scaled[y_encoded == 0][:, 0], X_scaled[y_encoded == 0][:, 1], marker='o', label='Class -1')
    plt.scatter(X_scaled[y_encoded == 1][:, 0], X_scaled[y_encoded == 1][:, 1], marker='x', label='Class 1')
    
    # Plot the decision boundary line
    plt.plot(np.arange(min_Xi, max_Xi), Xj_decision_boundary, color='red', label='Decision Boundary')
    
    # Set labels and show the plot
    plt.xlabel(f'{feature1} ({feature_1Name})')
    plt.ylabel(f'{feature2} ({feature_2Name})')
    plt.legend()
    plt.show()
