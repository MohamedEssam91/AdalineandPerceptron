import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000,MSE_threshold=0.02, use_bias=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.MSE_threshold =MSE_threshold
        if use_bias:
            self.bias = 0
        else:
            self.bias = None

    def compute_MSE(self,X,y):
       y_pred =self.predict(X)
       MSE = np.mean((y - y_pred)**2)
       return MSE
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iterations):

             for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights)
                if self.bias is not None:
                    linear_output += self.bias
                y_predicted = self.activation(linear_output)

                loss=(y[i] - y_predicted)
                update = self.learning_rate * loss
                self.weights += update * X[i]
                if self.bias is not None:
                    self.bias += update

                    MSE = self.compute_MSE(X,y)

                    if(MSE <= self.MSE_threshold ):
                     break


    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        if self.bias is not None:
            linear_output += self.bias
        y_predicted = self.activation(linear_output)
        return np.where(y_predicted >= 0, 1, -1)

    def activation(self, x):
        return np.where(x >= 0, 1, -1)


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

def Perceptron_GUI(class1, class2, feature1, feature2, eta, epochs, threshold, biass):
    df = pd.read_excel("Dry_Bean_Dataset.xlsx")
    df["MinorAxisLength"].fillna(df["MinorAxisLength"].mean(), inplace=True)
    df_BOMBAY = df[df['Class'] == "BOMBAY"]
    df_CALI = df[df['Class'] == "CALI"]
    df_SIRA = df[df['Class'] == "SIRA"]

    #Splitting data into 30 samples for training and 20 samples for testing for each class

    train_size = 30
    group1_train = df_BOMBAY.iloc[:train_size]
    group1_test = df_BOMBAY.iloc[train_size:]

    # Group 2
    group2_train = df_CALI.iloc[:train_size]
    group2_test = df_CALI.iloc[train_size:]

    # Group 3
    group3_train = df_SIRA.iloc[:train_size]
    group3_test = df_SIRA.iloc[train_size:]


    # Combining dataframes for training and testing
    training_df = pd.concat([group1_train, group2_train, group3_train], ignore_index=True)
    testing_df = pd.concat([group1_test, group2_test, group3_test], ignore_index=True)

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

    X_test = testing_df[[feature_1Name, feature_2Name]]
    y_test = testing_df['Class']


    prec = Perceptron(learning_rate=eta, n_iterations=epochs,MSE_threshold= threshold, use_bias=biass)
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)
    binary_encoded_labels = 2 * y_encoded - 1
    y_encoded_test = label_encoder.transform(y_test)
    binary_encoded_labels2 = 2 * y_encoded_test - 1


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the perceptron model on the training data.
    prec.fit(X_scaled , binary_encoded_labels)

    # Predict class labels on the test data.
    # Scale the test data using the same scaler as the training data
    X_scaled_test = scaler.fit_transform(X_test)
    predictions = prec.predict(X_scaled_test)

    # Calculate the confusion matrix and overall accuracy.
    confusion_matrix = calculate_confusion_matrix(binary_encoded_labels2, predictions)
    overall_accuracy = calculate_overall_accuracy(confusion_matrix)
    overall_accuracy *= 100

    print("Confusion Matrix:")
    print(confusion_matrix)
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
  
    W1i = prec.weights[1] if len(prec.weights) > 2 else prec.weights[0]
    W1j = prec.weights[2] if len(prec.weights) > 2 else prec.weights[1]
    b = prec.weights[0] if len(prec.weights) > 2 else 0

    # Define a range of values for Xi
    min_Xi = X_scaled[:, 0].min() - 1
    max_Xi = X_scaled[:, 0].max() + 1

    # Calculate Xj values for the decision boundary
    Xj_decision_boundary = (-W1i * np.arange(min_Xi, max_Xi) - b) / W1j

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