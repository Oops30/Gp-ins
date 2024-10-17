import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
iris = load_iris()
X, y = iris.data, iris.target
data = pd.DataFrame(X, columns=iris.feature_names)
data['target'] = y
print("T084-Nikita Hirap")
# Info
print("Dataset shape:", data.shape)
print("Sepal Length, Sepal Width, Petal Length, Petal Width:\n", data.describe())
print("Target classes:", iris.target_names)
# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
# Sigmoid functions
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_derivative = lambda x: x * (1 - x)
# Parameters
input_size, hidden_size, output_size, epochs, lr = X_train.shape[1], 10, 3, 10000, 0.01
weights_input_hidden = np.random.rand(input_size, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
print("\nNetwork Architecture:")
print(f"Input Layer Size: {input_size}, Hidden Layer Size: {hidden_size}, Output Layer Size: {output_size}")
# Training
mse_values = []
for epoch in range(epochs):
    hidden_output = sigmoid(X_train @ weights_input_hidden)
    final_output = sigmoid(hidden_output @ weights_hidden_output)
    error = np.eye(output_size)[y_train] - final_output
    weights_hidden_output += lr * hidden_output.T @ (error * sigmoid_derivative(final_output))
    weights_input_hidden += lr * X_train.T @ (sigmoid_derivative(hidden_output) * (error @ weights_hidden_output.T))
    mse_values.append(np.mean(error**2))
    if epoch % 1000 == 0: print(f'Epoch {epoch}, MSE: {mse_values[-1]}')
# Final weights
print("\nFinal Weights:")
print("Weights from Input to Hidden Layer:\n", weights_input_hidden)
print("Weights from Hidden to Output Layer:\n", weights_hidden_output)
# Evaluation
predictions = np.argmax(sigmoid(sigmoid(X_test @ weights_input_hidden) @ weights_hidden_output), axis=1)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
# Training Progress
plt.plot(mse_values)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Progress")
plt.show()
