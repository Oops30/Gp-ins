from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f"F1 Score: {metrics.f1_score(y_test, y_pred, average='macro')}")
    print(f"Precision: {metrics.precision_score(y_test, y_pred, average='macro')}")
    print(f"Recall: {metrics.recall_score(y_test, y_pred, average='macro')}")
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

# Step 1: Classification before adding noise
print("=== Classification Before Adding Noise ===")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svm_model = SVC(kernel='linear', C=1).fit(X_train, y_train)
evaluate_model(svm_model, X_test, y_test)

# Step 2: Classification after adding noise
print("\n=== Classification After Adding Noise ===")
np.random.seed(42)
X_noisy = X + np.random.normal(0, 0.5, X.shape)  # Adding noise
X_train_noisy, X_test_noisy, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)
svm_model_noisy = SVC(kernel='linear', C=1).fit(X_train_noisy, y_train)
evaluate_model(svm_model_noisy, X_test_noisy, y_test)

# Print the target and target names
print("\nTarget Values:", y)
print("Target Names:", iris.target_names)
       

