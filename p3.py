import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Libraries imported\nT106 TANVI SAKHALE")

# Load dataset and create DataFrame
iris = load_iris()
df = pd.DataFrame(np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['species'])

# Encode labels, split dataset
df['species'] = LabelEncoder().fit_transform(df['species'])
print("Null values in each column:\n",df.isnull().sum())
X_train, X_test, y_train, y_test = train_test_split(df.drop('species', axis=1), df['species'], test_size=0.3, random_state=101)

# Train and evaluate model
clf = DecisionTreeClassifier(random_state=0, criterion='gini').fit(X_train, y_train)
print(f"Test Accuracy: {accuracy_score(y_test, clf.predict(X_test)) * 100:.2f}")
print(f"Train Accuracy: {accuracy_score(y_train, clf.predict(X_train)) * 100:.2f}")

# Visualize Decision Tree
plt.figure(figsize=(10, 5))
plot_tree(clf, filled=True, feature_names=df.columns[:-1], class_names=iris.target_names)
plt.show()

# Second tree plot without labels
plt.figure(figsize=(10, 5))
plot_tree(clf, filled=True)
plt.show()

# Reports and confusion matrices
print("Test Set Report:\n", classification_report(y_test, clf.predict(X_test)))
print("Confusion Matrix (Test Set):\n", confusion_matrix(y_test, clf.predict(X_test)))
print("Train Set Report:\n", classification_report(y_train, clf.predict(X_train)))
print("Confusion Matrix (Train Set):\n", confusion_matrix(y_train, clf.predict(X_train)))

