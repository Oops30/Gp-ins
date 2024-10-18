
# Importing the libraries
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets

# Loading the Iris dataset from sklearn
def loaddata():
    # Load the dataset
    iris = datasets.load_iris()

    # Create a DataFrame from the iris dataset
    dataset = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['class'])
    return dataset

# Loading the dataset using the function
dataset = loaddata()
print("Iris Dataset Loaded Successfully:")
print(dataset.head())

# Selecting features and target variable
X = dataset.iloc[:, [0, 3]].values  # Features: Sepal Length and Petal Width
y = dataset.iloc[:, -1].values      # Target: Species

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("Predicted Test Results : ", y_pred)
print("~" * 20)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Calculate accuracy
ac = accuracy_score(y_test, y_pred)
print("Model Accuracy : ", ac)
print("~" * 20)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Model Confusion Matrix : ")
print(cm)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Visualizing Accuracy as a fraction (0 to 1)
plt.figure(figsize=(6, 4))
plt.bar(['Naive Bayes'], [ac], color='skyblue')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.ylabel('Accuracy (0 to 1)')
plt.show()

