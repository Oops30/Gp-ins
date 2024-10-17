import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Data generation
X, y = make_gaussian_quantiles(n_samples=2000, n_features=10, n_classes=3, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_leaf_nodes=8),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_leaf_nodes=8), n_estimators=300, algorithm="SAMME")
}

# Train, evaluate and print results
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"Error": 1 - acc, "CM": confusion_matrix(y_test, y_pred)}
    print(f"{name}: Error: {results[name]['Error']:.3f}, Accuracy: {acc:.3f}\n{classification_report(y_test, y_pred)}")

# Print specific misclassification errors
for name, res in results.items():
    print(f"{name}'s misclassification error: {res['Error']:.3f}")

# Plotting confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, metrics) in zip(axes, results.items()):
    sns.heatmap(metrics['CM'], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
plt.tight_layout()
plt.show()

# Plot misclassification errors
pd.DataFrame({"Classifier": results.keys(), "Error": [m['Error'] for m in results.values()]}).plot(
    x="Classifier", kind="bar", legend=False, color='orange')
plt.ylabel("Misclassification Error")
plt.title("Classifier Errors Comparison")
plt.xticks(rotation=0)
plt.show()

# Weak learners' errors and weights for AdaBoost
weak_learners_info = pd.DataFrame({
    "Number of Trees": range(1, 301),
    "Errors": classifiers["AdaBoost"].estimator_errors_,
    "Weights": classifiers["AdaBoost"].estimator_weights_
})

# Plot weak learners' errors and weights
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(weak_learners_info["Number of Trees"], weak_learners_info["Errors"], marker='o', color='tab:blue')
axs[1].plot(weak_learners_info["Number of Trees"], weak_learners_info["Weights"], marker='o', color='tab:orange')
axs[0].set(title="Weak Learner's Training Errors", xlabel="Number of Trees", ylabel="Training Error")
axs[1].set(title="Weak Learner's Weights", xlabel="Number of Trees", ylabel="Weight")
fig.suptitle("Weak Learner's Errors and Weights for AdaBoostClassifier")
plt.tight_layout()
plt.show()
