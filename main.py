# Step 1: Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 2: Loading and Visualizing the Data
# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

# Display the first 5 rows of the dataset
print("First 5 rows of the Iris dataset:")
print(iris_df.head())

# Describe basic statistics of the dataset
print("\nBasic statistics of the dataset:")
print(iris_df.describe())

# Step 3: Exploratory Data Analysis (EDA)
# Create scatter plots to visualize relationships between features
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=iris_df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="target",
    palette="deep",
)
plt.title("Sepal Length vs Sepal Width")
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=iris_df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="target",
    palette="deep",
)
plt.title("Petal Length vs Petal Width")
plt.show()

# Create histograms for each feature
iris_df.hist(figsize=(12, 10), bins=15)
plt.suptitle("Histograms of Iris Features")
plt.show()

# Create a heatmap for the features correlation
plt.figure(figsize=(10, 6))
sns.heatmap(iris_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.show()

# Step 4: Data Preprocessing
# Split the dataset into features (X) and target (y)
X = iris_df.drop("target", axis=1)
y = iris_df["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Training and Evaluating the Decision Tree
# Train a decision tree classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# To determine which pair of variables has the highest positive correlation
# in the Iris dataset, you can calculate the correlation coefficients using
# the `pandas` library. Here's how to do it:

# Load the Iris dataset
# data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Calculate the correlation matrix
# correlation_matrix = data.corr()

# Find the pair of variables with the highest positive correlation
# positive_correlations = correlation_matrix[correlation_matrix > 0].stack()
# highest_positive_correlation = positive_correlations.nlargest(3)
# print(highest_positive_correlation)
