# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate some example data (replace this with your own dataset)
# Here we create a binary classification problem
np.random.seed(42)
X = 3 * np.random.rand(100, 1) - 1.5  # Features
y = (X > 0).astype(int)               # Labels: 1 if X > 0, otherwise 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

# Plot the data and decision boundary
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
# plt.plot(X_test, model.predict_proba(X_test)[:, 1], color='red', label='Predicted Probability')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
