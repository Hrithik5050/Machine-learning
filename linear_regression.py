import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#creating psuedo dataset
np.random.seed(42)
x=2*np.random.rand(100,1)
y=4+3*x+ np.random.randn(100,1)

#train and test split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
print(mse)

# Plotting
plt.figure(figsize=(10, 6))

# Plot training data points
plt.scatter(x_train, y_train, color="blue", label="Training data")

# Plot testing data points
plt.scatter(x_test, y_test, color="green", label="Testing data")

# Plot the regression line
X_line = np.linspace(0, 2, 100).reshape(100, 1)  # Generate values for a smooth line
y_line = model.predict(X_line)  # Predict values for the regression line
plt.plot(X_line, y_line, color="black", label="Regression line")

# Labels and title
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
