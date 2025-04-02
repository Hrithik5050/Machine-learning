import numpy as np

# Dataset
X = np.array([[1, 1, 2],   # [bias, x1, x2]
              [1, 2, 3],
              [1, 3, 4],
              [1, 4, 5]])

y = np.array([5, 7, 9, 11])  # Target values

# Parameters (theta), initialized to 0
theta = np.zeros(3)

# Hyperparameters
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Number of training examples
#m = len(y)

# Gradient Descent Algorithm
def gradient_descent(X, y, theta, alpha, iterations):
    #m = len(y)
    for _ in range(iterations): #you can use an underscore (_) as a variable name in a for loop, especially when you 
        #don’t actually need to use the variable. 
        #Using _ is a common Python convention to signal that the variable is being ignored.
        # Compute the hypothesis h = X * theta
        h = np.dot(X, theta) #np.dot is matrix multiplication
         
        # Compute the gradient
        #gradient = (1/m) * np.dot(X.T, (h - y)) #X.T is transpse of X
        gradient = np.dot(X.T, (h - y)) #X.T is transpse of X
        # Update the parameters
        theta = theta - alpha * gradient
    
    return theta

# Running gradient descent
theta_optimized = gradient_descent(X, y, theta, alpha, iterations)

print(f"Optimized parameters: {theta_optimized}")