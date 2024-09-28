import numpy as np
X = np.array([[1, 1], [1, 2], [1, 3]])  # Design matrix with bias term
y = np.array([1, 2, 3])  # Target values
theta  = np.linalg.inv(X.T @ X) @(X.T@y)
print(theta)