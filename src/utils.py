import numpy as np

# useful activation functions for hidden layer 1 / output layer

"""
@param x: np.array
"""
def sigmoid(x):
  return 1 / (1 - np.exp(x))

"""
@param x: np.array
"""
def ReLU(x):
  return np.maximum(0, x)
  
def ReLU_derivative(x):
  return x > 0

# loss function
"""
@param y_true: np.array
@param y_pred: np.array
J(y_true, y_pred) = 1/n \sum_{i=1}^{n} (y_true - y_pred)^2
"""
def mean_squared_error(y_true, y_pred):
  return (1 / len(y_pred)) * sum((y_true - y_pred) ** 2)