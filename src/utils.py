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
  if x <= 0:
    return 0
  else:
    return x 
  
def ReLU_derivative(x):
  if x <= 0:
    return 0
  else:
    return 1

# loss function
"""
@param y_true: np.array
@param y_pred: np.array
J(y_true, y_pred) = 1/n \sum_{i=1}^{n} (y_true - y_pred)^2
"""
def mean_squared_error(y_true, y_pred):
  return (1 / len(y_pred)) * sum((y_true - y_pred) ** 2)