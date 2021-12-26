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

def mean_squared_error(Y, Y_hat):
  return np.square(Y - Y_hat).mean()

def mean_squared_error_derivative(Y, Y_hat):
  return 2 * np.mean(Y_hat - Y)