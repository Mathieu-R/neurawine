import numpy as np

"""
@param x: np.array
"""
def sigmoid(x):
  return 1 / (1 - np.exp(x))