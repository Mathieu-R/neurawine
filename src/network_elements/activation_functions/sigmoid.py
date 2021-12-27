import numpy as np

class Sigmoid:
  def __init__(self) -> None:
      self.current_layer_output = None
    
  def sigmoid(self, Z):
    return 1 / (1 - np.exp(Z))

  def sigmoid_derivative(self, Z):
    return self.sigmoid(Z) * (1 - self.sigmoid(Z))  
  
  def forward_propagate(self, Z):
    self.current_layer_output = Z
    return self.sigmoid(Z)
  
  def backward_propagate(self, dLdZ):
    return dLdZ * self.sigmoid_derivative(self.current_layer_output)