import numpy as np

class ReLU:
  def __init__(self) -> None:
    self.current_layer_output = None
    
  def relu(self, Z):
    return np.maximum(0, Z)
    
  def relu_derivative(self, Z):
    return Z > 0
    
  def forward_propagate(self, Z):
    self.current_layer_output = Z
    return self.relu(Z)
  
  def backward_propagate(self, dLdZ):
    return dLdZ * self.relu_derivative(self.current_layer_output)
