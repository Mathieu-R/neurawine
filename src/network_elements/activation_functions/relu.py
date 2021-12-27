import numpy as np

from src.network_elements.network_element import NetworkElement

class ReLU(NetworkElement):
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
    if self.current_layer_output is None:
      raise ValueError("Please forward propagate information before backward propagating.")
    
    return dLdZ * self.relu_derivative(self.current_layer_output)
