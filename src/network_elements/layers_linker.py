import numpy as np

from src.network_elements.network_element import NetworkElement

class LayersLinker(NetworkElement):
  def __init__(self, previous_layer_dimension, next_layer_dimension) -> None:
      self.previous_layer_dimension = previous_layer_dimension
      self.next_layer_dimension = next_layer_dimension
      
      self.W = self.init_random_uniform_matrix(size=(previous_layer_dimension, next_layer_dimension))
      self.B = self.init_random_uniform_matrix(size=(1, next_layer_dimension))
      
      self.previous_layer_activated_output = None
      
      self.dLdW = None
      self.dLdB = None
      
  def init_random_uniform_matrix(self, size):
    low = - np.sqrt(1 / np.sum(size))
    high = np.sqrt(1 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)
    
  def init_random_gaussian_matrix(self, size, mean=0.0, variance=1.0):
    return np.random.normal(loc=mean, scale=np.sqrt(variance), size=size)

  def forward_propagate(self, A):
    self.previous_layer_activated_output = A
    
    Z = (A @ self.W) + self.B
    return Z
  
  def backward_propagate(self, dLdZ):
    if self.previous_layer_activated_output is None:
      raise ValueError("Please forward propagate information before backward propagating.")
    
    (batch_size, _) = dLdZ.shape
    
    self.dLdW = self.previous_layer_activated_output.T @ dLdZ
    self.dLdB = np.ones(batch_size).T @ dLdZ
    
    return dLdZ @ self.W.T
  
  def update_weights_and_bias(self, learning_rate):
    if self.dLdW is None and self.dLdB is None:
      raise ValueError("Please forward propagate and backward propagate before updating parameters.")
    
    self.W -= learning_rate * self.dLdW
    self.B -= learning_rate * self.dLdB