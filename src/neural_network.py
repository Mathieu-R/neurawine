import numpy as np
import matplotlib.pyplot as plt

#from src.utils import ReLU, ReLU_derivative, mean_squared_error, mean_squared_error_derivative  
from src.network_elements.layers_linker import LayersLinker
from src.network_elements.activation_functions.relu import ReLU
from src.network_elements.activation_functions.sigmoid import Sigmoid

class NeuralNetwork:
  def __init__(self, input_dimension, layers_architecture, activations) -> None:
    """[summary]
    NOTATION:
    X = training input (A0)
    Y = training output (y_true) 
    Y_hat = predicted output (y_pred) = = activated output associated with the last layer (that is the output layer)
    Wi = weight matrix associated with i-th layer
    Bi = bias matrix associated with i-th layer
    Zi = (A_{i-1} \cdot Wi) + Bi = output matrix associated with i-th layer
    Ai = f(Zi) = activated output associated with i-th layer where f is the activation function (ex: ReLU)
    
    L = Loss function (ex: MSE)
    Args:
        input_dimension
        layers_architecture
        activations
    """
    
    self.input_dimension = input_dimension
    self.layers_architecture = layers_architecture
    self.activations = activations
    
    self.network_elements = []
    
    previous_layer_dimension = input_dimension
    
    for i in range(layers_architecture.size):
      next_layer_dimension = layers_architecture[i]
      activation = activations[i]
      
      self.network_elements.append(LayersLinker(previous_layer_dimension, next_layer_dimension))
      
      if activation == "relu":
        self.network_elements.append(ReLU())
      elif activation == "sigmoid":
        self.network_elements.append(Sigmoid())
        
      previous_layer_dimension = next_layer_dimension
    
  def forward_propagate(self, X):
    Z = X 
    for network_element in self.network_elements:
      Z = network_element.forward_propagate(Z)
      
    return Z
      
  def backward_propagate(self, dLdAN):
    dLdZ = dLdAN
    for network_element in self.network_elements[::-1]:
      dLdZ = network_element.backward_propagate(dLdZ)
    
    return dLdZ
  
  def update_weights_and_bias(self, learning_rate):
    for network_element in self.network_elements:
      network_element.update_weights_and_bias(learning_rate)
    