import numpy as np
import matplotlib.pyplot as plt

from src.utils import ReLU, ReLU_derivative, mean_squared_error

class NeuralNetwork:
  def __init__(self, dataset, layers) -> None:
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
        architecture
    """
    
    # parameters of the neural network
    self.parameters = {}
    
    # partial derivatives to update the parameters (weight, bias) of the neural network
    self.derivatives = {}
    
    # number of the last layer (the output layer)
    # number of layers - 1 because numerotation begins at 0.
    self.N = layers.size - 1
    
    # number of entries in the dataset
    self.m = dataset[:,0].size
    
    #print(dataset[0].size)
    
    # initialize neural network parameters
    for i in range(1, self.N + 1):
      self.parameters[f"W{str(i)}"] = np.random.uniform(size=(layers[i - 1], layers[i]))
      self.parameters[f"B{str(i)}"] = np.random.uniform(size=(self.m, layers[i]))
      
      self.parameters[f"Z{str(i)}"] = np.ones((self.m, layers[i]))
      self.parameters[f"A{str(i)}"] = np.ones((self.m, layers[i]))
    
    # initialize cost function value
    self.parameters["C"] = 1
    
  def forward_propagate(self, X):
    # initial the neural network with the input dataset
    self.parameters["A0"] = X
    
    for key, value in self.parameters.items():
      if type(value) is int:
        print(key, value)
      else:
        print(key, value.shape)
    
    print("================================")
    
    # forward propagate each subsequent layers
    for i in range(1, self.N + 1):
      # Z^i = (A^{i-1} \cdot W^i) + B^i
      Zi = (self.parameters[f"A{str(i-1)}"] @ self.parameters[f"W{str(i)}"]) + self.parameters[f"B{str(i)}"]
      self.parameters[f"Z{str(i)}"] = Zi
      # A^i = f(Z^i)
      self.parameters[f"A{str(i)}"] = ReLU(Zi)
      
    for key, value in self.parameters.items():
      if type(value) is int:
        print(key, value)
      else:
        print(key, value.shape)
    
    print("================================")
      
  def backward_propagate(self, X, Y):
    # compute derivatives of our loss function
    # we go backward
    
    # partial derivatives for the last layer
    dL_dAN = ((1 / self.m) * sum((self.parameters[f"A{str(self.N)}"]) - Y))[0]
    
    dL_dZN = dL_dAN * ReLU_derivative(self.parameters[f"Z{str(self.N)}"])
    
    #print(dL_dZN.T.shape, self.parameters[f"A{str(self.N - 1)}"].shape)
    dL_dWN = dL_dZN.T @ self.parameters[f"A{str(self.N - 1)}"]
    
    dL_dBN = dL_dZN
    
    self.derivatives[f"dLdZ{str(self.N)}"] = dL_dZN
    self.derivatives[f"dLdW{str(self.N)}"] = dL_dWN.T
    self.derivatives[f"dLdB{str(self.N)}"] = dL_dBN
    
    for key, value in self.derivatives.items():
      if type(value) is int:
        print(key, value)
      else:
        print(key, value.shape)
    
    print("================================")

    # partial derivatives for the subsequent layers
    for i in range(self.N - 1, 0, -1):   
      # "*" is to multiply entries 1 by 1 => shape: (m, n) * (m, n) for any m, n 
      dL_dZi =  ((self.derivatives[f"dLdZ{str(i + 1)}"]) @ self.parameters[f"W{str(i + 1)}"].T) * ReLU_derivative(self.parameters[f"Z{str(i)}"])
      dL_dWi = self.parameters[f"A{str(i - 1)}"].T @ dL_dZi
      dL_dBi = dL_dZi
      
      self.derivatives[f"dLdZ{str(i)}"] = dL_dZi
      self.derivatives[f"dLdW{str(i)}"] = dL_dWi
      self.derivatives[f"dLdB{str(i)}"] = dL_dBi
      
    for key, value in self.derivatives.items():
      if type(value) is int:
        print(key, value)
      else:
        print(key, value.shape)
    
    print("================================")
  
  def update_weights_and_bias(self, learning_rate):
    for i in range(1, self.N + 1):
      self.parameters[f"W{str(i)}"] -= learning_rate * self.derivatives[f"dLdW{str(i)}"]
      self.parameters[f"B{str(i)}"] -= learning_rate * self.derivatives[f"dLdB{str(i)}"]
  
  def gradient_descent(self, X, Y, epoch, learning_rate):
    for i in range(epoch):
      self.forward_propagate(X)
      self.backward_propagate(X, Y)
      self.update_weights_and_bias(learning_rate)
      
      if (i % 10 == 0):
        print(f"iteration: {i}")
        predictions = np.argmax(self.parameters[f"A{str(self.N)}"], 0)
        
        print(predictions, Y)
        print(np.sum(predictions == Y) / Y.size)