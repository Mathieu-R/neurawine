import numpy as np

from src.utils import ReLU, ReLU_derivative, mean_squared_error

class NeuralNetwork:
  def __init__(self, training_inputs, training_output, NUMBER_OF_DATA, NUMBER_OF_INPUTS, NUMBER_OF_NODES, LEARNING_RATE) -> None:
    self.training_inputs = training_inputs
    self.training_output = training_output
    
    self.NUMBER_OF_DATA = NUMBER_OF_DATA
    self.NUMBER_OF_INPUTS = NUMBER_OF_INPUTS
    self.NUMBER_OF_NODES = NUMBER_OF_NODES
    self.LEARNING_RATE = LEARNING_RATE
    
    # dimension of each layer subsequent to the inputs
    # is a matrix of size #current_nodes x #previous_nodes
    
    # hidden layer 1 : 10 nodes (arbitrary) ; 11 nodes for inputs
    # return a matrix 10x11 of values between [0, 1]
    self.weight_hl1 = np.random.uniform(size=(NUMBER_OF_DATA, NUMBER_OF_NODES))
    self.bias_hl1 = np.random.uniform(size=(NUMBER_OF_INPUTS, NUMBER_OF_NODES))
    # output layer : 10 nodes (arbitrary) ; 10 nodes for hidden layer 1
    # return a matrix 10x10 of values between [0, 1]
    self.weight_ol = np.random.uniform(size=(NUMBER_OF_NODES, NUMBER_OF_DATA))
    self.bias_ol = np.random.uniform(size=(NUMBER_OF_INPUTS, NUMBER_OF_DATA))
    
  def forward_propagate(self):
    # compute first hidden layer: \sum (inputs * weight) + bias (linear combination)
    self.hl1 = np.dot(self.training_inputs, self.weight_hl1) + self.bias_hl1
    # make the hidden layer non-linear by applying an activation function on it
    self.activated_hl1 = ReLU(self.hl1)
    
    # compute the output layer
    self.ol = np.dot(self.activated_hl1, self.weight_ol) + self.bias_ol
    self.activated_ol = ReLU(self.ol)
    
  def back_propagate(self):
    # compute derivatives of our loss function
    # we go backward
    
    y_true = self.training_output
    y_pred = self.activated_ol
    
    ## output layer ##
    
    # with respect to the weights
    # dL/dW2 = dL/dypred * dypred/dZ2 * dZ2/dW2 
    
    # with respect to the bias
    # dL/dB2 = dL/dypred * dypred/dZ2 * dZ2/dB2
    
    dL_dypred = (1 / self.NUMBER_OF_DATA) * 2 * sum( (y_true - y_pred) * (-1) )
    dypred_dZ2 = ReLU_derivative(self.ol)
    dZ2_dW2 = self.activated_hl1
    
    dZ2_dB2 = 1
    
    ## hidden_layer 1 ##
    
    # with respect to the weights
    # dL/dW1 = dL/dypred * dypred/dZ2 * dZ2/dfZ1 * dfZ1/dZ1 * dZ1/dW1
    
    # with respect to the bias
    # dL/dB2 = dL/dypred * dypred/dZ2 * dZ2/dfZ1 * dfZ1/dZ1 * dZ1/dB1
    
    dZ2_dfZ1 = self.weight_ol
    dfZ1_dZ1 = ReLU_derivative(self.hl1)
    dZ1_dW1 = self.training_inputs
    
    dZ1_dB1 = 1
    
    # compute the "full derivatives"
    self.dL_dW2 = dL_dypred * dypred_dZ2 * dZ2_dW2
    self.dL_dB2 = dL_dypred * dypred_dZ2 * dZ2_dB2
    
    self.dL_dW1 = dL_dypred * dypred_dZ2 * dZ2_dfZ1 * dfZ1_dZ1 * dZ1_dW1
    self.dL_dB1 = dL_dypred * dypred_dZ2 * dZ2_dfZ1 * dfZ1_dZ1 * dZ1_dB1
  
  def update_weights_and_bias(self):
    self.weight_ol -= self.LEARNING_RATE * self.dL_dW2
    self.bias_ol -= self.LEARNING_RATE * self.dL_dB2
    
    self.weight_hl1 -= self.LEARNING_RATE * self.dL_dW1
    self.bias_hl1 -= self.LEARNING_RATE * self.dL_dB1
  
  def gradient_descent(self):
    EPOCH = 1000
    for i in range(EPOCH):
      self.forward_propagate()
      self.back_propagate()
      self.update_weights_and_bias()
      
      if (i % 10 == 0):
        print(f"iteration: {i}")
        predictions = np.argmax(self.ol, 0)
        
        print(predictions, self.training_output)
        print(np.sum(predictions == self.training_output) / self.training_output.size)