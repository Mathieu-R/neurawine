import numpy as np

class MSE:
  def __init__(self) -> None:
    # current prediction (Y_hat)
    self.current_predictions = None
    
  def mse(self, Y, Y_hat):
    return np.square(Y - Y_hat).mean()

  def mse_derivative(self, Y, Y_hat):
    return 2 * np.mean(Y - Y_hat)
  
  def compute_loss(self, Y, Y_hat):
    self.current_predictions = Y_hat
    return self.mse(Y, Y_hat)
  
  def backward_propagate(self, Y):
    if self.current_predictions is None:
      raise ValueError("Please compute loss function before backward propagating it.")
    
    return self.mse_derivative(Y=Y, Y_hat=self.current_predictions)