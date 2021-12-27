import numpy as np

from src.network_elements.loss_functions.mse import MSE

class Trainer:
  def __init__(self, network, batch_size, nb_epoch, learning_rate, loss_function) -> None:
    self.network = network 
    self.batch_size = batch_size 
    self.nb_epoch = nb_epoch
    self.learning_rate = learning_rate 
    
    if loss_function == "mse":
      self.loss = MSE()
      
  def train(self, input_dataset, output_dataset):
    for epoch in range(self.nb_epoch):
      batch_input_dataset = np.array_split(input_dataset, self.batch_size)
      batch_output_dataset = np.array_split(output_dataset, self.batch_size)
      
      for batch_input, batch_output in zip(batch_input_dataset, batch_output_dataset):
        # 1. Forward Propagate 
        batch_prediction = self.network.forward_propagate(batch_input)
        # 2. Compute Error
        cost = self.loss.compute_loss(Y=batch_output, Y_hat=batch_prediction)
        # 3. Backward Propagate
        dLdAN = self.loss.backward_propagate(Y=batch_output)
        self.network.backward_propagate(dLdAN)
        # 4. Update Weights and Bias
        self.network.update_weights_and_bias(self.learning_rate)
        
      print(f"Epoch {epoch} out of {self.nb_epoch} completed.")
      
  def compute_loss(self, input_dataset, output_dataset):
    predictions = self.network.forward_propagate(input_dataset)
    cost = self.loss.compute_loss(Y=output_dataset, Y_hat=predictions)
    return cost