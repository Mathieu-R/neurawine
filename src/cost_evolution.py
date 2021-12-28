import numpy as np 
import matplotlib.pyplot as plt

def plot_cost_evolution(cost_history, nb_epoch):
  plt.title("Cost Evolution")

  plt.plot(range(nb_epoch), cost_history)
  
  plt.xlabel("Epoch")
  plt.ylabel("Cost")
  
  plt.tight_layout()
  plt.show() 