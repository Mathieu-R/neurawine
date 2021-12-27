import numpy as np
import matplotlib.pyplot as plt 

def plot_confusion_matrix(confusion_matrix):
  plt.title("Confusion Matrix")
  plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Greens)
  plt.colorbar()
  
  plt.xlabel("Predicted labels")
  plt.ylabel("True labels")
  
  plt.tight_layout()
  plt.show()