import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from src.NeuralNetwork import NeuralNetwork

# read red wine dataset
# 11 categories => 11 inputs ; 1 output (wine quality)
red_wine_dataset = pd.read_csv('./dataset/winequality-red.csv', sep=";").to_numpy()

red_wine_inputs_dataset = red_wine_dataset[:,0:-1]
red_wine_output_dataset = red_wine_dataset[:,-1]

# splitting dataset: 80% training / 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(red_wine_inputs_dataset, red_wine_output_dataset, train_size=0.8, test_size=0.2, shuffle=False)

# get dimensions: n lines / m columns
n, m = red_wine_dataset.shape

# number of different wines in the dataset
# we split the dataset in two so we have n/2
NUMBER_OF_DATA = int(n/2)

# number of categories = number of inputs = m - 1 (because 1 column is the true output = y)
NUMBER_OF_NODES_INPUTS = m - 1

# number of nodes for hidden layers 
NUMBER_OF_NODES_HIDDEN_LAYER = 10

# number of nodes for output layer
NUMBER_OF_NODES_OUTPUT_LAYER = 1


# architecture of the neural network
# number of nodes by layer
layers = np.array([NUMBER_OF_NODES_INPUTS, NUMBER_OF_NODES_HIDDEN_LAYER, NUMBER_OF_NODES_OUTPUT_LAYER])
activations = np.array(["relu", "relu"])

if __name__ == "__main__":
  NN = NeuralNetwork(X=X_train, layers=layers)
  NN.gradient_descent(X=X_train, Y=Y_train, nb_epoch=1000, learning_rate=0.0005)