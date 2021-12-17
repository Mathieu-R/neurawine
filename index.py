from os import sep
import numpy as np
import pandas as pd

from src.NeuralNetwork import NeuralNetwork

# read training dataset
# 11 categories => 11 inputs ; 1 output (wine quality)
red_wine_dataset = pd.read_csv('./dataset/winequality-red.csv', sep=";").to_numpy()

# get dimensions: n lines / m columns
n, m = red_wine_dataset.shape

# number of different wines in the dataset
# we split the dataset in two so we have n/2
NUMBER_OF_DATA = int(n/2)

# number of categories = number of inputs = m - 1 (because 1 column is the true output = y)
NUMBER_OF_INPUTS = m - 1

# number of nodes for hidden layers and output layers (arbitrary)
NUMBER_OF_NODES = 10

# learning rate (eta)
LEARNING_RATE = 0.1

# split dataset into two : a training dataset / a testing dataset
training_dataset = red_wine_dataset[0:NUMBER_OF_DATA].T
training_inputs = training_dataset[0:m - 1] # wine classifying categories (acidity, sugar, ph,...)
training_output = training_dataset[m - 1] # wine quality
#print(training_output)

testing_dataset = red_wine_dataset[NUMBER_OF_DATA:n].T
testing_inputs = testing_dataset[0:m - 1] # wine classifying categories (acidity, sugar, ph,...)
testing_output = testing_dataset[m - 1] # wine quality

if __name__ == "__main__":
  NN = NeuralNetwork(training_inputs, training_output, NUMBER_OF_DATA, NUMBER_OF_INPUTS, NUMBER_OF_NODES, LEARNING_RATE)
  NN.gradient_descent()