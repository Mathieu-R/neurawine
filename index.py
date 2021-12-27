import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from src.neural_network import NeuralNetwork
from src.trainer import Trainer
from src.confusion_matrix import plot_confusion_matrix

# read red wine dataset
# 11 categories => 11 inputs ; 1 output (wine quality)
red_wine_dataset = pd.read_csv('./dataset/winequality-red.csv', sep=";").to_numpy()

red_wine_inputs_dataset = red_wine_dataset[:,0:-1]
red_wine_output_dataset = red_wine_dataset[:,-1]

# splitting dataset: 80% training / 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(red_wine_inputs_dataset, red_wine_output_dataset, train_size=0.8, test_size=0.2, shuffle=False)

# normalise data to optimize the training
scaler = MinMaxScaler()
# fit scaler on data (so it can retrieve min and max value in the dataset)
scaler.fit(X_train)

X_train_normalized = scaler.transform(X_train)
X_test_normalized = scaler.transform(X_test)

# get dimensions: n lines / m columns
n, m = red_wine_dataset.shape

# dimension of the input
input_dimension = m - 1

# architecture of the neural network : number of nodes by layer
# we do not include the input layer
# e.g. for "l" hidden layers : [nb_nodes_hidden_layer_1, ..., nb_nodes_hidden_layer_l, nb_nodes_output_layer]
layers_architecture = np.array([100, 1])
# activation function for each i-th layer
activations = np.array(["relu", "relu"])

network = NeuralNetwork(input_dimension=input_dimension, layers_architecture=layers_architecture, activations=activations)
trainer = Trainer(network=network, batch_size=10, nb_epoch=1000, learning_rate=0.05, loss_function="mse")

# train our network on training dataset
trainer.train(X_train, Y_train)

# evaluate error on predictions
print(f"[Training dataset] Error on predictions = {trainer.compute_loss(X_train_normalized, Y_train)}")
print(f"[Testing dataset] Error on preditions = {trainer.compute_loss(X_test_normalized, Y_test)}")

# test our network on testing dataset
predictions = network.forward_propagate(X_test).argmax(axis=1).squeeze()
true_output = Y_test.argmax(axis=0).squeeze()

accuracy = (predictions == true_output).mean()

print(f"[Testing dataset] Predictions accuracy = {accuracy}")

# get confusion matrix
cm = confusion_matrix(y_true=true_output, y_pred=predictions)
plot_confusion_matrix(confusion_matrix=cm)