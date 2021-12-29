import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from src.neural_network import NeuralNetwork
from src.trainer import Trainer
from src.confusion_matrix import plot_confusion_matrix
from src.cost_evolution import plot_cost_evolution

# import wine dataset 
wine_dataset = datasets.load_wine()

wine_input_dataset = wine_dataset.data 
wine_output_dataset = wine_dataset.target

# splitting dataset: 80% training / 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(wine_input_dataset, wine_output_dataset, train_size=0.8, test_size=0.2, shuffle=True)

# dataset already normalised by sklearn

# # normalise data to optimize the training
# scaler = MinMaxScaler()
# # fit scaler on data (so it can retrieve min and max value in the dataset)
# scaler.fit(X_train)

# X_train_normalized = scaler.transform(X_train)
# X_test_normalized = scaler.transform(X_test)

(n_samples, m_inputs) = wine_input_dataset.shape

# architecture of the neural network : number of nodes by layer
# we do not include the input layer
# e.g. for "l" hidden layers : [nb_nodes_hidden_layer_1, ..., nb_nodes_hidden_layer_l, nb_nodes_output_layer]
layers_architecture = np.array([100, 1])
# activation function for each i-th layer
activations = np.array(["relu", "sigmoid"])

network = NeuralNetwork(input_dimension=m_inputs, layers_architecture=layers_architecture, activations=activations)
trainer = Trainer(network=network, batch_size=10, nb_epoch=1000, learning_rate=0.005, loss_function="mse")

# train our network on training dataset
trainer.train(X_train, Y_train)

# evaluate error on predictions
print(f"[Training dataset] Error on predictions = {trainer.compute_loss(X_train, Y_train)}")
print(f"[Testing dataset] Error on predictions = {trainer.compute_loss(X_test, Y_test)}")

# test our network on testing dataset
print(network.forward_propagate(X_test))
print(Y_test)

predictions = network.forward_propagate(X_test).argmax(axis=1).squeeze()
true_output = Y_test.argmax(axis=0).squeeze()

accuracy = (predictions == true_output).mean()

print(f"[Testing dataset] Predictions accuracy = {accuracy}")

# plot cost evolution over epoch
plot_cost_evolution(cost_history=trainer.cost_history, nb_epoch=1000)

# plot confusion matrix
cm = confusion_matrix(y_true=true_output, y_pred=predictions)
plot_confusion_matrix(confusion_matrix=cm)