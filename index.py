import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.metrics import confusion_matrix

from src.neural_network import NeuralNetwork
from src.trainer import Trainer
from src.confusion_matrix import plot_confusion_matrix
from src.cost_evolution import plot_cost_evolution

# import wine dataset 
wine_dataset = datasets.load_wine()

wine_input_dataset = wine_dataset.data 
wine_output_dataset = wine_dataset.target

# one hot encoding
# converts the output into an matrix of dimensions (n_samples, number_of_possible_outputs)
# where each row contains one "1" and other are "0" so that the "1" is at the index corresponding to the output (the output value is converted into an index) 
# so that we can use argmax and it is consistent with the predictions output (made of probability density) of the neural network
one_hot_encoder = LabelBinarizer()
wine_output_dataset_one_hot = one_hot_encoder.fit_transform(wine_output_dataset)

# splitting dataset: 80% training / 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(wine_input_dataset, wine_output_dataset_one_hot, train_size=0.8, test_size=0.2, shuffle=True)

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
# here the output is can be 3 differents cultivators of the wine (represented by number 0, 1, 2)
# so the dimension of the output layer is 3
layers_architecture = np.array([100, 3])
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

# the neural network return a probability density for the output to be one of the three cultivators.abs
# for each sample correspond a prediction output of dimension 3 that contains probability density to be one of the 3 cultivators. 
# we take the highest probability density to be the prediction
# for example, if for a given sample, the higest probability density is at index 1, it means that the prediction is that the cultivator is the one who is at the first index in the list of the possible labels output ["cultivator 1", "cultivator 2", "cultivator 3"]

predictions = network.forward_propagate(X_test).argmax(axis=1).squeeze()
Y_test = Y_test.argmax(axis=1).squeeze()

accuracy = (predictions == Y_test).mean()

print(f"[Testing dataset] Predictions accuracy = {accuracy}")

# plot cost evolution over epoch
plot_cost_evolution(cost_history=trainer.cost_history, nb_epoch=1000)

# plot confusion matrix
cm = confusion_matrix(y_true=Y_test, y_pred=predictions)
plot_confusion_matrix(confusion_matrix=cm)