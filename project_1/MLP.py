import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class NeuralNetwork:

    @staticmethod
    def sigma(x):
        return 1 / (np.exp(-x) + 1)

    @staticmethod
    def sigma_deriv(x):
        sig = NeuralNetwork.sigma(x)
        deriv = sig * (1 - sig)
        return deriv

    @staticmethod
    def softmax(x):
        # softmax function ---------
        t = np.exp(x - np.max(x))
        return t / np.sum(t, axis=1, keepdims=True)

    @staticmethod
    def get_activation_function(name):
        if name == 'sigma':
            return NeuralNetwork.sigma
        elif name == 'linear':
            return lambda x: x
        elif name == 'softmax':
            return NeuralNetwork.softmax
        elif name == 'tanh':
            return np.tanh
        elif name == 'relu':
            return lambda x: np.maximum(0, x)

    @staticmethod
    def get_activation_derivative(name):
        if name == 'sigma':
            return NeuralNetwork.sigma_deriv
        elif name == 'linear':
            return lambda x: np.ones_like(x)
        elif name == 'softmax':
            # jacobian of softmax - unused -------
            def softmax_deriv(x):
                value = NeuralNetwork.softmax(x)
                SM = value.reshape((-1, 1))
                jac = np.diagflat(value) - np.dot(SM, SM.T)
                return jac

            return softmax_deriv
        elif name == 'tanh':
            return lambda x: 1 - np.tanh(x)**2
        elif name == 'relu':
            return lambda x: x > 0


    @staticmethod
    def get_loss_function(name):
        if name == 'mse':
            return lambda x_pred, x: np.linalg.norm(x_pred - x)
        elif name == 'crossentropy':
            return lambda x_pred, x: -np.sum(x*np.log(x_pred))

    @staticmethod
    def get_loss_derivative(name):
        # currently unused
        if name == 'mse':
            return lambda x_pred, x: (x_pred - x)
        elif name == 'crossentropy':
            return lambda x_pred, x: x*(-1/x_pred)

    class Layer:
        def __init__(self, input_width, layer_width, activation_function):
            # self.weights = np.random.uniform(0, 1, (input_width + 1, layer_width))
            self.weights = np.random.normal(0, 1, (input_width + 1, layer_width))
            self.activation_function = activation_function

        def predict(self, x):
            activation_function = NeuralNetwork.get_activation_function(self.activation_function)
            return activation_function(np.dot(x, self.weights))

        def feedforward_step(self, x):
            activation_function = NeuralNetwork.get_activation_function(self.activation_function)
            activation = np.dot(x, self.weights)
            response = activation_function(activation)
            return response, activation

    def __init__(self, input_width, output_width, activation_function='sigma', loss_function='mse', bias_exists=True):
        ###
        # create a new nn object. activation_function specifies activation used on hidden layers
        # loss_function affects loss printed to console
        ###
        self.input_width = input_width
        self.output_width = output_width
        self.layers = []
        self.activation_function = activation_function
        self.loss_function = NeuralNetwork.get_loss_function(loss_function)
        self.loss_derivative = NeuralNetwork.get_loss_derivative(loss_function)
        self.bias_exists = bias_exists

    def add_layer(self, layer_width):
        ###
        # add a hidden layer with specified number of neurons
        ###
        if len(self.layers) == 0:
            self.layers.append(NeuralNetwork.Layer(self.input_width, layer_width, self.activation_function))
        else:
            self.layers.append(
                NeuralNetwork.Layer(self.layers[-1].weights.shape[1], layer_width, self.activation_function))


    def predict(self, x):
        ###
        # predict responses on new data
        ###
        values = np.copy(x)
        for layer in self.layers:
            values = np.hstack((values, np.ones((values.shape[0], 1)) if self.bias_exists else np.zeros((values.shape[0], 1))))
            values = layer.predict(values)
        return values

    def create_output_layer(self, activation_function='linear'):
        ###
        # create output layer with specified activation function. Use after adding all hidden layers and before training
        ###
        if len(self.layers) == 0:
            self.layers.append(NeuralNetwork.Layer(self.input_width, self.output_width, activation_function))
        else:
            self.layers.append(
                NeuralNetwork.Layer(self.layers[-1].weights.shape[1], self.output_width, activation_function))

    def feedforward(self, x):
        response = np.copy(x)
        response = response.reshape(1, -1)
        response = np.hstack((response, np.ones((values.shape[0], 1)) if self.bias_exists else np.zeros((values.shape[0], 1))))
        response_s = [response]
        activation_s = []
        for i in range(len(self.layers) - 1):
            response, activation = self.layers[i].feedforward_step(response_s[i])
            activation_s.append(activation)
            response = response.reshape(1, -1)
            response = np.hstack((response, np.ones((values.shape[0], 1)) if self.bias_exists else np.zeros((values.shape[0], 1))))
            response_s.append(response)
        response, activation = self.layers[-1].feedforward_step(response_s[-1])
        activation_s.append(activation)
        response = response.reshape(1, -1)
        response_s.append(response)
        return (response_s, activation_s)

    def backpropagation(self, x, y, r_s, a_s):
        e_s = [None] * len(self.layers)

        ###
        # derivative of loss function in respect to final layer weights
        # assuming loss is crossentropy and output layer is softmax
        # formula is the same as in the case of linear outputs and mse
        ###
        e_s[-1] = a_s[-1] - y

        for i in reversed(range(1, len(e_s))):
            unbiased_weights = self.layers[i].weights[0:(self.layers[i].weights.shape[0] - 1), :]
            e_s[i-1] = NeuralNetwork.get_activation_derivative(self.layers[i-1].activation_function)(a_s[i-1])*(e_s[i].dot(unbiased_weights.T))
        gradient = [r_s[j].T.dot(e_s[j]) for j in range(0, len(self.layers))]
        return gradient

    def train(self, x, y, batch_size=10, epochs=100, lr=0.01, method='basic', method_param=0.0):
        ###
        # train network. method can be 'basic', 'momentum' or 'rmsprop'.
        # method_param specifies lambda in momentum or beta in rmsprop
        ###
        errors = []
        eps = 1e-8
        momentum = [np.zeros(layer.weights.shape) for layer in self.layers]
        for e in range(epochs):
            permutation = np.random.permutation(y.shape[0])
            x = x[permutation, :]
            y = y[permutation, :]
            i = 0
            while i < y.shape[0]:
                deltas = [np.zeros(layer.weights.shape) for layer in self.layers]
                x_batch = x[i:i + batch_size, :]
                y_batch = y[i:i + batch_size, :]
                i = i + batch_size
                for j in range(0, y_batch.shape[0]):
                    r_s, a_s = self.feedforward(x_batch[j, :])
                    gradient = self.backpropagation(x_batch[j, :], y_batch[j, :], r_s, a_s)
                    for k in range(0, len(deltas)):
                        deltas[k] = deltas[k] - gradient[k]

                if method == 'momentum':
                    momentum = [delta + method_param * mom for mom, delta in zip(momentum, deltas)]
                elif method == 'rmsprop':
                    momentum = [method_param * mom + (1 - method_param)*np.square(delta) for mom, delta in zip(momentum, deltas)]

                for j in range(0, len(deltas)):
                    if method == 'momentum':
                        self.layers[j].weights = self.layers[j].weights + lr*momentum[j]
                    elif method == 'rmsprop':
                        self.layers[j].weights = self.layers[j].weights + lr * (deltas[j] / (np.sqrt(momentum[j]) + eps))
                    else:
                        self.layers[j].weights = self.layers[j].weights + lr*deltas[j]

                error = self.loss_function(self.predict(x_batch), y_batch)
                print("loss on batch = {}".format(error))
                errors.append(error)
        return errors


# TESTS ------------------------------------------------------------------------
# set path to dataset folders
classification = "mio1/classification/"
regression = "mio1/regression/"

# side functions

def nnerrors(inputw, outputw, activation, X, y, layers1, layers2, layers3, output_activation='linear', loss='mse', epochs=4):
    nn1 = NeuralNetwork(inputw, outputw, activation, loss)
    for layer in layers1:
        nn1.add_layer(layer)
    nn1.create_output_layer(output_activation)
    nn2 = NeuralNetwork(inputw, outputw, activation, loss)
    for layer in layers2:
        nn2.add_layer(layer)
    nn2.create_output_layer(output_activation)
    nn3 = NeuralNetwork(inputw, outputw, activation, loss)
    for layer in layers3:
        nn3.add_layer(layer)
    nn3.create_output_layer(output_activation)

    errors1 = nn1.train(X, y, epochs=epochs, batch_size=50, lr=.02, method='rmsprop', method_param=0.2)
    errors2 = nn2.train(X, y, epochs=epochs, batch_size=50, lr=.02, method='rmsprop', method_param=0.2)
    errors3 = nn3.train(X, y, epochs=epochs, batch_size=50, lr=.02, method='rmsprop', method_param=0.2)

    return [errors1, errors2, errors3]


def make_plots(inputw, outputw, X, y, layers1, layers2, layers3, output_activation, loss, label1, label2, label3, suptitle, ylim=(0, 20), epochs=4):
    np.random.seed(42)
    sigma_error = nnerrors(inputw, outputw, 'sigma', X, y, layers1, layers2, layers3, output_activation, loss, epochs)
    tanh_error = nnerrors(inputw, outputw, 'tanh', X, y, layers1, layers2, layers3, output_activation, loss, epochs)
    relu_error = nnerrors(inputw, outputw, 'relu', X, y, layers1, layers2, layers3, output_activation, loss, epochs)

    x_err = range(len(sigma_error[0]))

    fig, axis = plt.subplots(1, 3, figsize=(10,5))
    plt.setp(axis, ylim=ylim)
    axis[0].plot(x_err, sigma_error[0], label=label1)
    axis[0].plot(x_err, sigma_error[1], label=label2)
    axis[0].plot(x_err, sigma_error[2], label=label3)
    axis[0].set_title('sigma')
    axis[0].set_ylabel(loss)
    axis[1].plot(x_err, tanh_error[0])
    axis[1].plot(x_err, tanh_error[1])
    axis[1].plot(x_err, tanh_error[2])
    axis[1].set_title('tanh')
    axis[2].plot(x_err, relu_error[0])
    axis[2].plot(x_err, relu_error[1])
    axis[2].plot(x_err, relu_error[2])
    axis[2].set_title('relu')
    fig.legend()
    fig.suptitle(suptitle)
    fig.show()

# loading datasets
f = open(regression + "steps-large-training.csv")
X, y = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(1,2), unpack=True)
f.close()
X = (X-np.mean(X))/np.std(X)
y = (y-np.mean(y))/np.std(y)
X_steps = X.reshape(-1,1)
y_steps = y.reshape(-1,1)

f = open(regression + "multimodal-large-training.csv")
X, y = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(1,2), unpack=True)
f.close()
X = (X-np.mean(X))/np.std(X)
y = (y-np.mean(y))/np.std(y)
X_multimodal = X.reshape(-1,1)
y_multimodal = y.reshape(-1,1)

f = open(classification + "rings3-regular-training.csv")
rings3 = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1, 2))
f.close()
X = rings3[:, 0:2]
y = rings3[:, 2]
y = y.reshape(-1, 1)
encoder_rings3 = OneHotEncoder(sparse=False)
y_rings3 = encoder_rings3.fit_transform(y)
X_rings3 = (X - np.mean(X)) / np.std(X)

f = open(classification + "rings5-regular-training.csv")
rings5 = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1, 2))
f.close()
X = rings5[:, 0:2]
y = rings5[:, 2]
y = y.reshape(-1, 1)
encoder_rings5 = OneHotEncoder(sparse=False)
y_rings5 = encoder_rings5.fit_transform(y)
X_rings5 = (X - np.mean(X)) / np.std(X)


# Linear activation ----------------------------------------------------------

# Neural network with only linear activations will give a linear output:
# Example on square simple dataset

f = open(regression + 'square-simple-training.csv')
X, y = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(1,2), unpack=True)
f.close()

X = (X-np.mean(X))/np.std(X)
y = (y-np.mean(y))/np.std(y)
X = X.reshape(-1,1)
y = y.reshape(-1,1)
nn = NeuralNetwork(1, 1, 'linear')

# Multiple layers
nn.add_layer(5)
nn.add_layer(5)

nn.create_output_layer()
nn.errors = nn.train(X, y, epochs=5, batch_size=32, lr=.0005, method='basic', method_param=0.2)
pred = nn.predict(X)

# Plot example

plt.scatter(X.flatten(), y.flatten(), label='real data')
plt.scatter(X.flatten(), pred.flatten(), label='network predictions')
plt.legend()
plt.suptitle('Neural network with linear activation')
plt.show()

# In next examples we will omit linear activation and focus on ReLU, tanh and sigmoid



# How Changing layer number affects training time for different activation functions---------------------------

# Steps-large
make_plots(1, 1, X_steps, y_steps, [10], [10, 10], [10, 10, 10], 'linear', 'mse', 'one layer', 'two layers', 'three layers', 'steps_large', (0, 10), epochs=2)

# multimodal-large
make_plots(1, 1, X_multimodal, y_multimodal, [10], [10, 10], [10, 10, 10], 'linear', 'mse', 'one layer', 'two layers', 'three layers', 'multimodal_large')

# rings3
make_plots(2, 3, X_rings3, y_rings3, [10], [10, 10], [10, 10, 10], 'softmax', 'crossentropy', 'one layer', 'two layers', 'three layers', 'rings 3', (40, 200), epochs=4)

# rings5
make_plots(2, 5, X_rings5, y_rings5, [10], [10, 10], [10, 10, 10], 'softmax', 'crossentropy', 'one layer', 'two layers', 'three layers', 'rings 5', (60, 120))



# 3 layers with different number of neurons---------------------------

# Steps-large
make_plots(1, 1, X_steps, y_steps, [10, 10, 10], [5, 10, 5], [10, 5, 10],
           'linear', 'mse', '10-10-10 neurons', '5-10-5 neurons', '10-5-10 neurons', 'steps_large', (0, 10), epochs=2)

# multimodal-large
make_plots(1, 1, X_multimodal, y_multimodal, [10], [10, 10], [10, 10, 10],
           'linear', 'mse', '10-10-10 neurons', '5-10-5 neurons', '10-5-10 neurons', 'multimodal_large', epochs=4)

# rings3
make_plots(2, 3, X_rings3, y_rings3, [10], [10, 10], [10, 10, 10],
           'softmax', 'crossentropy', '10-10-10 neurons', '5-10-5 neurons', '10-5-10 neurons', 'rings 3', (20, 100), epochs=10)

# rings5
make_plots(2, 5, X_rings5, y_rings5, [10], [10, 10], [10, 10, 10],
           'softmax', 'crossentropy', '10-10-10 neurons', '5-10-5 neurons', '10-5-10 neurons', 'rings 5', (50, 140), epochs=4)

# Conclusion:
# Relu seems to behave very unpredictably and most of the time performs worse than rest of the functions
# Tanh seems to reach local minimas faster than sigmoid, but its prediction error tends to vary more,
# probably due to tanh being steeper than sigmoid
# Adding more layers usually means slower learning, but tends to lead to stronger fit. Depending on the situation this
# may cause overfitting
# Changing numbers of neurons in layers has different effects depending on the activation function and dataset