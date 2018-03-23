#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:47:18 2017

Code from: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
"""

"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.c1 = .001
        self.c2 = .01
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def feedforward_new_params(self, a, new_biases, new_weights):
        for b, w in zip(new_biases, new_weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def CGD(self, training_data, epochs):
        previous_grad_w = []
        previous_grad_b = []

        direction_w = []
        direction_b = []

        for j in range(0, epochs):

            # Compute the gradient
            current_grad_b = [np.zeros(b.shape) for b in self.biases]
            current_grad_w = [np.zeros(w.shape) for w in self.weights]
            for x, y in training_data:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                current_grad_b = [nb + dnb for nb, dnb in zip(current_grad_b, delta_nabla_b)]
                current_grad_w = [nw + dnw for nw, dnw in zip(current_grad_w, delta_nabla_w)]

            # Find the search direction
            if j == 0:
                direction_b = current_grad_b
                direction_w = current_grad_w
            else:
                num = sum([np.sum(b * b) for b in current_grad_b])
                num += sum([np.sum(w * w) for w in current_grad_w])
                den = sum([np.sum(b * b) for b in previous_grad_b])
                den += sum([np.sum(w * w)] for w in previous_grad_w)
                m = num/den
                direction_b = [cur_b + (m * pre_b) for cur_b, pre_b in zip(current_grad_b, previous_grad_b)]
                direction_w = [cur_w + (m * pre_w) for cur_w, pre_w in zip(current_grad_w, previous_grad_w)]

            # Do line search
            alpha = self.line_search(training_data, direction_b, direction_w)

            # Update weights and biases
            self.weights = [w + (alpha * nw) for w, nw in zip(self.weights, direction_w)]
            self.biases = [b + (alpha * nb) for b, nb in zip(self.biases, direction_b)]
            print("Epoch {0} complete".format(j))

    def line_search(self, training_data, direction_b, direction_w):
        alpha_0 = 0
        alpha_i = alpha_0 + .2
        alpha_1 = alpha_i
        previous_alpha = alpha_0

        phi_alpha_0 = self.phi(training_data, alpha_0, direction_b, direction_w)
        phi_prime_0 = self.phi_derev(training_data, alpha_0, direction_b, direction_w)

        while True:
            phi_alpha_i = self.phi(training_data, alpha_i, direction_b, direction_w)
            if phi_alpha_i > phi_alpha_0 + (self.c1 * alpha_i * phi_prime_0):
                return self.zoom(training_data, previous_alpha, alpha_i, direction_b, direction_w)
            if alpha_i != alpha_1 and phi_alpha_i >= phi_previous_alpha:
                return self.zoom(training_data, previous_alpha, alpha_i, direction_b, direction_w)

            phi_prime_i = self.phi_derev(training_data, alpha_i, direction_b, direction_w)
            if abs(phi_alpha_i) <= abs(self.c2 * phi_prime_0):
                return alpha_i

            if phi_prime_i >= 0:
                return self.zoom(training_data, alpha_i, previous_alpha, direction_b, direction_w)

            phi_previous_alpha = phi_alpha_i
            previous_alpha = alpha_i
            alpha_i += .2

        return -1

    def zoom(self, training_data, alpha_low, alpha_high, direction_b, direction_w):
        phi_alpha_0 = self.phi(training_data, 0, direction_b, direction_w)
        phi_prime_0 = self.phi_derev(training_data, 0, direction_b, direction_w)

        while True:
            print('Help! stuck in ZOOOOOM!')
            alpha_i = (alpha_low + alpha_high)/2
            phi_alplha_i = self.phi(training_data, alpha_i, direction_b, direction_w)

            if phi_alplha_i > phi_alpha_0 + (self.c1 + alpha_i * phi_prime_0) or \
                phi_alplha_i >= self.phi(training_data, alpha_low, direction_b, direction_w):
                alpha_high = alpha_i
            else:
                phi_prime_i = self.phi_derev(training_data, alpha_i, direction_b, direction_w)

                if abs(phi_prime_i) <= abs(self.c2 * phi_prime_0):
                    return alpha_i
                if phi_prime_i * (alpha_high - alpha_low) >= 0:
                    alpha_high = alpha_low
                alpha_low = alpha_i

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def backprop_new_params(self, x, y, new_weights, new_biases):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(new_weights, new_biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(new_weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivative if the output layer is a single node"""
        if y == 1:
            return [1 / (output_activations + 1)]
        elif y == 0:
            return [(-1 * output_activations) / (1 + output_activations)]

    def cost(self, output_activations, y):
        """Returns the log likelihood cost for a vector of outputs and labels."""
        passed = [i for i in range(0, len(y)) if y[i] == 1]
        failed = [i for i in range(0, len(y)) if y[i] == 0]

        return -1 * (np.sum(np.log(np.array(output_activations)[passed])) + np.sum(np.log(1 - np.array(output_activations)[failed])))

    def phi(self, training_data, alpha, direction_b, direction_w):
        output_activations = []
        Y = []
        w = [x + (alpha * d) for x, d in zip(self.weights, direction_w)]
        b = [x + (alpha * d) for x, d in zip(self.biases, direction_b)]

        for x, y in training_data:
            output_activations.append(self.feedforward_new_params(x, w, b))
            Y.append(y)

        return self.cost(output_activations, Y)

    def phi_derev(self, training_data, alpha, direction_b, direction_w):
        w = [x + (alpha * d) for x, d in zip(self.weights, direction_w)]
        b = [x + (alpha * d) for x, d in zip(self.biases, direction_b)]

        nabla_b = [np.zeros(b.shape) for b in b]
        nabla_w = [np.zeros(w.shape) for w in w]
        for x, y in training_data:
            delta_nabla_b, delta_nabla_w = self.backprop_new_params(x, y, w, b)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        retval = sum([np.sum(db * dirb) for db, dirb in zip(nabla_b, direction_b)])
        retval += sum([np.sum(dw * dirw) for dw, dirw in zip(nabla_w, direction_w)])

        return retval

# Miscellaneous functions


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def main():
    data = [(np.array([1, 1]), 0),
            (np.array([1, 0]), 1),
            (np.array([0, 1]), 1),
            (np.array([0, 0]), 0)]

    net = Network([2, 2, 1])
    net.CGD(data, 10)

    print(net.feedforward([1, 1]))

main()
