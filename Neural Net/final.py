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
            a = sigmoid(np.dot(w, a) + b)
        return a

    def feedforward_new_params(self, a, new_biases, new_weights):
        for b, w in zip(new_biases, new_weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def CGD(self, training_data, epochs):
        previous_neg_grad_w = []
        previous_neg_grad_b = []

        for j in range(0, epochs):

            # Compute the gradient
            current_neg_grad_b = [np.zeros(b.shape) for b in self.biases]
            current_neg_grad_w = [np.zeros(w.shape) for w in self.weights]

            for x, y in training_data:
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                current_neg_grad_b = [nb - dnb for nb, dnb in zip(current_neg_grad_b, delta_nabla_b)]
                current_neg_grad_w = [nw - dnw for nw, dnw in zip(current_neg_grad_w, delta_nabla_w)]

            # Find the search direction
            if j == 0:
                direction_b = current_neg_grad_b
                direction_w = current_neg_grad_w
            else:
                num = sum([np.sum(b * b) for b in current_neg_grad_b])
                num += sum([np.sum(w * w) for w in current_neg_grad_w])
                den = sum([np.sum(b * b) for b in previous_neg_grad_b])
                den += sum([np.sum(w * w) for w in previous_neg_grad_w])
                m = num / den
                direction_b = [cur_b + (m * pre_b) for cur_b, pre_b in zip(current_neg_grad_b, previous_neg_grad_b)]
                direction_w = [cur_w + (m * pre_w) for cur_w, pre_w in zip(current_neg_grad_w, previous_neg_grad_w)]

            # Do line search
            # alpha = self.line_search(training_data, direction_b, direction_w)
            alpha = .1
            # Update weights and biases
            self.weights = [w + (alpha * nw) for w, nw in zip(self.weights, direction_w)]
            self.biases = [b + (alpha * nb) for b, nb in zip(self.biases, direction_b)]

            previous_neg_grad_b = current_neg_grad_b
            previous_neg_grad_w = current_neg_grad_w

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
            if abs(phi_alpha_i) <= -1 * self.c2 * phi_prime_0:
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
            alpha_i = (alpha_low + alpha_high) / 2
            phi_alplha_i = self.phi(training_data, alpha_i, direction_b, direction_w)

            if phi_alplha_i > phi_alpha_0 + (self.c1 + alpha_i * phi_prime_0) or \
                            phi_alplha_i >= self.phi(training_data, alpha_low, direction_b, direction_w):
                print('stuck 1')

                if phi_alplha_i >= self.phi(training_data, alpha_low, direction_b, direction_w):
                    print('mo fn bshit')
                if phi_alplha_i > phi_alpha_0 + (self.c1 + alpha_i * phi_prime_0):
                    print('Yay!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                alpha_high = alpha_i
            else:
                phi_prime_i = self.phi_derev(training_data, alpha_i, direction_b, direction_w)

                if abs(phi_prime_i) <= -1 * self.c2 * phi_prime_0:
                    return alpha_i
                if phi_prime_i * (alpha_high - alpha_low) >= 0:
                    print('failure at second wolfe condition')
                    alpha_high = alpha_low
                alpha_low = alpha_i

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
        delta = self.cost_derivative(activations[-1][:, 0], np.array([y])) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def backprop_new_params(self, x, y, new_biases, new_weights):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for w, b in zip(new_weights, new_biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
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
        """Return the vector of partial derivative"""
        return 2 * (output_activations - y)

    def cost(self, output_activations, y):
        """Returns the square error cost for a vector of outputs and labels."""
        return np.sum((np.array(output_activations) - np.array(y)) ** 2)

    def phi(self, training_data, alpha, direction_b, direction_w):
        output_activations = []
        Y = []
        w = [x + (alpha * d) for x, d in zip(self.weights, direction_w)]
        b = [x + (alpha * d) for x, d in zip(self.biases, direction_b)]

        for x, y in training_data:
            output_activations.append(self.feedforward_new_params(x, b, w))
            Y.append(y)

        return self.cost(output_activations, Y)

    def phi_derev(self, training_data, alpha, direction_b, direction_w):
        w = [x + (alpha * d) for x, d in zip(self.weights, direction_w)]
        b = [x + (alpha * d) for x, d in zip(self.biases, direction_b)]

        nabla_b = [np.zeros(b.shape) for b in b]
        nabla_w = [np.zeros(w.shape) for w in w]
        for x, y in training_data:
            delta_nabla_b, delta_nabla_w = self.backprop_new_params(x, y, b, w)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        retval = sum([np.sum(db * dirb) for db, dirb in zip(nabla_b, direction_b)])
        retval += sum([np.sum(dw * dirw) for dw, dirw in zip(nabla_w, direction_w)])

        return retval


# Miscellaneous functions


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def main():
    data = [(np.array([[1], [1]]), np.array([0])),
            (np.array([[1], [0]]), np.array([1])),
            (np.array([[0], [1]]), np.array([1])),
            (np.array([[0], [0]]), np.array([0]))]

    print('Creating Network.')
    net = Network([2, 2, 1])
    print('Training Network.')
    net.CGD(data, 5000)

    print('Testing Network')
    print(net.feedforward(np.array([[1], [1]])), ' is the output when given (1, 1)')
    print(net.feedforward(np.array([[0], [1]])), ' is the output when given (0, 1)')
    print(net.feedforward(np.array([[1], [0]])), ' is the output when given (1, 0)')
    print(net.feedforward(np.array([[0], [0]])), ' is the output when given (0, 0)')

main()
