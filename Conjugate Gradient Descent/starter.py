#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:13:45 2017

@author: Mike
"""


import numpy as np
import matplotlib.pyplot as plt
from wolfeConditionAlgorithms import alg_3_5

line_search_alphas = []
line_search_betas = []


def load_data(file_name):
    data = []
    with open(file_name, 'r') as data_file:
        for line in data_file:
            split_line = line.split(',')
            line_tuple = split_line[0].strip(), split_line[1].strip()
            data.append(line_tuple)
    data.pop(0)  # Remove the header
    for i in range(0, len(data)):
        data[i] = float(data[i][0]), int(data[i][1])
    x, y = zip(*data)
    return np.array(x), np.array(y)


def logistic(x, alpha, beta):
    return 1 / (1 + np.exp(-1 * (alpha + beta * x)))


def least_squares_cost(f_x, y):
    return np.sum((y - f_x)**2)


def log_likelihood_cost(f_x, y):
    passed = [i for i in range(0, len(y)) if y[i] == 1]
    failed = [i for i in range(0, len(y)) if y[i] == 0]

    return -1 * (np.sum(np.log(f_x[passed])) + np.sum(np.log(1 - f_x[failed])))


def grad_log_likelihood(x, y, beta):
    '''
    Takes x, y, beta
    '''
    passed = [i for i in range(0, len(y)) if y[i] == 1]
    failed = [i for i in range(0, len(y)) if y[i] == 0]

    passed_alpha = 1 / (np.exp(beta[0] + beta[1] * x[passed]) + 1)
    failed_alpha = (-1 * np.exp(beta[0] + beta[1] *
                                x[failed])) / (1 + np.exp(beta[0] + beta[1] * x[failed]))

    passed_beta = x[passed] / (np.exp(beta[0] + beta[1] * x[passed]) + 1)
    failed_beta = (-1 * x[failed] * np.exp(beta[0] + beta[1] *
                                           x[failed])) / (1 + np.exp(beta[0] + beta[1] * x[failed]))

    return -1 * np.array([np.sum(passed_alpha) + np.sum(failed_alpha), np.sum(passed_beta) + np.sum(failed_beta)])


def phi(x, y, cost, gamma, alpha, beta, direction):
    alpha, beta = np.array([alpha, beta]) + (gamma * direction)

    global line_search_alphas
    line_search_alphas.append(alpha)
    global line_search_betas
    line_search_betas.append(beta)

    prediction = logistic(x, alpha, beta)
    return cost(prediction, y)


def phi_prime(x, y, grad, gamma, alpha, beta, direction):
    alpha, beta = np.array([alpha, beta]) + (gamma * direction)
    grad_at_new_spot = grad(x, y, alpha, beta)
    return np.dot(grad_at_new_spot, direction)


def line_search(x, y, alpha, beta, direction, cost, grad):
    c_1 = .1
    c_2 = .2

    gamma_0 = 0
    gamma_max = 10**2
    gamma_1 = gamma_0 + .2
    gamma_i = gamma_1
    previous_gamma = 0

    def phi_short(gamma):
        return phi(x, y, cost, gamma, alpha, beta, direction)

    def phi_prime_short(gamma):
        return phi_prime(x, y, grad, gamma, alpha, beta, direction)

    def zoom_short(gamma_low, gamma_high):
        return zoom(x, y, cost, grad, gamma_low, gamma_high, alpha, beta, direction, c_1, c_2)

    phi_at_gamma_0 = phi_short(0)
    phi_prime_at_gamma_0 = phi_prime_short(0)

    while gamma_i < gamma_max:
        phi_at_gamma_i = phi_short(gamma_i)
        if phi_at_gamma_i > phi_at_gamma_0 + (c_1 * gamma_i * phi_prime_at_gamma_0):
            return zoom_short(previous_gamma, gamma_i)
        if gamma_1 != gamma_i and phi_at_gamma_i >= phi_at_previous_gamma:
            return zoom_short(previous_gamma, gamma_i)

        phi_prime_at_gamma_i = phi_prime_short(gamma_i)
        if abs(phi_prime_at_gamma_i) <= -1 * c_2 * phi_prime_at_gamma_0:
            return gamma_i

        if phi_prime_at_gamma_i >= 0:
            return zoom_short(gamma_i, previous_gamma)

        previous_gamma = gamma_i
        gamma_i += .2
        phi_at_previous_gamma = phi_at_gamma_i

    return gamma_max


def zoom(x, y, cost, grad, gamma_low, gamma_high, alpha, beta, direction, c_1, c_2):
    def phi_short(gamma):
        return phi(x, y, cost, gamma, alpha, beta, direction)

    def phi_prime_short(gamma):
        return phi_prime(x, y, grad, gamma, alpha, beta, direction)

    phi_at_0 = phi_short(0)
    phi_prime_at_0 = phi_prime_short(0)

    while True:
        gamma_i = (gamma_low + gamma_high) / 2
        phi_at_gamma_i = phi_short(gamma_i)

        if phi_at_gamma_i > phi_at_0 + (c_1 * gamma_i * phi_prime_at_0) or phi_at_gamma_i >= phi_short(gamma_low):
            gamma_high = gamma_i
        else:
            phi_prime_at_gamma_i = phi_prime_short(gamma_i)

            if abs(phi_prime_at_gamma_i) <= -1 * c_2 * phi_prime_at_0:
                return gamma_i
            if phi_prime_at_gamma_i * (gamma_high - gamma_low) >= 0:
                gamma_high = gamma_low
            gamma_low = gamma_i


def grad_descent(x, y, cost, grad):
    alpha = 0
    beta = 0
    alphas = [alpha]
    betas = [beta]
    c_i = cost(logistic(x, alpha, beta), y)
    prior = None

    while True:
        if prior is not None:
            direction = -1 * grad(x, y, alpha, beta) + .4 * prior
        else:
            direction = -1 * grad(x, y, alpha, beta)
        gamma = line_search(x, y, alpha, beta, direction, cost, grad)
        alpha, beta = np.array([alpha, beta]) + (gamma * direction)
        alphas.append(alpha)
        betas.append(beta)

        prior = direction
        c_previous = c_i
        c_i = cost(logistic(x, alpha, beta), y)
        if abs(c_i - c_previous) < .00001:
            break

    print(len(alphas))
    print(alphas[-1], betas[-1])
    global line_search_alphas
    global line_search_betas
    plt.plot(alphas, betas, 'bs')
    plt.plot(line_search_alphas, line_search_betas, '+')
    plt.show()


def conjugate_gradient_descent(X, Y, objective, gradient, x0):
    x_prev = x0
    # f_prev = objective(x0)
    grad_f_prev = gradient(X, Y, x0)

    p_prev = -grad_f_prev
    k = 0
    while np.linalg.norm(grad_f_prev) > 0.001:
        alpha_prev = alg_3_5(x_prev, p_prev, 1, objective, X, Y, 1e-4, 0.2)
        x_next = x_prev + alpha_prev * p_prev
        grad_f_next = gradient(x_next)
        beta = (np.dot(grad_f_next.transpose(), grad_f_next))/(np.dot(grad_f_prev.transpose(), grad_f_prev))
        p_next = -grad_f_next + np.dot(beta, p_prev)
        k += 1
        p_prev = p_next
        x_prev = x_next
        
    return x_next 

x, y = load_data('data.txt')
# grad_descent(x, y, log_likelihood_cost, grad_log_likelihood)
betas = conjugate_gradient_descent(x, y, log_likelihood_cost, grad_log_likelihood, np.array([0, 0]))
