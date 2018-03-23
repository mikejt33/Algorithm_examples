
import numpy as np
import matplotlib.pyplot as plt


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


def grad_least_squares(x, y, alpha, beta):
    a = np.exp(alpha + (beta * x))
    b = y * a + y - a
    c = (a + 1)**(-3)
    partial_in_alpha = np.sum(-2 * a * b * c)
    d = np.exp(-(alpha + beta * x))
    e = y - (1/(d + 1))
    f = (d + 1)**(-2)
    partial_in_beta = np.sum(-2 * x * d * e * f)
    return np.array([partial_in_alpha, partial_in_beta])


def grad_log_likelihood(x, y, alpha, beta):
    passed = [i for i in range(0, len(y)) if y[i] == 1]
    failed = [i for i in range(0, len(y)) if y[i] == 0]

    passed_alpha = 1 / (np.exp(alpha + beta * x[passed]) + 1)
    failed_alpha = (-1 * np.exp(alpha + beta * x[failed]))/(1 + np.exp(alpha + beta * x[failed]))

    passed_beta = x[passed] / (np.exp(alpha + beta * x[passed]) + 1)
    failed_beta = (-1 * x[failed] * np.exp(alpha + beta * x[failed]))/(1 + np.exp(alpha + beta * x[failed]))

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
        gamma_i = (gamma_low + gamma_high)/2
        phi_at_gamma_i = phi_short(gamma_i)

        if phi_at_gamma_i > phi_at_0 + (c_1 * gamma_i * phi_prime_at_0) or phi_at_gamma_i >= phi_short(gamma_low):
            gamma_high = gamma_i
        else:
            phi_prime_at_gamma_i = phi_prime_short(gamma_i)

            if abs(phi_prime_at_gamma_i) <= -1 * c_2 * phi_prime_at_0:
                return gamma_i
            if phi_prime_at_gamma_i*(gamma_high - gamma_low) >= 0:
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


x, y = load_data('data.txt')
grad_descent(x, y, log_likelihood_cost, grad_log_likelihood)

