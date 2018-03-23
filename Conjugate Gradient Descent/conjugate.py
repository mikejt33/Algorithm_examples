#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:25:41 2017

@author: Mike
"""

import numpy as np
import matplotlib.pyplot as plt

hours  = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
hours = np.c_[np.ones(hours.shape[0]), hours]
passed = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

evaluationCount = 0

def logisticFunction(x):
    return 1/(1+np.exp(-x))


'''
The negative of the log-likelihood objective function. We invert it so that
when we minimize it, we are maximizing the likelihood.
Y is a vector of observed 1/0 (pass/fail), X is our vector of predictor
observations (data), and Beta is a vector of parameters, one for each
predictor variable, plus beta0 for the intercept.
'''
def logLikelihoodObjectiveFunction(X, Y, Beta):
    P = logisticFunction(X.dot(Beta)) # fitted probabilities
    
    # try:
    LL_val = (Y.dot(np.log(P)) + (1 - Y).dot(np.log(1 - P))) # LL function
    # except:
    #     pdb.set_trace()

    # Handle NaNs which occur when we take the log of 0
    if np.isnan(LL_val):
        tmp1 = Y * np.log(P) + (1 - Y) * np.log(1 - P) # elementwise vector mult
        tmp2 = X.dot(Beta)
        tmp2 = tmp2[np.isnan(tmp1)]
        tmp1[np.isnan(tmp1)] = tmp2 - np.log(np.exp(tmp2 + 1))
        LL_val = np.sum(tmp1)

    return -1 * LL_val # negative of LL function

# def logLikelihoodObjectiveFunction(x,y,params):
#     f_x = logisticFunction(params[1]*x+params[0])

    # a = y*np.log(f_x)
    # b = (1-y)*np.log(1-f_x)
    # objectiveFunctionValue = -sum(a+b)
    # if np.isnan(objectiveFunctionValue): 
    #     tmp1 = y*np.log(f_x)+(1-y)*np.log(1-f_x)
    #     tmp2 = params[1]*x+params[0]
    #     tmp2 = tmp2[np.isnan(tmp1)]
    #     tmp1[np.isnan(tmp1)] = tmp2 - np.log(np.exp(tmp2+1))
    #     objectiveFunctionValue = -sum(tmp1)
    # objectiveFunctionGrad = -np.array([sum(y-f_x),sum((y-f_x)*x)])
    # return objectiveFunctionValue, objectiveFunctionGrad

# def squareErrorObjectiveFunction(x,y,params):
#     f_x = logisticFunction(params[1]*x+params[0])
#     error = y-f_x
#     objectiveFunctionValue = np.dot(error,error)
#     objectiveFunctionGrad = -2*np.array([np.dot(f_x*(1-f_x),error),np.dot(f_x*(1-f_x)*x,error)])
#     return objectiveFunctionValue, objectiveFunctionGrad


def objectiveFunc(fnc,x,y,params):
    evaluationCount += 1
    return fnc(x,y,params)


'''
This is the gradient of our LL function. We invert it so that it's the gradient
of the negative LL function.
The log-likelihood function can be brutally twisted around to get a closed-form
solution for the gradient: (1/N)*X^T %*% (Y - Y_hat). (using our definition of
X, with a column of 1's in the beginning, and where Y_hat is the predicted
probabilities).
This has a ridiculous proof, which I can't type here:
    https://math.stackexchange.com/a/477261
'''
def gradient_LL(X, Y, Beta):
    P = logisticFunction(X.dot(Beta)) # Fitted probabilities

    E = Y - P # Error

    return -1 * (X.T.dot(E)) # negative of closed form solution of gradient


def interpolate(phi, phi_prime, alpha_lo, alpha_hi, method='quadratic'):

    # I'm not sure about the interpolation part.

    methods = ['quadratic', 'bisection']
    if method == 'quadratic':
        alpha_j = -1 * (phi_prime(alpha_lo) * alpha_hi**2) / \
                (2 * (phi(alpha_hi) - phi(alpha_lo) - \
                      phi_prime(alpha_lo) * alpha_hi))
        return alpha_j
    elif method == 'bisection':
        return (alpha_lo + alpha_hi) / 2.0
    else:
        raise ValueError('Invalid method. Valid methods are {}'.format(methods))

    
'''
    Algorithm 3.6 in Nocedal and Wright.

    phi -- Function phi(alpha) defined in Algorithm 3.6
    phi_prime -- Derivative phi'(alpha)
    alpha_lo -- Lower bound for alpha_star
    alpha_hi -- Upper bound for alpha_star
    c1 -- Scaling factor for the first Wolfe condition (default 1e-4)
    c2 -- Scaling factor for the second Wolfe condition (default 0.9)
    Returns alpha_star, the acceptable alpha step size
'''
def zoom(phi, phi_prime, alpha_lo, alpha_hi, c1=1e-4, c2=0.9):

    phi_0 = phi(0)
    phi_prime_0 = phi_prime(0)

    while True:

        alpha_j = interpolate(phi, phi_prime, alpha_lo, alpha_hi, 'bisection')

        phi_j = phi(alpha_j)

        if (phi_j > phi_0 + c1 * alpha_j * phi_prime_0) or \
        (phi_j >= phi(alpha_lo)):
            alpha_hi = alpha_j

        else:
            phi_prime_j = phi_prime(alpha_j)
            if abs(phi_prime_j) <= -c2 * phi_prime_0:
                alpha_star = alpha_j
                return alpha_star
            elif phi_prime_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo

            alpha_lo = alpha_j


    return alpha_j


'''
    Algorithm 3.5 in Nocedal and Wright.

    phi -- Function phi(alpha) defined in Algorithm 3.5
    phi_prime -- Derivative phi'(alpha)
    c1 -- Scaling factor for the first Wolfe condition (default 1e-4)
    c2 -- Scaling factor for the second Wolfe condition (default 0.9)
    alpha_max -- Maximum step size allowed
    step_sz -- Step size to increment alpha by (default 1)

    Returns alpha_star, the acceptable step size which satisfies the strong
    Wolfe conditions.
'''
def line_search(phi, phi_prime, c1=1e-4, c2=0.9, step_sz=1):

    # phi_alpha_prev is not actually used for first iteration, so don't bother
    # computing phi(alpha_prev)
    alphas = [0] # alpha_0 is set to 0

    # alpha_1 is set between 0 and alpha_max
    alphas.append(0 + step_sz)

    # store these for repeated use in our loop
    phi_0 = phi(0)
    phi_prime_0 = phi_prime(0)

    # Set alpha_0 and phi_0 to start with
    alpha_i = 0
    phi_i = phi_0

    i = 1
    # During this loop, our invariant is that alphas[-1] is alpha_i, and
    # alphas[-2] is alpha_{i-1}
    while True:
        alpha_i_minus_1 = alpha_i
        alpha_i = alphas[-1]

        phi_i_minus_1 = phi_i
        phi_i = phi(alpha_i)

        phi_prime_i = phi_prime(alpha_i)

        # if our new alpha estimate alpha_i violates the first Wolfe condition,
        # then use zoom() to return an alpha between our prior alpha and
        # alpha_i
        if (phi_i > phi_0 + c1 * alpha_i * phi_prime_0) or \
        (phi_i >= phi_i_minus_1 and i > 1):
            alpha_star = zoom(phi, phi_prime, alpha_i_minus_1, alpha_i)
            return alpha_star

        # if our new alpha estimate alpha_i satisfies both the first and second
        # Wolfe conditions, then we're done so return it
        elif abs(phi_prime_i) <= -c2 * phi_prime_0:
            alpha_star = alpha_i
            return alpha_star

        # if alpha_i satisfied the first Wolfe condition but not the second,
        # and the slope at alpha_i is positive, then we're heading up away from
        # a local minimum, so backtrack with zoom
        elif phi_prime_i >= 0:
            alpha_star = zoom(phi, phi_prime, alpha_i, alpha_i_minus_1)
            return alpha_star

        # if alpha_i satisfied the first Wolfe condition but not the second,
        # yet the slope at alpha_i is negative, then we can still decreate phi
        # by continuing forward, so move alpha_i forward by step_size
        else:
            # increment i by appending to our list, and set alpha_{i+1} in between alpha_i and alpha_max
            alphas.append(alpha_i + step_sz)
            i += 1
   
    
def conjugate_gradient_descent(X, Y, objective, gradient, x0):
    x_prev = x0
    # f_prev = objective(x0)
    grad_f_prev = gradient(X, Y, x0)
    p_prev = -grad_f_prev
    k = 0
    x_outs = []
    while np.linalg.norm(grad_f_prev) > 0.001:
        x_outs.append(x_prev)
        # The phi function is a cross-section of our objective function
        # from our current Beta in the direction of p_k
        phi = lambda alpha : objective(X, Y, x_prev + alpha * p_prev)

        # The derivative of phi w.r.t. alpha is the gradient of f (LL)
        # evaluated at Beta + alpha * p_k, dotted with p_k
        phi_prime = lambda alpha : gradient(X, Y, x_prev + alpha *
                                                        p_prev).dot(p_prev)

        alpha_prev = line_search(phi, phi_prime, c2=0.1)
        x_next = x_prev + alpha_prev * p_prev
        grad_f_next = gradient(X, Y, x_next)
        beta = (np.dot(grad_f_next.transpose(), grad_f_next))/(np.dot(grad_f_prev.transpose(), grad_f_prev))
        if k % 2 == 0:
            beta = 0
        p_next = -grad_f_next + np.dot(beta, p_prev)
        k += 1
        p_prev = p_next
        x_prev = x_next
        grad_f_prev = grad_f_next

    return np.array(x_outs)

#objectiveFunction = squareErrorObjectiveFunction
objectiveFunction = logLikelihoodObjectiveFunction

betas = np.array([0,0])
cnt = 0
alpha = 1

def main():
   x_outs = conjugate_gradient_descent(hours, passed, objectiveFunction, gradient_LL, betas) 
   fig, ax = plt.subplots(1,1)
   ax.plot(x_outs[:,0], x_outs[:,1], '-o', lw=2)
   ax.text(x_outs[0,0]+0.1, x_outs[0,1]+0.1, "Start")
   ax.text(x_outs[-1,0]+0.1, x_outs[-1,1]+0.1, "End")
   ax.set_ylabel('$\\beta_1$')
   ax.set_xlabel('$\\beta_0$')
   ax.set_title('Fletcher/Reeves CGD with {} evaluations'.format(x_outs.shape[0]))
   ax.grid()
   fig.savefig("./fletch.png", bbox_inches="tight")
   plt.show()

main()
