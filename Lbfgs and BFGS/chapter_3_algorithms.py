#!/usr/bin/python
import numpy as np
import sys
#import ipdb

'''
    Interpolation step of Algorithm 3.6 in Nocedal and Wright. Uses equation
    3.58 from the book.

    phi -- Function phi(alpha) defined in Algorithm 3.6
    phi_prime -- Derivative phi'(alpha)
    alpha_lo -- Lower bound for astar
    alpha_hi-- Upper bound for astar
    method -- Interpolation method to use. Must be in `methods` (default 'quadratic')
'''
def interpolate(phi, phi_prime, alpha_lo, alpha_hi, method='quadratic'):

    methods = ['quadratic', 'bisection']

    if method == 'quadratic':
        denominator = phi(alpha_hi) - phi(alpha_lo) - phi_prime(alpha_lo) * (alpha_hi - alpha_lo)

        if np.abs(denominator) < 1e-10:
            # numerical underflow issues, so avoid
            return interpolate(phi, phi_prime, alpha_lo, alpha_hi, 'bisection')

        alpha_j = alpha_lo + phi_prime(alpha_lo) * (alpha_hi - alpha_lo) * (alpha_hi - alpha_lo) / denominator / 2

        if alpha_j <= min([alpha_lo, alpha_hi]) or alpha_j >= max([alpha_lo, alpha_hi]):
            return interpolate(phi, phi_prime, alpha_lo, alpha_hi, 'bisection')
        
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
def zoom(phi, phi_prime, phi_0, phi_prime_0, alpha_lo, alpha_hi, c1=1e-4, c2=0.1):

    while True:

        alpha_j = interpolate(phi, phi_prime, alpha_lo, alpha_hi, 'quadratic')

        phi_j = phi(alpha_j)

        if (phi_j > phi_0 + c1 * alpha_j * phi_prime_0) or \
        (phi_j >= phi(alpha_lo)):
            alpha_hi = alpha_j

        else:
            phi_prime_j = phi_prime(alpha_j)
            if abs(phi_prime_j) <= abs(c2 * phi_prime_0):
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
def line_search(phi, phi_prime, c1=1e-4, c2=0.1, alpha_scale=4):

    # phi_alpha_prev is not actually used for first iteration, so don't bother
    # computing phi(alpha_prev)
    alphas = [0] # alpha_0 is set to 0

    # store these for repeated use in our loop
    phi_0 = phi(0)
    phi_prime_0 = phi_prime(0)

    if phi_prime_0 > 0:
        # This should not happen in general. However, for simple momentum
        # this may happen. So, just reverse direction.
        print("ERROR: phi_prime_0 > 0 in line search.")
        alphas.append(-1)
    elif phi_prime_0 == 0:
        print("WARNING: phi_prime == 0 ??")
        return 0
    else:
        # alpha_1 is set between 0 and alpha_max
        alphas.append(1)

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
            alpha_star = zoom(phi, phi_prime, phi_0, phi_prime_0, alpha_i_minus_1, alpha_i, c1, c2)
            return alpha_star

        # if our new alpha estimate alpha_i satisfies both the first and second
        # Wolfe conditions, then we're done so return it
        elif np.abs(phi_prime_i) <= -c2 * phi_prime_0:
            alpha_star = alpha_i
            return alpha_star

        # if alpha_i satisfied the first Wolfe condition but not the second,
        # and the slope at alpha_i is positive, then we're heading up away from
        # a local minimum, so backtrack with zoom
        elif phi_prime_i >= 0:
            alpha_star = zoom(phi, phi_prime, phi_0, phi_prime_0, alpha_i, alpha_i_minus_1, c1, c2)
            return alpha_star

        # if alpha_i satisfied the first Wolfe condition but not the second,
        # yet the slope at alpha_i is negative, then we can still decreate phi
        # by continuing forward, so move alpha_i forward by step_size
        else:
            # increment i by appending to our list, and set alpha_{i+1} in between alpha_i and alpha_max
            alphas.append(alpha_i * alpha_scale)
            i += 1


def main():
    pass


if __name__ == '__main__':
    main()
