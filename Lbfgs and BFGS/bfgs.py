#usr/bin/python
import numpy as np

from collections import deque

# Using globals rather than OOP
m = 13
d = deque(maxlen = m)


'''
Implements BFGS algorithm

H_k: Numpy matrix, previous hessian inverse
s_k: Numpy vector, x_{k+1} - x_k
y_k: Numpy vector, grad_{k+1} - grad_k
'''
def update_inverse_hessian(H_k, s_k, y_k):

    rho_k = 1.0 / y_k.dot(s_k)

    n = s_k.size

    I = np.identity(n)

    V_k = I - (rho_k * np.outer(y_k, s_k))

    H_kp1 = V_k.T.dot(H_k).dot(V_k) + rho_k * np.outer(s_k, s_k)

    return H_kp1


''' Implements recursive LBFGS algorithm for computing H_k * grad_k '''
def algorithm7_4(grad_k,hk_0):
    q = grad_k

    alphas = [] # store alpha_i's

    for rho, s, y in list(reversed(d)):
        #rho is a scalar, s, y, and q are vectors

        alpha = rho * s.dot(q) # alpha is a scalar
        alphas.insert(0,alpha) # prepend

        q = q - alpha * y

    r = hk_0.dot(q)

    for t, alpha in zip(d,alphas):

        rho, s, y = t

        beta = rho * y.dot(r)

        r += s * (alpha - beta)

    return r


''' Save (s_k, y_k) pair to circular buffer for LBFGS.  '''
def add_to_buffer(s_k, y_k):

    rho_k = 1.0 / y_k.dot(s_k)
    d.append((rho_k, s_k, y_k))

