import numpy as np
import pdb

def phi_and_deriv(x, alpha, p, fnc, data_x, data_y):
    # pdb.set_trace()
    params = x + alpha * p
    obj, grad = fnc(data_x, data_y, params)
    return obj, np.dot(grad, p)


def alg_3_6(alphaL, alphaH, phiL, phiH, phiDL, phiDH, phi0, phiD0, x, p, objectiveFunction, data_x, data_y, c1, c2):
    while True:
        # Assume a parabola of the form (phiL+phiDL*(alpha-alphaL)+C*(alpha-alphaL)^2)
        # If we want this formula to equal phiH when alpha=alphaH, then C=(phiH-phiL-phiDL*(alpha-alphaL))/(alpha-alphaL)^2
        # Finally, if we are looking for alpha where the derivative of this formula would equal 0, then alpha = alphaL + phiDL/(2*C)
        denominator = phiH - phiL - phiDL * (alphaH - alphaL)
        if np.abs(denominator) < 1e-10:
            # may have numerical underflow issues, so avoid
            return (alphaL + alphaH) / 2
        alpha = alphaL + phiDL * (alphaH - alphaL) * \
            (alphaH - alphaL) / denominator / 2
        if alpha <= min([alphaL, alphaH]) or alpha >= max([alphaL, alphaH]):
            alpha = (alphaL + alphaH) / 2
        phi, phiD = phi_and_deriv(
            x, alpha, p, objectiveFunction, data_x, data_y)
        if phi > phi0 + c1 * phiD0 or phi >= phiL:
            alphaH = alpha
            phiH = phi
            # phiDH=phiD   #not using
        else:
            if np.abs(phiD) <= -c2 * phiD0:
                return alpha
            if phiD * (alphaH - alphaL) >= 0:
                alphaH = alphaL
                phiD = phi
                # phiDH=phiDL   #not using
            alphaL = alpha
            phiL = phi
            phiDL = phiD


def alg_3_5(x, p, alpha0, objectiveFunction, data_x, data_y, c1, c2):
    alpha = alpha0
    alpha_prev = 0
    i = 1
    phi0, phiD0 = phi_and_deriv(x, 0, p, objectiveFunction, data_x, data_y)
    if phiD0 > 0:  # This should not happen for standard methods we are using
        phiD0 = -phiD0  # However, for simple momentum, it may happen
        p = -p  # So, just reverse direction
    elif phiD0 == 0:
        return 0
    phi_prev = phi0
    phiD_prev = phiD0

    while True:
        phi, phiD = phi_and_deriv(
            x, alpha, p, objectiveFunction, data_x, data_y)
        if (phi > phi0 + c1 * alpha * phiD0) or (i > 1 and phi > phi_prev):
            return alg_3_6(alpha_prev, alpha, phi_prev, phi, phiD_prev, phiD, phi0, phiD0, x, p, objectiveFunction, data_x, data_y, c1, c2)
        if np.abs(phiD) <= -c2 * phiD0:
            return alpha
        if phiD >= 0:
            return alg_3_6(alpha, alpha_prev, phi, phi_prev, phiD, phiD_prev, phi0, phiD0, x, p, objectiveFunction, data_x, data_y, c1, c2)

        alpha_prev = alpha
        phi_prev = phi
        phiD_prev = phiD

        alpha = 4 * alpha
        i = i + 1
