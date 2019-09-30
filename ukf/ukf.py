import numpy as np
import car_params as params
import scipy as sp 

from IPython.core.debugger import Pdb

def unwrap(phi):
    # Pdb().set_trace()
    # phi[phi >= np.pi] -= 2 * np.pi # not sure that this will work
    # phi[phi < -np.pi] += 2 * np.pi
    while phi >= np.pi:
        phi = phi - 2 * np.pi
    while phi < -np.pi:
        phi = phi + 2 * np.pi
    return phi


class UKF:
    def __init__(self, t):
        self.dt = t

    def propagateState(self, state, v, w):
        theta = state[2]
        st = np.sin(theta)
        stw = np.sin(theta + w * self.dt)
        ct = np.cos(theta)
        ctw = np.cos(theta + w * self.dt)

        A = np.array([-v/w * st + v/w * stw,
                    v/w * ct - v/w * ctw,
                    w * self.dt])
        temp = state + A
        temp[2] = unwrap(temp[2])
        return temp

    def update(self, mu, Sigma, z, v, w):
        mu_a, Sig_a = self.augmentState(mu, Sigma, v, w)

    def augmentState(self, mu, Sigma, v, w):
        M = np.diag([params.alpha1 * v**2 + params.alpha2 * w**2, params.alpha3 * v**2 + params.alpha4 * w**2])
        Q = np.diag([params.sigma_r**2, params.sigma_theta**2])

        mu_a = np.concatenate((mu, np.zeros(4)))
        Sig_a = sp.linalg.block_diag(Sigma, M, Q)

        Pdb().set_trace()

        return mu_a, Sig_a