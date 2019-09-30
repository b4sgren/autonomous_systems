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
    
    def propagateSigmaPts(self, Chi_x, Chi_u):
        theta = Chi_x[2,:]
        st = np.sin(theta)
        stw = np.sin(theta + Chi_u[1,:] * self.dt)
        ct = np.cos(theta)
        ctw = np.cos(theta + Chi_u[1,:] * self.dt)

        Pdb().set_trace()
        v = Chi_u[0,:]
        w = Chi_u[0,:]
        A = np.array([v/w * (-st + stw),
                     v/w * (ct - ctw),
                     w * self.dt])
        Chi_bar = Chi_x + A

        return Chi_bar

    def update(self, mu, Sigma, z, v, w):
        mu_a, Sig_a = self.augmentState(mu, Sigma, v, w)

        L = sp.linalg.cholesky(Sig_a, lower=True)
        Chi_a = self.generateSigmaPoints(mu_a, L)

        Chi_x_bar = self.propagateSigmaPts(Chi_a[0:3,:], Chi_a[3:5,:])

    def augmentState(self, mu, Sigma, v, w):
        M = np.diag([params.alpha1 * v**2 + params.alpha2 * w**2, params.alpha3 * v**2 + params.alpha4 * w**2])
        Q = np.diag([params.sigma_r**2, params.sigma_theta**2])

        mu_a = np.concatenate((mu, np.zeros(4)))
        Sig_a = sp.linalg.block_diag(Sigma, M, Q)

        return mu_a, Sig_a

    def generateSigmaPoints(self, mu_a, L):
        gamma = np.sqrt(params.n + params.lamb) # Is this supposed to be a vector?
        Chi_a = np.zeros((params.n, 2 * params.n + 1))

        Chi_a[:,0] = mu_a
        Chi_a[:,1:params.n+1] = mu_a + gamma * L
        Chi_a[:, params.n+1:] = mu_a - gamma * L

        return Chi_a