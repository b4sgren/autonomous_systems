import numpy as np
import car_params as params
import scipy as sp

from IPython.core.debugger import Pdb

def unwrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi


class ParticleFilter:
    def __init__(self, t):
        self.dt = t
        self.R = np.diag([params.sigma_r**2, params.sigma_theta**2])

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
    
    def propagateParticles(self, Chi, vc, wc):
        #Add noise to velocities to separate the particles
        v = vc + np.sqrt(params.alpha1 * vc**2 + params.alpha2 * wc**2) * np.random.randn(params.M)
        w = wc + np.sqrt(params.alpha3 * vc**2 + params.alpha4 * wc**2) * np.random.randn(params.M)

        thetas = Chi[2,:]
        st = np.sin(thetas)
        stw = np.sin(thetas + w*self.dt)
        ct = np.cos(thetas)
        ctw = np.cos(thetas + w * self.dt)

        A = np.array([-v/w * st + v/w * stw,
                      v/w * ct - v/w * ctw,
                      w * self.dt])
        temp = Chi + A
        temp[2] = unwrap(temp[2])
        return temp

    def getExpectedMeasurements(self, x):
        xy = x[0:2]
        ds = xy.reshape((2,1)) - params.lms

        r = np.linalg.norm(ds, axis=0)
        psi = np.arctan2(ds[1,:], ds[0,:])
        psi = unwrap(psi)

        return np.vstack((r, psi))
    
    def getProbability(self, e, Sigma):
        p = 1.0
        D = np.linalg.det(2 * np.pi * Sigma) ** (-0.5)
        for i in range(e.shape[1]):
            p *= D * np.exp(-0.5 * e[:,i].T @ np.linalg.inv(Sigma) @ e[:,i])
        return p
    
    def measurement_update(self, Chi, z):
        Pdb().set_trace()
        for i in range(params.M): # See if I can vectorize this later
            #weight is the product of the probabilities for each measurement
            z_hat = self.getExpectedMeasurements(Chi[:,i])
            w = self.getProbability(z-z_hat, self.R)

    def update(self, mu, Sigma, Chi, z, v, w):
        Chi = self.propagateParticles(Chi, v, w)
        Chi, w = self.measurement_update(Chi, z)

        return mu, Sigma, Chi