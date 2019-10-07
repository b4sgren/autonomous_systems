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
    
    def propagateParticles(self, particles, vc, wc):
        #Add noise to velocities to separate the particles
        v = vc + np.sqrt(params.alpha1 * vc**2 + params.alpha2 * wc**2) * np.random.randn(params.M)
        w = wc + np.sqrt(params.alpha3 * vc**2 + params.alpha4 * wc**2) * np.random.randn(params.M)

        thetas = particles[2,:]
        st = np.sin(thetas)
        stw = np.sin(thetas + w*self.dt)
        ct = np.cos(thetas)
        ctw = np.cos(thetas + w * self.dt)

        A = np.array([-v/w * st + v/w * stw,
                      v/w * ct - v/w * ctw,
                      w * self.dt])
        temp = particles + A
        temp[2] = unwrap(temp[2])
        return temp

    def update(self, mu, Sigma, particles, z, v, w):
        Pdb().set_trace()
        particles_bar = self.propagateParticles(particles, v, w)

        return mu, Sigma, particles