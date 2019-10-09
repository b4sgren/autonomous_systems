import numpy as np
import car_params as params
import scipy as sp

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
        ds = params.lms - xy.reshape((2,1))

        r = np.linalg.norm(ds, axis=0)
        psi = np.arctan2(ds[1,:], ds[0,:]) - x[2]
        psi = unwrap(psi)

        return np.vstack((r, psi))

    def getProbability(self, e, Sigma):
        p = 1.0
        pr = 1.0/np.sqrt(2*np.pi*Sigma[0,0]) * np.exp(-0.5 * e[0,:]**2/Sigma[0,0])
        p_psi = 1.0/np.sqrt(2 * np.pi * Sigma[1,1]) * np.exp(-0.5 * e[1,:]**2/Sigma[1,1])
        p = np.prod(pr * p_psi)
        if p < .0001: # This is to keep the probability from going to 0 on all the particles. If they are all 0 will be a uniform distribution
            p = .0001
        return p

    def measurement_update(self, Chi, z):
        Chi_bar = np.zeros_like(Chi)
        w = np.zeros(Chi_bar.shape[1])
        for i in range(params.M): # See if I can vectorize this later
            #weight is the product of the probabilities for each measurement
            z_hat = self.getExpectedMeasurements(Chi[:,i])
            w[i] = self.getProbability(z-z_hat, self.R)
            Chi_bar[:,i] = Chi[:,i] # Is this correct? Should I just return Chi?
        w = w/np.sum(w) #make sure they sum to one
        return Chi_bar, w

    def lowVarianceSampling(self, Chi, w):
        Chi_bar = np.zeros_like(Chi)
        M = params.M
        r = np.random.uniform(0, 1.0/M)
        c = w.item(0)
        i = 0

        for m in range(M): #Can I vectorize this?
            U = r + (m) * 1.0/M
            while U > c:
                i += 1
                c += w.item(i)
            Chi_bar[:,m] = Chi[:,i]
        return Chi_bar

    def recoverMeanAndCovar(self, Chi, w):
        mu = np.mean(Chi, axis=1)
        temp_x = Chi - mu.reshape((3,1))
        Sigma = np.cov(temp_x)

        return mu, Sigma

    def update(self, mu, Sigma, Chi, z, v, w):
        Chi = self.propagateParticles(Chi, v, w)
        Chi, w = self.measurement_update(Chi, z)
        Chi = self.lowVarianceSampling(Chi, w) # I think I'm getting all my weights on too few particles
        mu, Sigma = self.recoverMeanAndCovar(Chi, w)

        return mu, Sigma, Chi
