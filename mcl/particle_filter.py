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
        a1 = params.alpha1 * 1
        a2 = params.alpha2 * 1
        a3 = params.alpha3 * 1
        a4 = params.alpha4 * 1
        a5 = params.alpha5 * 1
        a6 = params.alpha6 * 1

        v = vc + np.sqrt(a1 * vc**2 + a2 * wc**2) * np.random.randn(params.M)
        w = wc + np.sqrt(a3 * vc**2 + a4 * wc**2) * np.random.randn(params.M)
        gamma = np.sqrt(a5 * vc**2 + a6 * wc**2) * np.random.randn(params.M)

        thetas = Chi[2,:]
        st = np.sin(thetas)
        stw = np.sin(thetas + w*self.dt)
        ct = np.cos(thetas)
        ctw = np.cos(thetas + w * self.dt)

        A = np.array([-v/w * st + v/w * stw,
                      v/w * ct - v/w * ctw,
                      w * self.dt + gamma * self.dt])
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
        e[1] = unwrap(e[1])
        p = 1.0
        pr = 1.0/np.sqrt(2*np.pi*Sigma[0,0]) * np.exp(-0.5 * e[0,:]**2/Sigma[0,0])
        p_psi = 1.0/np.sqrt(2 * np.pi * Sigma[1,1]) * np.exp(-0.5 * e[1,:]**2/Sigma[1,1])
        p = np.prod(pr * p_psi)
        return p

    def measurement_update(self, Chi, z):
        w = np.zeros(Chi.shape[1])
        for i in range(params.M): # See if I can vectorize this later
            #weight is the product of the probabilities for each measurement
            z_hat = self.getExpectedMeasurements(Chi[:,i])
            w[i] = self.getProbability(z-z_hat, self.R)
        w = w/np.sum(w) #make sure they sum to one
        return Chi, w

    def lowVarianceSampling(self, Chi, w):
        num_pts = Chi.shape[1]
        num_pts_inv = 1 / num_pts

        start_comb = num_pts_inv * np.random.rand()
        wght_cumulative = np.cumsum(w)

        teeth = np.arange(num_pts)
        comb = start_comb + teeth * num_pts_inv

        diff_mat = wght_cumulative - comb[:,None]
        diff_truth = diff_mat > 0

        wght_idx = np.argmax(diff_truth, axis=1)

        chi_pts_ret = Chi[:,wght_idx]
        return chi_pts_ret

    def recoverMeanAndCovar(self, Chi, w):
        mu = np.mean(Chi, axis=1)
        temp_x = Chi - mu.reshape((3,1))
        temp_x[2] = unwrap(temp_x[2])
        Sigma = np.cov(temp_x)

        return mu, Sigma

    def update(self, mu, Sigma, Chi, z, v, w):
        Chi = self.propagateParticles(Chi, v, w)
        Chi, w = self.measurement_update(Chi, z)
        Chi = self.lowVarianceSampling(Chi, w) # I think I'm getting all my weights on too few particles
        mu, Sigma = self.recoverMeanAndCovar(Chi, w)

        return mu, Sigma, Chi
