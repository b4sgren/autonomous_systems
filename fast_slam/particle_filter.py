import numpy as np
import car_params as params
import scipy as sp
from ekf import EKF

def unwrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi

class ParticleFilter:
    def __init__(self, t):
        self.dt = t
        self.R = np.diag([params.sigma_r**2, params.sigma_theta**2])
        self.lm_filters = [[EKF(params.dt, i) for i in range(params.num_lms)] for i in range(params.M)]  #List of lists. Inside list is EKF for each LM. Outer list is each particle

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
        a1 = params.alpha1
        a2 = params.alpha2
        a3 = params.alpha3
        a4 = params.alpha4

        v = vc + np.sqrt(a1 * vc**2 + a2 * wc**2) * np.random.randn(params.M)
        w = wc + np.sqrt(a3 * vc**2 + a4 * wc**2) * np.random.randn(params.M)

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

    def getExpectedMeasurements(self, x): #Can I vectorize this function to get z for all LM's for every particle
        #Currently vectorized for all LM's for a single particle
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

    def measurement_update(self, Chi, w, z, ind):
        for i in range(params.M): # For each particle
            for j in range(ind.size): # For each LM
                lm = ind[j]
                if not self.lm_filters[i][lm].found:
                    self.lm_filters[i][lm].found = True
                    #Initialize filter
                    self.lm_filters[i][lm].initialize(z[:,j], Chi[:,i])
                    w[i] = 1.0 / len(w) # Is this right? Do I reinitialize the weight if a new LM is found?
        return Chi, w

    def lowVarianceSampling(self, Chi, w):  # May need to edit this to resample kalman filters also!!
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

        #Combating particle deprivation
        P = np.cov(Chi)
        uniq = np.unique(wght_idx).size
        if uniq/params.M < 0.1:
            Q = P/((params.M * uniq) ** (1/3)) #3 is size of the state space
            chi_pts_ret += Q @ np.random.randn(*chi_pts_ret.shape)

        return chi_pts_ret

    def update(self, Chi, wc, z, ind, v, w):
        Chi = self.propagateParticles(Chi, v, w)
        Chi, wc = self.measurement_update(Chi, wc, z, ind)
        # Chi = self.lowVarianceSampling(Chi, wc)

        return Chi
