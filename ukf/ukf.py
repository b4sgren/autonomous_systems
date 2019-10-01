import numpy as np
import car_params as params
import scipy as sp

def unwrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi


class UKF:
    def __init__(self, t):
        self.dt = t

        self.wm = np.zeros(2 * params.n + 1)
        self.wc = np.zeros_like(self.wm)

        self.wm[0] = params.lamb / (params.n + params.lamb)
        self.wc[0] = self.wm[0] + (1 - params.alpha**2 + params.beta)

        self.wm[1:] = 1.0 / (2 * (params.n + params.lamb))
        self.wc[1:] = 1.0 / (2 * (params.n + params.lamb))

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
        # temp[2] = unwrap(temp[2])
        return temp

    def propagateSigmaPts(self, Chi_x, Chi_u, v, w):
        theta = Chi_x[2,:]
        v = v + Chi_u[0,:]
        w = w + Chi_u[1,:]

        st = np.sin(theta)
        stw = np.sin(theta + w * self.dt)
        ct = np.cos(theta)
        ctw = np.cos(theta + w * self.dt)

        A = np.array([-v/w * st + v/w * stw,
                     v/w * ct - v/w * ctw,
                     w * self.dt])
        Chi_bar = Chi_x + A

        return Chi_bar

    def update(self, mu, Sigma, z, v, w):
        mu_a, Sig_a = self.augmentState(mu, Sigma, v, w)

        #Generate Sigma Points
        L = sp.linalg.cholesky(Sig_a, lower=True)
        Chi_a = self.generateSigmaPoints(mu_a, L)

        #propagation step
        Chi_x_bar = self.propagateSigmaPts(Chi_a[0:3,:], Chi_a[3:5,:], v, w) 
        mu_bar = np.sum(self.wm * Chi_x_bar, axis=1)
        # mu_bar[2] = unwrap(mu_bar[2])
        temp_x = Chi_x_bar - mu_bar.reshape((3,1))
        # temp_x[2,:] = unwrap(temp_x[2,:])
        Sigma_bar = np.sum(self.wc.reshape(2*params.n + 1, 1, 1) * np.einsum('ij, kj->jik', temp_x, temp_x), axis=0)

        #Measurement updates 
        for i in range(z.shape[1]):
        # for i in range(1): #Start with just the first landmark only
            Z_bar = self.generateObservationSigmas(Chi_x_bar, Chi_a[5:, :], params.lms[:,i])
            z_hat = np.sum(self.wm * Z_bar, axis=1)
            temp_z = Z_bar - z_hat.reshape((2, 1))
            # temp_z[1,:] = unwrap(temp_z[1,:])
        
            S = np.sum(self.wc.reshape(2 * params.n + 1, 1, 1) * np.einsum('ij, kj->jik', temp_z, temp_z), axis=0)
            Sigma_xz = np.sum(self.wc.reshape(2 * params.n+1, 1, 1) * np.einsum('ij, kj->jik', temp_x, temp_z), axis=0)
        
            #Calculate the kalman gain
            K = Sigma_xz @ np.linalg.inv(S)
            innov = z[:,i] - z_hat
            innov[1] = unwrap(innov[1])
            mu_bar = mu_bar + K @ (innov)
            Sigma_bar = Sigma_bar - K @ S @ K.T

            #redraw sigma points (if not the last lm) and then reset stuff
            if not i == (z.shape[1] - 1):
                mu_a, Sig_a = self.augmentState(mu_bar, Sigma_bar, v, w)
                L = sp.linalg.cholesky(Sig_a, lower=True)
                Chi_a = self.generateSigmaPoints(mu_a, L)
                Chi_x_bar = Chi_a[0:3,:]
                temp_x = Chi_x_bar - mu_bar.reshape((3,1))

        return mu_bar, Sigma_bar, K


    def augmentState(self, mu, Sigma, v, w):
        M = np.diag([params.alpha1 * v**2 + params.alpha2 * w**2, params.alpha3 * v**2 + params.alpha4 * w**2])
        Q = np.diag([params.sigma_r**2, params.sigma_theta**2])

        mu_a = np.concatenate((mu, np.zeros(4)))
        Sig_a = sp.linalg.block_diag(Sigma, M, Q)

        return mu_a, Sig_a

    def generateSigmaPoints(self, mu_a, L):
        gamma = np.sqrt(params.n + params.lamb)
        Chi_a = np.zeros((params.n, 2 * params.n + 1))

        Chi_a[:,0] = mu_a
        Chi_a[:,1:params.n+1] = mu_a.reshape((params.n,1)) + gamma * L
        Chi_a[:, params.n+1:] = mu_a.reshape((params.n,1)) - gamma * L

        return Chi_a

    def generateObservationSigmas(self, Chi_x, Chi_z, lm):
        xy = Chi_x[0:2,:]
        thetas = Chi_x[2,:]

        ds = lm.reshape((2,1)) - xy
        r = np.sqrt(np.sum(ds**2, axis=0))

        psi = np.arctan2(ds[1,:], ds[0,:]) - thetas
        psi = unwrap(psi)

        Z = np.vstack((r, psi)) + Chi_z
        return Z
