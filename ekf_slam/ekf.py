import numpy as np
import car_params as params
import scipy as sp

def unwrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi

class EKF:
    def __init__(self, t):
        self.dt = t
        if params.gen_lms:
            self.num_lms = params.num_lms
        else:
            self.num_lms = 3

        self.Sigma = np.diag(np.ones(3 + 2 * self.num_lms)* 1e5) #Would use inf but causes issues. Using a really big number instead
        self.Sigma[0:3, 0:3] = np.eye(3) * 0 
        self.mu = np.zeros(3 + 2 * self.num_lms) 

        self.F = np.eye(3, 3 + 2 * self.num_lms)

        self.lms_found = {i: False for i in range(self.num_lms)}

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

    def update(self, z, lm_ind, v, w):
        G, V, M, Q = self.getJacobians(self.mu, v, w)
        R = V @ M @ V.T

        mu_bar = self.propagateState(self.mu[:3], v, w)
        self.mu[:3] = mu_bar
        # Gt = np.eye(3 + 2 * self.num_lms) + self.F.T @ G @ self.F  #Scipy.blockdiag w/ G and I. Add identity to G in getJacobians function
        Gt = sp.linalg.block_diag(G, np.eye(2 * self.num_lms))
        Rt = sp.linalg.block_diag(R, np.zeros((2 * self.num_lms, 2 * self.num_lms)))
        self.Sigma = Gt @ self.Sigma @ Gt.T + Rt # self.F.T @ R @ self.F #Scipy.blockdiag w/ R and I

        self.measurementUpdate(z, lm_ind, Q)

    def measurementUpdate(self, z, lm_ind, Q):
       for i in range(lm_ind.size):  # This will need to be modified when FOV is introduced
            lm = lm_ind.item(i)
            if not self.lms_found[lm]:
                self.lms_found[lm] = True
                theta = self.mu[2]
                phi = z[1, i]
                D = np.array([np.cos(phi + theta), np.sin(phi + theta)]) * z[0,i]
                self.mu[3 + lm * 2: 5 + lm*2] = self.mu[:2] + D 
            #Get expected measurement
            lm_pos = self.mu[3 + lm*2: 5 + lm*2]
            ds = lm_pos - self.mu[0:2]
            r = np.sqrt(ds @ ds)
            theta = unwrap(np.arctan2(ds[1], ds[0]) - self.mu[2])
            z_hat = np.array([r, theta])

            F = np.zeros((5, 2 * self.num_lms + 3))
            F[0:3, 0:3] = np.eye(3)
            F[3:, 2*lm+3:2*lm+5] = np.eye(2)

            tempH = np.array([[-r * ds[0], -r * ds[1], 0, r * ds[0], r * ds[1]],
                            [ds[1], -ds[0], -r**2, -ds[1], ds[0]]])
            H = 1/(r**2) * tempH @ F # This operation can be sped up by finding where the values in tempH/(r**2) go in H. H is a 2x2*N+3
            
            K = self.Sigma @ H.T @ np.linalg.inv(H @ self.Sigma @ H.T + Q)

            innov = z[:,i] - z_hat
            innov[1] = unwrap(innov[1])
            self.mu = self.mu + K @ (innov)
            self.mu[2] = unwrap(self.mu[2])
            self.Sigma = (np.eye(3 + 2 * self.num_lms) - K @ H) @ self.Sigma

    def getJacobians(self, mu, v, w):
        theta = mu[2]
        ct = np.cos(theta)
        st = np.sin(theta)
        cwt = np.cos(theta + w * self.dt)
        swt = np.sin(theta + w * self.dt)

        #Jacobian of motion model wrt the states
        # G = np.zeros((3,3))
        G = np.eye(3)
        G[0,2] = -v/w * ct + v/w * cwt
        G[1,2] = -v/w * st + v/w * swt

        #Jacobian of motion model wrt inputs
        V = np.array([[(-st + swt)/w, v * (st - swt)/w**2 + v * cwt * self.dt/w],
                      [(ct - cwt)/w, -v * (ct - cwt)/w**2 + v * swt * self.dt/w],
                      [0, self.dt]])

        #Process noise in motion model
        M = np.diag([params.alpha1 * v**2 + params.alpha2 * w**2,
                     params.alpha3 * v**2 + params.alpha4 * w**2])

        #Measurement Noise
        Q = np.diag([params.sigma_r**2, params.sigma_theta**2])

        return G, V, M, Q
