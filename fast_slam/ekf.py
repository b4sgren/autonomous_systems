import numpy as np
import car_params as params

def unwrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi

class EKF:
    def __init__(self, t):
        self.dt = t
        self.Sigma = np.eye(2) * 1e6 # No idea where LM is
        self.mu = np.zeros(2)
        self.Q = np.diag([params.sigma_r**2, params.sigma_theta**2])
        self.found = False

    def update(self, mu, z, v, w):
        for i in range(z.shape[1]):
            lm = params.lms[:,i]
            ds = lm - self.mu[0:2]

            r = np.sqrt(ds @ ds)
            phi = np.arctan2(ds[1], ds[0]) - self.mu[2] 
            phi = unwrap(phi)
            z_hat = np.array([r, phi])

            # I believe that H changes for this
            H = np.array([[-(lm[0] - self.mu[0])/r, -(lm[1] - self.mu[1])/r, 0],
                          [(lm[1] - self.mu[1])/r**2, -(lm[0] - self.mu[0])/r**2, -1]])

            S = H @ self.Sigma @ H.T + self.Q
            K = self.Sigma @ H.T @ np.linalg.inv(S)

            innov = z[:,i] - z_hat
            innov[1] = unwrap(innov[1])
            self.mu = self.mu + K @ (innov) 
            self.mu[2] = unwrap(self.mu[2])
            self.Sigma = (np.eye(3) - K @ H) @ self.Sigma

        self.mu[2] = unwrap(self.mu[2])
