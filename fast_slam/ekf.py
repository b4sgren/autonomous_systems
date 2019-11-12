import numpy as np
import car_params as params

def unwrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi

class EKF:
    def __init__(self, t, ind):
        self.dt = t
        self.Sigma = np.eye(2) * 1e6 # No idea where LM is
        self.mu = np.zeros(2)
        self.Q = np.diag([params.sigma_r**2, params.sigma_theta**2])
        self.found = False
        self.lm_pose = params.lms[:,ind]
    
    def initialize(self, z, pose):
        r = z.item(0)
        theta = z.item(1)
        
        # Initialize the pose
        phi = pose[2]
        temp = np.array([np.cos(theta + phi), np.sin(theta + phi)])
        self.mu = pose[:2] + r * temp

        #initialize covariance
        _, H = self.getExpectedMeasurement(z, pose)
        H_inv = np.linalg.inv(H)    # Invert beacuse we used the inverse measurement function
        self.Sigma = H_inv @ self.Q @ H_inv.T
    
    def getExpectedMeasurement(self, z, pose):
        ds = self.mu - pose[:2]
        r = np.sqrt(ds @ ds)
        q = r * r 
        theta = unwrap(np.arctan2(ds[1], ds[0]) - pose[2])

        z_hat = np.array([r, theta])

        #Jacobian
        H = np.array([[ds[0]/r, ds[1]/r], [-ds[1]/q, ds[0]/q]]) # Is this the right jacobian or is it the negative of this

        return z_hat, H

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
