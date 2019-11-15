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
        phi = unwrap(pose[2])
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
        theta = unwrap(np.arctan2(ds[1], ds[0]) - unwrap(pose[2]))

        z_hat = np.array([r, theta])

        #Jacobian
        H = np.array([[ds[0]/r, ds[1]/r], [-ds[1]/q, ds[0]/q]])

        return z_hat, H

    def update(self, z, pose):
        z_hat, H = self.getExpectedMeasurement(z, pose) 

        S = H @ self.Sigma @ H.T + self.Q   #Innovation Covariance
        K = self.Sigma @ H.T @ np.linalg.inv(S)

        innov = z - z_hat
        innov[1] = unwrap(innov[1])
        self.mu = self.mu + K @ (innov) 
        self.Sigma = (np.eye(2) - K @ H) @ self.Sigma #Should sigma decrease here on the initialization update

        return S, innov
