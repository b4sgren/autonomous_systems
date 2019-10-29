import numpy as np
import car_params as params

def unwrap(angle):
    """wrap an angle in rads, -pi <= theta < pi"""
    angle -= 2*np.pi * np.floor((angle + np.pi) * 0.5/np.pi)
    return angle

class EIF:
    def __init__(self, t):
        self.dt = t
        self.Omega = np.eye(3)
        self.covar = np.linalg.inv(self.Omega)

    def propagateState(self, state, v, w):
        theta = state[2]
        st = np.sin(theta)
        ct = np.cos(theta)

        A = np.array([v * ct, v * st, w]) * self.dt
        temp = state + A
        temp[2] = unwrap(temp[2])
        return temp

    def update(self, xi, z, v, w):
        G, V, M, Q = self.getJacobians(mu, v, w)

        mu_bar = self.propagateState(mu, v, w)
        Sigma_bar = G @ self.Sigma @ G.T + V @ M @ V.T

        for i in range(z.shape[1]):
            lm = params.lms[:,i]
            ds = lm - mu_bar[0:2]

            r = np.sqrt(ds @ ds)
            phi = np.arctan2(ds[1], ds[0]) - mu_bar[2] 
            phi = unwrap(phi)
            z_hat = np.array([r, phi])

            H = np.array([[-(lm[0] - mu_bar[0])/r, -(lm[1] - mu_bar[1])/r, 0],
                          [(lm[1] - mu_bar[1])/r**2, -(lm[0] - mu_bar[0])/r**2, -1]])

            S = H @ Sigma_bar @ H.T + Q
            K = Sigma_bar @ H.T @ np.linalg.inv(S)

            innov = z[:,i] - z_hat
            innov[1] = unwrap(innov[1])
            mu_bar = mu_bar + K @ (innov) 
            mu_bar[2] = unwrap(mu_bar[2])
            Sigma_bar = (np.eye(3) - K @ H) @ Sigma_bar

        self.Sigma = Sigma_bar
        mu_bar[2] = unwrap(mu_bar[2])
        return mu_bar, self.Sigma, K

    def getJacobians(self, mu, v, w):
        theta = mu[2]
        ct = np.cos(theta)
        st = np.sin(theta)

        #Jacobian of motion model wrt the states
        G = np.eye(3) #wrt the states is just I i believe
        G[0,2] = -v * st * self.dt
        G[0,3] = v * ct * self.dt

        #Jacobian of motion model wrt inputs
        V = np.array([[ct * self.dt, 0.0],
                      [st * self.dt, 0.0],
                      [0, self.dt]])

        #Process noise in motion model
        M = np.diag([params.sigma_v**2, params.sigma_w**2])

        #Measurement Noise
        Q = np.diag([params.sigma_r**2, params.sigma_theta**2])

        return G, V, M, Q
