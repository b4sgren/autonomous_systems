import numpy as np

class EKF:
    def __init__(self, t):
        self.mu = np.array([0.0, 0.0, np.pi/2.0])
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
        return state + A
