import numpy as np
import car_params as params

class MDPPlanner:
    def __init__(self):
        self.map = params.map
        self.policy = np.zeros_like(self.map) #Will be filled with a number deterining the direction of the arrow
    
    def createPolicy(self):
        idx = np.argwhere(self.map == 0)
        self.map[idx[0,:], idx[1,:]] = -2
        epsilon = 1

        # while diff < epsilon:
