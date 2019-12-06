import numpy as np
import car_params as params

class POMDPPlanner:
    def __init__(self):
        self.steps = params.steps
        self.T = params.T
        self.Z = params.Z
        self.R = params.R
        self.P = params.P
        self.Y = np.array([0,0])
        self.gamma = params.gamma
    
    def prune(self):
        debug = 1
    
    def createPolicy(self):
        for tau in range(self.steps):
            debug = 1