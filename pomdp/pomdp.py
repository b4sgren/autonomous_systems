import numpy as np
import car_params as params
from itertools import permutations

class POMDPPlanner:
    def __init__(self):
        self.steps = params.steps
        self.T = params.T
        self.Z = params.Z
        self.R = params.R
        self.P = params.P
        self.Y = np.array([[0,0]])
        self.gamma = params.gamma
    
    def getMax(self):
        return np.max(self.Y @ self.P, axis = 0), self.P[0]
    
    def prune(self):
        ind = np.unique(np.argmax(self.Y @ self.P, axis=0))
        self.Y = self.Y[ind]
    
    def sense(self):
        Pz_1 = np.diag(self.Z[:,0])
        Pz_2 = np.diag(self.Z[:,1])

        Vz_1 = self.Y @ Pz_1
        Vz_2 = self.Y @ Pz_2

        perm1 = np.array([[i,i] for i in range(self.Y.shape[0])])
        if self.Y.shape[0] == 1:
            perm = perm1 
        else:
            perm2 = np.array(list(permutations([0, 1])))
            perm = np.array([*perm1, *perm2])

        self.Y = Vz_1[perm[:,0]] + Vz_2[perm[:,1]]
    
    def predict(self):
        self.Y = self.Y @ (self.T) 
        self.Y = self.Y - np.ones_like(self.Y)
        self.Y = np.array([*self.R, *self.Y])
        debug = 1
    
    def createPolicy(self):
        for tau in range(self.steps):
            self.sense()
            self.prune()
            self.predict()
            self.prune()
            