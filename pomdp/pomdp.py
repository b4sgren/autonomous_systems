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
            perm2 = np.array(list(permutations([i for i in range(self.Y.shape[0])], 2)))
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
    
    def getOptimalAction(self, bel):
        policy = self.Y @ self.P
        ind = np.argmin(np.abs(self.P[0] - bel))
        alpha_vec_ind = np.argmax(policy[:,ind])

        if alpha_vec_ind == 0 or alpha_vec_ind == 1:
            return alpha_vec_ind
        else: #turning is the best policy
            return 2
    
    def getReward(self, u, true_state):
        if u == 2: #If turning
            return -1
        elif u == 0: #If going forward
            if true_state == 0:
                return -100
            else:
                return 100
        else: #If going backward
            if true_state == 0:
                return 100
            else:
                return -50
    
    def propagateAction(self, true_state, bel, u, u_succ):
        if u == 2:
            if u_succ < 0.8:
                true_state = 1 - true_state
            bel = .8 - 0.6 * bel
        return bel, true_state
    
    def getMeasurement(self, true_state, bel):
        num = np.random.uniform()
        if true_state == 0:
            if num < 0.7:
                z = 0
                z_prob = 0.7
            else:
                z = 1
                z_prob = 0.3
        else:
            if num < 0.7:
                z = 1
                z_prob = 0.7
            else:
                z = 0
                z_prob = 0.3
        
        if z == 0:
            bel = (0.7 * bel)/(0.4 * bel + 0.3)
        else:
            bel = (0.3 * bel)/(-0.4 * bel + 0.7)
        
        return z, num, bel

    
    def simulate(self, bel, true_state):
        u = 4
        while not u == 0 and not u == 1: #as long as it has not decided to go forward or backward
            print('P(x1): ', bel)
            u = self.getOptimalAction(bel)
            r = self.getReward(u, true_state)
            u_succ = np.random.uniform()
            bel, true_state = self.propagateAction(true_state, bel, u, u_succ)
            z, z_succ, bel = self.getMeasurement(true_state, bel)
            print('Action: ', u)
            print('Reward: ', r)
            print('Action Success: ', u_succ < 0.8)
            print('True State: ', true_state)
            print('Good Measurement: ', z_succ < 0.7)
            print('Measurement: ', z)
            print('Updated P(x1): ', bel)
            print('\n')