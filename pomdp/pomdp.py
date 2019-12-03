import numpy as np
import car_params as params

class POMDPPlanner:
    def __init__(self):
        self.steps = params.steps
        self.T = params.T
        self.Z = params.Z
        self.R = params.R
        self.Y = {0:[np.zeros(2) for i in range(self.T.shape[1])]}
        self.gamma = params.gamma
    
    def prune(self):
        debug = 1
    
    def createPolicy(self):
        N = self.T.shape[1]
        for tau in range(self.steps):
            Yp = {}
            V = []
            for up, alphas in self.Y.items():
                Vu = []
                for u in range(self.T.shape[0]):
                    Vz = []
                    for z in range(self.Z.shape[1]):
                        Vj = []
                        for j in range(N):
                            v = np.array(alphas).T #Each column is an alpha vector
                            pz_x = self.Z[z,:]
                            pxp_ux = self.T[u,j,:]
                            vp = np.sum(v * pz_x * pxp_ux, axis=1) # Not sure this is exactly what I want to do
                            Vj.append(vp)
                        Vz.append(Vj)
                    Vu.append(Vz)
                V.append(Vu)
            V = np.array(V)
            for u in range(self.T.shape[0]):
                for k1 in range(self.Z.shape[0]):
                    for k2 in range(self.Z.shape[0]):
                        VV = []
                        for i in range(N):
                            r = self.R[i,u]
                            v = np.sum(V[0,u,:,k1, i]) # Not sure this is right
                            vp = self.gamma * (r + v)
                            VV.append(vp)
                            debug = 1