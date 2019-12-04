import numpy as np
import car_params as params

class POMDPPlanner:
    def __init__(self):
        self.steps = params.steps
        self.T = params.T
        self.Z = params.Z
        self.R = params.R
        self.Y = {0:[[0.0, 0.0] for i in range(self.T.shape[1])]}
        self.gamma = params.gamma
    
    def prune(self):
        debug = 1
    
    def createPolicy(self):
        N = self.T.shape[1]
        for tau in range(self.steps):
            Yp = {}
            V = []
            K = len(self.Y.keys())
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
                            vp = np.sum(v * pz_x * pxp_ux, axis=1) # I think that this is right
                            Vj.append(vp)   # Not sure if this setup is the best way to do things
                        Vz.append(Vj)
                    Vu.append(Vz)
                V.append(Vu)
            V = np.array(V)
            for u in range(self.T.shape[0]):
                for k1 in range(K):
                    for k2 in range(K):
                        VV = []
                        for i in range(N):
                            r = self.R[i,u]
                            v1 = np.sum(V[0,u,:,k1, i]) # The 0 in the front is wrong. Need a new way to organize V
                            v2 = np.sum(V[0,u,:,k2,i])
                            vp = self.gamma * (r + v1 + v2)

                            VV.append(vp)
                            debug = 1
                        if not u in Yp.keys():
                            Yp[u] = [VV]
                        else:
                            if not VV in Yp[u]:
                                Yp[u].append(VV)
            #Do pruning here
            self.Y = Yp
            debug = 1