import numpy as np
import car_params as params

def wrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi
    
class OccupancyGrid:
    def __init__(self):
        self.x_size = params.l + 2
        self.y_size = params.w + 2
        self.map = np.ones((self.x_size, self.y_size)) * 0.5 

        xi = np.arange(0, self.x_size, 1) #+ 1
        yi = np.arange(0, self.y_size, 1) #+ 1

        self.Xi, self.Yi = np.meshgrid(xi, yi) # center of mass of each grid in the map
        self.l0 = np.log(0.5/0.5) #should be 0 for our case
        self.l_occ = np.log(params.p_emp/(1 - params.p_emp)) #These are switched so empty space comes up white instead of black
        self.l_emp = np.log(params.p_occ/(1 - params.p_occ))
    
    def updateMap(self, pose, z):
        L = np.zeros_like(self.map)
        for i in range(self.x_size):
            for j in range(self.y_size):
                dx = self.Xi[i,j] - pose[0]
                dy = self.Yi[i,j] - pose[1]

                if i == 99 and j == 25 and pose[1] ==90 and pose[0] == 25:
                    debug = 1

                r = np.sqrt(dx**2 + dy**2)
                phi = wrap(np.arctan2(dy,dx) - pose[2])

                k = np.argmin(np.abs(phi - params.thk))

                if r > np.minimum(params.z_max, z[0,k] + params.alpha/2.0) or np.abs(phi - z[1,k]) > params.beta/2.0:
                    L[j,i] = self.l0 # should this be j,i or i,j
                elif z[0,k] < params.z_max and np.abs(r - z[0,k]) < params.alpha/2.0:
                    L[j,i] = self.l_occ
                elif r < z[0,k]:
                    L[j,i] = self.l_emp
        L_map = np.log(self.map/(1 - self.map))
        L_map += L - self.l0
        self.map = 1 - 1.0 / (1 + np.exp(L_map))
        debug = 1
        
    def updateMap2(self, pose, z):
        dx = self.Xi - pose[0]
        dy = self.Yi - pose[1]

        r = np.sqrt(dx**2 + dy**2) # matrix of range to each grid cell
        phi = np.arctan2(dy, dx) - pose[2] #matrix of bearing to each grid cell
        phi = wrap(phi)
        mat_thk = np.ones((params.thk.size, params.l, params.w)) * params.thk[:,None,None] 
        dphi = phi - mat_thk  
        k = np.argmin(dphi, axis=0) # matrix indicating which beam of range finder would hit this cell

        L = np.zeros_like(self.map) #matrix of Log probabilities
        for i in range(params.l):
            for j in range(params.w):
                t = k[i,j]
                if r[i,j] > np.minimum(params.z_max, z[0,t] + params.alpha/2.0) or np.abs(phi[i,j] - z[1,t]) < params.beta/2.0:
                    L[j,i] = self.l0 
                elif z[0,t] < params.z_max and np.abs(r[i,j] - z[0,t]) < params.alpha/2.0:
                    L[j,i] = self.l_occ
                else:
                    L[j,i] = self.l_emp
        
        # dphi_k = wrap(phi - z[1,k])
        # L1 = (r > np.minimum(params.z_max, z[0,k] + params.alpha/2.0))
        # L2 = (np.abs(dphi_k) > params.beta/2.0)
        # temp1 = np.logical_or(L1, L2).astype(int) * self.l0
        # L += np.logical_or(L1, L2).astype(int) * self.l0
        
        # L3 = (z[0,k] < params.z_max)
        # L4 = (np.abs(r - z[0,k]) < params.alpha/2.0)
        # temp2 = np.logical_and(L3, L4).astype(int) * np.log(params.p_occ/(1-params.p_occ))
        # L += np.logical_and(L3, L4).astype(int) * np.log(params.p_occ/(1-params.p_occ))

        # temp3 = (r <= z[0,k]).astype(int) * np.log(params.p_emp/(1 - params.p_emp))
        # L += (r <= z[0,k]).astype(int) * np.log(params.p_emp/(1 - params.p_emp))

        L_map = np.log(self.map/(1-self.map))
        L_map += L - self.l0
        self.map = 1 - 1/(1+np.exp(L_map)) 