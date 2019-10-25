import numpy as np
import car_params as params

def wrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi
    
class OccupancyGrid:
    def __init__(self):
        self.map = np.ones((params.l, params.w)) * 0.5 

        xi = np.arange(0, 100, 1) + 0.5
        yi = np.arange(0, 100, 1) + 0.5

        self.Xi, self.Yi = np.meshgrid(xi, yi) # center of mass of each grid in the map
        self.l0 = np.log(0.5/0.5) #should be 0 for our case
    
    def updateMap(self, pose, z):
        dx = self.Xi - pose[0]
        dy = self.Yi - pose[1]

        r = np.sqrt(dx**2 + dy**2) # matrix of range to each grid cell
        phi = np.arctan2(dy, dx) - pose[2] #matrix of bearing to each grid cell
        phi = wrap(phi)
        mat_thk = np.ones((params.l, params.w, params.thk.size)) * params.thk[None,None,:] #want mat_thk to be 11x100x100 not 100x100x11
        dphi = wrap(-(mat_thk - phi))  #Issues with this subtraction. Want to subtrach phi from every page in mat_thk
        k = np.argmin(dphi, axis=1) # matrix indicating which beam of range finder would hit this cell

        L = np.zeros_like(self.map) #matrix of Log probabilities
        dphi_k = wrap(phi - params.thk[k])
        L += (r > np.min(params.z_max, z[0,k] + params.alpha/2.0) or np.abs(dphi_k) > params.beta/2.0).astype(int) * self.l0
        L += (z[k,0] < params.z_max and np.abs(r - z[0,k]) < params.alpha/2.0).astype(int) * np.log(params.p_oc/(1-params.p_occ))
        L += (r <= z[0,k]).astype(int) * np.log(params.p_emp/(1 - params.p_emp))

        L_map = np.log(self.map/(1-self.map))
        L_map += L - self.l0
        self.map = 1 - 1/(1+np.exp(L_map))