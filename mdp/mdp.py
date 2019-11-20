import numpy as np
import car_params as params

class MDPPlanner:
    def __init__(self):
        self.map = np.zeros_like(params.map) # Will need to fill in walls and obstacles at the end
        self.policy = np.ones_like(self.map) * np.nan #Will be filled with a number deterining the direction of the arrow
        self.gamma = params.gamma
        self.pf = params.p_forward
        self.pr = params.p_right
        self.pl = params.p_left
    
    def correcStaticCells(self):
        idx_walls = np.argwhere(params.walls < 0)
        idx_goal = np.argwhere(params.goal > 0)
        idx_pit = np.argwhere(params.goal < 0)
        idx_obs = np.argwhere(params.obs < 0)

        self.map[idx_walls[:,0], idx_walls[:,1]] = params.r_walls
        self.map[idx_goal[:,0], idx_goal[:,1]] = params.r_goal 
        if not params.read_file:
            self.map[idx_pit[:,0], idx_pit[:,1]] = - params.r_goal 
        self.map[idx_obs[:,0], idx_obs[:,1]] = params.r_obs
        self.map = np.flip(self.map, axis=0)
        self.policy = np.flip(self.policy, axis=0)
    
    def createPolicy(self):
        idx = np.argwhere(params.map == 0)
        # self.map[idx[:,0], idx[:,1]] = params.r_else
        r = np.zeros_like(self.map)
        r[idx[:,0], idx[:,1]] = params.r_else
        diff = 1e6
        epsilon = .001

        while diff > epsilon:
            temp_diff = []
            for i in range(1,params.r-1):
                for j in range(1,params.c-1):
                   if r[i,j] == -2:
                        V_north = self.pf * (params.walls[i+1, j] + params.obs[i+1, j] + params.goal[i+1, j] + self.map[i+1,j]) + \
                                  self.pr * (params.walls[i, j+1] + params.obs[i, j+1] + params.goal[i, j+1] + self.map[i, j+1]) + \
                                  self.pl * (params.walls[i, j-1] + params.obs[i, j-1] + params.goal[i, j-1] + self.map[i, j-1]) + r[i,j]
                        V_south = self.pf * (params.walls[i-1, j] + params.obs[i-1, j] + params.goal[i-1, j] + self.map[i-1, j]) + \
                                  self.pr * (params.walls[i, j+1] + params.obs[i, j+1] + params.goal[i, j+1] + self.map[i, j+1]) + \
                                  self.pl * (params.walls[i, j-1] + params.obs[i, j-1] + params.goal[i, j-1] + self.map[i, j-1]) + r[i,j]
                        V_east = self.pf * (params.walls[i, j+1] + params.obs[i, j+1] + params.goal[i, j+1] + self.map[i, j+1]) + \
                                  self.pr * (params.walls[i+1, j] + params.obs[i+1, j] + params.goal[i+1, j] + self.map[i+1, j]) + \
                                  self.pl * (params.walls[i-1, j] + params.obs[i-1, j] + params.goal[i-1, j] + self.map[i-1, j]) + r[i,j]
                        V_west = self.pf * (params.walls[i, j-1] + params.obs[i, j-1] + params.goal[i, j-1] + self.map[i, j-1]) + \
                                  self.pr * (params.walls[i-1, j] + params.obs[i-1, j] + params.goal[i-1, j] + self.map[i-1, j]) + \
                                  self.pl * (params.walls[i+1, j] + params.obs[i+1, j] + params.goal[i+1, j] + self.map[i+1, j]) + r[i,j]
                        V = [V_north, V_south, V_east, V_west]
                        max = np.max(V)
                        argmax = np.argmax(V)
                        self.policy[i, j] = argmax
                        temp_diff.append(np.abs(self.map[i,j] - max))
                        self.map[i,j] = max
            diff = np.sum(temp_diff)
        debug = 1

    def createPolicyVectorized(self):
        idx = np.argwhere(self.map == 0)
        self.map[idx[0,:], idx[1,:]] = params.r_else
        r = np.zeros_like(self.map)
        r[idx[0,:], idx[1,:]] = params.r_else
        diff = 1e6
        epsilon = 1
        idx_r0 = 1
        idx_r = params.r - 1
        idx_c0 = 1
        idx_c = params.c - 1

        while diff > epsilon:
            #Value for the traveling north reward policy. Need to add the cose of being in that state
            V_north = self.pf *self.map[idx_r0+1:idx_r+1, idx_c0:idx_c] + self.pr * self.map[idx_r0:idx_r, idx_c0+1:idx_c+1] + self.pl * self.map[idx_r0:idx_r, idx_c0-1:idx_c-1] 
            #Value for going south
            V_south = self.pf *self.map[idx_r0-1:idx_r-1, idx_c0:idx_c] + self.pr * self.map[idx_r0:idx_r, idx_c0+1:idx_c+1] + self.pl * self.map[idx_r0:idx_r, idx_c0-1:idx_c-1] 
            #Value fof going East
            V_east = self.pf * self.map[idx_r0:idx_r, idx_c0+1:idx_c+1] + self.pr * self.map[idx_r0-1:idx_r-1, idx_c0:idx_c] + self.pl * self.map[idx_r0+1:idx_r+1, idx_c0:idx_c]
            #Value for Going West
            V_west = self.pf * self.map[idx_r0:idx_r, idx_c0-1:idx_c-1] + self.pr * self.map[idx_r0-1:idx_r-1, idx_c0:idx_c] + self.pl * self.map[idx_r0+1:idx_r+1, idx_c0:idx_c]

            V = np.stack((V_north, V_south, V_east, V_west), axis=2)
            self.policy = np.argmax(V,axis=2) #use for updating the policy
            max = np.max(V, axis=2) #use for updating map
            debug = 1
            diff = np.sum(max - self.map[idx_r0:idx_r, idx_c0:idx_c])
            self.map[idx_r0:idx_r, idx_c0:idx_c] = max

            #reset obstacles
            idx_walls = np.argwhere(not params.walls == 0)
            self.map[idx_walls[0,:], idx_walls[1,:]] = -100
            idx_obs = np.argwhere(not params.obs == 0)
            self.map[idx_obs[0,:], idx_obs[1,:]] = -5000
            idx_goal = np.argwhere(not params.goal == 0)
            self.map[idx_goal[0,:], idx_goal[1,:]] = np.sign(params.goal[idx_goal[0,:], idx_goal[1,:]]) * 1e6
        debug = 1

           


