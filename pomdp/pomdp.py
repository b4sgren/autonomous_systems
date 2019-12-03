import numpy as np
import car_params as params

class POMDPPlanner:
    def __init__(self):
        debug = 1
    
    def createPolicy(self):
        idx = np.argwhere(params.map == 0)
        # self.map[idx[:,0], idx[:,1]] = params.r_else
        r = np.zeros_like(self.map)
        r[idx[:,0], idx[:,1]] = params.r_else
        diff = 1e6
        if params.read_file:
            epsilon = 300
        else:
            epsilon = .001 

        while diff > epsilon:
            temp_diff = []
            for i in range(1,params.r-1):
                for j in range(1,params.c-1):
                   if r[i,j] == -2:
                        V_north = (self.pf * (params.walls[i+1, j] + params.obs[i+1, j] + params.goal[i+1, j] + self.map[i+1,j]) + \
                                  self.pr * (params.walls[i, j+1] + params.obs[i, j+1] + params.goal[i, j+1] + self.map[i, j+1]) + \
                                  self.pl * (params.walls[i, j-1] + params.obs[i, j-1] + params.goal[i, j-1] + self.map[i, j-1])) + r[i,j]
                        V_south = (self.pf * (params.walls[i-1, j] + params.obs[i-1, j] + params.goal[i-1, j] + self.map[i-1, j]) + \
                                  self.pr * (params.walls[i, j+1] + params.obs[i, j+1] + params.goal[i, j+1] + self.map[i, j+1]) + \
                                  self.pl * (params.walls[i, j-1] + params.obs[i, j-1] + params.goal[i, j-1] + self.map[i, j-1])) + r[i,j]
                        V_east = (self.pf * (params.walls[i, j+1] + params.obs[i, j+1] + params.goal[i, j+1] + self.map[i, j+1]) + \
                                  self.pr * (params.walls[i+1, j] + params.obs[i+1, j] + params.goal[i+1, j] + self.map[i+1, j]) + \
                                  self.pl * (params.walls[i-1, j] + params.obs[i-1, j] + params.goal[i-1, j] + self.map[i-1, j])) + r[i,j]
                        V_west = (self.pf * (params.walls[i, j-1] + params.obs[i, j-1] + params.goal[i, j-1] + self.map[i, j-1]) + \
                                  self.pr * (params.walls[i-1, j] + params.obs[i-1, j] + params.goal[i-1, j] + self.map[i-1, j]) + \
                                  self.pl * (params.walls[i+1, j] + params.obs[i+1, j] + params.goal[i+1, j] + self.map[i+1, j])) + r[i,j]
                        V = [V_north, V_south, V_east, V_west]
                        max = np.max(V)
                        argmax = np.argmax(V)
                        self.policy[i, j] = argmax
                        temp_diff.append(np.abs(self.map[i,j] - max * self.gamma))
                        self.map[i,j] = max * self.gamma
            diff = np.sum(temp_diff)
            self.iter += 1
        debug = 1

           


