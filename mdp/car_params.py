import numpy as np
import scipy.io as sio

read_file = False
r_obs = -5000
r_goal = 1e6
r_walls = -100
r_else = -2
if read_file:
    data = sio.loadmat('mdp_map.mat')
    r = data['Mm']
    c = data['Nm']
    obs = data['obs'] * r_obs
    goal = data['goal'] * r_goal
    walls = data['walls'] * r_walls
    map = data['map']
    x0 = 28 #Column in matrix
    y0 = 20 #Row in matrix
else:
    r = 5
    c = 6
    obs = np.zeros((r,c))
    obs[2,2] = 1 * r_obs
    walls = np.array([[0, 1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1, 0]]) * r_walls
    goal = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0]]) * r_goal # The -1 is the weighted error zone
    map = walls + obs + goal
    x0 = 1 # Column in matrix
    y0 = 3 # row in matrix

p_forward = 0.8
p_left = 0.1
p_right = 0.1
gamma = 1.0
