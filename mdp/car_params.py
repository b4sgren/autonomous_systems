import numpy as np
import scipy.io as sio

read_file = False
if read_file:
    data = sio.loadmat('mdp_map.mat')
    r = data['Mm']
    c = data['Nm']
    obs = data['obs'] * -5000
    goal = data['goal'] * 1e6
    walls = data['walls'] * -100
    map = data['map']
    xm = data['xm']
    ym = data['ym']
    x0 = 28 #Column in matrix
    y0 = 20 #Row in matrix
    # # idx = np.argwhere(map == 0)
    # map[idx[0,:], idx[1,:]] = -2
else:
    r = 5
    c = 6
    obs = np.zeros((r,c))
    obs[2,2] = 1 * -5000
    walls = np.array([[0, 1, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [0, 1, 1, 1, 1, 0]]) * -100
    goal = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, -1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]]) * 1e6 # The -1 is the weighted error zone
    map = walls + obs + goal
    xm = []
    ym = []
    for i in range(r):
        for j in range(c):
            xm.append(i)
            ym.append(j)
    xm = np.array(xm)
    ym = np.array(ym)
    x0 = 1 # Column in matrix
    y0 = 3 # row in matrix
    # idx = np.argwhere(map==0)
    # map[idx[0,:], idx[1,:]] = -2


p_forward = 0.8
p_left = 0.1
p_right = 0.1
gamma = 1.0
