import numpy as np

steps = 20 # Time horizon 
gamma = 1.0 #Currently not using this

# Probability of transitioning to state p(s'|s,u): T(u, s, s')
T = np.array([[0.2, 0.8], [0.8, 0.2]])

#Probability of observing measurement zi in state x p(z|s'): Z(s', z)
Z = np.array([[0.7, 0.3], [0.3, 0.7]])

#Reward for executing u in state s: R(s, u)
R = np.array([[-100.0, 100.0, -1.0], [100.0, -50.0, -1.0]]).T

step = .0001
P = np.array([[np.arange(0, 1, step)],[np.arange(1,0,-step)]]).squeeze()