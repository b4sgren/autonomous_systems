import numpy as np

N = 2 # Time horizon 

# Probability of transitioning to state p(s'|s,u): T(u, s, s')
T = np.zeros((3,2,2))
T[2,:,:] = np.array([[0.2, 0.8], [0.8, 0.2]])

#Probability of observing measurement zi in state x p(z|s'): Z(s', z)
Z = np.array([[0.7, 0.3], [0.3, 0.7]])

#Reward for executing u in state s: R(s, u)
R = np.array([[-100.0, 100.0, -1.0], [100.0, -50.0, -1.0]])