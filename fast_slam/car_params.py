import numpy as np

#velocity motion model noise params
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1

# Sensor noise params
sigma_r = 0.1 #m
sigma_theta = 0.05 #rad

#number of particles
M = 1000

#landmark locations
gen_lms = False
num_lms = 8
if gen_lms:
    lms = np.random.uniform(low=-10.0, high=10.0, size=(2, num_lms))
else:
    lms = np.array([[6, -7, 6], [4, 8, -4]])

dt = 0.1
tf = 50.0
