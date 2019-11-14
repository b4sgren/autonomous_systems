import numpy as np

#velocity motion model noise params
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1

# Sensor noise params
sigma_r = 0.1 #m
sigma_theta = 0.05 #rad
fov = np.deg2rad(360/2.0) #Radians to each side

#number of particles
M = 100

#landmark locations
gen_lms = True
if gen_lms:
    num_lms = 8
else:
    num_lms = 3
if gen_lms:
    lms = np.random.uniform(low=-10.0, high=10.0, size=(2, num_lms))
else:
    lms = np.array([[6, -7, 6], [4, 8, -4]])

dt = 0.1
tf = 50.0
