import numpy as np

#velocity motion model noise params
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1

# Sensor noise params
sigma_r = 0.1 #m
sigma_theta = 0.05 #rad
fov = np.deg2rad(180/2.0) #Radians to each side

#number of particles
M = 100

#landmark locations
gen_lms = True
if gen_lms:
    num_lms = 12 # per quadrant
    n = int(num_lms/4)
else:
    num_lms = 3
if gen_lms:
    lms1x = np.random.uniform(low=0.0, high=10.0, size=(n))
    lms2x = np.random.uniform(low=-10.0, high=0.0, size=(n))
    lms3x = np.random.uniform(low=0.0, high=10.0, size=(n))
    lms4x = np.random.uniform(low=-10.0, high=0.0, size=(n))
    lms1y = np.random.uniform(low=0.0, high=10.0, size=(n))
    lms2y = np.random.uniform(low=0.0, high=10.0, size=(n))
    lms3y = np.random.uniform(low=-10.0, high=0.0, size=(n))
    lms4y = np.random.uniform(low=-10.0, high=0.0, size=(n))
    lmsx = np.hstack((lms1x, lms2x, lms3x, lms4x))
    lmsy = np.hstack((lms1y, lms2y, lms3y, lms4y))
    lms = np.vstack((lmsx, lmsy))
else:
    lms = np.array([[6, -7, 6], [4, 8, -4]])

dt = 0.1
tf = 40.0
