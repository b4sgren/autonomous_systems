import numpy as np

#initial position
x0 = -5
y0 = -3
theta0 = np.pi / 2.0

#velocity motion model noise params
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1

# Sensor noise params
sigma_r = 0.1 #m
sigma_theta = 0.05 #rad
fov = np.deg2rad(180/2.0) #Radians to each side

#landmark locations
gen_lms = True #False 
num_lms = 50
if gen_lms:
    lms = np.random.uniform(low=-15.0, high=15.0, size=(2, num_lms))
else:
    lms = np.array([[6, -7, 6], [4, 8, -4]])

dt = 0.1
tf = 50.0
