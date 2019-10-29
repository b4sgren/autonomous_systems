import numpy as np
import scipy.io as sio

data = sio.loadmat('midterm_data.mat')
truth = data['X_tr']
lms = data['m']
v = data['v'].flatten()
vc = data['v_c'].flatten()
w = data['om'].flatten()
wc = data['om_c'].flatten()
t = data['t'].flatten()
z_r = data['range_tr']
z_phi = data['bearing_tr']

#initial position
x0 = -5.0
y0 = 0.0
theta0 = np.pi / 2.0

#velocity motion model noise params
sigma_v = 0.15 # m/s
sigma_w = 0.1 # rad/s

# Sensor noise params
sigma_r = 0.2 #m
sigma_theta = 0.1 #rad

dt = 0.1 #s
tf = 30.0 #s