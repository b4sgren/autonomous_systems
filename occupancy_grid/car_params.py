import numpy as np
import scipy.io as sio

data = sio.loadmat('state_meas_data.mat')
x = data['x'] # 3x759 of the true posisions
z = data['z'] #2x11x759 of the measurements (range and bearing). Nan = no hit
thk = data['thk'] #1x11 of the angles of the 9 lasers on our sensor between -pi/2 and pi/2. Equally spaced

#initial position
x0 = x[0,0]
y0 = x[1,0]
theta0 = x[2,0]

# inverse range sensor model
alpha = 1 #m
beta = np.deg2rad(5) #rad
z_max = 150 #m

#grid size
l = 100 #m
w = 100 #m

# probability values
p_occ = 0.65 #probability of being occupied if hit detected
p_emp = 0.35 #probability of being empty if no hit detected