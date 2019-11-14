import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
import car_params as params
from particle_filter import ParticleFilter, unwrap
from ekf import EKF

def generateVelocities(t):
    v = 1 + .5 * np.cos(2 * np.pi * 0.2 * t)
    w = -0.2 + .1 * np.cos(2 * np.pi * 0.2 * t)

    return v, w

def getMeasurements(state): #Will need to change if using not 360 deg vision
    z = np.zeros_like(params.lms, dtype=float)
    
    ds = params.lms - state[0:2].reshape(2,1)
    r = np.sqrt(np.sum(ds**2, axis=0))
    theta = np.arctan2(ds[1], ds[0]) - state[2]

    z[0] = r + np.random.normal(0, params.sigma_r, size=r.size) #Measurement noise seems to be what is making everything do really bad
    z[1] = theta + np.random.normal(0, params.sigma_theta, size=theta.size)
    z[1] = unwrap(z[1])

    ind = np.argwhere(np.abs(z[1]) < params.fov)
    z = z[:, ind][:,:,0]

    return z, ind.squeeze()

def recoverMeanAndCovar(Chi):   # I think there is an issue recovering the mean around theta = pi b/c average between -pi and pi is 0
        mu = np.mean(Chi, axis=1)
        temp_x = Chi - mu.reshape((3,1))
        temp_x[2] = unwrap(temp_x[2])
        Sigma = np.cov(temp_x)

        return mu, Sigma

if __name__ == "__main__":
    t = np.arange(0, params.tf, params.dt)
    vc, wc = generateVelocities(t)
    v = vc + np.sqrt(params.alpha1 * vc**2 + params.alpha2 * wc**2) * np.random.randn(vc.size)
    w = wc + np.sqrt(params.alpha3 * vc**2 + params.alpha4 * wc**2) * np.random.randn(wc.size)

    Car = CarAnimation()
    filter = ParticleFilter(params.dt)

    x_hist = []
    mu_hist = []
    err_hist = []
    x_covar_hist = []
    y_covar_hist = []
    psi_covar_hist = []

    state = np.zeros(3)
    dead_reckon = np.zeros(3)
    Chi = np.zeros((3, params.M))
    wp= np.ones(params.M)  #Evenly distributed weights
    mu = np.mean(Chi, axis=1)
    Sigma = np.cov(mu.reshape((3,1)) - Chi)
    j = 0 #Index of the best particle

    for i in range(t.size):
        #stuff for plotting
        x_hist.append(state)
        mu_hist.append(mu)
        err = state - mu
        err[2] = unwrap(err[2])
        err_hist.append(err)
        x_covar_hist.append(Sigma[0,0])
        y_covar_hist.append(Sigma[1,1])
        psi_covar_hist.append(Sigma[2,2])

        Car.animateCar(state, mu, dead_reckon, Chi, filter.lm_filters[j])
        plt.pause(0.02)

        state = filter.propagateState(state, v[i], w[i])
        dead_reckon = filter.propagateState(dead_reckon, vc[i], wc[i])
        zt, ind = getMeasurements(state)
        Chi, j, wp = filter.update(Chi, wp, zt, ind, vc[i], wc[i])
        mu, Sigma = recoverMeanAndCovar(Chi)
        mu = Chi[:,j] #Use the mean of the best particle
        mu[2] = unwrap(mu[2])
        wp = np.ones_like(wp)

    fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True)
    x_hist = np.array(x_hist).T
    mu_hist = np.array(mu_hist).T
    ax1[0].plot(t, x_hist[0,:], label="Truth")
    ax1[0].plot(t, mu_hist[0,:], label="Est")
    ax1[0].set_ylabel("x (m)")
    ax1[0].legend()
    ax1[1].plot(t, x_hist[1,:], label="Truth")
    ax1[1].plot(t, mu_hist[1,:], label="Est")
    ax1[1].set_ylabel("y (m)")
    ax1[1].legend()
    ax1[2].plot(t, x_hist[2,:], label="Truth")
    ax1[2].plot(t, mu_hist[2,:], label="Est")
    ax1[2].set_xlabel("Time (s)")
    ax1[2].set_ylabel("$\psi$ (rad)")
    ax1[2].legend()
    ax1[0].set_title("Estimate vs Truth")

    fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True)
    err_hist = np.array(err_hist).T
    x_err_bnd = np.sqrt(np.array(x_covar_hist)) * 2
    y_err_bnd = np.sqrt(np.array(y_covar_hist)) * 2
    psi_err_bnd = np.sqrt(np.array(psi_covar_hist)) * 2
    ax2[0].plot(t, err_hist[0,:], label="Err")
    ax2[0].plot(t, x_err_bnd, 'r', label="2 $\sigma$")
    ax2[0].plot(t, -x_err_bnd, 'r')
    ax2[0].set_ylabel("Err (m)")
    ax2[0].legend()
    ax2[1].plot(t, err_hist[1,:], label="Err")
    ax2[1].plot(t, y_err_bnd, 'r', label="2 $\sigma$")
    ax2[1].plot(t, -y_err_bnd, 'r')
    ax2[1].set_ylabel("Err (m)")
    ax2[1].legend()
    ax2[2].plot(t, err_hist[2,:], label="Err")
    ax2[2].plot(t, psi_err_bnd, 'r', label="2 $\sigma$")
    ax2[2].plot(t, -psi_err_bnd, 'r')
    ax2[2].set_ylabel("Err (m)")
    ax2[2].set_xlabel("Time (s)")
    ax2[2].legend()
    ax2[0].set_title("Error vs Time")

    plt.show()
    print("Finished")
    plt.close()
