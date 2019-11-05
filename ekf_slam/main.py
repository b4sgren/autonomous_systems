import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
import car_params as params
import scipy.io as sio
from ekf import EKF
from ekf import unwrap
from mpl_toolkits.mplot3d import Axes3D


def generateVelocities(t):
    v = 1 + .5 * np.cos(2 * np.pi * 0.2 * t)
    w = -0.2 + 2 * np.cos(2 * np.pi * 0.6 * t)

    return v, w

def getMeasurements(state):
    z = np.zeros_like(params.lms, dtype=float)
    
    ds = params.lms - state[0:2].reshape(2,1)
    r = np.sqrt(np.sum(ds**2, axis=0))
    theta = np.arctan2(ds[1], ds[0]) - state[2]

    z[0] = r + np.random.normal(0, params.sigma_r, size=r.size)
    z[1] = theta + np.random.normal(0, params.sigma_theta, size=theta.size)
    z[1] = unwrap(z[1])

    ind = np.argwhere(np.abs(z[1]) < params.fov)
    z = z[:, ind][:,:,0]

    return z, ind

if __name__ == "__main__":
    t = np.arange(0, params.tf, params.dt)
    vc, wc = generateVelocities(t)
    v = vc + np.sqrt(params.alpha1 * vc**2 + params.alpha2 * wc**2) * np.random.randn(vc.size)
    w = wc + np.sqrt(params.alpha3 * vc**2 + params.alpha4 * wc**2) * np.random.randn(wc.size)

    Car = CarAnimation()
    ekf = EKF(params.dt)

    x_hist = []
    mu_hist = []
    err_hist = []
    x_covar_hist = []
    y_covar_hist = []
    psi_covar_hist = []
    K_hist = []

    state = np.zeros(3) #np.array([params.x0, params.y0, params.theta0])
    dead_reckon = np.zeros(3) #np.array([params.x0, params.y0, params.theta0])
    mu = ekf.mu
    Sigma = ekf.Sigma

    for i in range(t.size):
        #stuff for plotting
        x_hist.append(state)
        mu_hist.append(ekf.mu[:3])
        err = state - ekf.mu[:3]
        err[2] = unwrap(err[2])
        err_hist.append(err)
        x_covar_hist.append(ekf.Sigma[0,0])
        y_covar_hist.append(ekf.Sigma[1,1])
        psi_covar_hist.append(ekf.Sigma[2,2])

        Car.animateCar(state, ekf.mu[:2], dead_reckon, ekf.mu[3:], ekf.Sigma[3:,3:], ekf.lms_found)
        plt.pause(0.02)

        state = ekf.propagateState(state, v[i], w[i])
        zt, lm_ind = getMeasurements(state)
        ekf.update(zt, lm_ind, vc[i], wc[i])
        dead_reckon = ekf.propagateState(dead_reckon, vc[i], wc[i])


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

    tempx = np.arange(3 + 2 * ekf.num_lms)
    tempy = np.arange(3 + 2 * ekf.num_lms)
    xx, yy = np.meshgrid(tempx, tempy)
    fig3 = plt.figure(4)
    ax3 = fig3.add_subplot(111, projection='3d')
    # ax3.bar3d(xx, yy, np.zeros_like(ekf.Sigma), 1, 1, ekf.Sigma, shade=True)
    ax3.bar3d(xx.ravel(), yy.ravel(), 0, 1, 1, np.abs(ekf.Sigma.ravel()), shade=True)

    plt.show()
    print("Finished")
    plt.close()

