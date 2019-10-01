import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
import car_params as params
import scipy.io as sio
from ukf import UKF
from ukf import unwrap

from IPython.core.debugger import Pdb


def generateVelocities(t):
    v = 1 + .5 * np.cos(2 * np.pi * 0.2 * t)
    w = -0.2 + 2 * np.cos(2 * np.pi * 0.6 * t)

    return v, w

def readFile():
    data = sio.loadmat("hw2_soln_data.mat")
    t = data["t"].flatten()
    v = data["v"].flatten()
    w = data["om"].flatten()

    return t, v, w

def getMeasurements(state):
    ds = params.lms - state[0:2].reshape((2,1))
    r = np.sqrt(np.sum(ds**2, axis=0)) + np.random.normal(0, params.sigma_r, size=(params.lms.shape[1]))
    theta = (np.arctan2(ds[1,:], ds[0,:]) - state[2]) + np.random.normal(0, params.sigma_theta, size=(params.lms.shape[1]))
    # theta = unwrap(theta)
    for i in range(theta.size):
        theta[i] = unwrap(theta.item(i))

    z = np.array([[r.flatten()], [theta.flatten()]]).reshape((2,3))

    return z

if __name__ == "__main__":
    read_file = False
    if read_file:
        t, v, w = readFile()
        vc, wc = generateVelocities(t)
    else:
        t = np.arange(0, params.tf, params.dt)
        vc, wc = generateVelocities(t)
        v = vc + np.sqrt(params.alpha1 * vc**2 + params.alpha2 * wc**2) * np.random.randn(vc.size)
        w = wc + np.sqrt(params.alpha3 * vc**2 + params.alpha4 * wc**2) * np.random.randn(wc.size)

    Car = CarAnimation()
    ukf = UKF(params.dt)

    x_hist = []
    mu_hist = []
    err_hist = []
    x_covar_hist = []
    y_covar_hist = []
    psi_covar_hist = []
    K_hist = []

    x0 = params.x0
    y0 = params.y0
    phi0 = params.theta0
    state = np.array([x0, y0, phi0])
    dead_reckon = np.array([x0, y0, phi0])
    mu = np.array([x0, y0, phi0])
    Sigma = np.eye(3)

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

        Car.animateCar(state, mu, dead_reckon)
        plt.pause(0.02)

        state = ukf.propagateState(state, v[i], w[i])
        zt = getMeasurements(state)
        mu, Sigma, K = ukf.update(mu, Sigma, zt, vc[i], wc[i])
        dead_reckon = ukf.propagateState(dead_reckon, vc[i], wc[i])

        K_hist.append(K)

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
    ax2[0].plot(t, -x_err_bnd, 'r', label="2 $\sigma$")
    ax2[0].set_ylabel("Err (m)")
    ax2[0].legend()
    ax2[1].plot(t, err_hist[1,:], label="Err")
    ax2[1].plot(t, y_err_bnd, 'r', label="2 $\sigma$")
    ax2[1].plot(t, -y_err_bnd, 'r', label="2 $\sigma$")
    ax2[1].set_ylabel("Err (m)")
    ax2[1].legend()
    ax2[2].plot(t, err_hist[2,:], label="Err")
    ax2[2].plot(t, psi_err_bnd, 'r', label="2 $\sigma$")
    ax2[2].plot(t, -psi_err_bnd, 'r', label="2 $\sigma$")
    ax2[2].set_ylabel("Err (m)")
    ax2[2].set_xlabel("Time (s)")
    ax2[2].legend()
    ax2[0].set_title("Error vs Time")

    plt.figure(4)
    K_hist = np.array(K_hist)
    plt.plot(t, K_hist[:,0,0])
    plt.plot(t, K_hist[:,1,0])
    plt.plot(t, K_hist[:,2,0])
    plt.plot(t, K_hist[:,0,1])
    plt.plot(t, K_hist[:,1,1])
    plt.plot(t, K_hist[:,2,1])
    plt.xlabel("Time (s)")
    plt.ylabel("Kalman Gain")
    plt.title("Kalman Gain vs Time")

    plt.show()
    print("Finished")
    plt.close()
