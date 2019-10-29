import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
import car_params as params
from eif import EIF
from eif import unwrap


if __name__ == "__main__":
    Car = CarAnimation()
    eif = EIF(params.dt)

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
    state = params.truth[:,0]
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

        state = ekf.propagateState(state, v[i], w[i])
        zt = getMeasurements(state)
        mu, Sigma, K = ekf.update(mu, zt, vc[i], wc[i])
        dead_reckon = ekf.propagateState(dead_reckon, vc[i], wc[i])

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

'''
Different Input Velocities: Change linear velocity doesn't do much. Maybe increase the covariance. Same with angular velocity
Different Landmark locations: Doesn't affect the quality of the estimate too much. Changes the gains a little bit
Number of Landmarks: Decreasing the number of landmarks decreases the quality of the est, increases covariance and increase abs_val of K. Increasing does the oppopsite
Sensor Noise: Increasing noise decreases quality of estimate but not by much. This seems to be offset by the number of landmarks we are measuring.
Control/Motion Noise: Doesn't do much to the estimate. Offset by number of landmarks we measure. Does affect the gain
Yes the EKF behaves as expected
'''
