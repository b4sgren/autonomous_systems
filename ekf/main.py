import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
import car_params as params
import scipy.io as sio
from ekf import EKF

from IPython.core.debugger import Pdb

def generateVelocities(t):
    v = np.zeros_like(t)
    w = np.zeros_like(t)

    for i in range(t.size):
        v[i] = 1 + 0.5 * np.cos(2 * np.pi * 0.2 * t[i])
        w[i] = -0.2 + 2 * np.cos(2 * np.pi * 0.6 * t[i])

    return v, w

def getMeasurements(state):
    z = np.zeros_like(params.lms, dtype=float)

    for i in range(z.shape[1]):
        lm = params.lms[:,i]
        ds = lm - state[0:2]

        r = np.sqrt(np.sum(ds**2))
        theta = np.arctan2(ds[1], ds[0]) - state[2]

        z[0,i] = r + np.random.normal(0, params.sigma_r)
        z[1,i] = theta + np.random.normal(0, params.sigma_theta)

    return z

if __name__ == "__main__":
    t = np.arange(0, params.tf, params.dt)
    vc, wc = generateVelocities(t)
    v = vc + np.sqrt(params.alpha1 * vc**2 + params.alpha2 * wc**2) * np.random.randn(vc.size)
    w = wc + np.sqrt(params.alpha3 * vc**2 + params.alpha4 * wc**2) * np.random.randn(wc.size)

    Car = CarAnimation()
    ekf = EKF(params.dt)

    x0 = params.x0
    y0 = params.y0
    phi0 = params.theta0
    state = np.array([x0, y0, phi0])
    dead_reckon = np.array([x0, y0, phi0])
    mu = np.array([x0, y0, phi0])

    for i in range(t.size):
        Car.animateCar(state, mu, dead_reckon)
        plt.pause(0.02)

        zt = getMeasurements(state)

        state = ekf.propagateState(state, v[i], w[i])
        dead_reckon = ekf.propagateState(dead_reckon, vc[i], wc[i])

    print("Finished")
    plt.waitforbuttonpress()
    plt.close()
