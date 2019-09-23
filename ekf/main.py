import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
import car_params as params
import scipy.io as sio
from ekf import EKF

def generateVelocities(t):
    v = np.zeros_like(t)
    w = np.zeros_like(t)

    for i in range(t.size): #Note: still need to add noise to these
        v[i] = 1 + 0.5 * np.cos(2 * np.pi * 0.2 * t[i])
        w[i] = -0.2 + 2 * np.cos(2 * np.pi * 0.6 * t[i])

    return v, w

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

        state = ekf.propagateState(state, v[i], w[i])
        dead_reckon = ekf.propagateState(dead_reckon, vc[i], w[i])

    print("Finished")
    plt.waitforbuttonpress()
    plt.close()
