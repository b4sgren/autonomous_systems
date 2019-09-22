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
    v, w = generateVelocities(t)
    Car = CarAnimation()
    ekf = EKF(params.dt)

    x = np.array([params.x0, params.y0, params.theta0])
    x_dr = np.array([params.x0, params.y0, params.theta0]) # not sure I can do this. Will need the v and w wo noise

    for i in range(t.size):
        Car.animateCar(state)
        plt.pause(0.1)

        state = ekf.propagateState(x, v[i], w[i])

    print("Finished")
