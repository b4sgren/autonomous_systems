import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
import car_params as params

if __name__ == "__main__":
    Car = CarAnimation()

    x = params.x
    z = params.z
    thk = params.thk

    for i in range(x.shape[1]):
        Car.animateCar(x[:,i])
        plt.pause(0.002)

    plt.show()
    print("Finished")
    plt.close()