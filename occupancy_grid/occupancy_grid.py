import numpy as np
import car_params as params

def wrap(phi):
    while phi >= np.pi:
        phi = phi - 2 * np.pi
    while phi < -np.pi:
        phi = phi + 2 * np.pi
    return phi

class OccupancyGrid:
    def __init__(self):
        self.map = np.ones((params.l, params.w))  # maybe use np.meshgrid?