import numpy as np
import car_params as params

from IPython.core.debugger import Pdb

def unwrap(phi):
    # Pdb().set_trace()
    # phi[phi >= np.pi] -= 2 * np.pi # not sure that this will work
    # phi[phi < -np.pi] += 2 * np.pi
    while phi >= np.pi:
        phi = phi - 2 * np.pi
    while phi < -np.pi:
        phi = phi + 2 * np.pi
    return phi
