import numpy as np
import matplotlib.pyplot as plt
from car_animation2 import App
from pyqtgraph.Qt import QtGui
import car_params as params
import sys

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    Car = App()
    Car.show()

    x = params.x
    z = params.z
    thk = params.thk

    for i in range(x.shape[1]):
        # Car.animateCar(x[:,i])
        plt.pause(0.002)

    print("Finished")
    sys.exit(app.exec_())