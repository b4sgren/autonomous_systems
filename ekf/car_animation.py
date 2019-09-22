import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import car_params as params

class CarAnimation:
    def __init__(self):
        self.flagInit = True
        self.fig, self.ax = plt.subplots() #creates the subplots
        self.handle = []

        self.line = np.array([[0, 0.5], [0, 0]]) #car initially facing north

        self.ax.scatter(params.lms[0,:], params.lms[1,:], marker='x', color='k')
        plt.axis([-10, 10, -10, 10])
        self.ax.grid(b=True)

    def animateCar(self, state):
        self.drawCar(state)
        self.drawLine(state)
        self.flagInit = False

    def drawCar(self, state):
        theta = state[2]
        xy = state[0:2]

        if self.flagInit:
            self.handle.append(patches.CirclePolygon(xy,radius = .5, resolution = 15, fc = 'limegreen', ec = 'black'))
            self.ax.add_patch(self.handle[0])
        else:
            self.handle[0]._xy = xy

    def drawLine(self, state):
        theta = state[2]
        x = state[0]
        y = state[1]

        #rotate car before translating
        R = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]], np.float)
        xy = R @ self.line
        xy = xy + np.array([[x, x], [y, y]])

        if self.flagInit:
            self.handle.append(Line2D(xy[0,:], xy[1,:], color='k'))
            self.ax.add_line(self.handle[1])
        else:
            self.handle[1].set_xdata(xy[0,:])
            self.handle[1].set_ydata(xy[1,:])

    def drawStates(self, truth, ekf, dr):
        debug = 1
