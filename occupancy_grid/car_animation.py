import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
import car_params as params
from occupancy_grid import OccupancyGrid


class App(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)

        #### Create Gui Elements ###########
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        self.view = self.canvas.addViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(QtCore.QRectF(0,0, 100, 100))

        #### Set Data  #####################
        self.idx = 10
        self.x = np.linspace(0,50., num=100)
        self.X,self.Y = np.meshgrid(self.x,self.x)

        #  image plot
        self.img = pg.ImageItem(border='w')
        self.turtlebot = TurtleBotItem(self.X[:,self.idx], 1.5) 
        self.view.addItem(self.img)
        self.view.addItem(self.turtlebot)

        self.counter = 0
        self.fps = 0.
        self.lastupdate = time.time()

        # Variables used for creating and visualizing the map
        self.grid = OccupancyGrid()
        self.data = self.grid.map * 255

        #### Start  #####################
        self._update()

    def _update(self):
        if self.counter < params.x.shape[1]:
            self.grid.updateMap(params.x[:,self.counter], params.z[:,:,self.counter])
            self.turtlebot.setPose(params.x[:,self.counter])
            self.counter += 1

        self.data = self.grid.map * 255.0       
        self.img.setImage(self.data) #self.data will be the map

        now = time.time()
        dt = (now-self.lastupdate)
        if dt <= 0:
            dt = 0.000000000001
        fps2 = 1.0 / dt
        self.lastupdate = now
        self.fps = self.fps * 0.9 + fps2 * 0.1
        tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps )
        self.label.setText(tx)
        QtCore.QTimer.singleShot(1, self._update)

class TurtleBotItem(pg.GraphicsObject):
    def __init__(self, pose, radius):
        pg.GraphicsObject.__init__(self)
        self.pose = QtCore.QPointF(*pose[:2])
        self.R = radius 
        pt = pose[:2] + np.array([np.cos(pose[2]), np.sin(pose[2])]) * self.R
        self.pt = QtCore.QPointF(*(pose[:2] + pt))
        self.generatePicture()

    def setPose(self, pose):
        self.pose.setX(pose[0])
        self.pose.setY(pose[1])
        pt = pose[:2] + np.array([np.cos(pose[2]), np.sin(pose[2])]) * self.R
        self.pt.setX(pt[0])
        self.pt.setY(pt[1])
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(QtGui.QPen(Qt.black, 0.5, Qt.SolidLine))
        p.setBrush(QtGui.QBrush(Qt.yellow, Qt.SolidPattern))
        p.drawEllipse(self.pose, self.R, self.R)
        p.drawLine(self.pose, self.pt)
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    thisapp = App()
    thisapp.show()
    sys.exit(app.exec_())