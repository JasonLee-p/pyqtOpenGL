import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtOpenGL.GLViewWiget import GLViewWidget
from pyqtOpenGL.items.GLAxisItem import GLAxisItem
from pyqtOpenGL.items.GLGridItem import GLGridItem
from pyqtOpenGL.items.GLBoxTextureItem import GLBoxTextureItem
from pyqtOpenGL.items.GLScatterPlotItem import GLScatterPlotItem
from pyqtOpenGL.items.GLArrowPlotItem import GLArrowPlotItem
from pyqtOpenGL.items.GLSurfacePlotItem import GLSurfacePlotItem
from pyqtOpenGL.items.GLModelItem import GLModelItem, Matrix4x4, Vector3
from PyQt5 import QtGui, QtCore
from time import time


class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.ax = GLAxisItem(size=(4, 4, 4))
        # self.box = GLBoxTextureItem(size=(2, 2, 2))
        self.grid = GLGridItem(size=(8, 8))
        self.sca = GLScatterPlotItem(size=3)
        self.arrow = GLArrowPlotItem(
            start_pos= [[0,0,0], [1,2,1], [0,1,0]],
            end_pos= [[0.5,0.5,0], [-0.5,1,0], [0.3,-0.1,0.2]],
            color=[1,1,0]
        )
        self.model = GLModelItem("J:\\learnopengl-python\\resources\\objects\\nanosuit\\nanosuit.obj")
        # self.model = GLModelItem("J:\\learnopengl-python\\resources\\objects\\cyborg\\cyborg.obj")
        self.zmap = np.random.uniform(0, 0.2, (15,15)) + 1
        # self.zmap = np.array([[1.10627519, 1.03413945],
        #                         [1.15691676, 1.09466489]], dtype='f4')
        self.surf = GLSurfacePlotItem(zmap=self.zmap, x_size=4)
        # print(np.random.rand(20, 20))

        self.addItem(self.ax)
        self.addItem(self.grid)
        self.addItem(self.sca)
        # self.addItem(self.box)
        self.addItem(self.arrow)
        self.addItem(self.model)
        self.addItem(self.surf)
        self.model.scale(0.3, 0.25, 0.3)
        self.model.setLight(pos=[0, 5, 10], color=(2.5, 2.5, 2.5))
        pos = np.random.randint(-5, 5, size=(5, 3)).astype('f4')
        self.sca.setData(pos=pos, color=np.ones(3, dtype='f4'), size=0.6)
        self.arrow.setData(color=([0,0,1],[1,0,0],[0,1,0]))

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(30)

        self.t0 = time()
        self.cnt = 10

    def onTimeout(self):
        self.cnt += 1
        # 随机产生 100 个箭头的起点和终点， 以及颜色
        # end_pos = np.random.uniform(-5, 5, (1000, 3)).astype('f4')
        # st_pos = np.random.uniform(-5, 5, (1000, 3)).astype('f4')
        # color = np.random.uniform(0, 1, (1000, 3)).astype('f4')
        # self.sca.setData(st_pos, color)
        # self.zmap += np.random.uniform(-0.01, 0.01, (100,100))
        self.surf.setData(zmap=np.random.uniform(1, 2, (self.cnt,self.cnt)))
        trans = Matrix4x4.fromAxisAndAngle(0, 1, 0, 1)
        self.model.setLight(transform=trans)



if __name__ == '__main__':
    # pg.exec()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())