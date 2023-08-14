import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtOpenGL.items import *

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
from gelslimpi.utils import VideoReader
from time import time
import cv2
from pyqtOpenGL.items import *
from pyqtOpenGL.items.GLGelSlimItem import GLModelItem
from pyqtOpenGL.items.GLGelSlimDepthItem import GLSurfacePlotItem
from pyqtOpenGL.GLViewWiget import GLViewWidget
from pyqtOpenGL.transform3d import Matrix4x4

def sphere_depth(r_pix):
    x_span = np.linspace(-r_pix-2, r_pix+2, 2*int(r_pix)+5)
    y_span = np.array(x_span)
    X, Y = np.meshgrid(x_span, y_span)

    depth = r_pix**2 - X**2 * 4 - Y**2 * 4 -  X * Y*4
    depth[depth<0] = 0
    depth = np.sqrt(depth)
    return depth.astype(np.float32)

class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.ax = GLAxisItem(size=(4, 4, 4))
        self.ax.translate(-9, -9, 0)
        # self.grid = GLGridItem(size=(8, 8))
        # self.text = GLTextItem(text="Hello Worldj", pos=(0.1, 0.3, -1), color=(1, 1, 0), fixed=False)
        self.model = GLModelItem("J:\\pyqt-opengl\\GelSlim_obj\\GelSlim.obj")
        # self.zmap = np.random.uniform(0, 1, (105,105))
        self.zmap = sphere_depth(50)/5
        self.flag = False
        cv2.GaussianBlur(self.zmap, (5,5), 0, dst=self.zmap)
        self.zmap[[0,1,2,-3,-2,-1], :] = 0.0
        self.zmap[:, [0,1,2,-3,-2,-1]] = 0.0
        # self.surf = GLSurfacePlotItem(zmap=sphere_depth(30)/5, x_size=12)
        self.surf = GLSurfacePlotItem(zmap=self.zmap, x_size=12)
        self.surf._material = self.model.meshes[0]._material
        # print(self.model.meshes[1]._normals.shape)
        # self.arrow = GLArrowPlotItem(self.model.meshes[1]._vertexes, self.model.meshes[1]._vertexes+self.model.meshes[1]._normals)
        # self.addItem(self.arrow)
        self.addItem(self.ax)
        self.addItem(self.model)
        self.addItem(self.surf)
        self.model.setLight(pos=[0, 5, 10])
        self.surf.setLight(pos=[0, 5, 10])

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(30)

        self.t0 = time()
        self.cnt = 10

    def onTimeout(self):
        if self.flag:
            z = np.random.uniform(-0.3, 0.3, (105,105))
            cv2.GaussianBlur(z, (5,5), 0, dst=z)
            z[[0,1,2,-3,-2,-1], :] = 0.0
            z[:, [0,1,2,-3,-2,-1]] = 0.0
            self.surf.setData(zmap=self.zmap/10 + z)
        else:
            self.surf.setData(zmap=self.zmap/10)
        trans = Matrix4x4.fromAxisAndAngle(0, 0, 1, 1)
        # self.model.setLight(transform=trans)
        # self.surf.setLight(transform=trans)

    def keyPressEvent(self, a0) -> None:
        """按键处理"""
        if a0.key() == Qt.Key.Key_Escape:
            self.close()
        # 视频快捷键
        if a0.text() in ['a', 'A']:
            self.surf._material.opacity -= 0.05
            self.surf.update()
        elif a0.text() in ['d', 'D']:
            self.surf._material.opacity += 0.05
            self.surf.update()
        if a0.text() in ['q', 'Q']:  #
            self.surf._material.diffuse[0] -= 0.02
            self.surf.update()
        elif a0.text() in ['w', 'W']:
            self.surf._material.diffuse[0] += 0.02
            self.surf.update()
        if a0.text() in ['e', 'E']:  #
            self.surf._material.diffuse[1] -= 0.02
            self.surf.update()
        elif a0.text() in ['r', 'R']:
            self.surf._material.diffuse[1] += 0.02
            self.surf.update()
        if a0.text() in ['t', 'T']:  #
            self.flag=True
            self.surf.update()
        elif a0.text() in ['y', 'Y']:
            self.flag=False
            self.surf.update()

if __name__ == '__main__':
    # pg.exec()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())