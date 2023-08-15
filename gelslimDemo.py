import cv2
import numpy as np
from time import time
from pyqtOpenGL.items import *
from pyqtOpenGL.GLViewWiget import GLViewWidget
from pyqtOpenGL.items.GLGelSlimItem import GLGelSimItem
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

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
        self.grid = GLGridItem(size=(200, 200), spacing=(10, 10))
        self.grid.rotate(90, 1, 0, 0)
        self.grid.translate(10, 0, -20)

        self.zmap = sphere_depth(100)/5
        cv2.GaussianBlur(self.zmap, (5,5), 0, dst=self.zmap)
        self.zmap[[0,1,2,-3,-2,-1], :] = 0.0
        self.zmap[:, [0,1,2,-3,-2,-1]] = 0.0

        self.flag = False
        self.light_move = False

        self.light = PointLight(pos=[-15, -5, 7], diffuse=(0, 0, 0.8))
        self.light1 = PointLight(pos=[0, 15, 3], diffuse=(0, .8, 0))
        self.light2 = PointLight(pos=[3, 5, 10], diffuse=(0.8, 0.8, 0.8))
        self.model = GLGelSimItem(lights=[self.light, self.light1, self.light2])

        self.addItem(self.ax)
        self.addItem(self.model)
        self.addItem(self.grid)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(30)

        self.t0 = time()
        self.cnt = 10

    def onTimeout(self):
        if self.flag:
            z = np.random.uniform(-0.3, 0.3, (205,205))
            cv2.GaussianBlur(z, (5,5), 0, dst=z)
            z[[0,1,2,-3,-2,-1], :] = 0.0
            z[:, [0,1,2,-3,-2,-1]] = 0.0
            self.model.setDepth(zmap=self.zmap/10 + z)
        else:
            self.model.setDepth(zmap=self.zmap/10)

        if self.light_move :
            self.light.rotate(0, 1, 0, 1)
            self.light1.rotate(1, 1, 0, -2)
            self.light2.rotate(0.5, 1., 0.6, 1.5)
        self.update()

    def keyPressEvent(self, a0) -> None:
        """按键处理"""
        material = self.model.gelslim_base.getMaterial(0)
        if a0.key() == Qt.Key.Key_Escape:
            self.close()
        # 视频快捷键
        if a0.text() in ['a', 'A']:
            material.opacity -= 0.05
            self.update()
        elif a0.text() in ['d', 'D']:
            material.opacity += 0.05
            self.update()
        elif a0.text() == '1':  #
            material.diffuse[0] -= 0.02
            self.update()
        elif a0.text() == '2':
            material.diffuse[0] += 0.02
            self.update()
        elif a0.text() == '3':  #
            material.diffuse[1] -= 0.02
            self.update()
        elif a0.text() == '4':
            material.diffuse[1] += 0.02
            self.update()
        elif a0.text() == '5':  #
            material.diffuse[2] -= 0.02
            self.update()
        elif a0.text() == '6':
            material.diffuse[2] += 0.02
            self.update()
        elif a0.text() == '9':  #
            self.light_move = True
            self.update()
        elif a0.text() == '0':
            self.light_move = False
            self.update()
        elif a0.text() in ['t', 'T']:  #
            self.flag=True
            self.update()
        elif a0.text() in ['y', 'Y']:
            self.flag=False
            self.update()
        elif a0.text() in ['q', 'Q']:  #
            material.shininess -= 10
            self.update()
        elif a0.text() in ['e', 'E']:
            material.shininess += 10
            self.update()

if __name__ == '__main__':
    # pg.exec()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())