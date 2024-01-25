import cv2
import numpy as np
from time import time
from pyqtOpenGL.items import *
from pyqtOpenGL import GLViewWidget, tb
from pyqtOpenGL.items.GLGelSlimItem import GLGelSimItem
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

def sphere_depth(r_pix):
    x_span = np.linspace(-r_pix-2, r_pix+2, 2*int(r_pix)+5)
    y_span = np.array(x_span)
    X, Y = np.meshgrid(x_span, y_span)

    depth = r_pix**2 / 32 - X**2 / 8 - Y**2 / 8 - X*Y / 8
    depth[depth<0] = 0
    depth = np.sqrt(depth)
    return depth.astype(np.float32)

class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.ax = GLAxisItem(size=(4, 4, 4))
        self.ax.translate(-9, -9, 0)
        self.grid = GLGridItem(
            size=(200, 200), spacing=(10, 10),
            lights=[
                PointLight(pos=[3, 5, 10], ambient=(1.5, 1.5, 1.5), diffuse=(1,1, 1), visible=False)
            ],
        )
        self.grid.rotate(90, 1, 0, 0)
        self.grid.translate(10, 0, -20)

        self.zmap = sphere_depth(100)[25:-25]

        self.flag = False
        self.light_move = False

        self.light = PointLight(pos=[-15, -5, 7], diffuse=(0, 0, 0.8))
        self.light1 = PointLight(pos=[0, 15, 3], diffuse=(0, .8, 0))
        self.light2 = PointLight(pos=[3, 5, 10], diffuse=(1,1, 1))

        self.model = GLGelSimItem(lights=[self.light, self.light1, self.light2])

        self.addItem(self.ax)
        self.addItem(self.model)
        self.addItem(self.grid)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(30)

        self.t0 = time()

        with tb.window("control", self, 10, size=(300, 200)):
            default_color = (0.53, 0.57, 0.75, 1.0)
            tb.add_drag_array("RGBA", value=default_color, min_val=0, max_val=1, step=0.01, decimals=2,
                              horizontal=False, format=["R %.2f", "G %.2f", "B %.2f", "A %.2f"],
                              callback=self.on_color_changed)
            tb.add_drag_value("Shininess", value=10, min_val=1, max_val=300, step=1, decimals=0,
                              callback=self.on_color_changed)
            tb.add_separator()
            tb.add_checkbox("light move", value=False)
            tb.add_checkbox("random surface", value=False)

    def onTimeout(self):
        if tb.get_value("random surface") :
            z = np.random.uniform(-2, 2, (155,205))
            cv2.GaussianBlur(z, (5,5), 0, dst=z)
            z[[0,1,2,-3,-2,-1], :] = 0.0
            z[:, [0,1,2,-3,-2,-1]] = 0.0
            self.model.setDepth(zmap=self.zmap + z)
        else:
            self.model.setDepth(zmap=self.zmap)

        if tb.get_value("light move") :
            self.light.rotate(0, 1, 0, 1)
            self.light1.rotate(1, 1, 0, -2)
            self.light2.rotate(0.5, 1., 0.6, 1.5)
        self.update()

    def on_color_changed(self, value):
        material = self.model.gelslim_base.getMaterial(0)
        label = self.sender().get_label()
        if label == 'RGBA':
            material.diffuse = value[:3]
            material.opacity = value[3]
        else:
            material.shininess = value
        self.update()

    def closeEvent(self, a0) -> None:
        tb.clean()
        return super().closeEvent(a0)


if __name__ == '__main__':
    # pg.exec()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())