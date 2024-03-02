import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtOpenGL.items import *
from pyqtOpenGL import GLViewWidget, tb, Matrix4x4, Vector3

class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.resize(1200, 900)
        self.camera.set_params((0., 0., 5.0), -60, 0., -90)

        # -- data
        # cube_vert, cube_norm, cube_uv = cube(2, 2, 2)
        ball_vert, ball_ind, ball_uv, ball_norm = sphere(1, 100, 100, True)

        # -- material
        self.material = Material()

        # -- light
        self.light = PointLight(pos=[15, 15, 15], ambient=(1, 1, 1), diffuse=(1, 1, 1),
                               specular=(1, 1, 1), visible=True, directional=False)

        # -- items
        self.ax = GLAxisItem(size=(2, 2, 2), width=3, tip_size=0.2)
        self.grid = GLGridItem(
            size=(15, 15), spacing=(0.5, 0.5), lineWidth=1,
            lights=[PointLight(pos=[-2, 1, 4], ambient=(.1, .1, .1), diffuse=(1, 1, 1),
                               specular=(1, 1, 1), visible=False, directional=True)]
        ).rotate(90, 1, 0, 0).translate(0, 0, -1)

        self.box = GLMeshItem(
            vertexes=ball_vert, normals=ball_norm,
            indices=ball_ind, texcoords=ball_uv,
            material=self.material, lights=[self.light],
            glOptions="translucent_cull"
        )

        self.addItem(self.ax)
        self.addItem(self.grid)
        self.addItem(self.box)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(20)
        self.init_control()

    def init_control(self):
        with tb.window(label="control", parent=self, pos=[-500, 0], size=[500, 200]):
            tb.add_drag_array("light pos", self.light.position.toPolar(),
                              min_val=[1.5, 0, 0], max_val=[50, 360, 90], step=1, decimals=[1, 0, 0],
                              format=["r %.1f", "theta %d deg", "phi %d deg"],
                              callback=self.on_change)
            with tb.group(label="material", horizontal=False):
                tb.add_drag_array("ambient", self.material.ambient, 0, 1, 0.01, 2,
                              format=["r %.2f", "g %.2f", "b %.2f"], callback=self.on_change)
                tb.add_drag_array("diffuse", self.material.diffuse, 0, 1, 0.01, 2,
                              format=["r %.2f", "g %.2f", "b %.2f"], callback=self.on_change)
                tb.add_drag_array("specular", self.material.specular, 0, 1, 0.01, 2,
                              format=["r %.2f", "g %.2f", "b %.2f"], callback=self.on_change)
                tb.add_drag_value("shininess", self.material.shininess, 0, 100, 1, 0, callback=self.on_change)
                tb.add_drag_value("opacity", self.material.opacity, 0, 1, 0.01, 2, callback=self.on_change)

    def onTimeout(self):
        # self.light.rotate(0, 1, 1, 1)
        self.update()

    def on_change(self):
        widget = self.sender()
        label = widget.get_label()

        if label == "light pos":
            self.light.set_data(pos=Vector3.fromPolar(*widget.value))
        else:
            self.material.set_data(**{label: widget.value})

    def closeEvent(self, a0) -> None:
        tb.clean()
        return super().closeEvent(a0)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())