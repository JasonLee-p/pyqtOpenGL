import numpy as np
from pyqtOpenGL.GLViewWidget import GLViewWidget
from pyqtOpenGL.items import *
from pyqtOpenGL.transform3d import *
from PyQt5 import QtCore,  QtWidgets
from PyQt5.QtGui import QCloseEvent
from pyqtOpenGL import tb


class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        # -- lights
        self.light = PointLight(pos=[0, 5, 4], ambient=(0.8, 0.8, 0.8), diffuse=(0.8, 0.8, 0.8), visible=False)
        self.ax = GLAxisItem(size=(8, 8, 8))
        self.grid = GLGridItem(
            size=(7, 7), lineWidth=1, color=(1, 1, 1, 1),
            lights=[
                PointLight(pos=[0, 5, 4],
                           ambient=(1.8, 1.8, 1.8), diffuse=(0.8, 0.8, 0.8),
                           visible=False)
            ]
        )
        # sphere
        ver1, ind1, uv1, norm1 = sphere(3, 22, 40, calc_uv_norm=True)
        self.sphere = GLMeshItem(
            vertexes=ver1,
            indices=ind1,
            normals=norm1,
            texcoords=uv1,
            lights=[self.light],
            material=Material(
                ambient=(0.8, 0.8, 0.8),
                diffuse=(0.8, 0.8, 0.8),
                specular=(0.8, 0.8, 0.8),
                shininess=10,
                opacity=1.0,
                textures_paths={"tex_diffuse": "./pyqtOpenGL/items/resources/textures/earth.png"}
            ),
        )

        self.addItem(self.ax)
        self.addItem(self.grid)
        self.addItem(self.sphere)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(20)

        with tb.window("control", 10):
            tb.add_array_float("pos", value=[0,0,0], callback=self.on_changed)
            tb.add_slider("rotx", value=0, min_val=-180, max_val=180, step=1, decimals=0, callback=self.on_changed)
            tb.add_slider("roty", value=0, min_val=-180, max_val=180, step=1, decimals=0, callback=self.on_changed)
            tb.add_slider("rotz", value=0, min_val=-180, max_val=180, step=1, decimals=0, callback=self.on_changed)

        with tb.window("demo", 10):
            tb.add_button("button", callback=self.log)
            tb.add_button("checkable", checkable=True, callback=self.log)
            tb.add_checkbox(label="checkbox", callback=self.log)
            tb.add_slider("slider_int", -1, 0, 100, 1, callback=self.log)
            tb.add_slider("slider_int2", 31, 0, 100, 7, callback=self.log)
            tb.add_slider("slider_float", 0.12, 0, 100, 0.05, 2, callback=self.log)
            tb.add_text_editor("input text", callback=self.log)
            tb.add_combo("combo", ["a", "b", "c"], 1, callback=self.log)
            tb.add_checklist(label="checklist", callback=self.log, exclusive=True)
            tb.add_array_int("array_int", [12, 13,12.0], callback=self.log)
            tb.add_array_float("array_float", [12, 13,12.0,1], callback=self.log)

            tb.add_spacer(size=20)
            tb.add_text("There is a spacer")
            tb.add_separator(horizontal=True)
            tb.add_spacer(size=20)

            with tb.group("Group Horizontal", horizontal=True):
                tb.add_button(label="button2", callback=self.log)
                tb.add_checkbox(label="checkbox2", callback=self.log)

            with tb.group("Group Vertical", horizontal=False):
                tb.add_button(label="button3", callback=self.log)
                tb.add_checkbox(label="checkbox3", callback=self.log)
                tb.add_array_float("float", [0.5], callback=self.log)


    def onTimeout(self):
        self.update()

    def on_changed(self):
        pos = tb.get_value("pos")
        rotx = tb.get_value("rotx")
        roty = tb.get_value("roty")
        rotz = tb.get_value("rotz")
        self.sphere.setTransform(Matrix4x4.fromTranslation(*pos) @ Matrix4x4.fromEulerAngles(rotx, roty, rotz))

    def log(self, value):
        print(self.sender())
        print(value)

    def closeEvent(self, a0: QCloseEvent) -> None:
        tb.clean()
        return super().closeEvent(a0)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())