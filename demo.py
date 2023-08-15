import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtOpenGL import GLViewWidget
from pyqtOpenGL.items  import *
from pyqtOpenGL.transform3d import *
import PIL.Image as Image


class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

        ver1, ind1 = sphere(2, 20, 20)
        normal1 = vertex_normal(ver1, ind1)
        ver2, ind2 = cylinder(radius=[1.2, 1], cols=20, rows=20, length=2.4)
        img = Image.open("./pyqtOpenGL/items/resources/textures/box.png")
        img = np.array(img, dtype='f4')

        self.img = GLImageItem(img, width_height=(0.3, 0.3))
        self.ax = GLAxisItem(size=(8, 8, 8))
        self.box = GLBoxTextureItem(size=(2, 2, 2))
        self.box.translate(0, 1.1, 0)
        self.grid = GLGridItem(size=(8, 8), lineWidth=1.5)
        self.scatter = GLScatterPlotItem(
            pos=np.random.uniform(-5, 5, size=(15, 3)).astype('f4'),
            color=np.random.uniform(0, 1, size=(15, 3)).astype('f4'),
            size=2
        )
        self.arrow = GLArrowPlotItem(start_pos=ver1+[5,-1,0], end_pos=ver1+normal1+[5,-1,0], color=[1,1,0])

        self.light = PointLight(pos=[0, 5, 4], ambient=(0.8, 0.8, 0.8),diffuse=(0.8, 0.8, 0.8))
        self.light1 = PointLight(pos=[0, -5, 1], diffuse=(0, .8, 0))
        self.light2 = PointLight(pos=[-12, 3, 2], diffuse=(0.8, 0, 0))

        self.model = GLModelItem(
            "./pyqtOpenGL/items/resources/objects/cyborg/cyborg.obj",
            lights=[self.light, self.light1, self.light2]
        )
        self.model.translate(0, 2, 0)
        self.mesh1 = GLInstancedMeshItem(
            pos=[[5,-1,0],[3,10,-5]],
            lights=[self.light, self.light1, self.light2],
            indices=ind1,
            vertexes=ver1,
            color=(0.7,0.8,0.8)
        )
        self.mesh2 = GLMeshItem(
            indices=ind2,
            vertexes=ver2,
            lights=[self.light, self.light1, self.light2],
            material=Material((0.4, 0.1, 0.1), diffuse=(0.6, 0.1, 0.3))
        )
        self.mesh2.translate(-4, 3, 0)

        self.zmap = np.random.uniform(0, 3, (41,41))
        self.surf = GLSurfacePlotItem(
            zmap=self.zmap, x_size=6, lights=[self.light, self.light1, self.light2],
            material=Material((0.2, 0.2, 0.2), diffuse=(0.5, 0.5, 0.5))
        )
        self.surf.rotate(-90, 1, 0, 0)
        self.surf.translate(-5, -2, 0)
        self.text = GLTextItem(text="Hello World", pos=(7, 2, -1), color=(1, 0.6, 1), fixed=False)

        self.addItem(self.text)
        self.addItem(self.img)
        self.addItem(self.ax)
        self.addItem(self.grid)
        self.addItem(self.scatter)
        self.addItem(self.box)
        self.addItem(self.arrow)
        self.addItem(self.model)
        self.addItem(self.surf)
        self.addItem(self.mesh1)
        self.addItem(self.mesh2)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(20)

    def onTimeout(self):
        self.light.rotate(0, 1, 0.4, 1)
        self.light1.rotate(1, 1, 0, -2)
        self.light2.rotate(0.2, 1., 0., 1.5)
        self.update()


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())