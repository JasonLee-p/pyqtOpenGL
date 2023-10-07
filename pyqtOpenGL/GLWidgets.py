from .GLViewWiget import GLViewWidget
from .transform3d import Matrix4x4
from .items import *
from PyQt5.QtCore import Qt
import numpy as np
from matplotlib import cm
import cv2

__all__ = [
    "DefaultViewWidget",
    "QSurfaceWidget",
    "QPointCloudWidget",
    "QQuiverWidget",
    "QSurfaceQuiverWidget",
    "QGelSlimWidget"
]

class DefaultViewWidget(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.camera.set_params((0,0,1000), pitch=-75, yaw=0, roll=-90)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # 坐标轴
        self.axis = GLAxisItem(size=(255, 255, 255), tip_size=40)
        self.axis.translate(-240, -320, 0)
        self.addItem(self.axis)
        # 网格曲面
        self.grid1 = GLGridItem(size=(480, 640), spacing=(40, 40))
        self.grid1.rotate(90, 1, 0, 0)
        self.grid1.translate(0, 0, 0)
        self.addItem(self.grid1)
        self.grid1.setVisible(True)

        self.grid2 = GLGridItem(size=(480, 640), spacing=(160, 160), color=(1, 1, 1, 0.4))
        self.grid2.rotate(90, 1, 0, 0)
        self.grid2.translate(0, 0, 255)
        self.addItem(self.grid2)
        self.grid2.setVisible(False)

    def set_grid(self, grid1, grid2, grid1_z, grid2_z):
        self.grid1.setVisible(grid1)
        self.grid1.resetTransform()
        self.grid1.translate(0, 0, grid1_z)

        self.grid2.setVisible(grid2)
        self.grid2.resetTransform()
        self.grid2.translate(0, 0, grid2_z)

    def mousePressEvent(self, ev):
        self.mousePos = ev.pos()


def get_color(zmap, colormap='coolwarm'):
    colormap = eval("cm." + colormap)
    return colormap(zmap).reshape(-1, 4)[:, :3]


class QSurfaceWidget(DefaultViewWidget):

    def __init__(self, parent=None, x_size=640):
        super().__init__(parent=parent)
        self.surface = GLColorSurfaceItem(x_size=x_size)
        self.surface.rotate(90, 0, 0, 1)
        self.addItem(self.surface)

    def setData(self, img, channel=0, colormap:str='coolwarm', cm_scale=1., cm_bias=0.):
        if img.shape[-1] == 3:
            img = img[..., channel]
        color = get_color(img / cm_scale + cm_bias, colormap)

        self.surface.setData(img, color)


class QPointCloudWidget(DefaultViewWidget):

    def __init__(self, parent=None, x_size=640, point_size=20, xy_cnt=40):
        super().__init__(parent)
        self.pc = GLScatterPlotItem(size=point_size)
        self.pc.translate(-240, -x_size/2, 0)
        self.pc.applyTransform(
            Matrix4x4(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])),
            local=True
        )
        self.addItem(self.pc)
        self.xy_cnt = xy_cnt
        self.x_size = x_size

    def setData(self, img, channel=0, colormap:str='coolwarm', cm_scale=1., cm_bias=0.):
        if img.shape[-1] == 3:
            img = img[..., channel]

        scale = self.x_size / img.shape[1]

        x_range = np.linspace(0, img.shape[1]-1, self.xy_cnt, dtype=int)
        y_range = np.linspace(0, img.shape[0]-1, self.xy_cnt, dtype=int)
        x, y = np.meshgrid(x_range, y_range)
        zmap = img[y, x]

        color = get_color(zmap / cm_scale + cm_bias, colormap)

        point_cloud = np.stack([x.ravel() * scale, y.ravel() * scale, zmap.ravel()], axis=1)

        self.pc.setData(point_cloud, color)


class QQuiverWidget(DefaultViewWidget):

    def __init__(self, parent=None, tip_size=[0.7, 1], width=1.0):
        super().__init__(parent)
        verts, faces = sphere(width*0.5, 8, 8)
        self.sphere = GLInstancedMeshItem(vertexes=verts, indices=faces, color=[0.1,0.1,0.1], calcNormals=False)
        self.quiver = GLArrowPlotItem(tip_size=tip_size, color=(1, 0.2, 0), width=width)

        tr = Matrix4x4.fromTranslation(-240, -320, 0).applyTransform(
            Matrix4x4(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])),
            local=True
        )

        self.sphere.applyTransform(tr)
        self.quiver.applyTransform(tr)

        self.addItem(self.sphere)
        self.addItem(self.quiver)

    def setData(self, start_pts, end_pts):
        arrow_lengths = np.linalg.norm(end_pts - start_pts, axis=1)
        arrow_colors = cm.autumn(0.8 - (arrow_lengths / 50))[:, :3]

        self.quiver.setData(start_pts, end_pts, arrow_colors)
        self.sphere.setData(pos = start_pts)


class QSurfaceQuiverWidget(QSurfaceWidget):

    def __init__(self, parent=None, x_size=640, tip_size=[0.7, 1], width=1.0):
        super().__init__(parent, x_size)
        verts, faces = sphere(width*0.35, 8, 8)
        self.sphere = GLInstancedMeshItem(vertexes=verts, indices=faces, color=[0.1,0.1,0.1], calcNormals=False)
        self.quiver = GLArrowPlotItem(tip_size=tip_size, color=(0.7, 0.2, 0), width=width)

        tr = Matrix4x4.fromTranslation(-240, -320, 0).applyTransform(
            Matrix4x4(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])),
            local=True
        )
        self.sphere.applyTransform(tr).translate(0, 0, 1)
        self.quiver.applyTransform(tr).translate(0, 0, 1)

        self.addItem(self.sphere)
        self.addItem(self.quiver)
        # self.setBackgroundColor('w')  # 背景颜色

    def setData(self, img, start_pts, end_pts,
                channel=0, colormap:str="coolwarm", cm_scale=1, cm_bias=0, **kwargs):
        if img.shape[-1] == 3:
            img = img[..., channel]

        color = get_color(img / cm_scale + cm_bias, colormap)

        arrow_lengths = np.linalg.norm(end_pts - start_pts, axis=1)
        arrow_colors = cm.autumn(0.8 - (arrow_lengths / 50))[:, :3]

        self.surface.setData(zmap=img, color=color)
        self.quiver.setData(start_pos=start_pts, end_pos=end_pts, color=arrow_colors)
        self.sphere.setData(pos = start_pts)


class QGelSlimWidget(GLViewWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.camera.set_params((0,-2,35), pitch=-40, yaw=0, roll=-45)

        self.ax = GLAxisItem(size=(3, 3, 3))
        self.ax.translate(-6.75*3/4, -6.75, 0.5)

        self.grid = GLGridItem(size=(200, 200), spacing=(10, 10))
        self.grid.rotate(90, 1, 0, 0)
        self.grid.translate(10, 0, -20)

        self.light = PointLight(pos=[-15, -5, 7], diffuse=(0.8, 0.8, 0.8), visible=False)
        self.light1 = PointLight(pos=[0, 15, 3], diffuse=(0.8, 0.8, 0.8), visible=False)
        self.light2 = PointLight(pos=[3, 5, 10], diffuse=(0.8, 0.8, 0.8), visible=False)

        self.gelslim_model = GLGelSimItem(lights=[self.light, self.light1, self.light2])


        vert, ind = sphere(0.08)
        self.pointcloud = GLInstancedMeshItem(
                            None, vert, ind,
                            lights=[self.light, self.light1, self.light2],
                            color=[1,0.7,0.]
                        )
        self.arrow = GLArrowPlotItem(None, None, tip_size=[0.02, 0.03], tip_pos=-0.13, width=2, color=[1, 0, 0])

        tr = Matrix4x4.fromTranslation(-6.75*3/4, -6.75, 0).applyTransform(
            Matrix4x4(np.array([
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])),
            local=True
        )
        self.pointcloud.applyTransform(tr)
        self.arrow.applyTransform(tr)

        self.addItem(self.arrow)
        self.addItem(self.ax)
        self.addItem(self.grid)
        self.addItem(self.gelslim_model)
        self.addItem(self.pointcloud)

        self.cnt = 10

    def setData(self, zmap, start_pts, end_pts, **kwargs):
        if zmap.ndim == 3:
            return

        h, w = zmap.shape[0:2]
        if h > 400:
            zmap = cv2.resize(zmap, (480, 360), interpolation=cv2.INTER_NEAREST)

        scale = self.gelslim_model.gelslim_gel._x_size / w
        self.gelslim_model.setDepth(zmap=-zmap)

        if start_pts is not None and end_pts is not None:
            start_pts[:, 2] = -start_pts[:, 2]
            end_pts[:, 2] = -end_pts[:, 2]
            self.pointcloud.setData(pos = end_pts*scale)
            self.arrow.setData(start_pos=start_pts*scale, end_pos=end_pts*scale)
