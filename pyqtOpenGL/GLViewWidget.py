from OpenGL.GL import *  # noqa
from math import radians, cos, sin, tan, sqrt
from PyQt5 import QtCore, QtWidgets, QtGui
from .camera import Camera
from .functions import mkColor
from .transform3d import Matrix4x4, Quaternion, Vector3
from typing import List, Set
from .GLGraphicsItem import GLGraphicsItem
from .items.light import PointLight

class GLViewWidget(QtWidgets.QOpenGLWidget):

    def __init__(
        self,
        cam_position = Vector3(0., 0., 10.),
        yaw = 0.,
        pitch = 0.,
        roll = 0.,
        fov = 45.,
        bg_color = (0.2, 0.3, 0.3, 1.),
        # bg_color = (0.95, 0.95, 0.95, 1.),
        parent=None,
    ):
        """
        Basic widget for displaying 3D data
          - Rotation/scale controls
        """
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.camera = Camera(cam_position, yaw, pitch, roll, fov)
        self.bg_color = bg_color
        self.items : List[GLGraphicsItem] = []
        self.lights: Set[PointLight] = set()

    def get_proj_view_matrix(self):
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(
            self.deviceWidth(),
            self.deviceHeight()
        )
        return proj * view

    def get_proj_matrix(self):
        return self.camera.get_projection_matrix(
            self.deviceWidth(),
            self.deviceHeight()
        )

    def get_view_matrix(self):
        return self.camera.get_view_matrix()

    def deviceWidth(self):
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr)

    def deviceHeight(self):
        dpr = self.devicePixelRatioF()
        return int(self.height() * dpr)

    def deviceRatio(self):
        return self.height() / self.width()

    def reset(self):
        self.camera.set_params(Vector3(0., 0., 10.), 0, 0, 0, 45)

    def addItem(self, item: GLGraphicsItem):
        self.items.append(item)
        item.setView(self)
        if hasattr(item, 'lights'):
            self.lights |= set(item.lights)
        self.items.sort(key=lambda a: a.depthValue())
        self.update()

    def addItems(self, items: List[GLGraphicsItem]):
        for item in items:
            self.addItem(item)

    def removeItem(self, item):
        """
        Remove the item from the scene.
        """
        self.items.remove(item)
        item._setView(None)
        self.update()

    def clear(self):
        """
        Remove all items from the scene.
        """
        for item in self.items:
            item._setView(None)
        self.items = []
        self.update()

    def setBackgroundColor(self, *args, **kwds):
        """
        Set the background color of the widget. Accepts the same arguments as
        :func:`~pyqtgraph.mkColor`.
        """
        self.bg_color = mkColor(*args, **kwds).getRgbF()
        self.update()

    def getViewport(self):
        return (0, 0, self.deviceWidth(), self.deviceHeight())

    def initializeGL(self):
        """initialize OpenGL state after creating the GL context."""
        PointLight.initializeGL()

    def paintGL(self):
        """
        viewport specifies the arguments to glViewport. If None, then we use self.opts['viewport']
        region specifies the sub-region of self.opts['viewport'] that should be rendered.
        Note that we may use viewport != self.opts['viewport'] when exporting.
        """
        glClearColor(*self.bg_color)
        glDepthMask(GL_TRUE)
        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
        self.drawItems()

    def drawItems(self):
        for it in self.items:
            try:
                it.drawItemTree()
            except:
                printExc()
                print("Error while drawing item %s." % str(it))

        # draw lights
        for light in self.lights:
            light.paint(self.get_proj_view_matrix())

    def pixelSize(self, pos=Vector3(0, 0, 0)):
        """
        depth: z-value in global coordinate system
        Return the approximate (y) size of a screen pixel at the location pos
        Pos may be a Vector or an (N,3) array of locations
        """
        pos = self.get_view_matrix() * pos  # convert to view coordinates
        fov = self.camera.fov
        return max(-pos[2], 0) * 2. * tan(0.5 * radians(fov)) / self.deviceHeight()

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePressPos = lpos
        self.cam_quat, self.cam_pos = self.camera.get_quat_pos()

    def mouseMoveEvent(self, ev):
        ctrl_down = (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift_down = (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        alt_down = (ev.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)

        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        diff = lpos - self.mousePressPos

        if ctrl_down:
            diff *= 0.1

        if alt_down:
            roll = -diff.x() / 5

        if shift_down:
            if abs(diff.x()) > abs(diff.y()):
                diff.setY(0)
            else:
                diff.setX(0)

        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if alt_down:
                self.camera.orbit(0, 0, roll, base=self.cam_quat)
            else:
                self.camera.orbit(diff.x(), diff.y(), base=self.cam_quat)
        elif ev.buttons() == QtCore.Qt.MouseButton.MiddleButton:
            self.camera.pan(diff.x(), -diff.y(), 0, base=self.cam_pos)
        self.update()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().x()
        if delta == 0:
            delta = ev.angleDelta().y()
        if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            self.camera.fov *= 0.999**delta
        else:
            self.camera.pos.z = self.camera.pos.z * 0.999**delta
        self.update()

    def readQImage(self):
        """
        Read the current buffer pixels out as a QImage.
        """
        return self.grabFramebuffer()

    def isCurrent(self):
        """
        Return True if this GLWidget's context is current.
        """
        return self.context() == QtGui.QOpenGLContext.currentContext()

    def keyPressEvent(self, a0) -> None:
        """按键处理"""
        if a0.text() == '1':
            pos, euler = self.camera.get_params()
            print(f"pos: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})  "
                  f"euler: ({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f})")
        elif a0.text() == '2':
            self.camera.set_params((0.00, 0.00, 886.87),
                                   pitch=-31.90, yaw=-0, roll=-90)
            # self.camera.set_params((1.72, -2.23, 27.53),pitch=-27.17, yaw=2.64, roll=-70.07)


import warnings
import traceback
import sys

def formatException(exctype, value, tb, skip=0):
    """Return a list of formatted exception strings.

    Similar to traceback.format_exception, but displays the entire stack trace
    rather than just the portion downstream of the point where the exception is
    caught. In particular, unhandled exceptions that occur during Qt signal
    handling do not usually show the portion of the stack that emitted the
    signal.
    """
    lines = traceback.format_exception(exctype, value, tb)
    lines = [lines[0]] + traceback.format_stack()[:-(skip+1)] + ['  --- exception caught here ---\n'] + lines[1:]
    return lines


def getExc(indent=4, prefix='|  ', skip=1):
    lines = formatException(*sys.exc_info(), skip=skip)
    lines2 = []
    for l in lines:
        lines2.extend(l.strip('\n').split('\n'))
    lines3 = [" "*indent + prefix + l for l in lines2]
    return '\n'.join(lines3)


def printExc(msg='', indent=4, prefix='|'):
    """Print an error message followed by an indented exception backtrace
    (This function is intended to be called within except: blocks)"""
    exc = getExc(indent=0, prefix="", skip=2)
    # print(" "*indent + prefix + '='*30 + '>>')
    warnings.warn("\n".join([msg, exc]), RuntimeWarning, stacklevel=2)
    # print(" "*indent + prefix + '='*30 + '<<')


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLViewWidget(None)
    win.show()
    sys.exit(app.exec_())