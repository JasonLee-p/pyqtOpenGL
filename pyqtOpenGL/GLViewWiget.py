from OpenGL.GL import *  # noqa
from math import radians, cos, sin, tan
from .camera import Camera
from .functions import mkColor
from PyQt5 import QtCore, QtWidgets
from .transform3d import Matrix4x4, Quaternion, Vector3

class GLViewWidget(QtWidgets.QOpenGLWidget):

    def __init__(
        self,
        cam_position = Vector3(0., 0., 10.),
        yaw = 0.,
        pitch = 0.,
        fov = 45.,
        bg_color = (0.2, 0.3, 0.3, 1.),
        parent=None,
    ):
        """
        Basic widget for displaying 3D data
          - Rotation/scale controls
        """
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.camera = Camera(cam_position, yaw, pitch, fov)
        self.bg_color = bg_color
        self.items = []

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

    def reset(self):
        self.camera.set_params(Vector3(0., 0., 10.), 0, 0, 45)

    def addItem(self, item):
        self.items.append(item)
        item.setView(self)
        self.update()

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

    def paintGL(self):
        """
        viewport specifies the arguments to glViewport. If None, then we use self.opts['viewport']
        region specifies the sub-region of self.opts['viewport'] that should be rendered.
        Note that we may use viewport != self.opts['viewport'] when exporting.
        """
        glClearColor(*self.bg_color)
        glClear( GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT )
        self.drawItems()

    def drawItems(self):
        items = [x for x in self.items if x.parentItem() is None]
        items.sort(key=lambda a: a.depthValue())
        for it in items:
            try:
                it.drawItemTree()
            except:
                printExc()
                print("Error while drawing item %s." % str(it))

    def pixelSize(self):
        """
        Return the approximate size of a screen pixel at the location pos
        Pos may be a Vector or an (N,3) array of locations
        """
        dist = self.camera.pos.z
        fov = self.camera.fov
        return dist * 2. * tan(0.5 * radians(fov)) / self.width()

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.mousePos = lpos

    def mouseMoveEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        diff = lpos - self.mousePos
        diff.setY(-diff.y())
        self.mousePos = lpos

        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.camera.pan(diff.x(), diff.y(), 0)
            else:
                self.camera.orbit(diff.x(), -diff.y())
        elif ev.buttons() == QtCore.Qt.MouseButton.MiddleButton:
            if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
                self.camera.pan(diff.x(), 0, diff.y())
            else:
                self.camera.pan(diff.x(), diff.y(), 0)
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