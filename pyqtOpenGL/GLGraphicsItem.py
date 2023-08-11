from OpenGL.GL import *  # noqa
from OpenGL import GL
from math import radians
from PyQt5 import QtCore
from .transform3d import Matrix4x4, Quaternion
import numpy as np


GLOptions = {
    'opaque': {
        GL_DEPTH_TEST: True,
        GL_BLEND: False,
        GL_ALPHA_TEST: False,
        GL_CULL_FACE: False,
    },
    'translucent': {
        GL_DEPTH_TEST: True,
        GL_BLEND: True,
        GL_ALPHA_TEST: False,
        GL_CULL_FACE: False,
        'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
    },
    'additive': {
        GL_DEPTH_TEST: False,
        GL_BLEND: True,
        GL_ALPHA_TEST: False,
        GL_CULL_FACE: False,
        'glBlendFunc': (GL_SRC_ALPHA, GL_ONE),
    },
}


class GLGraphicsItem(QtCore.QObject):

    def __init__(
        self,
        parentItem: 'GLGraphicsItem' = None,
        depthValue: int = 0,
    ):
        super().__init__()
        self.__parent: GLGraphicsItem | None = None
        self.__view = None
        self.__children: set[GLGraphicsItem] = set()
        self.__transform = Matrix4x4()
        self.__visible = True
        self.__initialized = False
        self.setParentItem(parentItem)
        self.setDepthValue(depthValue)
        self.__glOpts = {}

    def setParentItem(self, item: 'GLGraphicsItem'):
        """Set this item's parent in the scenegraph hierarchy."""
        if item is None:
            return

        if self.__parent is not None:
            self.__parent.__children.remove(self)

        item.__children.add(self)
        self.__parent = item

    def addChildItem(self, item):
        item.setParentItem(self)

    def parentItem(self):
        """Return a this item's parent in the scenegraph hierarchy."""
        return self.__parent

    def childItems(self):
        """Return a list of this item's children in the scenegraph hierarchy."""
        return list(self.__children)

    def setGLOptions(self, opts):
        """
        Set the OpenGL state options to use immediately before drawing this item.
        (Note that subclasses must call setupGLState before painting for this to work)

        The simplest way to invoke this method is to pass in the name of
        a predefined set of options (see the GLOptions variable):

        ============= ======================================================
        opaque        Enables depth testing and disables blending
        translucent   Enables depth testing and blending
                      Elements must be drawn sorted back-to-front for
                      translucency to work correctly.
        additive      Disables depth testing, enables blending.
                      Colors are added together, so sorting is not required.
        ============= ======================================================

        It is also possible to specify any arbitrary settings as a dictionary.
        This may consist of {'functionName': (args...)} pairs where functionName must
        be a callable attribute of OpenGL.GL, or {GL_STATE_VAR: bool} pairs
        which will be interpreted as calls to glEnable or glDisable(GL_STATE_VAR).

        For example::

            {
                GL_ALPHA_TEST: True,
                GL_CULL_FACE: False,
                'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
            }


        """
        if isinstance(opts, str):
            opts = GLOptions[opts]
        self.__glOpts = opts.copy()

    def updateGLOptions(self, opts):
        """
        Modify the OpenGL state options to use immediately before drawing this item.
        *opts* must be a dictionary as specified by setGLOptions.
        Values may also be None, in which case the key will be ignored.
        """
        self.__glOpts.update(opts)

    def setView(self, v):
        self.__view = v

    def view(self):
        return self.__view

    def setDepthValue(self, value):
        """
        Sets the depth value of this item. Default is 0.
        This controls the order in which items are drawn--those with a greater depth value will be drawn later.
        Items with negative depth values are drawn before their parent.
        (This is analogous to QGraphicsItem.zValue)
        The depthValue does NOT affect the position of the item or the values it imparts to the GL depth buffer.
        """
        self.__depthValue = value

    def depthValue(self):
        """Return the depth value of this item. See setDepthValue for more information."""
        return self.__depthValue

    def setTransform(self, tr):
        """Set the local transform for this object.

        Parameters
        ----------
        tr : transform3d.Matrix4x4
            Tranformation from the local coordinate system to the parent's.
        """
        self.__transform = Matrix4x4(tr)
        self.update()

    def resetTransform(self):
        """Reset this item's transform to an identity transformation."""
        self.__transform = Matrix4x4()
        self.update()

    def transform(self):
        """Return this item's transform object."""
        return self.__transform

    def viewTransform(self):
        """Return the transform mapping this item's local coordinate system to the
        view coordinate system."""
        tr = self.__transform
        p = self
        while True:
            p = p.parentItem()
            if p is None:
                break
            tr = p.transform() * tr
        return tr

    def setVisible(self, vis):
        """Set the visibility of this item."""
        self.__visible = vis
        self.update()

    def visible(self):
        """Return True if the item is currently set to be visible.
        Note that this does not guarantee that the item actually appears in the
        view, as it may be obscured or outside of the current view area."""
        return self.__visible

    def setupGLState(self):
        """
        This method is responsible for preparing the GL state options needed to render
        this item (blending, depth testing, etc). The method is called immediately before painting the item.
        """
        for k,v in self.__glOpts.items():
            if v is None:
                continue
            if isinstance(k, str):
                func = getattr(GL, k)
                func(*v)
            else:
                if v is True:
                    glEnable(k)
                else:
                    glDisable(k)

    def initialize(self):
        if not self.__initialized:
            if self.view() is None:
                self.setView(self.__parent.view())
            self.initializeGL()
            self.__initialized = True

    @property
    def isInitialized(self):
        return self.__initialized

    def drawItemTree(self, model_matrix=Matrix4x4()):
        model_matrix = model_matrix * self.transform()
        if not self.visible():
            return

        self.initialize()
        self.paint(model_matrix)
        for child in self.__children:
            child.drawItemTree(model_matrix)

    def update(self):
        """
        Indicates that this item needs to be redrawn, and schedules an update
        with the view it is displayed in.
        """
        v = self.view()
        if v is None:
            return
        v.update()

    def proj_view_matrix(self) -> Matrix4x4:
        return self.__view.get_proj_view_matrix()

    def proj_matrix(self) -> Matrix4x4:
        return self.__view.get_proj_matrix()

    def view_matrix(self) -> Matrix4x4:
        return self.__view.get_view_matrix()

    def moveTo(self, x, y, z):
        """
        Move the object to the absolute position (x,y,z) in its parent's coordinate system.
        """
        self.__transform.moveto(x, y, z)

    def translate(self, dx, dy, dz, local=False):
        """
        Translate the object by (*dx*, *dy*, *dz*) in its parent's coordinate system.
        If *local* is True, then translation takes place in local coordinates.
        """
        self.__transform.translate(dx, dy, dz, local=local)

    def rotate(self, angle, x, y, z, local=False):
        """
        Rotate the object around the axis specified by (x,y,z).
        *angle* is in degrees.

        """
        self.__transform.rotate(angle, x, y, z, local=local)

    def scale(self, x, y, z, local=True):
        """
        Scale the object by (*dx*, *dy*, *dz*) in its local coordinate system.
        If *local* is False, then scale takes place in the parent's coordinates.
        """
        self.__transform.scale(x, y, z, local=local)

    # The following methods must be implemented by subclasses:
    def initializeGL(self):
        """
        Called once in GLViewWidget.paintGL.
        The widget's GL context is made current before this method is called.
        (So this would be an appropriate time to generate lists, upload textures, etc.)
        """
        raise NotImplementedError()

    def paint(self, model_matrix=Matrix4x4()):
        """
        Called by the GLViewWidget to draw this item.
        The widget's GL context is made current before this method is called.
        It is the responsibility of the item to set up its own modelview matrix,
        but the caller will take care of pushing/popping.
        """
        self.setupGLState()
        raise NotImplementedError()


class Material:
    def __init__(self):
        pass