from pathlib import Path
import numpy as np
import OpenGL.GL as gl
from ..GLGraphicsItem import GLGraphicsItem
from .GLModelItem import GLModelItem
from .GLSurfacePlotItem import GLSurfacePlotItem
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLGelSimItem']


class GLGelSimItem(GLGraphicsItem):
    """ Displays a GelSlim model with a surface plot on top of it."""

    def __init__(
        self,
        lights: list,
        parentItem=None,
    ):
        super().__init__(parentItem=parentItem)

        self.gelslim_base = GLModelItem(
            path = BASE_DIR / "resources/objects/GelSlim_obj/GelSlim.obj",
            lights = lights,
            glOptions = "translucent_cull",
            texcoords_scale = 40,
            parentItem = self,
        )
        self.gelslim_base.setPaintOrder([1, 0])

        self.gelslim_gel = GLSurfacePlotItem(
            zmap = np.zeros((10, 10), dtype=np.float32),
            x_size = 12,
            lights = lights,
            glOptions = "translucent",
            parentItem = self,
        )
        self.gelslim_gel.setMaterial(self.gelslim_base.getMaterial(0))
        self.gelslim_gel.setDepthValue(1)


    def setDepth(self, zmap):
        self.gelslim_gel.setData(zmap)