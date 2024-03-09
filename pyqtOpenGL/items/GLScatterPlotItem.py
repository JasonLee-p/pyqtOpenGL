from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .shader import Shader
from .BufferObject import VAO, VBO
import numpy as np
import OpenGL.GL as gl
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLScatterPlotItem']


class GLScatterPlotItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""

    def __init__(
        self,
        pos = None,
        size = 1,
        color = [1.0, 1.0, 1.0],
        antialias = True,
        glOptions = 'opaque',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.antialias = antialias
        self.setGLOptions(glOptions)

        self._pos = None
        self._color = None
        self._size = None
        self._npoints = 0
        self._gl_update_flag = False
        self.setData(pos, color, size)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = VBO([None, None], [3, 3], usage=gl.GL_DYNAMIC_DRAW)

    def updateVBO(self):
        if not self._gl_update_flag:
            return

        self.vao.bind()
        self.vbo.updateData([0,1], [self._pos, self._color])
        self.vbo.setAttrPointer([0, 1], attr_id=[0, 1])
        self._gl_update_flag = False

    def setData(self, pos=None, color=None, size=None):
        """"
        ====================  ==================================================
        **Arguments:**
        pos                (N,3) array of floats specifying point locations.
        color                 (N,3) array of floats (0.0-1.0) specifying
                              spot colors OR a tuple of floats specifying
                              a single color for all spots.
        size                  a single value to apply to all spots.
        ====================  ==================================================
        """
        if pos is not None:
            self._pos = np.ascontiguousarray(pos, dtype=np.float32)
            self._npoints = int(self._pos.size / 3)
        if size is not None:
            self._size = np.float32(size)
        if color is not None:
            self._color = np.ascontiguousarray(color, dtype=np.float32)
            if self._color.size == 3 and self._npoints > 1:
                self._color = np.tile(self._color, (self._npoints, 1))
        self._gl_update_flag = True
        self.update()

    def paint(self, model_matrix=Matrix4x4()):
        if self._npoints == 0:
            return
        self.updateVBO()
        self.setupGLState()

        if self.antialias:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)

        with self.shader:
            self.shader.set_uniform("size", self._size, "float")
            self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.vao.bind()
            gl.glDrawArrays(gl.GL_POINTS, 0, self._npoints)


vertex_shader = """
#version 330 core

uniform float size;
uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec3 iColor;

out vec3 oColor;

void main() {
    // 根据 camPos 和 iPos 计算出距离
    gl_Position = proj * view * model * vec4(iPos, 1.0);
    float distance = gl_Position.z / 1.;
    gl_PointSize = 100 * size / distance;
    oColor = iColor;

}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;
in vec3 oColor;

void main() {
    FragColor = vec4(oColor, 1.0);
}
"""