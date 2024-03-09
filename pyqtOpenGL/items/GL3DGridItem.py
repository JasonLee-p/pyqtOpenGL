import numpy as np
import OpenGL.GL as gl
from pathlib import Path
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .BufferObject import VAO, VBO, EBO, c_void_p
from .MeshData import grid3d

BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GL3DGridItem']


class GL3DGridItem(GLGraphicsItem):

    def __init__(
        self,
        grid = None,
        color = (1, 1, 1),
        opacity = 1.,
        lineWidth = 2,
        glOptions = 'translucent',
        parentItem = None,
        fill = False
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._grid = grid
        self._fill = fill
        self._lineWidth = lineWidth
        self._shape = (0, 0)
        self._vert_update_flag = False
        self._indice_update_flag = False
        self._vertexes = None
        self._colors = None
        self._indices = None
        self._opacity = opacity
        self.setData(grid, color, opacity)

    def setData(self, grid=None, color=None, opacity=None):
        if grid is not None:
            self.update_vertexs(grid)
        if self._vertexes is None:
            return

        if color is not None:
            color = np.array(color, dtype=np.float32)
            if color.size == 3:
                self._color = np.tile(color, (self._vertexes.shape[0], 1))
            else:
                self._color = color
            self._vert_update_flag = True

        assert self._vertexes.size == self._color.size, \
            "vertexes and colors must have same size"

        if opacity is not None:
            self._opacity = opacity
        self.update()

    def update_vertexs(self, grid):
        self._vert_update_flag = True
        grid = np.array(grid, dtype=np.float32)

        # calc vertexes
        if self._shape != grid.shape:
            self._shape = grid.shape
            self._indice_update_flag = True

            self._vertexes, self._indices = grid3d(grid)

        else:
            self._vertexes = grid.reshape(-1, 3)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = VBO([None, None], [3, 3], usage = gl.GL_DYNAMIC_DRAW)
        self.ebo = EBO(None)

    def updateGL(self):
        if not self._vert_update_flag:
            return

        self.vao.bind()
        self.vbo.updateData([0, 1], [self._vertexes, self._color])
        self.vbo.setAttrPointer([0, 1], attr_id=[0, 1])

        if self._indice_update_flag:
            self.ebo.updateData(self._indices)

        self._vert_update_flag = False
        self._indice_update_flag = False

    def paint(self, model_matrix=Matrix4x4()):
        if self._shape[0] == 0:
            return
        self.updateGL()
        self.setupGLState()

        gl.glLineWidth(self._lineWidth)
        self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
        self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("opacity", self._opacity, "float")

        with self.shader:
            self.vao.bind()
            if not self._fill:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

            gl.glDrawElements(
                gl.GL_QUADS,
                self._indices.size,
                gl.GL_UNSIGNED_INT,
                c_void_p(0)
            )

            if not self._fill:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 oColor;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

void main() {
    gl_Position = proj * view * model * vec4(aPos, 1.0);
    oColor = aColor;
}
"""


fragment_shader = """
#version 330 core

uniform float opacity;

in vec3 oColor;
out vec4 fragColor;

void main() {
    fragColor = vec4(oColor, opacity);
}
"""
