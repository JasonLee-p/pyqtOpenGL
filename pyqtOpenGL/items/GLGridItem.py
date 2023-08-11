from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .BufferObject import VAO, VBO
import numpy as np
import OpenGL.GL as gl

__all__ = ['GLGridItem']


def make_grid_data(size, spacing):
    x, y = size
    dx, dy = spacing
    xvals = np.arange(-x/2., x/2. + dx*0.001, dx, dtype=np.float32)
    yvals = np.arange(-y/2., y/2. + dy*0.001, dy, dtype=np.float32)

    xlines = np.stack(
        np.meshgrid(xvals, [yvals[0], yvals[-1]], indexing='ij'),
        axis=2
    ).reshape(-1, 2)
    ylines = np.stack(
        np.meshgrid([xvals[0], xvals[-1]], yvals, indexing='xy'),
        axis=2
    ).reshape(-1, 2)
    data = np.concatenate([xlines, ylines], axis=0)
    data = np.pad(data, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)
    return data


class GLGridItem(GLGraphicsItem):
    """
    Displays xy plane.
    """
    def __init__(
        self,
        size = (1., 1.),
        spacing = (1.,1.),
        color = (1.,1.,1.,0.4),
        lineWidth = 1,
        antialias = True,
        glOptions = 'translucent',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.__size = size
        self.__color = np.array(color, dtype=np.float32).clip(0, 1)
        self.__lineWidth = lineWidth
        self.antialias = antialias
        self.setGLOptions(glOptions)
        self.line_vertices = make_grid_data(self.__size, spacing)
        x, y = size
        self.plane_vertices = np.array([
            -x/2., -y/2., 0,
            -x/2.,  y/2., 0,
            x/2.,  -y/2., 0,
            x/2.,   y/2., 0,
        ], dtype=np.float32)
        self.rotate(90, 1, 0, 0)
        self.setDepthValue(-1)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)

        self.vao = VAO()

        self.vbo1 = VBO(
            data = [self.line_vertices, self.plane_vertices],
            size = [3, 3],
        )

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()

        if self.antialias:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glLineWidth(self.__lineWidth)

        self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")

        with self.shader:
            self.vao.bind()
            self.shader.set_uniform("objColor1", self.__color, "vec4")
            self.vbo1.setAttrPointer(1, attr_id=0)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

            gl.glDisable(gl.GL_BLEND)
            self.shader.set_uniform("objColor1", Vector3([0, 0, 0, 1]), "vec4")
            self.vbo1.setAttrPointer(0, attr_id=0)
            gl.glDrawArrays(gl.GL_LINES, 0, len(self.line_vertices))


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;

layout (location = 0) in vec3 iPos;

void main() {
    gl_Position = view * model * vec4(iPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

uniform vec4 objColor1;


void main() {
    FragColor = objColor1;
}
"""