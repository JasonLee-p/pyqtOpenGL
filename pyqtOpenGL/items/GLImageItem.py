from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .shader import Shader
from .BufferObject import VAO, VBO
from .texture import Texture2D
import numpy as np
import OpenGL.GL as gl
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLImageItem']


class GLImageItem(GLGraphicsItem):
    """Display Image."""

    def __init__(
        self,
        img = None,
        left_bottom = (0, 0),  # 左下角坐标 0 ~ 1
        width_height = (1, 1),  # 宽高 0 ~ 1
        glOptions = 'opaque',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._tex_update_flag = False
        self._vbo_update_flag = False
        self._img = None
        self.texture = None
        self.left_bottom = None
        self.width_height = None
        self.vertices = np.array( [
            # 顶点坐标             # texcoord
            -1, -1, 0,   0.0, 0.0,
             1, -1, 0,   1.0, 0.0,
             1,  1, 0,   1.0, 1.0,
             1,  1, 0,   1.0, 1.0,
            -1,  1, 0,   0.0, 1.0,
            -1, -1, 0,   0.0, 0.0,
        ], dtype=np.float32).reshape(-1, 5)
        self.setData(img=img, left_bottom=left_bottom, width_height=width_height)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = VBO([self.vertices], [[3, 2]], usage=gl.GL_STATIC_DRAW)
        self.vbo.setAttrPointer([0], attr_id=[[0,1]])
        self.texture = Texture2D(None, flip_y=True)

    def updateGL(self):
        if not self._tex_update_flag and not self._vbo_update_flag:
            return
        if self._tex_update_flag:
            self.texture.updateTexture(self._img)
            self._tex_update_flag = False
        if self._vbo_update_flag:
            self.vbo.updateData([0], [self.vertices])
            self._vbo_update_flag = False

    def setData(self, img=None, left_bottom=None, width_height=None):
        if isinstance(img, np.ndarray):
            self._img = img.astype(np.uint8)
            self._tex_update_flag = True

        if left_bottom is not None or width_height is not None:
            self._vbo_update_flag = True
            if left_bottom is not None:
                self.left_bottom = left_bottom
            if width_height is not None:
                self.width_height = width_height

            l, b = self.left_bottom
            l = l * 2 - 1
            b = b * 2 - 1
            w, h = self.width_height
            w, h = w * 2, h * 2

            self.vertices[:, :2] = np.array([
                [l, b],
                [l + w, b],
                [l + w, b + h],
                [l + w, b + h],
                [l, b + h],
                [l, b]
            ])

        self.update()

    def paint(self, model_matrix=Matrix4x4()):
        if self._img is None:
            return
        self.updateGL()
        self.setupGLState()
        self.texture.bind()
        self.shader.set_uniform("texture1", self.texture, "sample2D")

        with self.shader:
            self.vao.bind()
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)


vertex_shader = """
#version 330 core

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec2 iTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(iPos, 1.0);
    TexCoord =iTexCoord;
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D texture1;

void main() {
    FragColor = vec4(texture(texture1, TexCoord).rgb, 1.0);
}
"""