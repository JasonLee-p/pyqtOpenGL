import numpy as np
import OpenGL.GL as gl
from pathlib import Path
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .BufferObject import VAO, VBO, EBO, c_void_p
from .MeshData import surface
from .texture import Texture2D

BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLTextureSurfaceItem']


class GLTextureSurfaceItem(GLGraphicsItem):

    def __init__(
        self,
        zmap = None,
        texture = None,
        x_size = 10, # scale width to this size
        opacity = 1.,
        glOptions = 'translucent',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._zmap = None
        self._img = None
        self._shape = (0, 0)
        self._x_size = x_size
        self._vert_update_flag = False
        self._tex_update_flag = False
        self._indice_update_flag = False
        self._vertexes = None
        self._indices = None
        self._opacity = opacity
        self.scale_ratio = 1
        self.setData(zmap, texture, opacity)

    def setData(self, zmap=None, texture=None, opacity=None):

        if zmap is not None:
            self.update_vertexs(np.array(zmap, dtype=np.float32))

        if isinstance(texture, np.ndarray):
            self._img = texture.astype(np.uint8)
            self._tex_update_flag = True

        if opacity is not None:
            self._opacity = opacity
        self.update()

    def update_vertexs(self, zmap):
        self._vert_update_flag = True
        h, w = zmap.shape
        self.scale_ratio = self._x_size / w

        # calc vertexes
        if self._shape != zmap.shape:

            self._shape = zmap.shape
            self._indice_update_flag = True

            self.xy_size = (self._x_size, self.scale_ratio * h)
            # left top: (-x_size/2, -y_size/2, 0)
            self._vertexes, self._indices = surface(zmap, self.xy_size)

        else:
            self._vertexes[:, 2] = zmap.reshape(-1) * self.scale_ratio

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = VBO([None], [3], usage = gl.GL_DYNAMIC_DRAW)
        self.ebo = EBO(None)
        self.texture = Texture2D(None, flip_x=False, flip_y=True)

    def updateGL(self):
        if not self._vert_update_flag and not self._tex_update_flag:
            return

        self.vao.bind()
        if self._vert_update_flag:
            self.vbo.updateData([0], [self._vertexes])
            self.vbo.setAttrPointer([0], attr_id=[0])
            self._vert_update_flag = False

        if self._indice_update_flag:
            self.ebo.updateData(self._indices)
            self._indice_update_flag = False

        if self._tex_update_flag:
            if self.texture is not None:
                self.texture.delete()
            self.texture.updateTexture(self._img)
            self._tex_update_flag = False


    def paint(self, model_matrix=Matrix4x4()):
        if self._shape[0] == 0 or self._img is None:
            return
        self.updateGL()
        self.setupGLState()

        self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("opacity", self._opacity, "float")

        self.texture.bind(0)
        self.shader.set_uniform("rgb_texture", self.texture.unit, "sampler2D")
        self.shader.set_uniform("texScale", self.xy_size, "vec2")

        with self.shader:
            self.vao.bind()
            gl.glDrawElements(gl.GL_TRIANGLES, self._indices.size, gl.GL_UNSIGNED_INT, c_void_p(0))


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 oColor;

uniform mat4 view;
uniform mat4 model;
uniform vec2 texScale;
uniform sampler2D rgb_texture;

void main() {
    gl_Position = view * model * vec4(aPos, 1.0);
    vec2 TexCoord = (aPos.xy + texScale/2) / texScale;
    oColor = texture(rgb_texture, TexCoord).rgb;
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
