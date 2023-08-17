import numpy as np
import OpenGL.GL as gl
from pathlib import Path
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .BufferObject import VAO, VBO, EBO, c_void_p
from .MeshData import Material
from .texture import Texture2D
from .light import LightMixin, light_fragment_shader

BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLSurfacePlotItem']


class GLSurfacePlotItem(GLGraphicsItem, LightMixin):

    def __init__(
        self,
        zmap = None,
        x_size = 10, # scale width to this size
        material = dict(),
        lights = list(),
        glOptions = 'translucent',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._zmap = None
        self._shape = (0, 0)
        self._x_size = x_size
        self._vert_update_flag = False
        self._indice_update_flag = False
        self._vertexes = None
        self._normals = None
        self._indices = None
        self.scale_ratio = 1
        self.setData(zmap)

        # material
        self.setMaterial(material)

        # light
        self.addLight(lights)

    def setData(self, zmap=None):
        if zmap is not None:
            self.update_vertexs(np.array(zmap, dtype=np.float32))

        self.update()

    def update_vertexs(self, zmap):
        self._vert_update_flag = True
        h, w = zmap.shape

        # calc vertexes
        if self._shape != zmap.shape:

            self._shape = zmap.shape
            self._indice_update_flag = True

            self.scale_ratio = self._x_size / w
            x_size = self._x_size
            y_size = self.scale_ratio * h
            zmap = zmap * self.scale_ratio
            x = np.linspace(-x_size/2, x_size/2, w, dtype='f4')
            y = np.linspace(y_size/2, -y_size/2, h, dtype='f4')

            xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
            self._vertexes = np.stack([xgrid, ygrid, zmap.astype('f4')], axis=-1).reshape(-1, 3)
            self.xy_size = (x_size, y_size)
        else:
            self._vertexes[:, 2] = zmap.reshape(-1) * self.scale_ratio

        # calc indices
        if self._indice_update_flag:
            self._indices = self.create_indices(w, h)

        # calc normals texture
        v = self._vertexes[self._indices]  # Nf x 3 x 3
        v = np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]) # face Normal Nf(c*r*2) x 3
        v = v.reshape(h-1, 2, w-1, 3).sum(axis=1, keepdims=False)  # r x c x 3
        self._normal_texture = v / np.linalg.norm(v, axis=-1, keepdims=True)

    @staticmethod
    def create_indices(cols, rows):
        cols -= 1
        rows -= 1
        if cols == 0 or rows == 0:
            return
        indices = np.empty((cols*rows*2, 3), dtype=np.uint)
        rowtemplate1 = np.arange(cols).reshape(cols, 1) + np.array([[0     , cols+1, 1]])
        rowtemplate2 = np.arange(cols).reshape(cols, 1) + np.array([[cols+1, cols+2, 1]])
        for row in range(rows):
            start = row * cols * 2
            indices[start:start+cols] = rowtemplate1 + row * (cols+1)
            indices[start+cols:start+(cols*2)] = rowtemplate2 + row * (cols+1)
        return indices

    def initializeGL(self):
        self.shader = Shader(vertex_shader, light_fragment_shader)
        self.vao = VAO()
        self.vbo = VBO([None], [3], usage = gl.GL_DYNAMIC_DRAW)
        self.ebo = EBO(None)
        self.texture = Texture2D(None, flip_x=False, flip_y=True)

    def updateGL(self):
        if not self._vert_update_flag:
            return

        self.vao.bind()
        self.vbo.updateData([0], [self._vertexes])

        self.vbo.setAttrPointer([0], attr_id=[0])
        if self._indice_update_flag:
            self.ebo.updateData(self._indices)
        if self.texture is not None:
            self.texture.delete()
        self.texture.updateTexture(self._normal_texture)
        self._vert_update_flag = False
        self._indice_update_flag = False

    def paint(self, model_matrix=Matrix4x4()):
        if self._shape[0] == 0:
            return
        self.updateGL()
        self.setupGLState()

        self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")

        self._material.set_uniform(self.shader, "material")
        self.setupLight()

        self.texture.bind(0)
        self.shader.set_uniform("norm_texture", self.texture.unit, "sampler2D")
        self.shader.set_uniform("texScale", self.xy_size, "vec2")

        with self.shader:
            self.vao.bind()
            gl.glDrawElements(gl.GL_TRIANGLES, self._indices.size, gl.GL_UNSIGNED_INT, c_void_p(0))

    def setMaterial(self, material):
        if isinstance(material, dict):
            self._material = Material(material)
        elif isinstance(material, Material):
            self._material = material

    def getMaterial(self):
        return self._material


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 model;
uniform vec2 texScale;
uniform sampler2D norm_texture;

void main() {
    vec2 TexCoords = (aPos.xy + texScale/2) / texScale;
    vec3 aNormal = texture(norm_texture, TexCoords).rgb;
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);

    FragPos = vec3(model * vec4(aPos, 1.0));
    gl_Position = view * vec4(FragPos, 1.0);
}
"""
