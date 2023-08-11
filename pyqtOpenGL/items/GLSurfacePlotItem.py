import numpy as np
import OpenGL.GL as gl
from pathlib import Path
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .BufferObject import VAO, VBO, EBO, c_void_p
from .MeshData import Material
from .GLBoxItem import GLBoxItem
from time import sleep
BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLSurfacePlotItem']


class GLSurfacePlotItem(GLGraphicsItem):

    def __init__(
        self,
        zmap = None,
        x_size = 10, # scale width to this size
        calc_normals = True,
        material = {
            "COLOR_AMBIENT": [1.0, 1.0, 1.0],
            "COLOR_DIFFUSE": [0.5, 0.5, 0.5],
            "COLOR_SPECULAR": [1.0, 1.0, 1.0],
            "SHININESS": 10.0,
            "OPACITY": 1.0,
        },
        glOptions = 'opaque',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        self._zmap = None
        self._shape = (0, 0)
        self._x_size = x_size
        self._calc_normals = calc_normals
        self._vert_update_flag = False
        self._indice_update_flag = False
        self._vertexes = None
        self._normals = None
        self._indices = None
        self.setData(zmap)
        self._material = Material(
            material["COLOR_AMBIENT"],
            material["COLOR_DIFFUSE"],
            material["COLOR_SPECULAR"],
            material["SHININESS"],
            material["OPACITY"],
        )
        self.lightPos = Vector3([4, 0.0, -5.0])
        self.lightColor = Vector3([2.0, 2.0, 2.0])

        self.rotate(-90, 1, 0, 0)

        self.lightBox = GLBoxItem(size=(0.2, 0.2, 0.2), color=(10, 0, 0))
        self.lightBox.moveTo(*self.lightPos)
        self.lightBox.rotate(90, 1, 0, 0)
        self.lightBox.translate(0, 0, 0.1, True)
        self.addChildItem(self.lightBox)


    def setData(self, zmap=None):
        if zmap is not None:
            self.update_vertexs(np.array(zmap, dtype=np.float32))

        # if colors is not None:
        #     self._colors = np.ascontiguousarray(colors, dtype=np.float32)
        #     if self._colors.size == 3 and self._vertexes.size > 3:
        #         self._color = np.tile(self._color, (self._shape[0]*self._shape[1], 1))

        self.update()

    def update_vertexs(self, zmap):
        self._vert_update_flag = True

        if self._shape != zmap.shape:

            self._shape = zmap.shape
            self._indice_update_flag = True

            h, w = zmap.shape
            x_size = self._x_size
            y_size = x_size / w * h
            x = np.linspace(-x_size/2, x_size/2, w, dtype='f2')
            y = np.linspace(y_size/2, -y_size/2, h, dtype='f2')

            xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
            self._vertexes = np.stack([xgrid, ygrid, zmap.astype('f2')], axis=-1).reshape(-1, 3)

        else:
            self._vertexes[:, 2] = zmap.reshape(-1)

        if self._indice_update_flag:
            h, w = zmap.shape
            self._indices = self.create_indices(w, h)
            if self._calc_normals and self._normals is None:
                # list mapping each vertex index to a list of face indexes that use the vertex.
                self._vert_to_face = [[] for i in range(len(self._vertexes))]
                for i in range(self._indices.shape[0]):
                    face = self._indices[i]
                    for ind in face:
                        self._vert_to_face[ind].append(i)
        s = self._vertexes[self._indices]
        if self._calc_normals and self._normals is None:
            self._normals = self.vert_normals()

    @staticmethod
    def create_indices(cols, rows):
        cols -= 1
        rows -= 1
        if cols == 0 or rows == 0:
            return
        indices = np.empty((cols*rows*2, 3), dtype=np.uint)
        rowtemplate1 = np.arange(cols).reshape(cols, 1) + np.array([[0, 1, cols+1]])
        rowtemplate2 = np.arange(cols).reshape(cols, 1) + np.array([[cols+1, 1, cols+2]])
        for row in range(rows):
            start = row * cols * 2
            indices[start:start+cols] = rowtemplate1 + row * (cols+1)
            indices[start+cols:start+(cols*2)] = rowtemplate2 + row * (cols+1)
        return indices

    def face_normals(self):
        if self._indices is None:
            return
        v = self._vertexes[self._indices]  # Nf x 3 x 3
        return  np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0])  # Nf x 3

    def vert_normals(self):
        """
        Return an array of normal vectors.
        By default, the array will be (N, 3) with one entry per unique vertex in the mesh.
        """
        faceNorms = self.face_normals()
        _normals = np.empty(self._vertexes.shape, dtype='f2')
        for vi in range(self._vertexes.shape[0]):
            faces = self._vert_to_face[vi]
            if len(faces) == 0:
                _normals[vi] = (0,0,0)
                continue
            norms = faceNorms[faces]  ## get all face normals
            norm = norms.sum(axis=0)       ## sum normals
            norm /= (norm**2).sum()**0.5  ## and re-normalize
            _normals[vi] = norm
        return _normals

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = None
        self.ebo = None

    def updateVBO(self):
        if not self._vert_update_flag:
            return

        self.vao.bind()
        if self.vbo is None:
            self.vbo = VBO(
                [self._vertexes, self._normals],
                [3, 3],
                usage = gl.GL_STATIC_DRAW
            )
        else:
            self.vbo.loadd([self._vertexes, self._normals])
            # self.vbo.loadData([0,1], [self._vertexes, self._normals])
        self.vbo.setAttrPointer([0, 1], attr_id=[0, 1])

        if self._indice_update_flag:
            if self.ebo is not None:
                self.ebo.delete()
            self.ebo = EBO(self._indices)

        self._vert_update_flag = False
        self._indice_update_flag = False


    def paint(self, model_matrix=Matrix4x4()):
        if self._shape[0] == 0:
            return
        self.updateVBO()
        self.setupGLState()
        gl.glDisable(gl.GL_LINE_SMOOTH)

        for idx, texture in enumerate(self._material.textures):
            texture.bind(idx)
            self.shader.set_uniform(name=texture.type, data=idx, type="Sampler2D")
        self.shader.set_uniform("opacity", self._material.opacity, "float")
        self.shader.set_uniform("objColor", self._material.diffuse, "vec3")

        self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("lightPos", self.lightPos, "vec3")
        self.shader.set_uniform("lightColor", self.lightColor, "vec3")

        self.shader.use()
        self.vao.bind()
        gl.glDrawElements(gl.GL_TRIANGLES, self._indices.size, gl.GL_UNSIGNED_INT, c_void_p(0))
        self.shader.unuse()


    def setLight(self, pos=None, color=None, transform: Matrix4x4=None):
        if pos is not None:
            self.lightPos = Vector3(pos)
        if color is not None:
            self.lightColor = Vector3(color)
        if transform is not None:
            self.lightPos = Vector3(transform * self.lightPos)
            # print(transform * self.lightPos.xyz)
            self.lightBox.moveTo(*self.lightPos)
        self.update()


vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 model;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = view * vec4(FragPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform float opacity;
uniform vec3 objColor;

void main() {
    // ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor * objColor;

    // diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = lightColor * (diff * objColor);

    vec3 result = ambient + diffuse;
    FragColor = vec4(result, opacity);
}
"""
