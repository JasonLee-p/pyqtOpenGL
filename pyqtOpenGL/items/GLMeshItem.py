import OpenGL.GL as gl
import numpy as np
from .shader import Shader
from .BufferObject import VBO, EBO, VAO
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .MeshData import cone, direction_matrixs, vertex_normal

__all__ = ['GLMeshItem']


class GLMeshItem(GLGraphicsItem):
    """
    Displays three lines indicating origin and orientation of local coordinate system.
    """

    def __init__(
        self,
        pos = [[0, 0, 0]],  # nx3
        vertexes = None,
        indices = None,
        color = [1., 1., 1.],
        size = 1.,
        glOptions = 'opaque',
        calcNormals = True,
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._vertices = vertexes
        self._indices = indices
        self._pos = None
        self._calcNormals = calcNormals
        if self._calcNormals:
            self._normals = vertex_normal(self._vertices, self._indices)
        else:
            self._normals = None
        self.setData(pos, color, size)

    def setData(self, pos=None, color=None, size=None):
        if pos is not None:
            self._pos = np.array(pos, dtype=np.float32).reshape(-1, 3)
            self._gl_update_flag = True
        if color is not None:
            self._color = np.array(color, dtype=np.float32)
            if self._color.size == 3 and self._pos is not None:
                self._color = np.tile(self._color, (self._pos.shape[0], 1))
            self._gl_update_flag = True
        if size is not None:
            self._size = size
        self.update()

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        # cone
        self.vbo_mesh = VBO(
            [self._vertices, self._normals],
            [3, 3], gl.GL_STATIC_DRAW,
        )
        if self._calcNormals:
            self.vbo_mesh.setAttrPointer([0, 1], divisor=0, attr_id=[0, 1])
        else:
            self.vbo_mesh.setAttrPointer([0], divisor=0, attr_id=[0])

        self.ebo = EBO(self._indices)

        # pos
        self.vbo_pos = VBO([self._pos, self._color], [3,3], usage=gl.GL_DYNAMIC_DRAW)
        self.vbo_pos.setAttrPointer([0, 1], [2, 3], divisor=1)

    def updateGL(self):
        if not self._gl_update_flag:
            return

        self.vao.bind()
        self.vbo_pos.updateData([0, 1], [self._pos, self._color])
        self.vbo_pos.setAttrPointer([0, 1], [2, 3], divisor=1)
        self._gl_update_flag = False

    def paint(self, model_matrix=Matrix4x4()):
        if self._pos is None:
            return
        self.updateGL()
        self.setupGLState()

        self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("size", self._size, "float")

        self.shader.set_uniform("lightPos", Vector3([300, 200.0, 200.0]), "vec3")
        self.shader.set_uniform("lightColor", Vector3([1.0, 1.0, 1.0]), "vec3")
        self.shader.set_uniform("calcNormal", self._calcNormals, "bool")

        self.vao.bind()
        self.ebo.bind()

        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        with self.shader:
            gl.glDrawElementsInstanced(
                gl.GL_TRIANGLES,
                self._indices.size,
                gl.GL_UNSIGNED_INT, None,
                self._pos.shape[0],
            )
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)


# vertex_shader = """
# #version 330 core

# uniform mat4 model;
# uniform mat4 view;
# uniform float size;

# layout (location = 0) in vec3 iPos;
# layout (location = 2) in vec3 stPos;
# layout (location = 3) in vec3 iColor;
# out vec3 oColor;

# void main() {
#     gl_Position =  view * model * vec4(stPos + iPos*size, 1.0);
#     oColor = iColor;
# }
# """

# fragment_shader = """
# #version 330 core

# in vec3 oColor;
# out vec4 fragColor;

# void main() {
#     fragColor = vec4(oColor, 1.0f);
# }
# """

vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform float size;
uniform bool calcNormal;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec3 iNormal;
layout (location = 2) in vec3 stPos;
layout (location = 3) in vec3 iColor;

out vec3 FragPos;
out vec3 Normal;
out vec3 oColor;

void main() {
    oColor = iColor;
    FragPos = vec3(model * vec4(stPos + iPos*size, 1.0));
    if (calcNormal){
        Normal = mat3(transpose(inverse(model))) * iNormal;
    } else {
        Normal = vec3(0, 0, 0);
    }
    gl_Position = view * vec4(FragPos, 1.0);
}
"""


fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 oColor;

uniform vec3 lightColor;
uniform vec3 lightPos;
uniform bool calcNormal;

void main() {
    if (calcNormal){
        // ambient
        float ambientStrength = 0.4;
        vec3 ambient = ambientStrength * lightColor * oColor;

        // diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = lightColor * (diff * oColor);

        vec3 result = ambient + diffuse;
        FragColor = vec4(result, 1.0);
    } else {
        FragColor = vec4(oColor, 1.0f);
    }
}
"""