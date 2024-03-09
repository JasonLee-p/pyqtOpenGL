import OpenGL.GL as gl
import numpy as np
from .shader import Shader
from .BufferObject import VBO, EBO, VAO
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .MeshData import cone, direction_matrixs

__all__ = ['GLArrowPlotItem']


class GLArrowPlotItem(GLGraphicsItem):
    """
    Displays Arrows.
    """

    def __init__(
        self,
        start_pos = None,
        end_pos = None,
        color = [1., 1., 1.],
        tip_size = [0.1, 0.2],  # radius, height
        tip_pos = 0,  # bias of tip position, end + tip_pos * (end - start)/norm(end - start)
        width = 1.,
        antialias=True,
        glOptions='opaque',
        parentItem=None
    ):
        super().__init__(parentItem=parentItem)
        self.antialias = antialias
        self.setGLOptions(glOptions)
        self._cone_vertices, self._cone_indices = cone(tip_size[0]*width,
                                                       tip_size[1]*width)
        self._cone_vertices += np.array([0, 0, tip_pos], dtype=np.float32)
        self._width = width
        self._st_pos = None
        self._end_pos = None
        self._color = None
        self._num = 0
        self._gl_update_flag = False
        self.setData(start_pos, end_pos, color)

    def setSize(self, x=None, y=None, z=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z.
        """
        x = x if x is not None else self.__size.x
        y = y if y is not None else self.__size.y
        z = z if z is not None else self.__size.z
        self.__size = Vector3(x,y,z)
        self.update()

    def size(self):
        return self.__size.xyz

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader, geometry_shader)
        self.shader_cone = Shader(vertex_shader_cone, fragment_shader)
        self.vao = VAO()
        # cone
        self.vbo_cone = VBO(
            [self._cone_vertices],
            [3], gl.GL_STATIC_DRAW,
        )
        self.vbo_cone.setAttrPointer(0, divisor=0, attr_id=7)
        self.ebo_cone = EBO(self._cone_indices)

        # line
        self.vbo_shaft = VBO(
            [None, None, None, None],
            [3, 3, 3, [4, 4, 4, 4]],
            usage=gl.GL_DYNAMIC_DRAW
        )

    def setData(self, start_pos=None, end_pos=None, color=None):
        if color is not None:
            self._color = np.ascontiguousarray(color, dtype=np.float32)

        if start_pos is not None:
            self._st_pos = np.ascontiguousarray(start_pos, dtype=np.float32)
        if end_pos is not None:
            self._end_pos = np.ascontiguousarray(end_pos, dtype=np.float32)

        if self._st_pos is not None and self._end_pos is not None:
            assert self._st_pos.size == self._end_pos.size, \
                    "start_pos and end_pos must have the same size"
            self._num = int(self._st_pos.size / 3)
            self._transform = direction_matrixs(self._st_pos.reshape(-1,3),
                                                self._end_pos.reshape(-1,3))
            if self._color is not None and self._color.size!=self._num*3 and self._color.size>=3:
                self._color = np.tile(self._color.ravel()[:3], (max(self._num,1), 1))

        self._gl_update_flag = True
        self.update()

    def updateGL(self):
        if not self._gl_update_flag:
            return

        self.vao.bind()
        self.vbo_shaft.updateData([0, 1, 2, 3],
                                [self._st_pos, self._end_pos, self._color, self._transform])

        self.vbo_shaft.setAttrPointer(
            [0, 1, 2, 3],
            attr_id=[0, 1, 2, [3,4,5,6]],
            divisor=[0, 0, 0, 1],
        )
        self._gl_update_flag = False

    def paint(self, model_matrix=Matrix4x4()):
        if self._num == 0:
            return
        self.updateGL()
        self.setupGLState()

        if self.antialias:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glLineWidth(self._width)

        self.vao.bind()
        with self.shader:
            self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.vbo_shaft.setAttrPointer(2, divisor=0, attr_id=2)
            gl.glDrawArrays(gl.GL_POINTS, 0, self._num)

        with self.shader_cone:
            self.shader_cone.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader_cone.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader_cone.set_uniform("model", model_matrix.glData, "mat4")
            self.vbo_shaft.setAttrPointer(2, divisor=1, attr_id=2)
            self.ebo_cone.bind()
            gl.glDrawElementsInstanced(
                gl.GL_TRIANGLES,
                self._cone_indices.size,
                gl.GL_UNSIGNED_INT, None,
                self._num,
            )


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 stPos;
layout (location = 1) in vec3 endPos;
layout (location = 2) in vec3 aColor;

out V_OUT {
    vec4 endPos;
    vec3 color;
} v_out;

void main() {
    mat4 matrix = proj * view * model;
    gl_Position =  matrix * vec4(stPos, 1.0);
    v_out.endPos = matrix * vec4(endPos, 1.0);
    v_out.color = aColor;
}
"""

vertex_shader_cone = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 7) in vec3 iPos;
layout (location = 2) in vec3 aColor;
layout (location = 3) in vec4 row1;
layout (location = 4) in vec4 row2;
layout (location = 5) in vec4 row3;
layout (location = 6) in vec4 row4;
// layout (location = 2) in vec3 aColor;
out vec3 oColor;

void main() {
    mat4 transform = mat4(row1, row2, row3, row4);
    gl_Position =  proj * view * model * transform * vec4(iPos, 1.0);
    oColor = aColor * vec3(0.9, 0.9, 0.9);
}
"""

geometry_shader = """
#version 330 core
layout(points) in;
layout(line_strip, max_vertices = 2) out;

in V_OUT {
    vec4 endPos;
    vec3 color;
} gs_in[];
out vec3 oColor;

void main() {
    oColor = gs_in[0].color;
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    gl_Position = gs_in[0].endPos;
    EmitVertex();
    EndPrimitive();
}
"""

fragment_shader = """
#version 330 core

in vec3 oColor;
out vec4 fragColor;

void main() {
    fragColor = vec4(oColor, 1.0f);
}
"""