import OpenGL.GL as gl
import numpy as np

from .shader import Shader
from .BufferObject import VBO, EBO, VAO
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3, Quaternion
from .MeshData import cone, direction_matrixs

__all__ = ['GLAxisItem']


class GLAxisItem(GLGraphicsItem):
    """
    Displays three lines indicating origin and orientation of local coordinate system.
    """
    stPos = np.array([
        # positions
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], dtype="f4")

    endPos = np.array([
        # positions
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ], dtype="f4")

    colors = np.array([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ], dtype="f4")

    def __init__(
        self,
        size=Vector3(1.,1.,1.),
        width=2,
        tip_size=1,
        antialias=True,
        glOptions='opaque',
        fix_to_corner=False,
        parentItem=None
    ):
        super().__init__(parentItem=parentItem)
        self.__size = Vector3(size)
        self.__width = width
        self.__fix_to_corner = fix_to_corner

        self.setGLOptions(glOptions)
        if fix_to_corner:
            # 保证坐标轴不会被其他物体遮挡
            self.updateGLOptions({"glClear": (gl.GL_DEPTH_BUFFER_BIT,)})
            self.setDepthValue(1000)  # make sure it is drawn last

        self.antialias = antialias
        self.cone_vertices, self.cone_indices = cone(0.06*width*tip_size, 0.15*width*tip_size)

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

        # line
        self.vao_line = VAO()

        self.vbo1 = VBO(
            data = [self.stPos, self.endPos, self.colors],
            size = [3, 3, 3],
        )
        self.vbo1.setAttrPointer([0, 1, 2])

        # cone
        self.transforms = direction_matrixs(self.stPos.reshape(-1,3)*self.__size ,
                                            self.endPos.reshape(-1,3)*self.__size )
        self.vao_cone = VAO()

        self.vbo2 = VBO(
            [self.cone_vertices, self.transforms],
            [3, [4,4,4,4]],
        )
        self.vbo2.setAttrPointer([0, 1], divisor=[0, 1], attr_id=[0, [1,2,3,4]])

        self.vbo1.bind()
        self.vbo1.setAttrPointer(2, divisor=1, attr_id=5)

        self.ebo = EBO(self.cone_indices)

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()

        if self.antialias:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glLineWidth(self.__width)

        proj_view = self.proj_view_matrix().glData
        with self.shader:
            self.shader.set_uniform("sizev3", self.size(), "vec3")
            self.shader.set_uniform("view", proj_view, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.vao_line.bind()
            gl.glDrawArrays(gl.GL_POINTS, 0, 3)

        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        with self.shader_cone:
            self.shader_cone.set_uniform("view", proj_view, "mat4")
            self.shader_cone.set_uniform("model", model_matrix.glData, "mat4")
            self.vao_cone.bind()
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.cone_indices.size, gl.GL_UNSIGNED_INT, None, 3)
        gl.glDisable(gl.GL_CULL_FACE)

    def proj_view_matrix(self):
        if self.__fix_to_corner:
            view = self.view()
            proj = Matrix4x4.create_projection(
                20, 1 / view.deviceRatio(), 1, 50.0
            )
            # 计算在这个投影矩阵下, 窗口右上角点在相机坐标系下的坐标
            pos = proj.inverse() * Vector3(0.75, 0.75, 1)
            return proj * Matrix4x4.fromTranslation(*(pos * 40)) * self.view().camera.quat
        else:
            return super().proj_view_matrix()


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform vec3 sizev3;

layout (location = 0) in vec3 stPos;
layout (location = 1) in vec3 endPos;
layout (location = 2) in vec3 iColor;

out V_OUT {
    vec4 endPos;
    vec3 color;
} v_out;

void main() {
    mat4 matrix = view * model;
    gl_Position =  matrix * vec4(stPos * sizev3, 1.0);
    v_out.endPos = matrix * vec4(endPos * sizev3, 1.0);
    v_out.color = iColor;
}
"""

vertex_shader_cone = """
#version 330 core

uniform mat4 model;
uniform mat4 view;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec4 row1;
layout (location = 2) in vec4 row2;
layout (location = 3) in vec4 row3;
layout (location = 4) in vec4 row4;
layout (location = 5) in vec3 iColor;
out vec3 oColor;

void main() {
    mat4 transform = mat4(row1, row2, row3, row4);
    gl_Position =  view * model * transform * vec4(iPos, 1.0);
    oColor = iColor;
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