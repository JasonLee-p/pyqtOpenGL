from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .BufferObject import VAO, VBO
import numpy as np
import OpenGL.GL as gl
from .light import LightMixin, PointLight
from typing import List

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


class GLGridItem(GLGraphicsItem, LightMixin):
    """
    Displays xy plane.
    """
    def __init__(
        self,
        size = (1., 1.),
        spacing = (1.,1.),
        color = (1.,1.,1.,1),
        lineWidth = 1,
        lights: List[PointLight] = list(),
        antialias: bool = True,
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
        ], dtype=np.float32).reshape(-1, 3)
        self.rotate(90, 1, 0, 0)
        self.setDepthValue(-1)
        self.addLight(lights)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)

        self.vao = VAO()

        self.vbo1 = VBO(
            data = [np.vstack([self.plane_vertices, self.line_vertices])],
            size = [3],
        )
        self.vbo1.setAttrPointer(0, attr_id=0)

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()
        self.setupLight(self.shader)

        if self.antialias:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glLineWidth(self.__lineWidth)

        with self.shader:
            self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")

            self.vao.bind()
            # draw surface
            self.shader.set_uniform("oColor", self.__color, "vec4")
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

            # draw lines
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_DEPTH_TEST)
            self.shader.set_uniform("oColor", np.array([0, 0, 0, 1], "f4"), "vec4")
            gl.glDrawArrays(gl.GL_LINES, 0, len(self.line_vertices))
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_BLEND)


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

layout (location = 0) in vec3 aPos;
out vec3 FragPos;
out vec3 Normal;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalize(mat3(transpose(inverse(model))) * vec3(0, 0, -1));
    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""



fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 ViewPos;
uniform vec4 oColor;

struct PointLight {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool directional;
};
#define MAX_POINT_LIGHTS 10
uniform PointLight pointLight[MAX_POINT_LIGHTS];
uniform int nr_point_lights;

float shininess = 32.0;

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewPos)
{
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 lightDir = vec3(0);
    if (light.directional)
        lightDir = normalize(light.position);
    else
        lightDir = normalize(light.position - fragPos);

    //vec3 halfwayDir = normalize(lightDir + viewDir);
    vec3 reflectDir = reflect(-lightDir, normal);
    // 漫反射着色
    float diff = max(dot(normal, lightDir), 0.0);;
    // 镜面光着色
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    //float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    // 合并结果
    vec3 ambient  = light.ambient * oColor.xyz * 0.3;
    vec3 diffuse  = light.diffuse * diff * oColor.xyz * 0.5;
    vec3 specular = light.specular * spec * oColor.xyz * 0.2;

    return ambient + specular + diffuse;
}

void main() {

    vec3 result = vec3(0);
    for(int i = 0; i < nr_point_lights; i++)
        result += CalcPointLight(pointLight[i], Normal, FragPos, ViewPos);
    if(nr_point_lights == 0){
        result = oColor.rgb;
    }
    FragColor = vec4(result.rgb, oColor.a);
}
"""
