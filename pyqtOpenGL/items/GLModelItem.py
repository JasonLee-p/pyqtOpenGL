from pathlib import Path
import time
import ctypes
import numpy as np
import OpenGL.GL as gl
import assimp_py as assimp
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .BufferObject import VAO, VBO, EBO
from .texture import Texture2D
from .GLBoxItem import GLBoxItem
from .MeshData import Material

__all__ = ['GLModelItem']


class Mesh():

    def __init__(
        self,
        vertexes,
        indices,
        texcoords = None,
        normals = None,
        material: tuple = None,
        directory = None,
        usage = gl.GL_STATIC_DRAW,
    ):
        self._vertexes = np.array(vertexes, dtype=np.float32)
        self._indices = np.array(indices, dtype=np.uint32)
        self._normals = np.array(normals, dtype=np.float32)
        self._texcoords = np.array(texcoords[0], dtype=np.float32)[..., :2]
        self._material = Material(
            material["COLOR_AMBIENT"],
            material["COLOR_DIFFUSE"],
            material["COLOR_SPECULAR"],
            material["SHININESS"],
            material["OPACITY"],
            material["TEXTURES"],
            directory,
        )

        self._usage = usage

    def initializeGL(self):
        self.vao = VAO()
        self.vbo = VBO(
            [self._vertexes, self._normals, self._texcoords],
            [3, 3, 2],
            usage=self._usage
        )
        self.vbo.setAttrPointer([0, 1, 2], attr_id=[0, 1, 2])

        self.ebo = EBO(self._indices)

        self._material.load_textures()

    def paint(self, shader):
        for idx, texture in enumerate(self._material.textures):
            texture.bind(idx)
            shader.set_uniform(name=texture.type, data=idx, type="Sampler2D")
        shader.set_uniform("opacity", self._material.opacity, "float")

        self.vao.bind()
        gl.glDrawElements(gl.GL_TRIANGLES, self._indices.size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))


class GLModelItem(GLGraphicsItem):

    def __init__(
        self,
        path,
        lightPos = [5, 3.0, 20.0],
        lightColor = [1.0, 1.0, 1.0],
        gamma=False,
        glOptions='translucent',
        parentItem=None,
    ):
        super().__init__(parentItem=parentItem)
        self._path = path
        self._directory = Path(path).parent
        self.meshes = list()
        self.gamma_correction = gamma
        self._load_model(path)
        self.setGLOptions(glOptions)

        # light
        self.lightPos = Vector3(lightPos)
        self.lightColor = Vector3(lightColor)

        self.lightBox = GLBoxItem(size=(2, 2, 2), color=(10, 10, 10))
        self.lightBox.moveTo(*self.lightPos)
        # self.addChildItem(self.lightBox)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)

        for m in self.meshes:
            m.initializeGL()
        self.view().addItem(self.lightBox)

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()

        self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("lightPos", self.lightPos, "vec3")
        self.shader.set_uniform("lightColor", self.lightColor, "vec3")
        with self.shader:
            for m in self.meshes:
                m.paint(self.shader)

    def _load_model(self, path):
        start_time = time.time()

        post_process = (assimp.Process_Triangulate |
                        assimp.Process_FlipUVs)
                        # assimp.Process_CalcTangentSpace 计算法线空间

        scene = assimp.ImportFile(path, post_process)
        if not scene:
            raise ValueError("ERROR:: Assimp model failed to load, {}".format(path))

        for m in scene.meshes:
            self.meshes.append(
                Mesh(
                    m.vertices,
                    m.indices,
                    m.texcoords,
                    m.normals,
                    scene.materials[m.material_index],
                    directory=self._directory
                )
            )

        print("Took {}s to load model {}".format(
                round(time.time()-start_time, 3), path))

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
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 model;

void main() {
    TexCoords = aTexCoords;
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = view * vec4(FragPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;

uniform sampler2D tex_diffuse;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform float opacity;

void main() {
    vec3 objColor = texture(tex_diffuse, TexCoords).rgb;
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
