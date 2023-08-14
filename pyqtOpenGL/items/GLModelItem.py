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
from .MeshData import Material, PointLight

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
        self._material.set_uniform(shader, "material")

        self.vao.bind()
        gl.glDrawElements(gl.GL_TRIANGLES, self._indices.size, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))


class GLModelItem(GLGraphicsItem):

    def __init__(
        self,
        path,
        lightPos = [5, 3.0, 20.0],
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
        self.light = PointLight(position=Vector3(lightPos), ambient=[0.25,0.25,0.25], specular=[1,1,1])

        self.lightBox = GLBoxItem(size=(0.5, 0.5, 0.5), color=self.light.diffuse)
        self.lightBox.moveTo(*self.light.position)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)

        for m in self.meshes:
            m.initializeGL()
        self.view().addItem(self.lightBox)

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()

        self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")
        self.light.set_uniform(self.shader, "pointLight[0]")
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

    def setLight(self, pos=None, transform: Matrix4x4=None):
        if pos is not None:
            self.light.set_pos(pos)
            self.lightBox.moveTo(*pos)
        if transform is not None:
            self.light.set_pos(transform * self.light.position)
            self.lightBox.moveTo(*self.light.position)
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

uniform vec3 ViewPos;

struct Material {
    float opacity;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    bool use_texture;
    sampler2D tex_diffuse;
};
uniform Material material;

struct PointLight {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};
#define NR_POINT_LIGHTS 1
uniform PointLight pointLight[NR_POINT_LIGHTS];

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewPos)
{
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 lightDir = normalize(light.position - fragPos);
    // 漫反射着色
    float diff = max(dot(normal, lightDir), 0.0);
    // 镜面光着色
    vec3 reflectDir = normalize(reflect(-lightDir, normal));
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), material.shininess);
    // 衰减
    //float distance    = length(light.position - fragPos);
    //float attenuation = 1.0 / (light.constant + light.linear * distance +
    //             light.quadratic * (distance * distance));
    // 合并结果
    vec3 ambient  = vec3(0);
    vec3 diffuse  = vec3(0);
    vec3 specular = vec3(0);
    if (material.use_texture) {
        ambient  = light.ambient  * vec3(texture(material.tex_diffuse, TexCoords));
        diffuse  = light.diffuse  * diff * vec3(texture(material.tex_diffuse, TexCoords));
        specular = light.specular * spec * vec3(texture(material.tex_diffuse, TexCoords));
    } else {
        ambient  = light.ambient  * material.ambient;
        diffuse  = light.diffuse  * diff * material.diffuse;
        specular = light.specular * spec * material.specular;
    }

    //ambient  *= attenuation;
    //diffuse  *= attenuation;
    //specular *= attenuation;
    return ambient + specular + diffuse;
}

void main() {
    vec3 result = vec3(0);
    for(int i = 0; i < NR_POINT_LIGHTS; i++)
        result += CalcPointLight(pointLight[i], Normal, FragPos, ViewPos);
    FragColor = vec4(result, material.opacity);
}
"""
