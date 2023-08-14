import numpy as np
import OpenGL.GL as gl
from pathlib import Path
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .shader import Shader
from .BufferObject import VAO, VBO, EBO, c_void_p
from .MeshData import Material, PointLight
from .GLBoxItem import GLBoxItem
from .texture import Texture2D

BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLSurfacePlotItem']


class GLSurfacePlotItem(GLGraphicsItem):

    def __init__(
        self,
        zmap = None,
        x_size = 10, # scale width to this size
        calc_normals = True,
        lightPos = [5, 3.0, 20.0],
        material = {
            'COLOR_AMBIENT': [0.8470588326454163, 0.8470588326454163, 0.8470588326454163],
            'COLOR_DIFFUSE': [0.677647054195404, 0.677647054195404, 0.677647054195404],
            'COLOR_SPECULAR': [0.8470588326454163, 0.8470588326454163, 0.8470588326454163],
            'SHININESS': 24.0,
            'OPACITY': 1.0,
        },
        glOptions = 'translucent',
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

        # light
        self.light = PointLight(position=Vector3(lightPos))

        self.lightBox = GLBoxItem(size=(0.5, 0.5, 0.5), color=self.light.diffuse.xyz*3)
        self.lightBox.moveTo(*self.light.position)

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

            scale = self._x_size / w
            x_size = self._x_size
            y_size = scale * h
            zmap = zmap * scale
            x = np.linspace(-x_size/2, x_size/2, w, dtype='f4')
            y = np.linspace(y_size/2, -y_size/2, h, dtype='f4')

            xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
            self._vertexes = np.stack([xgrid, ygrid, zmap.astype('f4')], axis=-1).reshape(-1, 3)
            self.xy_size = (x_size, y_size)
        else:
            self._vertexes[:, 2] = zmap.reshape(-1)

        # calc indices
        if self._indice_update_flag:
            self._indices = self.create_indices(w, h)

        # calc normals texture
        v = self._vertexes[self._indices]  # Nf x 3 x 3
        v = np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]) # Nf(c*r*2) x 3
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
        vert_to_face = [[] for i in range(len(self._vertexes))]
        for i in range(self._indices.shape[0]):
            face = self._indices[i]
            for ind in face:
                vert_to_face[ind].append(i)

        normals = np.empty(self._vertexes.shape, dtype='f4')
        for vi in range(self._vertexes.shape[0]):
            faces = vert_to_face[vi]
            if len(faces) == 0:
                normals[vi] = (0,0,0)
                continue
            norms = faceNorms[faces]  ## get all face normals
            norm = norms.sum(axis=0)       ## sum normals
            norm /= (norm**2).sum()**0.5  ## and re-normalize
            normals[vi] = norm
        return normals

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = VBO([None], [3], usage = gl.GL_DYNAMIC_DRAW)
        self.ebo = EBO(None)

        self.view().addItem(self.lightBox)

    def updateGL(self):
        if not self._vert_update_flag:
            return

        self.vao.bind()
        self.vbo.updateData([0], [self._vertexes])
        self.vbo.setAttrPointer([0], attr_id=[0])
        if self._indice_update_flag:
            self.ebo.updateData(self._indices)

        self.texture = Texture2D(self._normal_texture, flip_y=True, flip_x=False)
        self._vert_update_flag = False
        self._indice_update_flag = False

    def paint(self, model_matrix=Matrix4x4()):
        if self._shape[0] == 0:
            return
        self.updateGL()
        self.setupGLState()
        # gl.glLineWidth(2.0)  # 设置线条宽度
        gl.glDisable(gl.GL_LINE_SMOOTH)

        self._material.set_uniform(self.shader, "material")
        self.light.set_pos([0, 5, 10])
        self.light.set_uniform(self.shader, "pointLight[0]")
        self.light.set_pos([-25, -20.0, -25.0])
        self.light.set_uniform(self.shader, "pointLight[1]")

        self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")
        self.shader.set_uniform("texScale", self.xy_size, "vec2")

        self.texture.bind(0)
        self.shader.set_uniform("norm_texture", self.texture.unit, "sampler2D")

        with self.shader:
            self.vao.bind()
            gl.glDrawElements(gl.GL_TRIANGLES, self._indices.size, gl.GL_UNSIGNED_INT, c_void_p(0))
        # gl.glDrawArrays(gl.GL_POINTS, 0, self._vertexes.shape[0])

    def setLight(self, pos=None, color=None, transform: Matrix4x4=None):
        if pos is not None:
            self.lightPos = Vector3(pos)
            self.lightBox.moveTo(*self.lightPos)
        if color is not None:
            self.lightColor = Vector3(color)
        if transform is not None:
            self.lightPos = Vector3(transform * self.lightPos)
            self.lightBox.moveTo(*self.lightPos)
        self.update()


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


fragment_shader = """
#version 330 core
out vec4 FragColor;

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
#define NR_POINT_LIGHTS 2
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
    ambient  = light.ambient  * material.ambient;
    diffuse  = light.diffuse  * diff * material.diffuse;
    specular = light.specular * spec * material.specular;

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