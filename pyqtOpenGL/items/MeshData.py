import numpy as np
import math
import OpenGL.GL as gl
import assimp_py as assimp
from ctypes import c_float, sizeof, c_void_p, Structure
from .shader import Shader
from .BufferObject import VAO, VBO, EBO
from .texture import Texture2D
from ..transform3d import Vector3

Vec2 = (2 * c_float)
Vec3 = (3 * c_float)


TextureType = {
    assimp.TextureType_DIFFUSE: "tex_diffuse",  # map_Kd
    assimp.TextureType_SPECULAR: "tex_specular",  # map_Ks
    assimp.TextureType_AMBIENT: "tex_ambient",  # map_Ka
    assimp.TextureType_HEIGHT: "tex_height",  # map_Bump
}

class Material():

    def __init__(
        self,
        ambient,
        diffuse,
        specular,
        shininess,
        opacity,
        textures: dict = None,
        directory = None,
    ):
        self.ambient = Vector3(ambient)
        self.diffuse = Vector3(diffuse)
        self.specular = Vector3(specular)
        self.shininess = shininess
        self.opacity = opacity
        self.textures_path = textures
        self.directory = directory
        self.textures = list()

    def load_textures(self):
        # print(self.tex)
        for type, paths in self.textures_path.items():
            self.textures.append(
                Texture2D(self.directory / paths[0], tex_type=TextureType[type])
            )

    def set_uniform(self, shader: Shader, name: str):
        use_texture = False
        for i, tex in enumerate(self.textures):
            if tex.type == "tex_diffuse":
                tex.bind(i)
                shader.set_uniform(f"{name}.{tex.type}", i, "sampler2D")
                use_texture = True
        shader.set_uniform(name+".ambient", self.ambient, "vec3")
        shader.set_uniform(name+".diffuse", self.diffuse, "vec3")
        shader.set_uniform(name+".specular", self.specular, "vec3")
        shader.set_uniform(name+".shininess", self.shininess, "float")
        shader.set_uniform(name+".opacity", self.opacity, "float")
        shader.set_uniform(name+".use_texture", use_texture, "bool")

class PointLight():

    def __init__(
        self,
        position = Vector3(0.0, 0.0, 0.0),
        ambient = Vector3(0.05, 0.05, 0.05),
        diffuse = Vector3(0.8, 0.8, 0.8),
        specular = Vector3(1.0, 1.0, 1.0),
        constant = 1.0,
        linear = 0.09,
        quadratic = 0.032,
    ):
        self.position = position
        self.amibent = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic

    def set_uniform(self, shader: Shader, name: str):
        shader.set_uniform(name + ".position", self.position, "vec3")
        shader.set_uniform(name + ".ambient", self.amibent, "vec3")
        shader.set_uniform(name + ".diffuse", self.diffuse, "vec3")
        shader.set_uniform(name + ".specular", self.specular, "vec3")
        shader.set_uniform(name + ".constant", self.constant, "float32")
        shader.set_uniform(name + ".linear", self.linear, "float32")
        shader.set_uniform(name + ".quadratic", self.quadratic, "float32")

    def set_pos(self, pos):
        self.position = Vector3(pos)


def cone(radius, height, slices=12):
    slices = max(3, slices)
    vertices = np.zeros((slices+2, 3), dtype="f4")
    vertices[-2] = [0, 0, height]
    step = 360 / slices  # 圆每一段的角度
    for i in range(0, slices):
        p = step * i * 3.14159 / 180  # 转为弧度
        vertices[i] = [radius * math.cos(p), radius * math.sin(p), 0]
    # 构造圆锥的面索引
    indices = np.zeros((slices*6, ), dtype="uint32")
    for i in range(0, slices):
        indices[i*6+0] = i
        indices[i*6+1] = (i+1) % slices
        indices[i*6+2] = slices
        indices[i*6+3] = i
        indices[i*6+5] = (i+1) % slices
        indices[i*6+4] = slices+1
    return vertices, indices.reshape(-1, 3)

def direction_matrixs(starts, ends):
    arrows = ends - starts
    arrows = arrows.reshape(-1, 3)
    # 处理零向量，归一化
    arrow_lens = np.linalg.norm(arrows, axis=1)
    zero_idxs = arrow_lens < 1e-3
    arrows[zero_idxs] = [0, 0, 1e-3]
    arrow_lens[zero_idxs] = 1e-3
    arrows = arrows / arrow_lens[:, np.newaxis]
    # 构造标准箭头到目标箭头的旋转矩阵
    e = np.zeros_like(arrows)
    e[arrows[:, 0]==0, 0] = 1
    e[arrows[:, 0]!=0, 1] = 1
    b1 = np.cross(arrows, e)  # 第一个正交向量 (n, 3)
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)  # 单位化
    b2 = np.cross(arrows, b1)  # 第二个正交单位向量 (n, 3)
    transforms = np.stack((b1, b2, arrows, ends.reshape(-1, 3)), axis=1)  # (n, 4(new), 3)
    # 转化成齐次变换矩阵
    transforms = np.pad(transforms, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0)  # (n, 4, 4)
    transforms[:, 3, 3] = 1
    # 将 arrow_vert(n, 3) 变换至目标位置
    # vertexes = vertexes @ transforms  #  (n, 3)
    return transforms.copy()


def sphere(radius=1.0, rows=12, cols=12, offset=True):
        """
        Return a MeshData instance with vertexes and faces computed
        for a spherical surface.
        """
        verts = np.empty((rows+1, cols, 3), dtype=np.float32)

        ## compute vertexes
        phi = (np.arange(rows+1) * np.pi / rows).reshape(rows+1, 1)
        s = radius * np.sin(phi)
        verts[...,2] = radius * np.cos(phi)
        th = ((np.arange(cols) * 2 * np.pi / cols).reshape(1, cols))
        if offset:
            th = th + ((np.pi / cols) * np.arange(rows+1).reshape(rows+1,1))  ## rotate each row by 1/2 column
        verts[...,0] = s * np.cos(th)
        verts[...,1] = s * np.sin(th)
        verts = verts.reshape((rows+1)*cols, 3)[cols-1:-(cols-1)]  ## remove redundant vertexes from top and bottom

        ## compute faces
        faces = np.empty((rows*cols*2, 3), dtype=np.uint)
        rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 0, 1]])) % cols) + np.array([[0, cols, 0]])
        rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, cols, 0]])
        for row in range(rows):
            start = row * cols * 2
            faces[start:start+cols] = rowtemplate1 + row * cols
            faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols
        faces = faces[cols:-cols]  ## cut off zero-area triangles at top and bottom

        ## adjust for redundant vertexes that were removed from top and bottom
        vmin = cols-1
        faces[faces<vmin] = vmin
        faces -= vmin
        vmax = verts.shape[0]-1
        faces[faces>vmax] = vmax
        return verts, faces

def cylinder(radius=[1.0, 1.0], length=1.0, rows=12, cols=12, offset=False):
    """
    Return a MeshData instance with vertexes and faces computed
    for a cylindrical surface.
    The cylinder may be tapered with different radii at each end (truncated cone)
    """
    verts = np.empty(((rows+1)*cols+2, 3), dtype=np.float32)
    verts1 = verts[:(rows+1)*cols, :].reshape(rows+1, cols, 3)
    if isinstance(radius, int):
        radius = [radius, radius] # convert to list
    ## compute vertexes
    th = np.linspace(2 * np.pi, (2 * np.pi)/cols, cols).reshape(1, cols)
    r = np.linspace(radius[0],radius[1],num=rows+1, endpoint=True).reshape(rows+1, 1) # radius as a function of z
    verts1[...,2] = np.linspace(0, length, num=rows+1, endpoint=True).reshape(rows+1, 1) # z
    if offset:
        th = th + ((np.pi / cols) * np.arange(rows+1).reshape(rows+1,1))  ## rotate each row by 1/2 column
    verts1[...,0] = r * np.cos(th) # x = r cos(th)
    verts1[...,1] = r * np.sin(th) # y = r sin(th)
    verts1 = verts1.reshape((rows+1)*cols, 3) # just reshape: no redundant vertices...
    verts[-2] = [0, 0, 0] # zero at bottom
    verts[-1] = [0, 0, length] # length at top

    ## compute faces
    num_side_faces = rows * cols * 2
    num_cap_faces = cols
    faces = np.empty((num_side_faces + num_cap_faces*2, 3), dtype=np.uint)
    rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 0, 1]])) % cols) + np.array([[0, cols, 0]])
    rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, cols, 0]])
    for row in range(rows):
        start = row * cols * 2
        faces[start:start+cols] = rowtemplate1 + row * cols
        faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols

    # Bottom face
    bottom_start = num_side_faces
    bottom_row = np.arange(cols)
    bottom_face = np.column_stack((bottom_row, np.roll(bottom_row, -1), np.full(cols, (rows+1) * cols)))
    faces[bottom_start : bottom_start + num_cap_faces] = bottom_face

    # Top face
    top_start = num_side_faces + num_cap_faces
    top_row = np.arange(rows * cols, (rows+1) * cols)
    top_face = np.column_stack((np.roll(top_row, -1), top_row, np.full(cols, (rows+1) * cols+1)))
    faces[top_start : top_start + num_cap_faces] = top_face

    return verts, faces

def face_normal(v1, v2, v3):
    """计算一个三角形的法向量"""
    a = v2 - v1 # 三角形的一条边
    b = v3 - v1 # 三角形的另一条边
    return np.cross(a, b)

def vertex_normal(vert, ind):
    """计算每个顶点的法向量"""
    nv = len(vert) # 顶点的个数
    nf = len(ind) # 面的个数
    norm = np.zeros((nv, 3), np.float32) # 初始化每个顶点的法向量为零向量
    for i in range(nf): # 遍历每个面
        v1, v2, v3 = vert[ind[i]] # 获取面的三个顶点
        fn = face_normal(v1, v2, v3) # 计算面的法向量
        norm[ind[i]] += fn # 将面的法向量累加到对应的顶点上
    norm = norm / np.linalg.norm(norm, axis=1, keepdims=True) # 归一化每个顶点的法向量
    return norm

