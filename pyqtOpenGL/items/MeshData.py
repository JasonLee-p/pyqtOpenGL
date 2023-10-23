import numpy as np
import math
import OpenGL.GL as gl
import assimp_py as assimp
from ctypes import c_float, sizeof, c_void_p, Structure
from .shader import Shader
from .BufferObject import VAO, VBO, EBO
from .texture import Texture2D
from ..transform3d import Vector3
from ..functions import dispatchmethod

__all__ = [
    "Mesh", "Material", "direction_matrixs", "vertex_normal",
    "sphere", "cylinder", "cube", "cone", "plane"
]

Vec2 = (2 * c_float)
Vec3 = (3 * c_float)


TextureType = {
    assimp.TextureType_DIFFUSE: "tex_diffuse",  # map_Kd
    assimp.TextureType_SPECULAR: "tex_specular",  # map_Ks
    assimp.TextureType_AMBIENT: "tex_ambient",  # map_Ka
    assimp.TextureType_HEIGHT: "tex_height",  # map_Bump
}

class Material():

    @dispatchmethod
    def __init__(
        self,
        ambient = [0.4, 0.4, 0.4],
        diffuse = [1.0, 1.0, 1.0],
        specular = [0.2, 0.2, 0.2],
        shininess = 10,
        opacity = 1,
        textures: list = list(),
        textures_paths: dict = dict(),
        directory = None,
    ):
        self.ambient = Vector3(ambient)
        self.diffuse = Vector3(diffuse)
        self.specular = Vector3(specular)
        self.shininess = shininess
        self.opacity = opacity
        self.textures = list()
        self.textures.extend(textures)
        self.texture_paths = textures_paths
        self.directory = directory

    @__init__.register(dict)
    def _(self, material_dict: dict, directory=None):
        self.__init__(
            material_dict.get("COLOR_AMBIENT", [0.4, 0.4, 0.4]),  # Ka
            material_dict.get("COLOR_DIFFUSE", [1.0, 1.0, 1.0]),  # Kd
            material_dict.get("COLOR_SPECULAR", [0.2, 0.2, 0.2]),  # Ks
            material_dict.get("SHININESS", 10),
            material_dict.get("OPACITY", 1),
            textures_paths = material_dict.get("TEXTURES", None),
            directory = directory,
        )

    def load_textures(self):
        """在 initializeGL() 中调用 """
        for type, paths in self.texture_paths.items():
            self.textures.append(
                Texture2D(self.directory / paths[0], tex_type=TextureType[type])
            )

    def set_uniform(self, shader: Shader, name: str):
        use_texture = False
        for tex in self.textures:
            if tex.type == "tex_diffuse":
                tex.bind()
                shader.set_uniform(f"{name}.tex_diffuse", tex, "sampler2D")
                use_texture = True
        shader.set_uniform(name+".ambient", self.ambient, "vec3")
        shader.set_uniform(name+".diffuse", self.diffuse, "vec3")
        shader.set_uniform(name+".specular", self.specular, "vec3")
        shader.set_uniform(name+".shininess", self.shininess, "float")
        shader.set_uniform(name+".opacity", self.opacity, "float")
        shader.set_uniform(name+".use_texture", use_texture, "bool")

    def set_data(self, ambient=None, diffuse=None, specular=None, shininess=None, opacity=None):
        if ambient is not None:
            self.ambient = Vector3(ambient)
        if diffuse is not None:
            self.diffuse = Vector3(diffuse)
        if specular is not None:
            self.specular = Vector3(specular)
        if shininess is not None:
            self.shininess = shininess
        if opacity is not None:
            self.opacity = opacity


class Mesh():

    def __init__(
        self,
        vertexes,
        indices,
        texcoords = None,
        normals = None,
        material = None,
        directory = None,
        usage = gl.GL_STATIC_DRAW,
        texcoords_scale = 1,
        calc_normals = False,
    ):
        self._vertexes = np.array(vertexes, dtype=np.float32)

        if indices is not None:
            self._indices = np.array(indices, dtype=np.uint32)
        else:
            self._indices = None

        if calc_normals and normals is None:
            self._normals = vertex_normal(self._vertexes, self._indices)
        else:
            self._normals = np.array(normals, dtype=np.float32)

        if texcoords is None:
            self._texcoords = None
        else:
            self._texcoords = np.array(texcoords, dtype=np.float32)[..., :2] / texcoords_scale

        if isinstance(material, dict):
            self._material = Material(material, directory)
        elif isinstance(material, Material):
            self._material = material

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

        if self._indices is None:
            shader.set_uniform("material.use_texture", False, 'bool')

        self.vao.bind()
        if self._indices is not None:
            gl.glDrawElements(gl.GL_TRIANGLES, self._indices.size, gl.GL_UNSIGNED_INT, c_void_p(0))
        else:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._vertexes.size)

    def setMaterial(self, material=None):
        if isinstance(material, dict):
            self._material = Material(material)
        elif isinstance(material, Material):
            self._material = material

    def getMaterial(self):
        return self._material


def cone(radius, height, slices=12):
    slices = max(3, slices)
    vertices = np.zeros((slices+2, 3), dtype="f4")
    vertices[-2] = [0, 0, height]
    step = 360 / slices  # 圆每一段的角度
    for i in range(0, slices):
        p = step * i * 3.14159 / 180  # 转为弧度
        vertices[i] = [radius * math.cos(p), radius * math.sin(p), 0]
    # 构造圆锥的面索引
    indices = np.zeros((slices*6, ), dtype=np.uint32)
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
        faces = np.empty((rows*cols*2, 3), dtype=np.uint32)
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
    verts = np.empty(((rows+3)*cols+2, 3), dtype=np.float32)  # 顶面的点和底面的点重复一次, 保证法线计算正确
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
    # 顶面, 底面
    verts[(rows+1)*cols:(rows+2)*cols] = verts1[-cols:]
    verts[(rows+2)*cols:-2] = verts1[:cols]
    verts[-2] = [0, 0, 0] # zero at bottom
    verts[-1] = [0, 0, length] # length at top

    ## compute faces
    num_side_faces = rows * cols * 2
    num_cap_faces = cols
    faces = np.empty((num_side_faces + num_cap_faces*2, 3), dtype=np.uint32)
    rowtemplate1 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 0, 1]])) % cols) + np.array([[0, cols, 0]])
    rowtemplate2 = ((np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 1]])) % cols) + np.array([[cols, cols, 0]])
    for row in range(rows):
        start = row * cols * 2
        faces[start:start+cols] = rowtemplate1 + row * cols
        faces[start+cols:start+(cols*2)] = rowtemplate2 + row * cols

    # Bottom face
    bottom_start = num_side_faces
    bottom_row = np.arange((rows+2) * cols, (rows+3) * cols)
    bottom_face = np.column_stack((bottom_row, np.roll(bottom_row, -1), np.full(cols, (rows+3) * cols)))
    faces[bottom_start : bottom_start + num_cap_faces] = bottom_face

    # Top face
    top_start = num_side_faces + num_cap_faces
    top_row = np.arange((rows+1) * cols, (rows+2) * cols)
    top_face = np.column_stack((np.roll(top_row, -1), top_row, np.full(cols, (rows+3) * cols+1)))
    faces[top_start : top_start + num_cap_faces] = top_face

    return verts, faces

def cube(x, y, z):
    """
    Return a MeshData instance with vertexes and normals computed
    for a rectangular cuboid of the given dimensions.
    """
    vertices = np.array( [
        # 顶点坐标             # 法向量       # 纹理坐标
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 0.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  1.0, 1.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 1.0,
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  0.0, 0.0,

        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,
         0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 0.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  1.0, 1.0,
        -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 1.0,
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  0.0, 0.0,

        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,
        -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  1.0, 1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  0.0, 1.0,
        -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  0.0, 0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  1.0, 0.0,

         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  1.0, 1.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  0.0, 1.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  0.0, 0.0,
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  1.0, 0.0,

        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0,
         0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  1.0, 1.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  1.0, 0.0,
        -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  0.0, 0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  0.0, 1.0,

        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  1.0, 1.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  1.0, 0.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  0.0, 0.0,
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  0.0, 1.0,
    ], dtype="f4").reshape(-1, 8)
    verts = vertices[:, :3] * np.array([x,y,z], dtype="f4")
    normals = vertices[:, 3:6]
    texcoords = vertices[:, 6:]
    return verts, normals, texcoords


def plane(x, y):
    vertices = np.array([
        # 顶点坐标             # 法向量       # 纹理坐标
        -0.5, -0.5, 0.0,  0.0,  0.0, 1.0,  0.0, 0.0,
         0.5, -0.5, 0.0,  0.0,  0.0, 1.0,  1.0, 0.0,
         0.5,  0.5, 0.0,  0.0,  0.0, 1.0,  1.0, 1.0,
         0.5,  0.5, 0.0,  0.0,  0.0, 1.0,  1.0, 1.0,
        -0.5,  0.5, 0.0,  0.0,  0.0, 1.0,  0.0, 1.0,
        -0.5, -0.5, 0.0,  0.0,  0.0, 1.0,  0.0, 0.0,
    ], dtype=np.float32).reshape(-1, 8)
    verts = vertices[:, :3] * np.array([x,y,1.0], dtype="f4")
    normals = vertices[:, 3:6]
    texcoords = vertices[:, 6:]
    return verts, normals, texcoords


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


def surface(zmap, xy_size):
    x_size, y_size = xy_size
    h, w = zmap.shape
    scale = x_size / w
    zmap *= scale

    x = np.linspace(-x_size/2, x_size/2, w, dtype='f4')
    y = np.linspace(y_size/2, -y_size/2, h, dtype='f4')

    xgrid, ygrid = np.meshgrid(x, y, indexing='xy')
    verts = np.stack([xgrid, ygrid, zmap.astype('f4')], axis=-1).reshape(-1, 3)

    # calc indices
    ncol = w - 1
    nrow = h - 1
    if ncol == 0 or nrow == 0:
        raise Exception("cols or rows is zero")

    faces = np.empty((nrow, 2, ncol, 3), dtype=np.uint32)
    rowtemplate1 = np.arange(ncol).reshape(1, ncol, 1) + np.array([[[0     , ncol+1, 1]]])  # 1, ncols, 3
    rowtemplate2 = np.arange(ncol).reshape(1, ncol, 1) + np.array([[[ncol+1, ncol+2, 1]]])
    rowbase = np.arange(nrow).reshape(nrow, 1, 1) * (ncol+1)  # nrows, 1, 1
    faces[:, 0] = (rowtemplate1 + rowbase)  # nrows, 1, ncols, 3
    faces[:, 1] = (rowtemplate2 + rowbase)

    return verts, faces.reshape(-1, 3)


def grid3d(grid):
    # grid: (h, w, 3)
    h, w = grid.shape[:2]
    ncol, nrow = w-1, h-1

    rowtemplate = np.arange(ncol, dtype=np.uint32).reshape(1, ncol, 1) + \
        np.array([[[0, ncol+1, ncol+2, 1]]])  # 1, ncols, 4
    rowbase = np.arange(nrow, dtype=np.uint32).reshape(nrow, 1, 1) * (ncol+1)  # nrows, 1, 1
    faces = (rowtemplate + rowbase).reshape(-1, 4).astype(np.uint32)

    return grid.reshape(-1, 3).astype(np.float32), faces


def mesh_concat(verts: list, faces: list):
    """合并多个网格"""

    vert_nums = [len(v) for v in verts]
    id_bias = np.cumsum(vert_nums)
    for i in range(1, len(faces)):
        faces[i] += id_bias[i-1]

    verts = np.concatenate(verts, axis=0)
    faces = np.concatenate(faces, axis=0).astype(np.uint32)

    return verts, faces