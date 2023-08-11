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
                Texture2D(self.directory / paths[0], type=TextureType[type])
            )


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
    return vertices, indices


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