from pathlib import Path
import time
import ctypes
import numpy as np
import OpenGL.GL as gl
import assimp_py as assimp
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .MeshData import Mesh
from .light import LightMixin, light_fragment_shader
from typing import List

__all__ = ['GLModelItem']


class GLModelItem(GLGraphicsItem, LightMixin):

    def __init__(
        self,
        path,
        lights = list(),
        glOptions='translucent',
        texcoords_scale=1,
        parentItem=None,
    ):
        super().__init__(parentItem=parentItem)
        self._path = path
        self._directory = Path(path).parent
        self.setGLOptions(glOptions)
        # model
        self.meshes: List[Mesh] = list()
        self._load_model(path, texcoords_scale)
        # light
        self.addLight(lights)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, light_fragment_shader)

        for m in self.meshes:
            m.initializeGL()

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()
        self.setupLight(self.shader)

        with self.shader:
            self.shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")
            for i in self._order:
                self.meshes[i].paint(self.shader)

    def setMaterial(self, mesh_id, material):
        self.meshes[mesh_id].setMaterial(material)

    def getMaterial(self, mesh_id):
        return self.meshes[mesh_id]._material

    def _load_model(self, path, texcoords_scale):
        start_time = time.time()

        post_process = (assimp.Process_Triangulate |
                        assimp.Process_FlipUVs|
                        assimp.Process_GenNormals|
                        assimp.Process_PreTransformVertices
                        )
                        # assimp.Process_CalcTangentSpace 计算法线空间
        scene = assimp.ImportFile(str(path), post_process)
        if not scene:
            raise ValueError("ERROR:: Assimp model failed to load, {}".format(path))

        faces = 0
        is_dae = Path(path).suffix == ".dae"
        for m in scene.meshes:

            # 若模型是 dae 文件, 且其中 <up_axis>Z_UP</up_axis>, 则需要将原始坐标绕 x 轴旋转 90 度
            verts = np.array(m.vertices, dtype=np.float32).reshape(-1, 3)
            norms = np.array(m.normals, dtype=np.float32).reshape(-1, 3)
            if is_dae: # x, y, z -> x, -z, y
                verts[:, 2] = -verts[:, 2]
                verts[:, [1, 2]] = verts[:, [2, 1]]
                norms[:, 2] = -norms[:, 2]
                norms[:, [1, 2]] = norms[:, [2, 1]]

            self.meshes.append(
                Mesh(
                    verts,
                    m.indices,
                    m.texcoords[0] if len(m.texcoords) > 0 else None,
                    norms,
                    scene.materials[m.material_index],
                    directory=self._directory,
                    texcoords_scale=texcoords_scale,
                )
            )
            faces += len(m.indices)
        self._order = list(range(len(self.meshes)))
        print(f"Took {round(time.time()-start_time, 3)}s to load model {path} ({faces})")

    def setPaintOrder(self, order: list):
        """设置绘制顺序, order为mesh的索引列表"""
        assert max(order) < len(self.meshes) and min(order) >= 0
        self._order = order


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
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
    gl_Position = view * vec4(FragPos, 1.0);
}
"""
