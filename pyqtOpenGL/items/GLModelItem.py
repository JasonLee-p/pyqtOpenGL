from pathlib import Path
from ..GLGraphicsItem import GLGraphicsItem, PickColorManager
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .MeshData import Mesh
from .light import LightMixin, light_fragment_shader
from .GLMeshItem import vertex_shader
from typing import List

__all__ = ['GLModelItem']


class GLModelItem(GLGraphicsItem, LightMixin):

    def __init__(
            self,
            path,
            lights=None,
            glOptions='translucent',
            parentItem=None,
            selectable=False
    ):
        super().__init__(parentItem=parentItem, selectable=selectable)
        if lights is None:
            lights = list()
        self._path = path
        self._directory = Path(path).parent
        self.setGLOptions(glOptions)
        # model
        self.meshes: List[Mesh] = Mesh.load_model(path)
        self._order = list(range(len(self.meshes)))
        # light
        self.addLight(lights)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, light_fragment_shader)
        self.pick_shader = Shader(vertex_shader, self.pick_fragment_shader)
        for m in self.meshes:
            m.initializeGL()

    def paint(self, model_matrix=Matrix4x4()):
        if not self.selected():
            self.setupGLState()
            self.setupLight(self.shader)

            with self.shader:
                self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
                self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
                self.shader.set_uniform("model", model_matrix.glData, "mat4")
                self.shader.set_uniform("ViewPos", self.view_pos(), "vec3")
                for i in self._order:
                    self.meshes[i].paint(self.shader)
        else:
            self.paint_selected(model_matrix)

    def paint_selected(self, model_matrix=Matrix4x4()):
        self.paint_pickMode(model_matrix)  # TODO: 暂时的实现

    def paint_pickMode(self, model_matrix=Matrix4x4()):
        self.setupGLState()
        with self.pick_shader:
            self.pick_shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
            self.pick_shader.set_uniform("model", model_matrix.glData, "mat4")
            self.pick_shader.set_uniform("pickColor", self._pickColor, "vec3")
            for i in self._order:
                self.meshes[i].paint(self.pick_shader)

    def setMaterial(self, mesh_id, material):
        self.meshes[mesh_id].setMaterial(material)

    def getMaterial(self, mesh_id):
        return self.meshes[mesh_id]._material

    def setPaintOrder(self, order: list):
        """设置绘制顺序, order为mesh的索引列表"""
        assert max(order) < len(self.meshes) and min(order) >= 0
        self._order = order
