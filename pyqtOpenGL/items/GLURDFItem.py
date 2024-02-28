from pathlib import Path
from typing import Union, List, Dict
import numpy as np
import OpenGL.GL as gl
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .GLModelItem import GLModelItem
from .GLAxisItem import GLAxisItem
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
import xml.etree.ElementTree as ET
from dataclasses import dataclass

__all__ = ['GLURDFItem']


rad2deg = 180 / np.pi

def parse_origin(origin: ET.Element):
    rpy_str = origin.get('rpy', '0 0 0').split()
    rpy = [float(val) for val in rpy_str]
    xyz_str = origin.get('xyz', '0 0 0').split()
    xyz = [float(val) for val in xyz_str]
    return rpy, xyz

def rpy_xyz_to_mat(rpy: List[float], xyz: List[float]) -> Matrix4x4:
    """rpy: rad, xyz: m"""
    return (Matrix4x4.fromEulerAngles(rad2deg*rpy[0], rad2deg*rpy[1], rad2deg*rpy[2])
                    .moveto(xyz[0], xyz[1], xyz[2]))



class GLLinkItem(GLGraphicsItem):

    def __init__(
        self,
        link_elem: ET.Element,
        lights: list,
        base_dir: Path,
        glOptions: str = "translucent",
        parentItem: GLGraphicsItem=None,
    ):
        super().__init__(parentItem=parentItem)
        self.name = link_elem.get('name')
        self.visual_model = None
        # axis
        self.axis = GLAxisItem(size=(0.12, 0.12, 0.12), tip_size=0.12)
        self.addChildItem(self.axis)

        # visual model: optional
        visual = link_elem.find('visual')
        if visual is not None:
            mesh = visual.find('geometry').find('mesh')
            if mesh is None:
                return

            mesh_path = base_dir / mesh.get('filename')
            self.visual_model = GLModelItem(
                path = mesh_path,
                lights = lights,
                glOptions = glOptions,
            )

            # 加载的 dae 朝向 y, 让其朝向 z 与 urdf 原始定义保持一致
            self.visual_model.rotate(90, 1, 0, 0)
            # parse visual origin transform
            origin = visual.find('origin')
            if origin is not None:
                self.visual_model.applyTransform(rpy_xyz_to_mat(*parse_origin(origin)))

            self.addChildItem(self.visual_model)

    def childLinks(self) -> List['GLLinkItem']:
        return [item for item in self.childItems() if isinstance(item, GLLinkItem)]

    def set_data(
        self,
        axis_visiable: bool = None,
        visual_visiable: bool = None,
    ):
        if axis_visiable is not None:
            self.axis.setVisible(axis_visiable)
        if visual_visiable is not None and self.visual_model is not None:
            self.visual_model.setVisible(visual_visiable)



@dataclass
class Joint:
    child: GLLinkItem
    type: str  # revolute, prismatic. fixed 忽略
    axis: np.ndarray
    limit: np.ndarray
    origin: Matrix4x4
    _value: float = 0

    def __post_init__(self):
        self.set_value(self._value)

    @property
    def value(self):
        return self._value

    def set_value(self, value):
        self._value = np.clip(value, self.limit[0], self.limit[1])
        tf = Matrix4x4()
        # 根据关节类型设置关节值
        if self.type == 'revolute':
            tf = tf.fromAxisAndAngle(self.axis[0], self.axis[1],
                                     self.axis[2], rad2deg*value)
        elif self.type == 'prismatic':
            t = self.axis * value
            tf = tf.moveto(t[0], t[1], t[2])
        self.child.setTransform(self.origin * tf)


class GLURDFItem(GLGraphicsItem):
    """ Displays a GelSlim model with a surface plot on top of it."""

    def __init__(
        self,
        urdf_path: Union[str, Path],
        lights: list,
        glOptions: str = "translucent",
        parentItem=None,
        axis_visiable=False,
        **kwargs # 传递给 GLLinkItem.set_data
    ):
        super().__init__(parentItem=parentItem)

        # 解析xml文件
        self.urdf_path = Path(urdf_path)
        self.base_dir = self.urdf_path.parent
        self.urdf = ET.parse(urdf_path)
        self.links: Dict[str, GLLinkItem] = dict()
        self.joints: Dict[str, Joint] = dict()

        # 遍历每个link元素
        for link in self.urdf.findall('link'):
            name = link.get('name')

            self.links[name] = GLLinkItem(
                link_elem = link,
                lights = lights,
                base_dir = self.base_dir,
                glOptions = glOptions,
            )
            self.links[name].set_data(axis_visiable=axis_visiable, **kwargs)

        # 遍历每个joint元素
        for joint in self.urdf.findall('joint'):
            name = joint.get('name')
            type = joint.get('type')
            origin = rpy_xyz_to_mat(*parse_origin(joint.find('origin')))
            parent = self.links[joint.find('parent').get('link')]
            child = self.links[joint.find('child').get('link')]

            if type == 'fixed':  # 忽略固定关节
                child.setTransform(origin)

            if type in ['revolute', 'prismatic']:
                axis = np.array(joint.find('axis').get('xyz').split(), dtype=float)
                limit = np.array([joint.find('limit').get('lower'), joint.find('limit').get('upper')], dtype=float)
                self.joints[name] = Joint(
                    child = child,
                    type = type,
                    axis = axis,
                    limit = limit,
                    origin = origin
                )

            # 添加父子关系
            parent.addChildItem(child)

        # 设置 base_link
        self.base_link = None
        for link in self.links.values():
            if link.parentItem() is None:
                self.addChildItem(link)
                self.base_link = link
                break

        if self.base_link is None:
            raise ValueError("No base link found")

    def set_joint(self, name: Union[int, str], value):
        if isinstance(name, int):
            name = list(self.joints.keys())[name]
        self.joints[name].set_value(value)

    def set_joints(self, values: Union[list, np.ndarray]):
        for i, (name, joint) in enumerate(self.joints.items()):
            joint.set_value(values[i])

    def get_joints(self)->List[float]:
        return [joint.value for joint in self.joints.values()]

    def get_joints_name(self)->List[str]:
        return list(self.joints.keys())

    def get_joints_limit(self)->np.ndarray:
        return np.array([joint.limit for joint in self.joints.values()])

    def get_links_name(self)->List[str]:
        return list(self.links.keys())

    def set_link(self, name: Union[int, str], **kwargs): # axis_visiable, visual_visiable
        if isinstance(name, int):
            name = list(self.links.keys())[name]
        self.links[name].set_data(**kwargs)

    def print_links(self) -> str:
        """dfs print"""
        prefix = ' '
        print(self.urdf_path)
        stack = [(self.base_link, 1)]  # node, 缩进级别

        while(stack):
            node, level = stack.pop()
            print(prefix * level + node.name)

            for child in reversed(node.childLinks()):
                    stack.append((child, level+1))
        print()

    def set_data(self, visiable: bool):
        stack =[self.base_link]
        while stack:
            link = stack.pop()
            link.visual_model.setVisible(visiable)

    def print_joints(self):
        for name, joint in self.joints.items():
            print(f"{name} | {joint.type} | val: {joint.value} | "
                  f"axis: {joint.axis} | limit: {joint.limit}")
        print()