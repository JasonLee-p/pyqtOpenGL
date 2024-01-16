import sys
import typing
import yaml
from time import time
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication, QSizePolicy
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QSize, pyqtSlot, QPoint, QEventLoop, QThread, QRect
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent
from PyQt5 import QtGui
from matplotlib import cm
from typing import Callable
from ..GLWidgets import *

__all__ = [
    "QParamSlider",
    "QPathSelector",
    "QDirectorySelector",
    "QTextEditor",
    "QOptionSelector",
    "QStatusViewer",
    "QCheckBox",
    "QPushButton",
    "QCheckList",
    "QTableSlider",
    "QTablePanel",
    "QMenu",
    "QImageViewWidget",
    "VisualizeWidget",
    "qtcv",
    "QtThread",
    "QSynchronizer",
    "add_line",
    "create_layout"
]


#-- 输入工具
class QParamSlider(QWidget):
    """带参数显示的滑动条"""
    sigValueChanged = pyqtSignal()
    def __init__(self, parent, key, value, start, stop, step, type: str="int"):
        """type:  int / float """
        super().__init__(parent)
        self.key = key
        self.type = type
        self.step = abs(step)  # > 0
        self.start = start
        self.stop = stop
        self.initUI()
        self.slider.setValue(self.val2idx(value))
        self.spinbox.setValue(value)

        self.slider.valueChanged.connect(lambda idx: self.spinbox.setValue(self.idx2val(idx)))
        self.spinbox.valueChanged.connect(lambda val: self.slider.setValue(self.val2idx(val)))
        self.slider.valueChanged.connect(self.sigValueChanged)

    def idx2val(self, idx):
        return self.start + self.step * idx

    def val2idx(self, val):
        return round((val - self.start) / self.step)

    @property
    def index(self):
        return self.slider.value()

    def initUI(self):
        self.setMaximumWidth(450)
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(5)

        self.label = QtWidgets.QLabel(self.key, self)
        # self.label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignRight)

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setMinimumWidth(40)
        self.slider.setRange(0, int((self.stop - self.start) / self.step) )

        self.spinbox = QtWidgets.QDoubleSpinBox(self)
        self.spinbox.setRange(self.start, self.stop)
        self.spinbox.setSingleStep(self.step)
        self.spinbox.setMinimumWidth(60)
        if self.type == "int":
            self.spinbox.setDecimals(0)
        else:
            self.spinbox.setDecimals(3)

        hbox.addWidget(self.label, 2)
        hbox.addWidget(self.slider, 5)
        hbox.addWidget(self.spinbox, 2)


    @property
    def value(self):
        if self.type == "int":
            return int(self.spinbox.value())
        return self.spinbox.value()

    @value.setter
    def value(self, val):
        self.spinbox.setValue(val)

class QPathSelector(QWidget):
    sigValueChanged = pyqtSignal()
    def __init__(self, parent, name:str, default_path:str):
        super().__init__(parent)
        self.setMaximumWidth(450)
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        self.name_label = QtWidgets.QLabel(name, self)
        self.path_editor = QtWidgets.QLineEdit(default_path, self)
        # self.path_editor.setMaximumWidth(450)
        self.path_button = QtWidgets.QPushButton("...", self)
        self.path_button.setFixedWidth(30)
        hbox.addWidget(self.name_label, 2)
        hbox.addWidget(self.path_editor, 5)
        hbox.addWidget(self.path_button, 2)
        # 信号/槽
        self.path_button.clicked.connect(self.select_path)
        self.path_editor.editingFinished.connect(self.sigValueChanged)

    @pyqtSlot()
    def select_path(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Path", options=options)
        if path and path != self.value:
            self.path_editor.setText(path)
            self.sigValueChanged.emit()

    @property
    def value(self):
        return self.path_editor.text()

    @value.setter
    def value(self, val):
        if val != self.value:
            self.path_editor.setText(val)
            self.sigValueChanged.emit()

    def setTight(self):
        self.name_label.adjustSize()
        self.name_label.setFixedWidth(self.name_label.width())

class QDirectorySelector(QWidget):
    sigValueChanged = pyqtSignal()
    def __init__(self, parent, name:str, default_path:str):
        super().__init__(parent)
        self.setMaximumWidth(450)
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        self.name_label = QtWidgets.QLabel(name, self)
        self.dir_editor = QtWidgets.QLineEdit(default_path, self)
        # self.path_editor.setMaximumWidth(450)
        self.path_button = QtWidgets.QPushButton("...", self)
        self.path_button.setFixedWidth(30)
        hbox.addWidget(self.name_label, 2)
        hbox.addWidget(self.dir_editor, 5)
        hbox.addWidget(self.path_button, 2)
        # 信号/槽
        self.path_button.clicked.connect(self.select_path)
        self.dir_editor.editingFinished.connect(self.sigValueChanged)

    @pyqtSlot()
    def select_path(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        if dir and dir != self.value:
            self.dir_editor.setText(dir)
            self.sigValueChanged.emit()

    @property
    def value(self):
        return self.dir_editor.text()

    @value.setter
    def value(self, val):
        self.dir_editor.setText(val)

    def setTight(self):
        self.name_label.adjustSize()
        self.name_label.setFixedWidth(self.name_label.width())

class QTextEditor(QWidget):
    sigValueChanged = pyqtSignal()
    def __init__(self, parent, name:str, default:str, editable=False):
        super().__init__(parent)
        self.setMaximumWidth(450)
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        self.name_label = QtWidgets.QLabel(name, self)
        self.text_editor = QtWidgets.QLineEdit(default, self)
        if not editable:
            self.text_editor.setFocusPolicy(Qt.NoFocus)
        hbox.addWidget(self.name_label, 2)
        hbox.addWidget(self.text_editor, 7)
        # 信号/槽
        self.text_editor.returnPressed.connect(self.sigValueChanged)

    @property
    def value(self):
        return self.text_editor.text()

    @value.setter
    def value(self, val):
        self.text_editor.setText(val)

    def setTight(self):
        self.name_label.adjustSize()
        self.name_label.setFixedWidth(self.name_label.width())

class QOptionSelector(QWidget):
    sigValueChanged = pyqtSignal()
    def __init__(self, parent, name:str, items: typing.List[str]):
        super().__init__(parent)
        self.setMaximumWidth(450)
        self.hbox = QtWidgets.QHBoxLayout(self)
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.setSpacing(0)
        self.name_label = QtWidgets.QLabel(name, self)
        self.options = QtWidgets.QComboBox(self)
        self.options.setMinimumWidth(65)
        self.options.addItems(items)

        self.hbox.addWidget(self.name_label, 1)
        self.hbox.addWidget(self.options, 2)
        # 信号/槽
        self.options.currentIndexChanged.connect(self.sigValueChanged)

    @property
    def value(self):
        return self.options.currentText()

    @value.setter
    def value(self, val):
        self.options.setCurrentText(val)

    def updateItems(self, items):
        self.options.clear()
        self.options.addItems(items)

    def setTight(self):
        self.name_label.adjustSize()
        self.name_label.setFixedWidth(self.name_label.width())

class QStatusViewer(QWidget):
    def __init__(self, parent, key:str, val:bool):
        super().__init__(parent)
        self.key = key
        self._val = val
        self._color = QtGui.QColor(0,200,0) if val else QtGui.QColor(200,0,0)
        # 字体长度
        fm = QtGui.QFontMetrics(self.font())
        self.text_width = fm.width(self.key)
        self.setFixedSize(self.text_width + 55, 25)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        # painter.setFont(QtGui.QFont("Arial", 10))
        painter.setBrush(self._color)
        painter.drawText(0, 19, self.key)
        painter.drawEllipse(QPoint(self.text_width+20, 12), 8, 8)
        return super().paintEvent(a0)

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, val):
        if self._val != val:
            self._val = val
            self._color = QtGui.QColor(0,200,0) if val else QtGui.QColor(200,0,0)
            self.update()

class QCheckBox(QtWidgets.QCheckBox):
    def __init__(self, parent, key, val):
        super().__init__(key, parent)
        self.setChecked(val)

    @property
    def value(self):
        return self.isChecked()

    @value.setter
    def value(self, val):
        self.setChecked(val)

class QPushButton(QtWidgets.QPushButton):
    def __init__(self, parent, key, val=False, checkable=False):
        super().__init__(key, parent)
        self.setMaximumWidth(450)
        if checkable:
            self.setCheckable(True)
            self.setChecked(val)

    @property
    def value(self):
        return self.isChecked()

    @value.setter
    def value(self, val):
        self.setChecked(val)

class QCheckList(QWidget):
    """互斥 check list"""
    sigValueChanged = pyqtSignal()
    def __init__(self, parent, item_list:typing.List[str], id:int=0, type="h", exclusive=True):
        super().__init__(parent=parent)
        self.exclusive = exclusive
        if type == "v":
            self.box = QtWidgets.QVBoxLayout(self)
        elif type == "h":
            self.box = QtWidgets.QHBoxLayout(self)

        self.box.setContentsMargins(10, 10, 0, 0)
        self.box_group = QtWidgets.QButtonGroup(self)
        for i, item in enumerate(item_list):
            checkbox = QtWidgets.QCheckBox(str(item), self)
            checkbox.setChecked(False)
            self.box.addWidget(checkbox)
            self.box_group.addButton(checkbox, i)
        # 互斥
        self.box_group.setExclusive(exclusive)
        self.box_group.buttonClicked.connect(self.sigValueChanged)
        self.value = id

    @property
    def value(self):
        if self.exclusive:
            return self.box_group.checkedId()
        else:
            ret = []
            for i, button in enumerate(self.box_group.buttons()):
                if button.isChecked():
                    ret.append(i)
            return ret

    @value.setter
    def value(self, val: int):
        self.box_group.button(val).setChecked(True)

    @property
    def checked_name(self):
        return self.box_group.checkedButton().text()

class QTablePanel(QWidget):
    sigTableChanged = pyqtSignal(str)
    def __init__(self, parent, table: dict, name: str=None, type="v"):
        super().__init__(parent)
        if name is not None:
            self.setObjectName(name)
        self._table = dict()  # {key: value, }
        self._widgets = dict()
        # 定义布局
        if type == "v":
            self.box = QtWidgets.QVBoxLayout(self)
            spacer = QtWidgets.QSpacerItem(0, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        elif type == "h":
            self.box = QtWidgets.QHBoxLayout(self)
            spacer = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # 定义 widgets
        for key, val in table.items():
            if val[0] in ("int", "float"):  # 连续变量
                widget = QParamSlider(self, key, *val[1:5], val[0])
                widget.sigValueChanged.connect(self.onTableChanged)
                self._table[key] = val[1]
            elif val[0] == "path":  # path
                widget = QPathSelector(self, key, val[1])
                widget.sigValueChanged.connect(self.onTableChanged)
                self._table[key] = val[1]
            elif val[0] == "directory":  # path
                widget = QDirectorySelector(self, key, val[1])
                widget.sigValueChanged.connect(self.onTableChanged)
                self._table[key] = val[1]
            elif val[0] == "str":  # 文本输入
                widget = QTextEditor(self, key, val[1])
                widget.sigValueChanged.connect(self.onTableChanged)
                self._table[key] = val[1]
            elif val[0] == "bool":
                widget = QCheckBox(self, key, val[1])
                widget.clicked.connect(self.onTableChanged)
                self._table[key] = val[1]
            elif val[0] == "option":
                widget = QOptionSelector(self, key, val[2])
                widget.sigValueChanged.connect(self.onTableChanged)
                self._table[key] = val[2][val[1]]
            elif val[0] == "checkbutton":
                widget = QPushButton(self, key, val[1], True)
                widget.clicked.connect(self.onTableChanged)
                self._table[key] = val[1]
            elif val[0] == "button":
                widget = QPushButton(self, key)
                widget.clicked.connect(self.onTableChanged)
                self._table[key] = False
            elif val[0] == "status":  # 对外展示, 不发出 onTableChanged 信号
                widget = QStatusViewer(self, key, val[1])
                self._table[key] = False
            elif val[0] == "label":  # 对外展示, 不发出 onTableChanged 信号
                widget = QtWidgets.QLabel(key, self)
            elif val[0] == "checklist":
                widget = QCheckList(self, val[2], val[1], *val[3:])
                widget.sigValueChanged.connect(self.onTableChanged, 0)
                self._table[key] = val[1]
            elif val[0] == "line":
                line_type = 'v' if type=='h' else 'h'
                add_line(self.box, type=line_type)
                continue
            elif val[0] == "spacer":
                size = (1, val[1]) if type=='v' else (val[1], 1)
                spacer1 = QtWidgets.QSpacerItem(size[0], size[1], QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
                self.box.addItem(spacer1)
                continue
            widget.setObjectName(key)
            self.box.addWidget(widget)
            self._widgets[key] = widget
        self.box.setSpacing(10)
        self.box.addItem(spacer)

    @pyqtSlot()
    def onTableChanged(self):
        """更新 table 的值"""
        sender = self.sender()
        key = sender.objectName()
        self._table[key] = sender.value
        self.sigTableChanged.emit(key)

    @property
    def table(self):
        return self._table

    def update(self, new_table):
        for key, val in new_table.items():
            self.__setitem__(key, val)

    def __getitem__(self, key):
        return self._table[key]

    def __setitem__(self, key, val):
        if self._table[key] != val:
            self._widgets[key].value = val
            self._table[key] = self._widgets[key].value
            # self.sigTableChanged.emit(key)

    def __iter__(self):
        """dict(self) 将实例转化为 dict"""
        return iter(self._table.items())

    def __repr__(self):
        return f"QTablePanel({self._table.__repr__()})"

    def keys(self):
        return self._table.keys()

    def get(self, key, default=None):
        return self._table.get(key, default)

    def setStretch(self, index, stretch):
        self.box.setStretch(index, stretch)

    def setStretchs(self, stretchs: typing.List[int]):
        for i, v in enumerate(stretchs):
            self.box.setStretch(i, v)

    def widget(self, key):
        return self._widgets.get(key, None)

# 使 yaml 可以序列化 QTablePanel
yaml.add_representer(QTablePanel, lambda dumper, panel: dumper.represent_dict(dict(panel)))


def QColormapPanel():
    colormap_panel={
        "channel": ["checklist", 0, ["B", "G", "R"]],
        "colormap": ["option", 0, ["coolwarm", "viridis", "plasma", "inferno", "cividis", "jet"]],
        "cm_scale": ["float", 40, 1, 250, 1],
        "cm_bias": ["float", 0.2, -0.5, 0.5, 0.01],
    }
    colormap_panel = QTablePanel(None, colormap_panel, type="h")
    colormap_panel.widget("colormap").name_label.setText("cm")
    colormap_panel.widget("colormap").hbox.setStretch(1,1)

    colormap_panel.setStretchs([1,1,3,3])
    colormap_panel.box.setSpacing(10)
    colormap_panel.box.setContentsMargins(0, 0, 0, 0)
    colormap_panel.setMaximumWidth(900)
    colormap_panel.setMaximumHeight(30)
    return colormap_panel


def QGridMovepanel():
    grid_move_panel={
        "grid1": ["checkbutton", 1],
        "grid2": ["checkbutton", 0],
        "grid1_z": ["int", 0, -255, 255, 1],
        "grid2_z": ["int", 255, -255, 255, 1],
    }
    grid_move_panel = QTablePanel(None, grid_move_panel, type="h")
    # colormap_panel.setContentsMargins(0, 0, 0, 0)
    grid_move_panel.setStretchs([1,1,5,5])
    grid_move_panel.box.setSpacing(30)
    grid_move_panel.box.setContentsMargins(0, 0, 0, 0)
    grid_move_panel.setMaximumWidth(800)
    grid_move_panel.setMaximumHeight(30)
    # grid_move_panel.setStyleSheet("background-color: white; ")
    return grid_move_panel


class QImageViewWidget(QtWidgets.QLabel):
    sigKeyPressed = pyqtSignal(QtGui.QKeyEvent)
    sigMousePressed = pyqtSignal(tuple)
    sigMouseReleased = pyqtSignal(tuple)
    sigMousePressedMove = pyqtSignal(tuple)
    def __init__(self, parent=None, auto_scale=True, show_text=True):
        """auto_scale: 等比例缩放图片填充窗口"""
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setMouseTracking(True)
        self.x_mouse = 0
        self.y_mouse = 0
        self.x_img = 0
        self.y_img = 0
        self.q_img = None
        self.show_text = show_text
        self.auto_scale = auto_scale
        self.mouse_pressed = False

    def img_size(self):
        if self.q_img is None:
            return (640, 480)
        return (self.q_img.width(), self.q_img.height())

    def setData(self, img):
        img = img.astype(np.uint8)
        if img.ndim==3 and img.shape[2]==3:  # 彩色图
            self.q_img = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.size // img.shape[0],
                                 QtGui.QImage.Format_BGR888)#.scaled(self.size(), Qt.KeepAspectRatio)
        else:
            self.q_img = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.size // img.shape[0],
                           QtGui.QImage.Format_Grayscale8)#.scaled(self.size(), Qt.KeepAspectRatio)
        self.update()

    def resizeEvent(self, a0: QResizeEvent) -> None:
        if self.q_img is not None and self.auto_scale:
            self.update()
        return super().resizeEvent(a0)

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        self.sigKeyPressed.emit(ev)

    def mouseMoveEvent(self, event):
        self.x_mouse, self.y_mouse = event.x(), event.y()
        if self.q_img is not None:
            self.update()
        if self.mouse_pressed:
            self.sigMousePressedMove.emit((self.x_img, self.y_img))

    def mouseDoubleClickEvent(self, a0) -> None:
        """双击关闭 Pos, Color 展示"""
        self.show_text = not self.show_text
        return super().mouseDoubleClickEvent(a0)

    def mousePressEvent(self, ev) -> None:
        self.mouse_pressed = True
        self.sigMousePressed.emit((self.x_img, self.y_img))
        return super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:
        self.mouse_pressed = False
        self.sigMouseReleased.emit((self.x_img, self.y_img))
        return super().mouseReleaseEvent(ev)

    def update(self):
        if self.auto_scale:  # 缩放
            q_img = self.q_img.scaled(self.size(), Qt.KeepAspectRatio)
        else:
            q_img = self.q_img.copy()
        # 将鼠标坐标变换为图片坐标
        win_size = self.size()
        img_size = q_img.size()
        x = min(self.x_mouse, img_size.width()-1)
        y = self.y_mouse + int((-win_size.height() + img_size.height() - 1) / 2)
        y = min(max(y, 0), img_size.height()-1)
        if self.auto_scale:
            scale_rate = max(q_img.width() / self.q_img.width(), 1e-4)
            self.x_img = int(x / scale_rate)
            self.y_img = int(y / scale_rate)
        else:
            self.x_img = x
            self.y_img = y

        if self.show_text:
            # 获取当前位置 rgb
            color = self.q_img.pixelColor(self.x_img, self.y_img)
            r, g, b = color.red(), color.green(), color.blue()
            # 绘制信息
            painter = QtGui.QPainter()
            painter.begin(q_img)
            painter.setPen(QtGui.QColor(220,220,220))
            painter.drawText(10, 20, f"Pos: ({self.x_img}, {self.y_img})")
            painter.drawText(10, 36, f"RGB: ")
            painter.setPen(QtGui.QColor(255,0,0))
            painter.drawText(45, 36, str(r))
            painter.setPen(QtGui.QColor(0,255,0))
            painter.drawText(75, 36, str(g))
            painter.setPen(QtGui.QColor(0,100,255))
            painter.drawText(105, 36, str(b))
            painter.end()
        self.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def get_img(self):
        h, w = self.q_img.height(), self.q_img.width()
        bytes = self.q_img.bits().asstring(self.q_img.byteCount())
        img = np.frombuffer(bytes, np.uint8).reshape((h, w, self.q_img.depth() // 8))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

class VisualizeType():
    Image = 0
    Surface = 1
    PointCloud = 2
    Quiver = 3
    SurfaceQuiver = 4
    gelslim = 5


class VisualizeWidget(QtWidgets.QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.init_ui()
        self.setData = self.set_data
        self._screen = QtWidgets.QApplication.screenAt(QtGui.QCursor.pos())

    def init_ui(self):
        self.vbox = QtWidgets.QVBoxLayout(self)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(0)

        self.tab_widget = QtWidgets.QTabWidget(self)
        self.vbox.addWidget(self.tab_widget)

        self.color_map_panel = QColormapPanel()
        self.color_map_panel.setVisible(False)
        self.vbox.addWidget(self.color_map_panel)

        self.grid_move_panel = QGridMovepanel()
        self.grid_move_panel.setVisible(False)
        self.vbox.addWidget(self.grid_move_panel)

        self.image_view = QImageViewWidget(self, auto_scale=True)
        self.surface_view = QSurfaceWidget(self)
        self.pc_view = QPointCloudWidget(self, point_size=35)
        self.quiver_view = QQuiverWidget(self, width=4)
        self.surface_quiver_view = QSurfaceQuiverWidget(self, width=4)

        self.tab_widget.addTab(self.image_view, "image")  # 0
        self.tab_widget.addTab(self.surface_view, "surface")  # 1
        self.tab_widget.addTab(self.pc_view, "pointcloud")  # 2
        self.tab_widget.addTab(self.quiver_view, "quiver")  # 3
        self.tab_widget.addTab(self.surface_quiver_view, "surfacequiver")  # 4
        self.tab_widget.addTab(QGelSlimWidget(self), "gelslim")  # 5

        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        self.grid_move_panel.sigTableChanged.connect(self.on_grid_changed)

    def set_data(self, img=None, start_pts=None, stop_pts=None, tab_index=None):
        if tab_index is not None:
            self.tab_widget.setCurrentIndex(tab_index)

        if self.tab_widget.currentIndex() == VisualizeType.Image and img is not None:
            self.tab_widget.currentWidget().setData(img)

        elif self.tab_widget.currentIndex() in [VisualizeType.Surface, VisualizeType.PointCloud] and \
            img is not None:
            while (img.shape[0] > 800):
                img = cv2.pyrDown(img)
            self.tab_widget.currentWidget().setData(img, **self.color_map_panel.table)

        elif self.tab_widget.currentIndex() == VisualizeType.Quiver and start_pts is not None \
            and stop_pts is not None:
            self.tab_widget.currentWidget().setData(start_pts, stop_pts)

        elif self.tab_widget.currentIndex() == VisualizeType.SurfaceQuiver and img is not None and start_pts is not None \
            and stop_pts is not None:
            self.tab_widget.currentWidget().setData(img, start_pts, stop_pts, **self.color_map_panel.table)

        elif self.tab_widget.currentIndex() == VisualizeType.gelslim :
            self.tab_widget.currentWidget().setData(img, start_pts, stop_pts)

    @pyqtSlot()
    def on_tab_changed(self):
        idx = self.tab_widget.currentIndex()
        if idx in (1, 2, 4):
            self.grid_move_panel.setVisible(True)
            self.color_map_panel.setVisible(True)
        elif idx == 3:
            self.grid_move_panel.setVisible(True)
            self.color_map_panel.setVisible(False)
        else:
            self.grid_move_panel.setVisible(False)
            self.color_map_panel.setVisible(False)

    @pyqtSlot()
    def on_grid_changed(self):
        self.tab_widget.currentWidget().set_grid(**self.grid_move_panel.table)

    @property
    def type(self):
        return self.tab_widget.tabText(self.tab_widget.currentIndex())

    @property
    def qrect(self):
        """获取当前显示窗口的位置"""
        widget = self.tab_widget.currentWidget()

        # 如果是图像, 则直接返回图像大小, 而不是实际窗口大小, 因为无需截图
        if isinstance(widget, QImageViewWidget):
            w, h = widget.img_size()
            return QRect(0, 0, w, h)

        pos = widget.mapToGlobal(widget.pos())
        w, h = widget.width(), widget.height()
        return QRect(pos.x(), pos.y(), w, h)

    def grab_frame(self):
        """捕获当前显示图像"""
        if self.tab_widget.currentIndex() == VisualizeType.Image:
            return self.tab_widget.currentWidget().get_img()

        widget = self.tab_widget.currentWidget()
        pos = widget.mapToGlobal(widget.pos())
        w, h = widget.width(), widget.height()

        pixmap = self._screen.grabWindow(0).copy(QRect(pos.x(), pos.y(), w, h))
        qimage = pixmap.toImage()
        bytes = qimage.bits().asstring(qimage.byteCount())
        img = np.frombuffer(bytes, np.uint8).reshape((h, w, 4))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


class QTableSlider(QWidget):
    """使用字典定义连续变量, 并可视化控制
    字典格式 {key: [value, start, stop, step], }
    """
    sigTableChanged = pyqtSignal()
    def __init__(self, parent, table: dict):
        super().__init__(parent)
        self._table = dict()  # {key: value, }
        self.vbox = QtWidgets.QVBoxLayout(self)
        for key, val in table.items():
            paramSlider = QParamSlider(self, key, *val)
            paramSlider.setObjectName(key)
            paramSlider.sigValueChanged.connect(self.onTableChanged)
            self.vbox.addWidget(paramSlider)
            self.table[key] = val[0]

    @pyqtSlot()
    def onTableChanged(self):
        sender = self.sender()
        self._table[sender.key] = sender.value
        self.sigTableChanged.emit()

    @property
    def table(self):
        return self._table


class QMenu(QWidget):
    """根据菜单列表创建一组菜单按钮"""
    sigMenuChanged = pyqtSignal(str)
    def __init__(self, parent, menu_list: list, type="v"):
        super().__init__(parent)
        self.menu_list = menu_list
        # 定义布局
        if type == "v":
            self.box = QtWidgets.QVBoxLayout(self)
            spacer = QtWidgets.QSpacerItem(0, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        elif type == "h":
            self.box = QtWidgets.QHBoxLayout(self)
            spacer = QtWidgets.QSpacerItem(20, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.box.setContentsMargins(0,0,0,0)
        self.box.setSpacing(0) # 设置布局中的间隙为0
        self.button_group = QtWidgets.QButtonGroup(self)  # 创建一个按钮组
        self.button_group.setExclusive(True) # 设置按钮组为互斥的
        # 定义按钮
        for menu_item in self.menu_list:
            button = QtWidgets.QPushButton(menu_item) # 创建一个按钮，并设置文本
            button.setCheckable (True) # 设置按钮为可选中的
            self.box.addWidget(button) # 将按钮添加到布局中
            self.button_group.addButton(button)
        self.box.addItem(spacer)
        # 信号 / 槽
        self.button_group.buttonClicked.connect(self.on_menu_changed)

    def on_menu_changed(self, button):
        self.sigMenuChanged.emit(button.text())


def add_line(layout, type: str='h'):
    line = QtWidgets.QFrame(layout.parent())
    if type == "h":
        line.setFrameShape(QtWidgets.QFrame.HLine)
    if type == "v":
        line.setFrameShape(QtWidgets.QFrame.VLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    layout.addWidget(line)


def create_layout(
    parent,
    type : str = "v",
    widgets : list = None,
    stretchs : list = None,
    content_margins : tuple = (0, 0, 0, 0),
    spacing : int = 0,
):
    """创建布局"""
    widgets = widgets if widgets is not None else []
    stretchs = stretchs if stretchs is not None else []

    layout = QtWidgets.QVBoxLayout(parent) if type == "v" else QtWidgets.QHBoxLayout(parent)
    layout.setContentsMargins(*content_margins)
    layout.setSpacing(spacing)
    for i, widget in enumerate(widgets):
        if i > len(stretchs) - 1:
            layout.addWidget(widget)
        else:
            layout.addWidget(widget, stretchs[i])
    return layout


class QTCV():
    def __init__(self):
        self.app_is_init = False
        self.Key = Qt.Key

    def init(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        self.loop = QEventLoop()
        self.win_list = dict()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.on_timeout)
        self.key = None
        self.app_is_init = True

    def imshow(self, name: str, img, auto_scale=True):
        if not self.app_is_init:
            self.init()
        if not name in self.win_list:
            self.win_list[name] = QImageViewWidget(None, auto_scale=auto_scale)
            self.win_list[name].move(len(self.win_list)*100, len(self.win_list)*100)
            self.win_list[name].resize(img.shape[1], img.shape[0])
            self.win_list[name].setWindowTitle(name)
            self.win_list[name].keyPressEvent = self.keyPressEvent
            self.win_list[name].closeEvent = self.closeEvent
        # self.win_list[name].resize(img.shape[1], img.shape[0])
        self.win_list[name].setData(img)
        self.win_list[name].show()

    def waitKey(self, delay):
        if delay > 0:
            self.timer.start(delay)
        self.loop.exec()
        return self.key

    def on_timeout(self):
        self.key = None
        self.loop.quit()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        self.key = event.key()
        self.timer.stop()
        self.loop.quit()

    def closeEvent(self, event):
        self.key = None
        self.timer.stop()
        self.loop.quit()

    def destroyAllWindows(self):
        self.win_list = dict()

qtcv = QTCV()


class QtThread(QThread):
    sigFinished = pyqtSignal(tuple)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        res = self.func(*self.args, **self.kwargs)
        # 任务完成后发出信号
        self.sigFinished.emit(res)


class QSynchronizer():
    def __init__(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        self.loop = QEventLoop()

    def wait_until(self, condition: Callable, period=0.02, timeout=20, *args, **kwargs):
        self.condition = lambda: condition(*args, **kwargs)
        self.t_0 = time()
        self.timeout = timeout
        self.period = period

        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timeout)
        self.timer.start(int(period*1000))
        if not self.loop.isRunning():
            self.loop.exec()

    def on_timeout(self):
        if self.condition() or time()-self.t_0 > self.timeout:
            self.loop.quit()


if __name__ == '__main__':

    params = {
        "path": ["path", "a"],
        "str": ["str", ""],
        "int": ["int", -20, -30, 100, 1],
        "float": ["float", 0.89, -1, 1, 0.01],
        "Option": ["option", 0, ["a", "b", "c"]],
        "Option1": ["option", 0, [""]],
        "Bool": ["bool", 0],
        "Click": ["button", ],
        "Check": ["checkbutton", 1],
        "Bool1": ["status", 1],
        "Status1": ["status", 0],
        "CheckList": ["checklist", 0, ["r", "g", "b"]],
        "line1": ["line",]
    }
    app = QApplication(sys.argv)
    mainwindow = QTablePanel(None, params, 'v')
    mainwindow.show()
    sys.exit(app.exec())
