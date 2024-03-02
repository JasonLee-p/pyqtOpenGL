from contextlib import contextmanager
from typing import List, Union, Dict, Callable, Type, Tuple
from PyQt5.QtCore import Qt, QPoint, QSize, QEvent
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFocusEvent, QMouseEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
import numpy as np

Number = Union[int, float]
NumberTuple = Tuple[Number]

def create_layout(
    parent,
    horizontal : bool = True,
    widgets : list = None,
    stretchs : list = None,
    content_margins : tuple = (0, 0, 0, 0),
    spacing : int = 0,
):
    """创建布局"""
    widgets = widgets if widgets is not None else []
    stretchs = stretchs if stretchs is not None else []

    layout = QtWidgets.QHBoxLayout(parent) if horizontal else QtWidgets.QVBoxLayout(parent)
    layout.setContentsMargins(*content_margins)
    layout.setSpacing(spacing)
    for i, widget in enumerate(widgets):
        if i > len(stretchs) - 1:
            layout.addWidget(widget)
        else:
            layout.addWidget(widget, stretchs[i])
    return layout


class CollapseTitleBar(QtWidgets.QFrame):
    """折叠标题栏"""
    toggleCollapsed = QtCore.pyqtSignal(bool)

    def __init__(self, parent:QWidget):
        super().__init__(parent)
        self.setupUi()
        self.is_collapsed = False
        self.collapse_button.clicked.connect(self.on_click)
        self.close_button.clicked.connect(parent.close)

    def setLabel(self, label):
        self.label.setText(label)

    def setupUi(self):
        sizeFixedPolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Fixed)
        self.setSizePolicy(sizeFixedPolicy)
        self.setStyleSheet("background-color: #0f5687; color: white;")

        self.collapse_button = QtWidgets.QPushButton("▾", self)
        self.collapse_button.setFixedSize(QSize(20, 20))
        self.collapse_button.setStyleSheet(
            "QPushButton { border-radius: 10px; }"
            "QPushButton:hover { background-color: #288ad4; }"
        )

        self.close_button = QtWidgets.QPushButton("×", self)
        self.close_button.setFixedSize(QSize(25, 25))
        self.close_button.setStyleSheet(
            "QPushButton { border: 0px; }"
            "QPushButton:hover { background-color: #288ad4; }"
        )

        self.label = QtWidgets.QLabel(self)
        self.label.setMinimumSize(QSize(0, 25))

        self.hbox = create_layout(self, True,
                                  [self.collapse_button, self.label, self.close_button],
                                  [1, 1, 1], content_margins=(5, 0, 0, 0), spacing=5)

    def on_click(self):
        self.is_collapsed = not self.is_collapsed
        self.collapse_button.setText("▾▸"[self.is_collapsed])
        self.toggleCollapsed.emit(self.is_collapsed)


class ToolItem():

    def set_label(self, label:str):
        self.__label = label

    def get_label(self) -> str:
        return self.__label

    @property
    def value(self):
        raise NotImplementedError

    @value.setter
    def value(self, val):
        raise NotImplementedError


class ToolContainer():

    def add_item(self, item: Union[ToolItem, QWidget]):
        raise NotImplementedError

    def get_layout(self):
        raise NotImplementedError


class ToolGroup(QtWidgets.QWidget, ToolContainer):
    """不带边框的group"""

    def __init__(self, horizontal=True, spacing=5):
        super().__init__()
        self.container_box = QtWidgets.QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        self.container_box.setContentsMargins(0, 0, 0, 0)
        self.container_box.setSpacing(spacing)
        self.setMinimumWidth(200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setSizePolicy(sizePolicy)

    def add_item(self, item: Union[ToolItem, QWidget]):
        return self.container_box.addWidget(item)

    def get_layout(self):
        return self.container_box


class ToolGroupBox(QtWidgets.QGroupBox, ToolContainer):

    def __init__(self, label:str="", horizontal=True, margins:List[int]=(5,15,5,15), spacing=5):
        super().__init__()
        self.setTitle(label)
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #0f5687;  /* 设置边框样式和颜色 */
                border-radius: 5px;  /* 设置边框圆角 */
                background-color: #f2f2f2;  /* 设置背景颜色 */
                margin-top: 3ex;  /* 上边距 */
            }
            QGroupBox:title {
                subcontrol-origin: margin;
                subcontrol-position: top left; /* 设置标题位置 */
                padding: 0 0px;  /* 设置标题的内边距 */
                color:  #0f5687;  /* 设置标题的颜色 */
            }
        """)
        self.container_box = QtWidgets.QHBoxLayout(self) if horizontal else QVBoxLayout(self)
        self.container_box.setAlignment(Qt.AlignLeft)
        self.container_box.setContentsMargins(*margins)
        self.container_box.setSpacing(spacing)
        self.setMinimumWidth(200)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setSizePolicy(sizePolicy)

    def add_item(self, item: Union[ToolItem, QWidget]):
        return self.container_box.addWidget(item)

    def get_layout(self):
        return self.container_box


class ToolWindow(QtWidgets.QWidget, ToolContainer):

    def __init__(self, label="ToolWindow", spacing=5,
                 pos:Union[Tuple[int], List[int]]=(0, 0)):
        super().__init__()
        self.setupUi()
        self.title_bar.setLabel(label)
        self.container_box.setSpacing(spacing)

        # 用于保存鼠标点击位置的变量
        self.drag_position = QPoint()
        self.movable = False

        # signals
        self.title_bar.toggleCollapsed.connect(self.on_toggleCollapsed)
        self.move(pos[0], pos[1])
        self.show()

    def setupUi(self):
        # 设置窗口样式为无边框
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # 创建一个自定义标题栏
        self.title_bar = CollapseTitleBar(self)

        # 容器
        self.container = QtWidgets.QFrame()
        self.container.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                     QtWidgets.QSizePolicy.Minimum)

        # 创建主布局
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.container)
        self.main_layout.addItem(QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Expanding,
                                                       QtWidgets.QSizePolicy.Expanding))
        # 添加一个QSizeGrip到布局的右下角
        self.size_grip = QtWidgets.QSizeGrip(self)
        self.main_layout.addWidget(self.size_grip, 0, Qt.AlignBottom | Qt.AlignRight)

        # 创建容器的布局
        self.container_box = QVBoxLayout(self.container)

    def on_toggleCollapsed(self, is_collapsed):
        # 折叠/展开窗口
        if is_collapsed:
            self.last_height = self.height()
            self.container.setVisible(False)
            self.size_grip.setVisible(False)
            self.setMaximumHeight(25)
            self.adjustSize()
        else:
            self.setMaximumHeight(5000)
            self.container.setVisible(True)
            self.size_grip.setVisible(True)

    def mousePressEvent(self, event):
        self.setFocus()  # 设置窗口为焦点, 实现点击窗口任意位置, 取消子组件的焦点的效果
        # 保存鼠标点击位置
        self.movable = False
        if self.title_bar.hbox.geometry().contains(event.pos()):
            self.movable = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        # 移动窗口位置
        if event.buttons() == Qt.LeftButton and self.movable:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, a0) -> None:
        self.movable = False
        return super().mouseReleaseEvent(a0)

    def add_item(self, Item: ToolItem):
        self.container_box.addWidget(Item)

    def get_layout(self):
        return self.container_box


class ButtonItem(QPushButton, ToolItem):

    def __init__(self, label="Button", value=False, checkable=False, callback: Callable=None):
        super().__init__(label)
        self.set_label(label)
        self.setMaximumWidth(300)
        if checkable:
            self.setCheckable(True)
            self.setChecked(value)
        if callback is not None:
            self.clicked.connect(callback)

    @property
    def value(self):
        return self.isChecked()

    @value.setter
    def value(self, val):
        self.setChecked(val)


class CheckBoxItem(QtWidgets.QCheckBox, ToolItem):

    def __init__(self, label, value=False, callback: Callable=None):
        super().__init__(label, None)
        self.set_label(label)
        self.setMaximumWidth(300)
        self.setChecked(value)
        if callback is not None:
            self.clicked.connect(callback)

    @property
    def value(self):
        return self.isChecked()

    @value.setter
    def value(self, val):
        self.setChecked(val)


class CheckListItem(QtWidgets.QFrame, ToolItem):
    """互斥 checkboxes"""
    sigClicked = QtCore.pyqtSignal(object)

    def __init__(self, label:str, items: Tuple[str], value:Union[int, List[bool]]=None,
                 horizontal=True, exclusive=True, callback: Callable=None):
        """value=None: 全部不选中, value=int: 选中第value个, value=List[bool]: 选中对应的checkbox"""
        super().__init__()
        self.set_label(label)
        # 设置边框颜色和底色
        self.setStyleSheet("QFrame{border:1px solid #aaaaaa; border-radius: 3px; background-color: #ffffff;}")
        self.exclusive = exclusive
        self._items = tuple(items)
        if horizontal:
            self.box = QtWidgets.QHBoxLayout(self)
        else:
            self.box = QtWidgets.QVBoxLayout(self)

        self.box.setContentsMargins(10, 5, 0, 5)
        self.box_group = QtWidgets.QButtonGroup(self)
        for i, item in enumerate(items):
            checkbox = QtWidgets.QRadioButton(str(item), self)
            checkbox.setChecked(False)
            self.box.addWidget(checkbox)
            self.box_group.addButton(checkbox, i)
        # 互斥
        self.box_group.setExclusive(exclusive)

        # 当状态发生变化时, 发射信号
        self.box_group.idToggled.connect(self._on_toggled)
        if callback is not None:
            self.sigClicked.connect(callback)

        self.value = value

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
    def value(self, val: Union[int, List[bool]]):
        if val is None:
            return
        elif isinstance(val, int):
            self.box_group.button(val).setChecked(True)
        else:
            for v, button in zip(val, self.box_group.buttons()):
                button.setChecked(v)

    @property
    def items(self):
        return self._items

    def _on_toggled(self, button_id):
        button = self.box_group.button(button_id)
        return self.sigClicked.emit((button.text(), button.isChecked()))


class ComboItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.pyqtSignal(object)

    def __init__(self, label:str, items: Tuple[str], value: int=0, callback: Callable=None):
        super().__init__()
        self.set_label(label)
        self.setMaximumWidth(400)
        self.combo = QtWidgets.QComboBox(self)
        self.name_label = QtWidgets.QLabel(label, self)
        self.combo.addItems(items)
        self.combo.setCurrentIndex(value)
        self.box = create_layout(self, True, [self.combo, self.name_label], [5, 2], spacing=10)
        # 信号/槽
        self.combo.currentTextChanged.connect(self._on_changed)
        if callback:
            self.sigChanged.connect(callback)

    def _on_changed(self):
        return self.sigChanged.emit(self.value)

    @property
    def value(self):
        return self.combo.currentText()

    @value.setter
    def value(self, val):
        self.combo.setCurrentText(val)

    def updateItems(self, items):
        self.combo.clear()
        self.combo.addItems(items)

class TextEditorItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.pyqtSignal(object)

    def __init__(self, label:str, value:str, editable=True, callback: Callable=None):
        super().__init__()
        self.set_label(label)
        self.setMaximumWidth(400)
        self.name_label = QtWidgets.QLabel(label, self)
        self.text_editor = QtWidgets.QLineEdit(value, self)
        if not editable:
            self.text_editor.setFocusPolicy(Qt.NoFocus)
        self.box = create_layout(self, True, [self.text_editor, self.name_label], [5, 2], spacing=10)
        # 信号/槽
        self.text_editor.editingFinished.connect(self._on_changed)
        if callback is not None:
            self.sigChanged.connect(callback)

    @property
    def value(self):
        return self.text_editor.text()

    @value.setter
    def value(self, val):
        self.text_editor.setText(val)

    def _on_changed(self):
        return self.sigChanged.emit(self.value)


class SliderItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.pyqtSignal(object)
    def __init__(self, label:str, value, min_val, max_val, step, decimals=0, callback: Callable=None):
        """decimals: 小数位数"""
        super().__init__()
        self.set_label(label)
        self.step = step
        self.decimals = decimals
        value = max(min_val, min(value, max_val))
        l_steps = int((value - min_val) / step)
        r_steps = int((max_val - value) / step)
        self.min_val = value - l_steps * step
        self.steps = l_steps + r_steps

        self.name_label = QtWidgets.QLabel(label, self)

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self.steps+1)

        self.spinbox = QtWidgets.QDoubleSpinBox(self)
        self.spinbox.setRange(self.min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(decimals)

        self.box = create_layout(self, True,
                                 [self.slider, self.spinbox, self.name_label],
                                 [9, 1, 4], spacing=10)

        self.spinbox.setValue(value)
        self.slider.setValue(int((value - self.min_val) / self.step))

        self.slider.valueChanged.connect(self._on_changed)
        self.spinbox.valueChanged.connect(self._on_changed)
        if callback is not None:
            self.sigChanged.connect(callback)

    @property
    def value(self):
        val = self.spinbox.value()
        return int(val) if self.decimals == 0 else val

    @value.setter
    def value(self, val):
        self.spinbox.setValue(val)

    def _on_changed(self):
        if isinstance(self.sender(), QtWidgets.QSlider):
            val = self.min_val + self.slider.value() * self.step
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(val)
            self.spinbox.blockSignals(False)
        else:
            val = self.spinbox.value()
            self.slider.blockSignals(True)
            self.slider.setValue(int((val - self.min_val) / self.step))
            self.slider.blockSignals(False)

        self.sigChanged.emit(self.value)


class ArrayTypeItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.pyqtSignal(object)

    def __init__(self, label:str, value:NumberTuple,
                 type:Union[Type[int], Type[float]], editable=True, callback: Callable=None):
        super().__init__()
        self.set_label(label)
        self._value = list(value)
        self.length = len(value)
        self.type = type
        assert self.length > 0, "Array length must be greater than 0"

        self.name_label = QtWidgets.QLabel(label, self)
        self.inputs_frame = QtWidgets.QFrame(self)
        self.inputs_layout = QtWidgets.QHBoxLayout(self.inputs_frame)
        self.inputs_layout.setContentsMargins(0, 0, 0, 0)

        self.inputs = []
        validator = QtGui.QIntValidator() if type == int else QtGui.QDoubleValidator()
        for i in range(self.length):
            input = QtWidgets.QLineEdit(str(self.type(value[i])), self.inputs_frame)
            input.setMinimumWidth(10)
            input.setValidator(validator)
            if not editable:
                input.setFocusPolicy(Qt.NoFocus)
            input.editingFinished.connect(self._on_changed)
            input.setObjectName(str(i))  # 设置objectName, 用于区分信号来源

            self.inputs.append(input)
            self.inputs_layout.addWidget(input, 1)

        self.box = create_layout(self, True,
                                 [self.inputs_frame, self.name_label], [5, 2], spacing=10)
        if callback is not None:
            self.sigChanged.connect(callback)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        if self._value == val:
            return
        for i in range(self.length):
            if self._value[i] != val[i]:
                self._value[i] = val[i]
                self.inputs[i].setText(str(val[i]))
        self.sigChanged.emit(self._value)

    def _on_changed(self):
        id = int(self.sender().objectName())
        val = self.type(self.inputs[id].text())
        if self._value[id] != val:
            self._value[id] = val
            self.sigChanged.emit(self._value)


class DragValue(QtWidgets.QLineEdit):

    sigValueChanged = QtCore.pyqtSignal(object)

    def __init__(self, value, min_val, max_val, step, decimals=2, format: str=None,
                 parent=None):
        super().__init__(parent)
        self.decimal_format = "%." + str(decimals) + "f"
        if format is None:
            self.format = self.decimal_format
        else:
            self.format = format
        self.drag_position = QPoint()  # 记录鼠标按下的位置
        self.pressed = False

        self._value = None
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.value = value
        self._on_press_value = self._value  # 记录鼠标按下时的值

        # 设置验证器
        double_validator = QtGui.QDoubleValidator()
        double_validator.setDecimals(decimals)
        self.setValidator(double_validator)

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(20)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet(
        """ QLineEdit {background-color: #f7f7f7; border: 1px solid #aaaaaa; border-radius: 3px; }
            QLineEdit:hover { background-color: #d0eef9; }
        """)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        _value = min(self.max_val, max(self.min_val, val))
        if self._value != _value:
            self._value = _value
            self.sigValueChanged.emit(self._value)
        self.setText(self.format % self._value)

    def mouseDoubleClickEvent(self, event):
        """双击可编辑"""
        self.setFocus()
        event.accept()

    def mousePressEvent(self, event):
        if not self.hasFocus():
            self._on_press_value = self._value
            self.pressed = True
            self.drag_position = event.pos()
        event.ignore()  # 忽略事件, 使父类 ToolWindow 能够接收到事件, 并获取 Focus

    def mouseReleaseEvent(self, event):
        self.pressed = False
        event.accept()

    def mouseMoveEvent(self, event):
        """按下并拖动可改变数值"""
        if self.pressed and not self.hasFocus():
            scale = 1
            # 如果按下shift键, 则按照最小步长移动
            if event.modifiers() == Qt.ShiftModifier:
                scale = 0.02
            # value.setter
            self.value = self._on_press_value + \
                int((event.pos().x() - self.drag_position.x()) * scale) * self.step
        event.accept()

    def keyPressEvent(self, event):
        """ESC或回车退出编辑"""
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Return:
            self.clearFocus() # 清除焦点
        else:
            super().keyPressEvent(event) # 调用父类的方法

    def focusInEvent(self, a0: QFocusEvent) -> None:
        # 修改文字为只包含数字
        self.setText(self.decimal_format % self._value)
        self.setAlignment(Qt.AlignLeft)
        return super().focusInEvent(a0)

    def focusOutEvent(self, event):
        """失去焦点时恢复原来的样式"""
        self.setAlignment(Qt.AlignCenter)
        super().focusOutEvent(event)
        # value.setter
        self.value = float(self.text())


class DragValueItem(QtWidgets.QWidget, ToolItem):

        sigChanged = QtCore.pyqtSignal(object)

        def __init__(self, label:str, value, min_val, max_val, step, decimals:int=0,
                     format: str=None,
                     callback: Callable=None):
            super().__init__()
            self.set_label(label)

            self.name_label = QtWidgets.QLabel(label, self)
            self.value_drager = DragValue(value, min_val, max_val, step, decimals, format, self)

            self.box = create_layout(self, True,
                                    [self.value_drager, self.name_label],
                                    [5, 2], spacing=10)

            self.value_drager.sigValueChanged.connect(self._on_changed)

            if callback is not None:
                self.sigChanged.connect(callback)

        def _on_changed(self, val):
            return self.sigChanged.emit(val)

        @property
        def value(self):
            return self.value_drager.value

        @value.setter
        def value(self, val):
            self.value_drager.value = val


class DragArrayItem(QtWidgets.QWidget, ToolItem):

    sigChanged = QtCore.pyqtSignal(object)

    def __init__(self, label:str, value, min_val, max_val, step, decimals,
                    format: Union[str, Tuple[str]]=None, callback: Callable=None, horizontal=True):
        super().__init__()
        self.length = len(value)
        self.set_label(label)

        self._value = list(value)
        min_val = self._validate_arg(min_val)
        max_val = self._validate_arg(max_val)
        step = self._validate_arg(step)
        decimals = self._validate_arg(decimals)
        format = self._validate_arg(format)

        self.name_label = QtWidgets.QLabel(label, self)

        self.inputs_frame = QtWidgets.QFrame(self)
        self.inputs_layout = QtWidgets.QHBoxLayout(self.inputs_frame) if horizontal \
            else QtWidgets.QVBoxLayout(self.inputs_frame)
        self.inputs_layout.setContentsMargins(0, 0, 0, 0)

        self.inputs: List[DragValue] = []
        for i in range(self.length):
            input = DragValue(value[i], min_val[i], max_val[i], step[i], decimals[i], format[i], self.inputs_frame)
            input.sigValueChanged.connect(self._on_changed)
            input.setObjectName(str(i))  # 设置objectName, 用于区分信号来源
            self.inputs.append(input)
            self.inputs_layout.addWidget(input, 1)

        self.box = create_layout(self, True,
                                 [self.inputs_frame, self.name_label], [5, 2], spacing=10)

        if callback is not None:
            self.sigChanged.connect(callback)

    def _validate_arg(self, arg) -> List[Number]:
        if isinstance(arg, (list, tuple, np.ndarray)):
            assert len(arg) == self.length, "arg length must be equal to value length"
            return arg
        elif isinstance(arg, (int, float)) or arg is None:
            return [arg] * self.length
        else:
            raise TypeError(f"arg must be list, tuple, int or float, but got {type(arg)}")

    def _on_changed(self, val):
        id = int(self.sender().objectName())
        self._value[id] = val
        # return self.sigChanged.emit(self._value)
        return self.sigChanged.emit((id, val))

    @property
    def value(self) -> List[Number]:
        return self._value

    @value.setter
    def value(self, val):
        for i in range(self.length):
            # 如果值不相等, 会触发 DragValue.sigValueChanged 信号, 进而触发 self.sigChanged 信号
            # 进而 self._value 的更新在 self._on_changed 中完成
            self.inputs[i].value = val[i]


class ToolBox():

    Windows: Dict[str, ToolWindow] = {}
    Items: Dict[str, ToolItem] = {}
    ContainerStack: List[ToolContainer] = []

    @classmethod
    @contextmanager
    def window(cls, label="Toolbox", parent:QtWidgets=None, spacing=5,
               pos:Union[Tuple[int], List[int]]=(0, 0),
               size:Union[Tuple[int], List[int]]=None):
        try:
            if parent is not None:
                parent.show()
                base_pos = parent.pos()
            else:
                base_pos = QPoint(0, 0)
            pos = base_pos + QPoint(pos[0], pos[1])
            win = ToolWindow(label, spacing, (pos.x(), pos.y()))
            cls.Windows[label] = win
            cls.ContainerStack.append(win)
            if size is not None:
                win.resize(size[0], size[1])
            yield win
        finally:
            cls.ContainerStack.pop()

    @classmethod
    @contextmanager
    def group(cls, label, horizontal=True, spacing=5, show=True):
        """show: 是否显示group的边框"""
        try:
            if show:
                container = ToolGroupBox(label, horizontal, spacing=spacing)
            else:
                container = ToolGroup(horizontal, spacing=spacing)
            cls.ContainerStack[-1].add_item(container)
            cls.ContainerStack.append(container)
            yield container
        finally:
            cls.ContainerStack.pop()

    @classmethod
    def clean(cls):
        for box in cls.Windows.values():
            box.close()

    @classmethod
    def _add_item(cls, label:str, item: ToolItem):
        # check if label exists
        if label in cls.Items.keys():
            raise KeyError("label already exists")
        cls.ContainerStack[-1].add_item(item)
        cls.Items[label] = item

    @classmethod
    def get_value(cls, label):
        """获取某个控件的值"""
        return cls.Items[label].value

    @classmethod
    def get_widget(cls, label):
        return cls.Items[label]

    @classmethod
    def get_window(cls, label):
        return cls.Windows[label]

    # add items
    @classmethod
    def add_button(cls, label:str, value=False, checkable=False, callback: Callable=None) -> ButtonItem:
        button = ButtonItem(label, value, checkable, callback)
        cls._add_item(label, button)
        return button

    @classmethod
    def add_checkbox(cls, label:str, value=False, callback: Callable=None) -> CheckBoxItem:
        checkbox = CheckBoxItem(label, value, callback)
        cls._add_item(label, checkbox)
        return checkbox

    @classmethod
    def add_checklist(cls, label:str, items=Tuple[str], value=None, horizontal=True,
                      exclusive=True, callback: Callable=None) -> CheckListItem:
        checklist = CheckListItem(label, items, value, horizontal, exclusive, callback)
        cls._add_item(label, checklist)
        return checklist

    @classmethod
    def add_separator(cls, horizontal=True):
        """horizontal: True for horizontal line, False for vertical line"""
        line = QtWidgets.QFrame()
        if horizontal:
            line.setFrameShape(QtWidgets.QFrame.HLine)
        else:
            line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        cls.ContainerStack[-1].add_item(line)

    @classmethod
    def add_spacer(cls, size, horizontal=False):
        size = (size, 1) if horizontal else (1, size)
        spacer = QtWidgets.QSpacerItem(size[0], size[1], QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        cls.ContainerStack[-1].get_layout().addItem(spacer)

    @classmethod
    def add_text(cls, label):
        text = QtWidgets.QLabel(label)
        cls.ContainerStack[-1].add_item(text)

    @classmethod
    def add_combo(cls, label:str, items=Tuple[str], value=0, callback: Callable=None) -> ComboItem:
        combo = ComboItem(label, items, value, callback)
        cls._add_item(label, combo)
        return combo

    @classmethod
    def add_text_editor(cls, label:str, value:str="", editable=True, callback: Callable=None) -> TextEditorItem:
        text = TextEditorItem(label, value, editable, callback)
        cls._add_item(label, text)
        return text

    @classmethod
    def add_slider(cls, label:str, value, min_val, max_val, step, decimals:int=0, callback: Callable=None) -> SliderItem:
        slider = SliderItem(label, value, min_val, max_val, step, decimals, callback)
        cls._add_item(label, slider)
        return slider

    @classmethod
    def add_array_int(cls, label:str, value:List[int], editable=True, callback: Callable=None) -> ArrayTypeItem:
        array_int = ArrayTypeItem(label, value, int, editable, callback)
        cls._add_item(label, array_int)
        return array_int

    @classmethod
    def add_array_float(cls, label:str, value:List[float], editable=True, callback: Callable=None) -> ArrayTypeItem:
        array_float = ArrayTypeItem(label, value, float, editable, callback)
        cls._add_item(label, array_float)
        return array_float

    @classmethod
    def add_drag_value(cls, label:str, value, min_val, max_val, step, decimals=2, format:str=None,
                     callback: Callable=None) -> DragValueItem:
        drag_int = DragValueItem(label, value, min_val, max_val, step, decimals, format, callback)
        cls._add_item(label, drag_int)
        return drag_int

    @classmethod
    def add_drag_array(cls, label:str,
                       value: NumberTuple,
                       min_val: Union[Number, NumberTuple],
                       max_val: Union[Number, NumberTuple],
                       step: Union[Number, NumberTuple],
                       decimals: Union[int, Tuple[int]]=0,
                       format: Union[str, Tuple[str]]=None,
                       callback: Callable=None,
                       horizontal: bool=True) -> DragArrayItem:
        drag_array = DragArrayItem(label, value, min_val, max_val, step, decimals, format, callback, horizontal)
        cls._add_item(label, drag_array)
        return drag_array