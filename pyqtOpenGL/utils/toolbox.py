from contextlib import contextmanager
from typing import List, Union, Dict, Callable, Type
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton


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

    def __init__(self):
        super().__init__()
        self.setupUi()
        self.is_collapsed = False
        self.pushButton.clicked.connect(self.on_click)

    def setLabel(self, label):
        self.label.setText(label)

    def setupUi(self):
        sizeFixedPolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                QtWidgets.QSizePolicy.Fixed)
        self.setSizePolicy(sizeFixedPolicy)
        self.setStyleSheet("background-color: #0f5687; color: white;")
        self.hbox = QtWidgets.QHBoxLayout(self)
        self.hbox.setContentsMargins(5, 0, 0, 0)
        self.hbox.setSpacing(5)
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setFixedSize(QSize(20, 20))
        self.pushButton.setStyleSheet(
            "QPushButton { border-radius: 10px; }"
            "QPushButton:hover { background-color: #288ad4; }"
        )
        self.label = QtWidgets.QLabel(self)
        self.label.setMinimumSize(QSize(0, 25))
        self.label.setObjectName("label")
        self.hbox.addWidget(self.pushButton)
        self.hbox.addWidget(self.label)
        self.pushButton.setText("▾")

    def on_click(self):
        self.is_collapsed = not self.is_collapsed
        self.pushButton.setText("▾▸"[self.is_collapsed])
        self.toggleCollapsed.emit(self.is_collapsed)


class ToolItem():

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

    def __init__(self, label="ToolWindow", spacing=5):
        super().__init__()
        self.setupUi()
        self.title_bar.setLabel(label)
        self.container_box.setSpacing(spacing)

        # 用于保存鼠标点击位置的变量
        self.drag_position = QPoint()
        self.movable = False

        # signals
        self.title_bar.toggleCollapsed.connect(self.on_toggleCollapsed)
        self.show()

    def setupUi(self):
        # 设置窗口样式为无边框
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        # 创建一个自定义标题栏
        self.title_bar = CollapseTitleBar()

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

    def __init__(self, items: List[str], value:int=0, horizontal=True, exclusive=True, callback: Callable=None):
        super().__init__()
        # 设置边框颜色和底色
        self.setStyleSheet("QFrame{border:1px solid #aaaaaa;}")
        self.exclusive = exclusive
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

        self.box_group.buttonClicked.connect(self._on_click)
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
    def value(self, val: int):
        self.box_group.button(val).setChecked(True)

    @property
    def checked_name(self):
        return self.box_group.checkedButton().text()

    def _on_click(self):
        return self.sigClicked.emit(self.value)


class ComboItem(QtWidgets.QWidget, ToolItem):
    def __init__(self, label:str, items: List[str], value: int=0, callback: Callable=None):
        super().__init__()
        self.setMaximumWidth(400)
        self.combo = QtWidgets.QComboBox(self)
        self.label = QtWidgets.QLabel(label, self)
        self.combo.addItems(items)
        self.combo.setCurrentIndex(value)
        self.box = create_layout(self, True, [self.combo, self.label], [5, 2], spacing=10)
        # 信号/槽
        if callback:
            self.combo.currentTextChanged.connect(callback)
            # self.combo.currentIndexChanged.connect(callback)

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
        self.step = step
        self.decimals = decimals
        value = max(min_val, min(value, max_val))
        l_steps = int((value - min_val) / step)
        r_steps = int((max_val - value) / step)
        self.min_val = value - l_steps * step
        self.steps = l_steps + r_steps

        self.label = QtWidgets.QLabel(label, self)

        self.slider = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, self.steps+1)

        self.spinbox = QtWidgets.QDoubleSpinBox(self)
        self.spinbox.setRange(self.min_val, max_val)
        self.spinbox.setSingleStep(step)
        self.spinbox.setDecimals(decimals)

        self.box = create_layout(self, True,
                                 [self.slider, self.spinbox, self.label],
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

    def __init__(self, label:str, value:List[Union[int, float]],
                 type:Union[Type[int], Type[float]], editable=True, callback: Callable=None):
        super().__init__()
        self._value = value
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
        for i in range(self.length):
            self.inputs[i].setText(str(val[i]))

    def _on_changed(self):
        self._value = [self.type(input.text()) for input in self.inputs]
        return self.sigChanged.emit(self._value)


class ToolBox():

    Boxes: List[ToolWindow] = []
    Items: Dict[str, ToolItem] = {}
    ContainerStack: List[ToolContainer] = []

    @classmethod
    @contextmanager
    def window(cls, label="Toolbox", spacing=5):
        try:
            container = ToolWindow(label, spacing)
            cls.Boxes.append(container)
            cls.ContainerStack.append(container)
            yield container
        finally:
            cls.ContainerStack.pop()

    @classmethod
    @contextmanager
    def group(cls, label, horizontal=True, spacing=5, show=True):
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
        for box in cls.Boxes:
            box.close()

    @classmethod
    def get_value(cls, label):
        """获取某个控件的值"""
        return cls.Items[label].value

    @classmethod
    def get_widget(cls, label):
        return cls.Items[label]

    # add items
    @classmethod
    def add_button(cls, label:str, value=False, checkable=False, callback: Callable=None) -> ButtonItem:
        button = ButtonItem(label, value, checkable, callback)
        cls.ContainerStack[-1].add_item(button)
        cls.Items[label] = button
        return button

    @classmethod
    def add_checkbox(cls, label:str, value=False, callback: Callable=None) -> CheckBoxItem:
        checkbox = CheckBoxItem(label, value, callback)
        cls.ContainerStack[-1].add_item(checkbox)
        cls.Items[label] = checkbox
        return checkbox

    @classmethod
    def add_checklist(cls, label:str, items=["a", "b", "c"], value=0, horizontal=True,
                      exclusive=True, callback: Callable=None) -> CheckListItem:
        checklist = CheckListItem(items, value, horizontal, exclusive, callback)
        cls.ContainerStack[-1].add_item(checklist)
        cls.Items[label] = checklist
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
    def add_combo(cls, label:str, items=["a", "b", "c"], value=0, callback: Callable=None) -> ComboItem:
        combo = ComboItem(label, items, value, callback)
        cls.ContainerStack[-1].add_item(combo)
        cls.Items[label] = combo
        return combo

    @classmethod
    def add_text_editor(cls, label:str, value:str="", editable=True, callback: Callable=None) -> TextEditorItem:
        text = TextEditorItem(label, value, editable, callback)
        cls.ContainerStack[-1].add_item(text)
        cls.Items[label] = text
        return text

    @classmethod
    def add_slider(cls, label:str, value, min_val, max_val, step, decimals:int=0, callback: Callable=None) -> SliderItem:
        slider = SliderItem(label, value, min_val, max_val, step, decimals, callback)
        cls.ContainerStack[-1].add_item(slider)
        cls.Items[label] = slider
        return slider

    @classmethod
    def add_array_int(cls, label:str, value:List[int], editable=True, callback: Callable=None) -> ArrayTypeItem:
        array_int = ArrayTypeItem(label, value, int, editable, callback)
        cls.ContainerStack[-1].add_item(array_int)
        cls.Items[label] = array_int
        return array_int

    @classmethod
    def add_array_float(cls, label:str, value:List[float], editable=True, callback: Callable=None) -> ArrayTypeItem:
        array_float = ArrayTypeItem(label, value, float, editable, callback)
        cls.ContainerStack[-1].add_item(array_float)
        cls.Items[label] = array_float
        return array_float
