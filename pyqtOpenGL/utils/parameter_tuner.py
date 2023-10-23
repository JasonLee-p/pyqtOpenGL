import sys
import cv2
import numpy as np
from time import time
from pathlib import Path
from typing import Callable, Union
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, pyqtSlot

from .video_utils import VirableRateVideoWriter, VideoReader
from ..functions import Filter, increment_path

from .QtTools import (QTablePanel,
                    VisualizeWidget,
                    QPushButton,
                    QImageViewWidget,
                    create_layout)


class InputType():
    VIDEO = 'video'
    IMAGE = 'image'


class InputSource():
    """输入源"""
    def __init__(self, source: Union[str, Path]=None, type: str=None):
        """
        type: 'video', 'image'
        """
        self.input_source = None
        self.type = type
        self.update_source(source, type)
        self.paused = False

    @property
    def is_connected(self):
        return self.input_source is not None

    def update_source(self, source: Union[str, Path, np.ndarray], type: str):
        if type == InputType.VIDEO:
            assert not str(source).split('.')[-1] in ("jpg", "png")
            self.input_source = VideoReader(source)
            self.height, self.width = self.input_source.height, self.input_source.width

        elif type == InputType.IMAGE:
            if isinstance(source, np.ndarray):
                self.input_source = source
            else:
                self.input_source = cv2.imread(str(source))
            assert self.input_source is not None, "[ERROR] Reading image failed"
            self.height, self.width = self.input_source.shape[0:2]

        self.type = type

    def get_frame(self):
        if not self.is_connected:
            raise Exception("[ERROR] Input source is not connected")

        if self.type == InputType.VIDEO:
            if self.paused:
                return self.img_curr.copy(), self.stamp_curr
            return self.input_source.get_loop_frame()

        elif self.type == InputType.IMAGE:
            return np.array(self.input_source), time()

    def release(self):
        if self.type == InputType.VIDEO and self.input_source is not None:
            self.input_source.release()

    def pause_toggle(self, val: bool=None):
        """video 暂停切换"""
        if self.type != InputType.VIDEO:
            return

        self.paused = not self.paused if val is None else val
        self.img_curr, self.stamp_curr = self.input_source.get_loop_frame()

    def back(self, sec=1):
        """video 快退"""
        if self.type != InputType.VIDEO:
            return

        self.input_source.fast_back(sec)
        if self.paused:
            self.img_curr, self.stamp_curr = self.input_source.get_loop_frame()

    def forward(self, sec=1):
        """video 快进"""
        if self.type != InputType.VIDEO:
            return

        self.input_source.fast_forward(sec)
        if self.paused:
            self.img_curr, self.stamp_curr = self.input_source.get_loop_frame()

    def forward_frame(self, n=1):
        """步进 n 帧"""
        if self.type != InputType.VIDEO:
            return

        for _ in range(n):
            img, stamp = self.input_source.get_loop_frame()
        if self.paused:
            self.img_curr, self.stamp_curr = img, stamp

    def back_frame(self):
        "回退 n 帧"
        if self.type != InputType.VIDEO:
            return

        self.input_source.back_frame()

        if self.paused:
            self.img_curr, self.stamp_curr = self.input_source.get_loop_frame()


class VideoControlWidget(QtWidgets.QWidget):
    """控制视频暂停/后退/快进, 显示视频流时间戳等信息"""
    sigRecordVideo = pyqtSignal(bool)
    sigVideoChanged = pyqtSignal(bool)
    def __init__(self, parent, input_widget:InputSource):
        super().__init__(parent)
        self.input_widget = input_widget
        self.init_ui()

    def init_ui(self):
        self.playButton = QPushButton(self, "", False, checkable=True)
        self.playButton.setMaximumWidth(100)
        self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.on_play)

        self.backButton = QPushButton(self, "", False, checkable=False)
        self.backButton.setMaximumWidth(100)
        self.backButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward))
        self.backButton.clicked.connect(self.on_back)

        self.forwardButton = QPushButton(self, "", False, checkable=False)
        self.forwardButton.setMaximumWidth(100)
        self.forwardButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekForward))
        self.forwardButton.clicked.connect(self.on_forward)

        self.recordButton = QPushButton(self, "record", False, checkable=True)
        self.recordButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogNoButton))
        self.recordButton.setMaximumWidth(120)
        self.recordButton.clicked.connect(self.on_record)

        self.stamp_label = QtWidgets.QLabel(self)
        self.hbox = create_layout(
            self,
            type = "h",
            widgets = [self.backButton, self.playButton, self.forwardButton,
                       self.recordButton, self.stamp_label],
            stretchs = [1, 1, 1, 1, 1],
            spacing = 10,
        )

    @pyqtSlot()
    def on_play(self):
        if self.playButton.isChecked():
            self.input_widget.pause_toggle(True)
            self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        else:
            self.input_widget.pause_toggle(False)
            self.playButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    @pyqtSlot()
    def on_forward(self):
        self.input_widget.forward()
        self.sigVideoChanged.emit(True)

    @pyqtSlot()
    def on_back(self):
        self.input_widget.back()
        self.sigVideoChanged.emit(True)

    @pyqtSlot()
    def on_forward_frame(self):
        self.input_widget.forward_frame()
        self.sigVideoChanged.emit(True)

    @pyqtSlot()
    def on_back_frame(self):
        self.input_widget.back_frame()
        self.sigVideoChanged.emit(True)

    @pyqtSlot()
    def on_record(self):
        if self.recordButton.isChecked():
            self.sigRecordVideo.emit(True)
            self.recordButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton))
        else:
            self.sigRecordVideo.emit(False)
            self.recordButton.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogNoButton))

    def update_stamp(self):
        if self.input_widget.type == "video":
            video = self.input_widget.input_source
            self.stamp_label.setText(
                f"frame: {video.index:>4d}/{video.total_frames:>4d} | time:{video.stamp:>5.1f}/{video.duration:>5.1f}"
            )
        elif self.input_widget.type == "gelslim":
            gelslim = self.input_widget.input_source
            self.stamp_label.setText(
                f"GelSlim fps: {gelslim.fps:>5.1f} | Stamp:{gelslim.stamp:>7.1f}"
            )

    def pause_toggle(self):
        self.playButton.click()


class ParameterTuner(QWidget):

    panel_config = {
        "spacer0": ["spacer", 20],
        "Record Source Image": ["bool", 0],
        "Show Source Image": ["bool", 1],
        "spacer1": ["spacer", 20],
        "line1": ["line", ],
        "spacer2": ["spacer", 20],
        "paramters: ": ["label", ],
    }

    def __init__(
        self,
        func: Callable,
        params: dict = {},
        video: InputSource = None,
        preprocess: Callable = None,
    ):
        """
        :param func: 待调参的图片处理函数
        :param params: 参数字典, 格式为 {param1: [value, start, stop, step], ...}
        :param video: 视频或图片
        """
        # 初始化窗口
        super().__init__()
        self.func = func
        self.video = video
        self.preprocess = preprocess
        self.panel_config.update(params)
        self.param_keys = params.keys()

        self.fps = Filter(0)
        self.video_writer = None
        self.record_flag = 0  # 0 normal, 1 start, 2 recording, 3 close
        self.record_Transition ={(0,True): 1, (1,False): 3, (2,False): 3}

        self.setup_ui(self.panel_config)

        self.timer = QTimer(self)
        self.timer.start(5)
        self.timer.timeout.connect(self.on_timeout)

        self.vc_widget.sigRecordVideo.connect(
            lambda val:
                setattr(self, "record_flag",
                        self.record_Transition[(self.record_flag, val)])
        )
        self.vc_widget.sigVideoChanged.connect(self.on_video_changed)

    def setup_ui(self, params):
        self.resize(1565, 995)
        self.setWindowTitle("Parameter Tuner")
        self.left_frame = QtWidgets.QFrame(self)
        self.right_frame = QtWidgets.QFrame(self)
        self.src_img_widget = QImageViewWidget(self)
        self.src_img_widget.setMinimumWidth(150)
        self.hbox = create_layout(
            self,
            type = "h",
            widgets = [self.left_frame, self.right_frame, self.src_img_widget],
            stretchs = [1, 4, 4],
        )

        self.panel = QTablePanel(self, params)
        self.status_label = QtWidgets.QLabel(self)
        self.left_vbox = create_layout(
            self.left_frame,
            type = "v",
            widgets = [self.panel, self.status_label],
            stretchs = [10, 1],
        )

        self.vis_widget = VisualizeWidget(self)
        self.vc_widget = VideoControlWidget(self, self.video)
        self.right_vbox = create_layout(
            self.right_frame,
            type = "v",
            widgets = [self.vis_widget, self.vc_widget],
            stretchs = [15, 1],
        )

    def on_timeout(self):
        if not self.video.paused:
            self.update()

    def on_video_changed(self):
        if self.video.paused:
            self.update()

    def update(self):
        # 原图或处理后的图片
        img, stamp = self.video.get_frame()

        if self.preprocess is not None:
            img = self.preprocess(img)

        if self.panel["Show Source Image"]:
            self.src_img_widget.setVisible(True)
            self.src_img_widget.setData(img)
        else:
            self.src_img_widget.setVisible(False)

        t0 = time()
        ret = self.func(img, **(self._get_params()))
        self.fps.update(1 / (time()-t0+1e-5))  # 计算 fps

        # 更新图片
        if isinstance(ret, tuple):
            self.vis_widget.set_data(*ret)
        else:
            self.vis_widget.set_data(ret)

        # 视频更新状态栏
        self.status_label.setText(
            f"fps:{self.fps.data:>8.3f} \
            frame time:{1000/self.fps.data:>8.3f}ms"
        )
        self.vc_widget.update_stamp()

        self._record_video(img, stamp)

    def _record_video(self, img, stamp):
        if self.record_flag == 0:
            return

        elif self.record_flag == 1:
            print("[INFO] Start recording video")
            save_path = increment_path(Path.cwd() / "video.mp4")
            if self.panel["Record Source Image"]:
                w, h = img.shape[1], img.shape[0]
            else:
                qrect = self.vis_widget.qrect
                w, h = qrect.width(), qrect.height()

            self.video_writer = VirableRateVideoWriter(
                save_path,
                (w - w % 2, h - h % 2),
                bit_rate=2048000,
                auto_incr_stamp=True,
            )
            self.record_flag = 2

        elif self.record_flag == 2:
            if self.panel["Record Source Image"]:
                self.video_writer.write(img, stamp)
            else:
                img = self.vis_widget.grab_frame()
                self.video_writer.write(img, stamp)

        elif self.record_flag == 3:
            self.video_writer.release()
            print(f"[INFO] Video saved at {self.video_writer.video_path}")
            self.video_writer = None
            self.record_flag = 0

    def tune(func, params: dict, video: InputSource, **kwargs):
        """对外接口"""
        app = QApplication(sys.argv)
        mainwindow = ParameterTuner(func, params, video, **kwargs)
        mainwindow.show()
        sys.exit(app.exec())

    def keyPressEvent(self, a0):
        """按键处理"""
        if a0.key() == Qt.Key.Key_Escape:
            self.close()
        # 视频快捷键
        if a0.text() in ['a', 'A']:  # 快退一秒
            self.vc_widget.on_back()
        elif a0.text() in ['d', 'D']:
            self.vc_widget.on_forward() # 快进一秒
        elif a0.text() in ['f', 'F']:  # 步进
            self.vc_widget.on_forward_frame()
        elif a0.text() in ['b', 'B']:  # 步进
            self.vc_widget.on_back_frame()
        elif a0.key() == Qt.Key.Key_Space:  # 步进
            self.vc_widget.pause_toggle()
        elif a0.key() == Qt.Key.Key_J:
            img, stamp = self.video.get_frame()
            cv2.imwrite("./sample.jpg", img)
            print("save sample.jpg")

    def closeEvent(self, a0):
        """输出最终参数"""
        print("Final Parameters:")
        print(self._get_params())
        if self.video_writer is not None:
            self.video_writer.release()
        return super().closeEvent(a0)

    def _get_params(self):
        params = dict()
        for key in self.param_keys:
            params[key] = self.panel[key]
        return params




