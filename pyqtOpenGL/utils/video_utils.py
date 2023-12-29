"""视频读取和写入工具"""

import av
from fractions import Fraction
from itertools import tee

class VideoReader:
    def __init__(self, video_path, skip_time=0):
        # 视频容器
        self.container = av.open(str(video_path))
        self.stream = self.container.streams.video[0]  # 容器中的第一个视频流
        self.width, self.height = self.stream.codec_context.width, self.stream.codec_context.height
        self._generator = self.container.decode(self.stream)  # 创建生成器从 stream 中解码图像帧
        # 信息
        self.average_rate = float(self.stream.average_rate)  # 平均帧率
        self.time_base = float(self.stream.time_base)  # time = time_base * frame.pts
        self.index_base = self.time_base * self.average_rate  # cnt 约等于 index_base * frame.pts
        self.total_frames = self.stream.frames - 1  # 视频流总帧数
        self.duration = self.stream.duration * self.time_base  # 视频总时长 s
        self.pts_cur = 0   # 当前 pts
        self.pts_0 = max(0, int(skip_time / self.time_base))  # 跳过前 pts_0
        if(self.pts_0):
            self.jump(self.pts_0)

    @property
    def stamp(self):
        """当前时间戳"""
        return self.pts_cur * self.time_base

    def get_loop_frame(self):
        """循环读取图片帧
        返回(图片, 时间戳s)
        """
        try:
            frame = next(self._generator)
            image = frame.to_ndarray(format='bgr24')
            self.pts_cur = frame.pts
        except:  # 重置
            self.jump(self.pts_0)
            frame = next(self._generator)
            image = frame.to_ndarray(format='bgr24')
            self.pts_cur = frame.pts
        return image, frame.time  # 返回图片(bgr)和时间戳

    def get_frame(self):
        """读取一次视频"""
        try:
            frame = next(self._generator)
            image = frame.to_ndarray(format='bgr24')
            self.pts_cur = frame.pts
        except:
            return None, -1
        return image, frame.time

    def get_generator(self, fps=0):
        """返回生成器"""
        frame_time = None
        if fps > 0:
            frame_time = 1.0 / min(fps, self.average_rate)

        def _generator():
            last_time = None
            self.jump(self.pts_0)
            gen = self.container.decode(self.stream)

            for frame in gen:
                if frame_time is None:
                    yield frame.to_ndarray(format='bgr24'), frame.time
                else:
                    if last_time is None:
                        last_time = frame.time
                    elif frame.time - last_time >= frame_time:
                        last_time = frame.time
                        yield frame.to_ndarray(format='bgr24'), frame.time

        return _generator()

    def jump(self, pts):
        """跳转到 pts"""
        pts_goal = max(self.pts_0, int(pts))
        self.container.seek(pts_goal, stream=self.stream, backward=True)  # 上一个关键帧
        self._generator = self.container.decode(self.stream)
        for frame in self._generator:
            if pts_goal - frame.pts <= 1 / self.index_base:
                self.pts_cur = frame.pts
                break

    def back_key(self):
        """回退到上一个关键帧"""
        pts_goal = max(self.pts_0, int(self.pts_cur-10000))
        self.container.seek(pts_goal, stream=self.stream, backward=True)
        self._generator = self.container.decode(self.stream)
        iter1, self._generator = tee(self._generator)
        self.pts_cur = next(iter1).pts

    def back_frame(self):
        """回退一帧"""
        pts = self.pts_cur
        self.back_key()
        while self.pts_cur < pts - 3 / self.index_base:
            frame = next(self._generator)
            self.pts_cur = frame.pts

    def fast_back(self, n):
        """快退 n s"""
        self.jump(self.pts_cur - n / self.time_base)

    def fast_forward(self, n):
        """快进 n s"""
        self.jump(self.pts_cur + n / self.time_base)

    @property
    def index(self):
        """上一次读取帧的序号"""
        return int(self.pts_cur * self.index_base)

    @property
    def stamp(self):
        """上一次读取的时间戳 s"""
        return self.pts_cur * self.time_base

    def release(self):
        self.container.close()


class VirableRateVideoWriter():
    """ 可变帧率录制视频
    extended from https://github.com/PyAV-Org/PyAV/blob/main/examples/numpy/generate_video.py
    """
    def __init__(
        self,
        video_path,
        img_shape = (1280, 960),  # w, h
        auto_incr_stamp = True,
        bit_rate = 2048000,
        pix_fmt = 'yuv420p', # 'gray', 'yuv420p'
    ):
        self.video_path = str(video_path)
        self.container = av.open(str(video_path), mode='w')
        self.stream = self.container.add_stream('libx264', 24)
        self.stream.options['q:v'] = '0'
        self.stream.width = img_shape[0]
        self.stream.height = img_shape[1]
        self.stream.pix_fmt = pix_fmt
        self.stream.bit_rate = int(bit_rate)
        self.time_base = Fraction(1, 1000)
        self.stream.codec_context.time_base = self.time_base
        self.init_stamp = None
        self.last_stamp = -1  # last frame stamp
        self.auto_incr_stamp = auto_incr_stamp  # 自动递增 stamp

    def write(self, img, stamp):
        """ stamp(s) """
        if self.init_stamp is None:
            self.init_stamp = stamp

        # check stamp
        if self.last_stamp >= stamp:
            if not self.auto_incr_stamp:
                raise ValueError("stamp error")
            else: # 将 init_stamp 前移
                duration = self.last_stamp - self.init_stamp
                self.init_stamp = stamp - 1/self.stream.average_rate - duration

        self.last_stamp = stamp
        if self.stream.pix_fmt == 'gray':
            frame = av.VideoFrame.from_ndarray(img, format='gray')
        else:
            frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        frame.pts = int((stamp - self.init_stamp) / self.time_base)

        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def release(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        # Close the file
        self.container.close()

    def __def__(self):
        self.release()


