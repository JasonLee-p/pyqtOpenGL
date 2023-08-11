import OpenGL.GL as gl
from typing import List, Union
import numpy as np
from ctypes import c_uint, c_float, c_void_p
import ctypes

GL_Type = {
    np.dtype("f4"): gl.GL_FLOAT,
    np.dtype("u4"): gl.GL_UNSIGNED_INT,
    np.dtype("f2"): gl.GL_HALF_FLOAT,
}

class VBO():

    def __init__(
        self,
        data: List[np.ndarray],
        size: List[int],  # 3 for vec3, 16 for mat4
        usage = gl.GL_STATIC_DRAW,
    ):
        self._usage = usage
        self._vbo = gl.glGenBuffers(1)

        # 规划缓冲区结构
        self._buf_index = []  # attribute index, int for vec3, list for mat4
        self._buf_nbytes = []
        self._buf_dtype = []
        self._buf_size = []  # data size, 3 for vec3, [4,4,4,4] for mat4
        self._buf_offsets = None
        id = 0
        for _data, _size in zip(data, size):
            if _size == 16:
                _size = [4,4,4,4]
            if isinstance(_size, list):  # mat4
                self._buf_index.append(list(range(id, id+len(_size))))
                self._buf_size.append(_size)
                self._buf_dtype.append(_data.dtype)
                self._buf_nbytes.append(_data.nbytes)
                id += len(_size)
            elif _size <= 4:
                self._buf_index.append(id)
                self._buf_size.append(_size)
                self._buf_dtype.append(_data.dtype)
                self._buf_nbytes.append(_data.nbytes)
                id += 1
        self._buf_offsets = [0] + np.cumsum(self._buf_nbytes).tolist()[:-1]

        # 缓冲区数据
        self.bind()
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, sum(self._buf_nbytes), None, self._usage
        )
        self.loadData(range(len(data)), data)

    @property
    def isbind(self):
        return self._vbo == gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING)

    def bind(self):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)

    def loadData(self, id: List[int], data: List[np.ndarray]):
        for _id, _data in zip(id, data):
            self.loadSubData(_id, _data)

    def loadSubData(self, data_id, data: np.ndarray):
        assert data.dtype == self._buf_dtype[data_id], "data type not match"
        if not self.isbind:
            self.bind()

        # data.nbytes > self._buf_nbytes[id] 时，需要扩展缓冲区
        if data.nbytes > self._buf_nbytes[data_id]:
            self.extendBuffer(data_id, data.nbytes)

        gl.glBufferSubData(
            gl.GL_ARRAY_BUFFER, self._buf_offsets[data_id], self._buf_nbytes[data_id], data
        )

    def loadd(self, data):
        self.bind()
        buf = gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_WRITE_ONLY)
        data = np.concatenate(data, axis=0)
        ctypes.memmove(buf, data.ctypes.data, data.nbytes)
        gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        # glUnmapBuffer(GL_ARRAY_BUFFER)

    def getSubData(self, data_id):
        """get data from buffer"""
        if not self.isbind:
            self.bind()

        read_offset = self._buf_offsets[data_id]  # 读取数据的偏移量
        read_nbytes = self._buf_nbytes[data_id]  # 读取数据的大小
        buf_dtype = self._buf_dtype[data_id]
        data = np.empty(int(read_nbytes/buf_dtype.itemsize), dtype=self._buf_dtype[data_id])

        gl.glGetBufferSubData(
            gl.GL_ARRAY_BUFFER,
            read_offset, read_nbytes,
            data.ctypes.data_as(c_void_p)
        )
        return data.reshape(-1, self._buf_size[data_id])

    def setAttrPointer(self, id: Union[int, list], divisor=0, attr_id: Union[int, list]=None):
        """
        :param id: id of subdata, not attribute index
        :param divisor: divisor of attribute
        """
        if not self.isbind:
            self.bind()

        if isinstance(id, list):
            if not isinstance(divisor, list):
                divisor = [divisor] * len(id)
            for i, _id in enumerate(id):
                _new = None if attr_id is None else attr_id[i]
                self.setAttrPointer(_id, divisor[i], _new)
            return

        attr_id = attr_id if attr_id is not None else self._buf_index[id]
        attr_size = self._buf_size[id]
        dsize = np.dtype(self._buf_dtype[id]).itemsize
        if isinstance(attr_size, list):  # mat4
            stride = sum(attr_size) * dsize
            attr_offsets = [0] + np.cumsum(attr_size).tolist()[:-1]
            for i in range(len(attr_id)):
                gl.glVertexAttribPointer(
                    attr_id[i],
                    attr_size[i],
                    GL_Type[self._buf_dtype[id]],
                    gl.GL_FALSE,
                    stride,
                    c_void_p(self._buf_offsets[id] + attr_offsets[i]*dsize)
                )
                gl.glVertexAttribDivisor(attr_id[i], divisor)
                gl.glEnableVertexAttribArray(attr_id[i])
        else:
            gl.glVertexAttribPointer(
                attr_id,
                attr_size,
                GL_Type[self._buf_dtype[id]],
                gl.GL_FALSE,
                attr_size * dsize,
                c_void_p(self._buf_offsets[id])
            )
            gl.glVertexAttribDivisor(attr_id, divisor)
            gl.glEnableVertexAttribArray(attr_id)

    def delete(self):
        gl.glDeleteBuffers(1, [self._vbo])

    def extendBuffer(self, data_id, new_size):
        """extend a sub buffer to new_size"""
        old_nbytes = sum(self._buf_nbytes)
        new_buf_nbytes = [x for x in self._buf_nbytes]
        new_buf_nbytes[data_id] = new_size
        new_buf_offsets = [0] + np.cumsum(new_buf_nbytes).tolist()[:-1]
        new_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, new_vbo)
        gl.glBufferData(gl.GL_COPY_WRITE_BUFFER, old_nbytes, None, self._usage)
        gl.glCopyBufferSubData(gl.GL_ARRAY_BUFFER, gl.GL_COPY_WRITE_BUFFER, 0, 0, old_nbytes)

        gl.glBufferData(gl.GL_ARRAY_BUFFER, sum(new_buf_nbytes), None, self._usage)
        # not the first buffer
        if data_id > 0:
            gl.glCopyBufferSubData(
                gl.GL_COPY_WRITE_BUFFER,
                gl.GL_ARRAY_BUFFER,
                0,   # read offset
                0,   # write offset
                self._buf_offsets[data_id]  # size
            )
        # not the last buffer
        if data_id < len(self._buf_nbytes)-1:
            gl.glCopyBufferSubData(
                gl.GL_COPY_WRITE_BUFFER,
                gl.GL_ARRAY_BUFFER,
                self._buf_offsets[data_id+1],
                new_buf_offsets[data_id+1],
                sum(self._buf_nbytes[data_id+1:]),
            )

        self._buf_nbytes = new_buf_nbytes
        self._buf_offsets = new_buf_offsets

        gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, 0)
        gl.glDeleteBuffers(1, [new_vbo])


class VAO():

    def __init__(self):
        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

    @property
    def isbind(self):
        return self._vao == gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)

    def bind(self):
        gl.glBindVertexArray(self._vao)

    def unbind(self):
        gl.glBindVertexArray(0)


class EBO():

    def __init__(self, indices: np.ndarray):
        self._ebo = gl.glGenBuffers(1)
        self._size = indices.size
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        self.loadData(indices)

    @property
    def isbind(self):
        return self._ebo == gl.glGetIntegerv(gl.GL_ELEMENT_ARRAY_BUFFER_BINDING)

    def loadData(self, indices: np.ndarray):
        if not self.isbind:
            self.bind()
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices,
            gl.GL_STATIC_DRAW
        )
        self._size = indices.size

    @property
    def size(self):
        return self._size

    def bind(self):
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)

    def delete(self):
        gl.glDeleteBuffers(1, [self._ebo])
