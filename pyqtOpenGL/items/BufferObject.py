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


class MemoryBlock:

    def __init__(
        self,
        blocks: List[np.ndarray],
        dsize,
    ):
        self.block_lens = [0 if x is None else x.nbytes for x in blocks]
        self.block_used = np.array(self.block_lens, dtype=int)
        self.block_offsets = [0] + np.cumsum(self.block_lens).tolist()[:-1]
        self.sum_lens = sum(self.block_lens)
        self.dtype = [np.dtype('f4') if x is None else x.dtype for x in blocks]

        # attributes properties
        self.attr_size = [[4, 4, 4, 4] if x==16 else x for x in dsize]
        id = 0
        self.attr_idx = []
        for si in self.attr_size:
            if isinstance(si, list):
                self.attr_idx.append(list(range(id, id+len(si))))
                id += len(si)
            else:
                self.attr_idx.append(id)
                id += 1

    def setBlock(self, ids: List[int], blocks: List[int]):
        extend = False
        keep_blocks = []
        copy_blocks = []
        ptr = 0

        for id, len in zip(ids, blocks):
            t = self.block_offsets[id] - ptr
            if t > 0:
                keep_blocks.append([ptr, t, id])  # read offset, size, id(then convert to write offset)

            ptr = self.block_offsets[id] + self.block_lens[id]
            self.block_used[id] = len
            if len > self.block_lens[id]:
                self.block_lens[id] = len
                extend = True
        if ptr < self.sum_lens:
            keep_blocks.append([ptr, self.sum_lens-ptr, -1])

        if extend:
            self.block_offsets = [0] + np.cumsum(self.block_lens).tolist()[:-1]
            self.sum_lens = sum(self.block_lens)
            for kb in keep_blocks:  # calc write offset
                id = kb[2]
                end = self.block_offsets[id] if id!=-1 else self.sum_lens
                kb[2] = end - kb[1]

        for id, len in zip(ids, blocks):
            copy_blocks.append([self.block_offsets[id], len])  # read offset, size

        return copy_blocks, keep_blocks, extend

    def locBlock(self, id):
        return self.block_offsets[id], self.block_lens[id]

    @property
    def nblocks(self):
        return len(self.block_lens)

    @property
    def nbytes(self):
        return self.sum_lens

    def __len__(self):
        return self.sum_lens

    def __getitem__(self, id):
        return {
            "offset": self.block_offsets[id],
            "length": self.block_lens[id],
            "used": self.block_used[id],
            "dtype": self.dtype[id],
            "attr_size": self.attr_size[id],
            "attr_idx": self.attr_idx[id],
        }

    def __repr__(self) -> str:
        repr = "|"
        for i in range(len(self.block_lens)):
            repr += f"{self.block_offsets[i]}> {self.block_used[i]}/{self.block_lens[i]}|"
        return repr


class VBO():
    def __init__(
        self,
        data: List[np.ndarray],
        size: List[int],  # 3 for vec3, 16 for mat4
        usage = gl.GL_STATIC_DRAW,
    ):
        self._usage = usage
        self.blocks = MemoryBlock(data, size)

        # 缓冲区数据
        self._vbo = gl.glGenBuffers(1)
        if self.blocks.nbytes > 0:
            self.bind()
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.blocks.nbytes, None, self._usage)
            self.buffer_need_init = True
        self.updateData(range(len(data)), data)

    def _loadSubDatas(self, block_id: List[int], data: List[np.ndarray]):
        """load data to buffer"""
        self.bind()
        if self.buffer_need_init:
            self.buffer_need_init = False
            for id, da in zip(block_id, data):
                offset, size = self.blocks.locBlock(id)
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, offset, size, da)
        else:
            buf = gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_WRITE_ONLY)
            for id, da in zip(block_id, data):
                offset = self.blocks.block_offsets[id]
                length = self.blocks.block_used[id]
                ctypes.memmove(buf+offset, da.ctypes.data, length)
            gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def updateData(self, block_id: List[int], data: List[np.ndarray]):
        """update data to buffer, first check if need to extend buffer"""
        self.bind()
        old_nbytes = self.blocks.nbytes
        copy_blocks, keep_blocks, extend = self.blocks.setBlock(block_id,
                                                                [0 if x is None else x.nbytes for x in data])
        if self.blocks.nbytes == 0:
            return

        if extend:
            """extend a sub buffer to new_size"""
            new_vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, new_vbo)
            gl.glBufferData(gl.GL_COPY_WRITE_BUFFER, old_nbytes, None, self._usage)
            gl.glCopyBufferSubData(gl.GL_ARRAY_BUFFER, gl.GL_COPY_WRITE_BUFFER, 0, 0, old_nbytes)

            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.blocks.nbytes, None, self._usage)
            self.buffer_need_init = True
            for keep in keep_blocks:
                gl.glCopyBufferSubData(
                    gl.GL_COPY_WRITE_BUFFER,
                    gl.GL_ARRAY_BUFFER,
                    keep[0],   # read offset
                    keep[2],   # write offset
                    keep[1],   # size
                )

            gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, 0)
            gl.glDeleteBuffers(1, [new_vbo])

        self._loadSubDatas(block_id, data)

    @property
    def isbind(self):
        return self._vbo == gl.glGetIntegerv(gl.GL_ARRAY_BUFFER_BINDING)

    def bind(self):
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)

    def delete(self):
        gl.glDeleteBuffers(1, [self._vbo])

    def getData(self, id):
        """get data from buffer"""
        self.bind()
        offset, nbytes = self.blocks.locBlock(id)  # 读取数据的偏移量和大小
        dtype = self.blocks.dtype[id]
        data = np.empty(int(nbytes/dtype.itemsize), dtype=dtype)

        gl.glGetBufferSubData(
            gl.GL_ARRAY_BUFFER,
            offset, nbytes,
            data.ctypes.data_as(c_void_p)
        )
        asize = self.blocks.attr_size[id]
        return data.reshape(-1, asize if isinstance(asize, int) else sum(asize))

    def setAttrPointer(self, block_id: List[int], attr_id: List[int]=None, divisor=0):
        """
        :param id: id of block, not attribute index
        :param divisor: divisor of attribute
        """
        self.bind()
        if isinstance(block_id, int):
            block_id = [block_id]

        if attr_id is None:
            attr_id = [self.blocks.attr_idx[b_id] for b_id in block_id]
        elif isinstance(attr_id, int):
            attr_id = [attr_id]

        if isinstance(divisor, int):
            divisor = [divisor] * len(block_id)

        for b_id, a_id, div in zip(block_id, attr_id, divisor):
            a_size = self.blocks.attr_size[b_id]
            dtype = np.dtype(self.blocks.dtype[b_id])

            if isinstance(a_size, list):  # mat4
                stride = sum(a_size) * dtype.itemsize
                a_offsets = [0] + np.cumsum(a_size).tolist()[:-1]
                for i in range(len(a_size)):
                    gl.glVertexAttribPointer(
                        a_id[i],
                        a_size[i],
                        GL_Type[dtype],
                        gl.GL_FALSE,
                        stride,
                        c_void_p(self.blocks.block_offsets[b_id] + a_offsets[i]*dtype.itemsize)
                    )
                    gl.glVertexAttribDivisor(a_id[i], div)
                    gl.glEnableVertexAttribArray(a_id[i])
            else:
                gl.glVertexAttribPointer(
                    a_id,
                    a_size,
                    GL_Type[dtype],
                    gl.GL_FALSE,
                    a_size * dtype.itemsize,
                    c_void_p(self.blocks.block_offsets[b_id])
                )
                gl.glVertexAttribDivisor(a_id, div)
                gl.glEnableVertexAttribArray(a_id)


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
