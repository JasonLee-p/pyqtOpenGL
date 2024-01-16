import OpenGL.GL as gl
from PIL import Image
import numpy as np
from typing import Union

__all__ = ['Texture2D']


class Texture2D:

    Format = {
        1 : gl.GL_RED,
        3 : gl.GL_RGB,
        4 : gl.GL_RGBA,
    }

    InternalFormat = {
        (1, 'uint8') : gl.GL_R8,  # 归一化
        (3, 'uint8') : gl.GL_RGB8,
        (4, 'uint8') : gl.GL_RGBA8,
        (1, 'float32') : gl.GL_R32F,
        (3, 'float32') : gl.GL_RGB32F,
        (4, 'float32') : gl.GL_RGBA32F,
        (1, 'float16') : gl.GL_R16F,
        (3, 'float16') : gl.GL_RGB16F,
        (4, 'float16') : gl.GL_RGBA16F,
    }

    DataType = {
        'uint8' : gl.GL_UNSIGNED_BYTE,
        'float16' : gl.GL_HALF_FLOAT,
        'float32' : gl.GL_FLOAT,
    }

    UnitCnt = 0

    def __init__(
        self,
        source: Union[str, np.ndarray] = None,
        tex_type: str = "tex_diffuse",
        mag_filter = gl.GL_LINEAR,
        min_filter = gl.GL_LINEAR_MIPMAP_LINEAR,
        wrap_s = gl.GL_REPEAT,
        wrap_t = gl.GL_REPEAT,
        flip_y = False,
        flip_x = False,
        generate_mipmaps=True,
    ):
        self._id = None
        self.unit = None

        # if the texture image is updated, the flag is set to True,
        # meaning that the texture needs to be updated to the GPU.
        self._img_update_flag = False
        self._img = None  # the texture image

        self.flip_y = flip_y
        self.flip_x = flip_x
        self.mag_filter = mag_filter
        self.min_filter = min_filter
        self.wrap_s = wrap_s
        self.wrap_t = wrap_t
        self.type = tex_type
        self.generate_mipmaps = generate_mipmaps
        if source is not None:
            self.updateTexture(source)

    def updateTexture(self, img: Union[str, np.ndarray]):
        if not isinstance(img, np.ndarray):
            self._path = str(img)
            img = np.array(Image.open(self._path))
        self._img = flip_image(img, self.flip_x, self.flip_y)
        self._img_update_flag = True

    def bind(self):
        """ Bind the texture to the specified texture unit,
        if unit is None, the texture will be bound to the next available unit.
        Must be called after the OpenGL context is made current."""
        if self.unit is None:
            self.unit = Texture2D.UnitCnt
            Texture2D.UnitCnt += 1

        if self._img is None:
            raise ValueError('Texture not initialized.')

        # do this job in bind() instead of updateTexture() to make sure that
        # the context is current.
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.unit)

        if self._img_update_flag:  # bind and update texture
            channels = 1 if self._img.ndim==2 else self._img.shape[2]
            dtype = self._img.dtype.name

            # -- set alignment
            nbytes_row = self._img.shape[1] * self._img.dtype.itemsize * channels
            if nbytes_row % 4 != 0:
                gl.glPixelStorei( gl.GL_UNPACK_ALIGNMENT, 1)
            else:
                gl.glPixelStorei( gl.GL_UNPACK_ALIGNMENT, 4)

            self.delete()
            self._id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._id)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0,
                self.InternalFormat[(channels, dtype)],
                self._img.shape[1], self._img.shape[0], 0,
                self.Format[channels],
                self.DataType[dtype],
                self._img,
            )

            if self.generate_mipmaps:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            # -- texture wrapping
            gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, self.wrap_s)
            gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, self.wrap_t)
            # -- texture filterting
            gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, self.min_filter)
            gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, self.mag_filter)

            self._img_update_flag = False

        else:  # bind texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._id)

    def delete(self):
        if self._id is not None:
            gl.glDeleteTextures([self._id])
            self._id == None


def flip_image(img, flip_x=False, flip_y=False):
    if flip_x and flip_y:
        img = np.flip(img, (0, 1))
    elif flip_x:
        img = np.flip(img, 1)
    elif flip_y:
        img = np.flip(img, 0)
    return img