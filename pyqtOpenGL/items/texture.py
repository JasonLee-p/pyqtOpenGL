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

    def __init__(
        self,
        source = None,
        tex_type: str = None,
        mag_filter = gl.GL_LINEAR,
        min_filter = gl.GL_LINEAR_MIPMAP_LINEAR,
        wrap_s = gl.GL_REPEAT,
        wrap_t = gl.GL_REPEAT,
        flip_y = False,
        flip_x = False,
        generate_mipmaps=True,
    ):
        self._id = None
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

    def bind(self, unit):
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._id)
        self.unit = unit

    def updateTexture(self, img: Union[str, np.ndarray]):
        if not isinstance(img, np.ndarray):
            self._path = str(img)
            img = np.array(Image.open(self._path))
        img = flip_image(img, self.flip_x, self.flip_y)

        channels = 1 if img.ndim==2 else img.shape[2]
        dtype = img.dtype.name

        # -- set alignment
        nbytes_row = img.shape[1] * img.dtype.itemsize * channels
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
            img.shape[1], img.shape[0], 0,
            self.Format[channels],
            self.DataType[dtype],
            img,
        )

        if self.generate_mipmaps:
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        # -- texture wrapping
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, self.wrap_s)
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, self.wrap_t)
        # -- texture filterting
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, self.min_filter)
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, self.mag_filter)

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