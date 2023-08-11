import OpenGL.GL as gl
from PIL import Image

__all__ = ['Texture2D']


class Texture2D:

    def __init__(
        self,
        path,
        type: str = None,
        mag_filter = gl.GL_LINEAR,
        min_filter = gl.GL_LINEAR_MIPMAP_LINEAR,
        wrap_s = gl.GL_REPEAT,
        wrap_t = gl.GL_REPEAT,
        flip_y = False,
        flip_x = False,
        generate_mipmaps=True,
    ):
        self._id = gl.glGenTextures(1)
        self.type = type
        self._path = path

        img = Image.open(str(path))
        img = flip_image(img, flip_x, flip_y)

        _format = {
            1 : gl.GL_RED,
            3 : gl.GL_RGB,
            4 : gl.GL_RGBA,
        }.get(len(img.getbands()))

        gl.glBindTexture(gl.GL_TEXTURE_2D, self._id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, _format, img.width, img.height,
                        0, _format, gl.GL_UNSIGNED_BYTE, img.tobytes())
        if generate_mipmaps:
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        # -- texture wrapping
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap_s)
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap_t)
        # -- texture filterting
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, min_filter)
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, mag_filter)

    def bind(self, unit):
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._id)
        self.unit = unit



def flip_image(img, flip_y=False, flip_x=False):
    if flip_y:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_x:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
