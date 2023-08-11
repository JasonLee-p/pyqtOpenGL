from PyQt5 import QtGui
import numpy as np
import sys

Colors = {
    'b': QtGui.QColor(0,0,255,255),
    'g': QtGui.QColor(0,255,0,255),
    'r': QtGui.QColor(255,0,0,255),
    'c': QtGui.QColor(0,255,255,255),
    'm': QtGui.QColor(255,0,255,255),
    'y': QtGui.QColor(255,255,0,255),
    'k': QtGui.QColor(0,0,0,255),
    'w': QtGui.QColor(255,255,255,255),
    'd': QtGui.QColor(150,150,150,255),
    'l': QtGui.QColor(200,200,200,255),
    's': QtGui.QColor(100,100,150,255),
}


def clip_scalar(val, vmin, vmax):
    """ convenience function to avoid using np.clip for scalar values """
    return vmin if val < vmin else vmax if val > vmax else val

def mkColor(*args):
    """
    Convenience function for constructing QColor from a variety of argument
    types. Accepted arguments are:

    ================ ================================================
     'c'             one of: r, g, b, c, m, y, k, w
     R, G, B, [A]    integers 0-255
     (R, G, B, [A])  tuple of integers 0-255
     float           greyscale, 0.0-1.0
     int             see :func:`intColor() <pyqtgraph.intColor>`
     (int, hues)     see :func:`intColor() <pyqtgraph.intColor>`
     "#RGB"
     "#RGBA"
     "#RRGGBB"
     "#RRGGBBAA"
     QColor          QColor instance; makes a copy.
    ================ ================================================
    """
    err = 'Not sure how to make a color from "%s"' % str(args)
    if len(args) == 1:
        if isinstance(args[0], str):
            c = args[0]
            if len(c) == 1:
                try:
                    return Colors[c]
                except KeyError:
                    raise ValueError('No color named "%s"' % c)
            have_alpha = len(c) in [5, 9] and c[0] == '#'  # "#RGBA" and "#RRGGBBAA"
            if not have_alpha:
                # try parsing SVG named colors, including "#RGB" and "#RRGGBB".
                # note that QColor.setNamedColor() treats a 9-char hex string as "#AARRGGBB".
                qcol = QtGui.QColor()
                qcol.setNamedColor(c)
                if qcol.isValid():
                    return qcol
                # on failure, fallback to pyqtgraph parsing
                # this includes the deprecated case of non-#-prefixed hex strings
            if c[0] == '#':
                c = c[1:]
            else:
                raise ValueError(f"Unable to convert {c} to QColor")
            if len(c) == 3:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = 255
            elif len(c) == 4:
                r = int(c[0]*2, 16)
                g = int(c[1]*2, 16)
                b = int(c[2]*2, 16)
                a = int(c[3]*2, 16)
            elif len(c) == 6:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = 255
            elif len(c) == 8:
                r = int(c[0:2], 16)
                g = int(c[2:4], 16)
                b = int(c[4:6], 16)
                a = int(c[6:8], 16)
            else:
                raise ValueError(f"Unknown how to convert string {c} to color")
        elif isinstance(args[0], QtGui.QColor):
            return QtGui.QColor(args[0])
        elif np.issubdtype(type(args[0]), np.floating):
            r = g = b = int(args[0] * 255)
            a = 255
        elif hasattr(args[0], '__len__'):
            if len(args[0]) == 3:
                r, g, b = args[0]
                a = 255
            elif len(args[0]) == 4:
                r, g, b, a = args[0]
            elif len(args[0]) == 2:
                return intColor(*args[0])
            else:
                raise TypeError(err)
        elif np.issubdtype(type(args[0]), np.integer):
            return intColor(args[0])
        else:
            raise TypeError(err)
    elif len(args) == 3:
        r, g, b = args
        a = 255
    elif len(args) == 4:
        r, g, b, a = args
    else:
        raise TypeError(err)
    args = [int(a) if np.isfinite(a) else 0 for a in (r, g, b, a)]
    return QtGui.QColor(*args)


def glColor(*args, **kargs):
    """
    Convert a color to OpenGL color format (r,g,b,a) floats 0.0-1.0
    Accepts same arguments as :func:`mkColor <pyqtgraph.mkColor>`.
    """
    c = mkColor(*args, **kargs)
    return c.getRgbF()


def intColor(index, hues=9, values=1, maxValue=255, minValue=150, maxHue=360, minHue=0, sat=255, alpha=255):
    """
    Creates a QColor from a single index. Useful for stepping through a predefined list of colors.

    The argument *index* determines which color from the set will be returned. All other arguments determine what the set of predefined colors will be

    Colors are chosen by cycling across hues while varying the value (brightness).
    By default, this selects from a list of 9 hues."""
    hues = int(hues)
    values = int(values)
    ind = int(index) % (hues * values)
    indh = ind % hues
    indv = ind // hues
    if values > 1:
        v = minValue + indv * ((maxValue-minValue) // (values-1))
    else:
        v = maxValue
    h = minHue + (indh * (maxHue-minHue)) // hues

    return QtGui.QColor.fromHsv(h, sat, v, alpha)


# umath.clip was slower than umath.maximum(umath.minimum).
# See https://github.com/numpy/numpy/pull/20134 for details.
_win32_clip_workaround_needed = (
    sys.platform == 'win32' and
    tuple(map(int, np.__version__.split(".")[:2])) < (1, 22)
)

def clip_array(arr, vmin, vmax, out=None):
    # replacement for np.clip due to regression in
    # performance since numpy 1.17
    # https://github.com/numpy/numpy/issues/14281

    if vmin is None and vmax is None:
        # let np.clip handle the error
        return np.clip(arr, vmin, vmax, out=out)

    if vmin is None:
        return np.core.umath.minimum(arr, vmax, out=out)
    elif vmax is None:
        return np.core.umath.maximum(arr, vmin, out=out)
    elif _win32_clip_workaround_needed:
        if out is None:
            out = np.empty(arr.shape, dtype=np.find_common_type([arr.dtype], [type(vmax)]))
        out = np.core.umath.minimum(arr, vmax, out=out)
        return np.core.umath.maximum(out, vmin, out=out)

    else:
        return np.core.umath.clip(arr, vmin, vmax, out=out)