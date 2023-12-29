from typing import List, Optional, Union
import subprocess
import os
from pathlib import Path
from .video_utils import VideoReader
import av

__all__ = ["optimize_gif", "video2gif"]


def optimize_gif(
    sources: Union[List[str], str, List[Path], Path],
    destination: Optional[str] = None,
    optimize: bool = False,
    colors: int = 256,
    options: Optional[List[str]] = None
) -> None:
    """Apply gifsicle with given options to image at given paths.

    Parameters
    -----------------
    sources:Union[List[str], str, List[Path], Path],
        Path or paths to gif(s) image(s) to optimize.
    destination:Optional[str] = None
        Path where to save updated gif(s).
        By default the old image is overwrited.
        If multiple sources are specified, they will be merged.
    optimize: bool = False,
        Boolean flag to add the option to optimize image.
    colors:int = 256,
        Integer value representing the number of colors to use. Must be a power of 2.
    options:Optional[List[str]] = None
        List of options.

    Raises
    ------------------
    ValueError:
        If gifsicle is not installed.
    ValueError:
        If given source path does not exist.
    ValueError:
        If given source path is not a gif image.
    ValueError:
        If given destination path is not a gif image.

    References
    ------------------
    You can learn more about gifsicle at the project home page:
    https://www.lcdf.org/gifsicle/
    """
    if isinstance(sources, (str, Path)):
        sources = [sources]
    for source in sources:
        if isinstance(source, Path):
            source = str(source)  # should work on all windows, mac, and linux
        if not os.path.exists(source):
            raise ValueError(
                "Given source path `{}` does not exist.".format(source)
            )
        if not source.endswith(".gif"):
            raise ValueError(
                "Given source path `{}` is not a gif image.".format(source)
            )

    if destination is None:
        destination = sources[0]

    if not str(destination).endswith(".gif"):
        raise ValueError("Given destination path is not a gif image.")

    if options is None:
        options = []

    if optimize and "-O3" not in options:
        options.append("-O3")

    try:
        subprocess.call(["gifsicle", *options, *sources, "--colors",
                        str(colors), "--output", destination])
    except FileNotFoundError:
        raise FileNotFoundError((
            "The gifsicle library was not found on your system.\n"
            "On MacOS it is automatically installed using brew when you "
            "use the pip install command.\n"
            "On other systems, like Linux systems and Windows, it prompts the "
            "instructions to be followed for completing the installation.\n"
            "You can learn more on how to install gifsicle on "
            "the gifsicle and pygifsicle documentation."
        ))

def crop_img(img, rect):
    h, w = img.shape[:2]
    x = int(rect[0] * w)
    y = int(rect[1] * h)
    w = int(rect[2] * w)
    h = int(rect[3] * h)
    return img[y:y+h, x:x+w]

def video2gif(
    video_path,
    save_path=None,
    speed = 1,
    fps = 12,
    shape_scale = 1,
    img_shape = None,  # h, w
    optimize_options = ['-O3'], # '--colors', '128',
    crop_rect = (0, 0, 1, 1),  # x, y, w, h
):
    video = VideoReader(video_path)

    # calculate img shape
    img = video.get_frame()[0]
    img = crop_img(img, crop_rect)
    if img_shape is None:
        img_shape = (int(img.shape[0]*shape_scale), int(img.shape[1]*shape_scale))
    else:
        img_shape = (int(img_shape[0]*shape_scale), int(img_shape[1]*shape_scale))

    # create gif container
    if save_path is None:
        save_path = str(video_path).replace('.mp4', '.gif')

    gif_container = av.open(str(save_path), mode='w')
    stream = gif_container.add_stream('gif', rate=fps)
    stream.height = img_shape[0]
    stream.width = img_shape[1]
    stream.pix_fmt = 'rgb8'

    # write gif
    duration = video.duration
    pbar_width = 20
    cnt = 0
    for img, stamp in video.get_generator(fps=fps / speed):
        img = crop_img(img, crop_rect)
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        frame = frame.reformat(format='rgb8')
        gif_container.mux(stream.encode(frame))

        percent = stamp / duration
        done = int(percent*pbar_width) * '█'
        todo = (pbar_width - int(percent*pbar_width)) * ' '
        print(f'Converting to gif: |{done}{todo}| {percent*100:.2f}%', end='\r')
        cnt += 1

    print(f'Converting to gif: |{done}| {100:.2f}%, frames: {cnt:d}', end='\n')
    gif_container.close()

    print('Optimizing gif...')
    optimize_gif(save_path, options=optimize_options)


def avalible_codecs():
    codecs = av.codec.codec.codecs_available
    for codec in codecs:
        print(codec)


if __name__ == '__main__':
    video = 'J:\\test_imgs\\video.mp4'
    video2gif(video, fps=7, shape_scale=0.5)