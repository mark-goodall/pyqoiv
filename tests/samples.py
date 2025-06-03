from typing import Callable, Generator
import numpy as np
from numpy.typing import NDArray
from pyqoiv.types import ColourSpace


def create_static_video(
    width: int, height: int, frames: int
) -> Callable[[], Generator[NDArray[np.uint8]]]:
    """Create a static video with the specified width, height, and number of frames."""
    frame = np.full((height, width, 3), fill_value=128, dtype=np.uint8)

    def impl():
        for _ in range(frames):
            yield frame

    return impl


def create_ball_video(
    width: int, height: int, frames: int
) -> Callable[[], Generator[NDArray[np.uint8]]]:
    """Create a basic video with the specified width, height, and number of frames."""

    def impl():
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = int(width * 0.5 + (width / 3) * np.sin(i / 10))
            y = int(height * 0.5 + (height / 3) * np.cos(i / 10))
            radius = width // 10
            frame[y - radius : y + radius, x - radius : x + radius, 0] = 255
            frame[y - radius : y + radius, x - radius : x + radius, 1] = 255
            frame[y - radius : y + radius, x - radius : x + radius, 2] = 255
            yield frame

    return impl


short_test_sequences = [
    (create_static_video(64, 64, 20), 64, 64, 20, ColourSpace.sRGB, None, 6),
    (create_ball_video(64, 64, 20), 64, 64, 20, ColourSpace.sRGB, None, 6),
    (create_static_video(64, 64, 20), 64, 64, 20, ColourSpace.sRGB, 6, None),
    (create_ball_video(64, 64, 20), 64, 64, 20, ColourSpace.sRGB, 6, None),
]
