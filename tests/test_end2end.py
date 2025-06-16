from pyqoiv.encode import Encoder
from pyqoiv.decode import Decoder
from pyqoiv.types import ColourSpace
import numpy as np
from numpy.typing import NDArray
from typing import Generator, Optional, Callable
from io import BytesIO
import pytest
from .samples import short_test_sequences


@pytest.mark.parametrize(
    "video, width, height, frames, colourspace, keyframe_interval",
    short_test_sequences,
)
def test_end_to_end(
    video: Callable[[], Generator[NDArray[np.uint8]]],
    width: int,
    height: int,
    frames: int,
    colourspace: ColourSpace,
    keyframe_interval: Optional[int],
):
    file = BytesIO()
    encoder = Encoder(file, width, height, colourspace, keyframe_interval)
    for frame in video():
        encoder.push(frame)

    encoder.flush()

    file.seek(0)

    decoder = Decoder(file)

    for input_frame, (frame, details) in zip(video(), decoder):
        assert np.array_equal(input_frame, frame), (
            "Decoded frame does not match input frame."
        )
