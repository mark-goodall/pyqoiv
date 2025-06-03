from pyqoiv.encode import EncodedFrame, Opcode, Encoder
from pyqoiv.types import QovFrameHeader, FrameType, ColourSpace
from io import BufferedIOBase, BytesIO
from typing import Callable, Generator, Optional
import numpy as np
from numpy.typing import NDArray
import pytest
from .samples import short_test_sequences


class TestOpcode(Opcode):
    def write(self, file: BufferedIOBase):
        """Write the opcode to the file."""
        file.write(b"\x00")

    def __len__(self):
        """Return the size of the opcode in bytes."""
        return 1


def test_encoded_frame():
    frame = EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Key), opcodes=[TestOpcode()]
    )
    file = BytesIO()
    frame.write(file)
    file.seek(0)
    new_frame = QovFrameHeader.read(file)
    assert new_frame.frame_type == FrameType.Key
    first_byte = file.read(1)
    assert first_byte == b"\x00"
    assert file.read() == b""  # Ensure no extra data is read


@pytest.mark.parametrize(
    "video, width, height, frames, colourspace, keyframe_interval, max_keyframe_interval",
    short_test_sequences,
)
def test_encoder_shrinks_video(
    video: Callable[[], Generator[NDArray[np.uint8]]],
    width: int,
    height: int,
    frames: int,
    colourspace: ColourSpace,
    keyframe_interval: Optional[int],
    max_keyframe_interval: Optional[int],
):
    max_size = width * height * 3 * frames
    file = BytesIO()
    encoder = Encoder(
        file, width, height, colourspace, keyframe_interval, max_keyframe_interval
    )
    for frame in video():
        encoder.push(frame)

    encoder.flush()

    size = file.tell()

    # All of these simple tests should compress well
    assert size < max_size
