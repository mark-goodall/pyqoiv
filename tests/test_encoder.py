from pyqoiv.encode import EncodedFrame, Encoder
from pyqoiv.types import QovFrameHeader, FrameType, ColourSpace, PixelHashMap
from pyqoiv.opcodes import IndexOpcode, RgbOpcode, RunOpcode, DiffOpcode, Opcode
from io import BufferedIOBase, BytesIO
from typing import Callable, Generator, Optional
import numpy as np
from numpy.typing import NDArray
import pytest
from .samples import short_test_sequences


class FakeOpcode(Opcode):
    def write(self, file: BufferedIOBase):
        """Write the opcode to the file."""
        file.write(b"\x00")

    def __len__(self):
        """Return the size of the opcode in bytes."""
        return 1


def test_encoded_frame():
    frame = EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Key), opcodes=[FakeOpcode()]
    )
    file = BytesIO()
    frame.write(file)
    file.seek(0)
    new_frame = QovFrameHeader.read(file)
    assert new_frame.frame_type == FrameType.Key
    first_byte = file.read(1)
    assert first_byte == b"\x00"
    assert file.read() == b""  # Ensure no extra data is read

    assert len(frame) == 1
    frame.opcodes.append(FakeOpcode())
    assert len(frame) == 2


@pytest.mark.parametrize(
    "video, width, height, frames, colourspace, keyframe_interval",
    short_test_sequences,
)
def test_encoder_shrinks_video(
    video: Callable[[], Generator[NDArray[np.uint8]]],
    width: int,
    height: int,
    frames: int,
    colourspace: ColourSpace,
    keyframe_interval: Optional[int],
):
    max_size = width * height * 3 * frames
    file = BytesIO()
    encoder = Encoder(file, width, height, colourspace, keyframe_interval)
    for frame in video():
        encoder.push(frame)

    encoder.flush()

    size = file.tell()

    # All of these simple tests should compress well
    assert size < max_size


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.ones((10, 10, 3), dtype=np.uint8) * 255,
        np.ones((10, 10, 3), dtype=np.uint8) * 128,
    ],
)
def test_encoder_encodes_flat_frame_as_expected(frame: NDArray[np.uint8]):
    encoder = Encoder(BytesIO(), width=10, height=10, colourspace=ColourSpace.sRGB)
    encoded_frame = encoder.encode_keyframe(frame, PixelHashMap())
    assert len(encoded_frame.opcodes) == 3
    if frame[0, 0, 0] == 0:
        assert isinstance(encoded_frame.opcodes[0], IndexOpcode)
        assert encoded_frame.opcodes[0].index == 0
    else:
        assert isinstance(encoded_frame.opcodes[0], RgbOpcode)
    assert isinstance(encoded_frame.opcodes[1], RunOpcode)
    assert isinstance(encoded_frame.opcodes[2], RunOpcode)
    assert encoded_frame.opcodes[1].run == 62
    assert encoded_frame.opcodes[2].run == 100 - 1 - 62


def test_encoder_uses_diff_opcode_as_expected():
    encoder = Encoder(BytesIO(), width=10, height=10, colourspace=ColourSpace.sRGB)
    encoded_frame = encoder.encode_keyframe(
        np.array([[[1, 1, 1], [2, 2, 2], [2, 1, 1]]]), PixelHashMap()
    )
    assert len(encoded_frame.opcodes) == 3
    assert isinstance(encoded_frame.opcodes[0], RgbOpcode)
    assert isinstance(encoded_frame.opcodes[1], DiffOpcode)
    assert encoded_frame.opcodes[1].dr == 1
    assert encoded_frame.opcodes[1].dg == 1
    assert encoded_frame.opcodes[1].db == 1
    assert isinstance(encoded_frame.opcodes[2], DiffOpcode)
    assert encoded_frame.opcodes[2].dr == 0
    assert encoded_frame.opcodes[2].dg == -1
    assert encoded_frame.opcodes[2].db == -1


def test_encoder_uses_index_opcode_as_expected():
    encoder = Encoder(BytesIO(), width=10, height=10, colourspace=ColourSpace.sRGB)
    encoded_frame = encoder.encode_keyframe(
        np.array([[[1, 1, 1], [20, 20, 20], [1, 1, 1]]]), PixelHashMap()
    )
    assert len(encoded_frame.opcodes) == 3
    assert isinstance(encoded_frame.opcodes[0], RgbOpcode)
    assert isinstance(encoded_frame.opcodes[1], RgbOpcode)
    assert isinstance(encoded_frame.opcodes[2], IndexOpcode)
