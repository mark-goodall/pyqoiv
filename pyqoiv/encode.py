from .types import ColourSpace, QovHeader, PixelHashMap, QovFrameHeader, FrameType
from .opcodes import Opcode, RgbOpcode, IndexOpcode, DiffOpcode
from io import BytesIO
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from enum import Enum
from dataclasses import dataclass
from typing import List
from io import BufferedIOBase


class KeyFrameNow(Enum):
    """A quick enum to determine if the next frame should be a keyframe."""

    No = 0
    Maybe = 1
    Yes = 2


@dataclass
class EncodedFrame:
    """A helper class to represent how to encode a frame as a series of opcodes."""

    header: QovFrameHeader
    opcodes: List[Opcode]

    def __len__(self):
        """Report the size of the frame in bytes."""
        return sum([len(opcode) for opcode in self.opcodes])

    def write(self, file: BufferedIOBase) -> None:
        """Convert and write the frame to the provided file handle."""
        self.header.write(file)
        for opcode in self.opcodes:
            opcode.write(file)


class Encoder:
    """This class is responsible for encoding frames into the QOV format."""

    def __init__(
        self,
        file: BytesIO,
        width: int,
        height: int,
        colourspace: ColourSpace,
        keyframe_interval: Optional[int] = None,
        max_keyframe_interval: Optional[int] = 600,
    ):
        """Construct a new encoder."""
        self.header = QovHeader(width=width, height=height, colourspace=colourspace)
        self.file = file
        self.keyframe_interval = keyframe_interval
        self.max_keyframe_interval = max_keyframe_interval
        self.last_keyframe: Optional[NDArray[np.uint8]] = None
        self.frames_since_last_keyframe: int = -1
        self.header.write(file)
        self.pixels = PixelHashMap()

    def trigger_keyframe(self) -> None:
        """Ensure that the next frame is a keyframe."""
        self.frames_since_last_keyframe = -1

    @property
    def is_next_frame_keyframe(self) -> KeyFrameNow:
        """Partly determine if the next frame is a keyframe."""
        if self.frames_since_last_keyframe == -1:
            return KeyFrameNow.Yes

        if self.keyframe_interval is not None:
            if self.frames_since_last_keyframe >= self.keyframe_interval:
                self.frames_since_last_keyframe = 0
                return KeyFrameNow.Yes

        if self.max_keyframe_interval is not None:
            if self.frames_since_last_keyframe >= self.max_keyframe_interval:
                self.frames_since_last_keyframe = 0
                return KeyFrameNow.Yes
            else:
                return KeyFrameNow.Maybe

        return KeyFrameNow.No

    def encode_keyframe(
        self, frame: NDArray[np.uint8], pixels: PixelHashMap
    ) -> EncodedFrame:
        """Encode a frame as a keyframe."""
        opcodes: List[Opcode] = []

        last_pixel: Optional[NDArray[np.uint8]] = None
        for pixel in frame.reshape(-1, 3):
            opcode = None

            if last_pixel is not None:
                # TODO Handle runs
                if np.array_equal(pixel, last_pixel):
                    pass
                else:
                    pass

                diff = pixel - last_pixel
                if -2 <= diff[0] < 2 and -2 <= diff[1] < 2 and -2 <= diff[2] < 2:
                    opcode = DiffOpcode(diff[0], diff[1], diff[2])

            if pixel in pixels:
                opcode = IndexOpcode(index=pixels.push(pixel))
            # TODO determine Opcode to use
            pixels.push(pixel)
            last_pixel = pixel
            if opcode is None:
                opcode = RgbOpcode(r=pixel[0], g=pixel[1], b=pixel[2])
            opcodes.append(opcode)
        return EncodedFrame(
            header=QovFrameHeader(frame_type=FrameType.Key), opcodes=opcodes
        )

    def encode_predicted(
        self, frame: NDArray[np.uint8], pixels: PixelHashMap
    ) -> EncodedFrame:
        """Encode a predicted frame."""
        # TODO encode a predicted frame
        return self.encode_keyframe(frame, pixels)

    def push(self, frame: NDArray[np.uint8]) -> None:
        """Push a new frame into the encoder."""
        is_keyframe = self.is_next_frame_keyframe

        if is_keyframe == KeyFrameNow.Yes:
            self.pixels.clear()
            encoded = self.encode_keyframe(frame, self.pixels)
            encoded.write(self.file)

        elif is_keyframe == KeyFrameNow.No:
            encoded = self.encode_predicted(frame, self.pixels)
            encoded.write(self.file)
        else:
            p = self.encode_predicted(frame, self.pixels)
            new_pixels = PixelHashMap()
            k = self.encode_keyframe(frame, new_pixels)
            if len(p) > len(k):
                p.write(self.file)
            else:
                k.write(self.file)
                self.pixels = new_pixels

    def flush(self) -> None:
        """Flush the encoder to the file."""
        self.file.flush()
