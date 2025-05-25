import struct
from dataclasses import dataclass
from enum import IntEnum
from io import BufferedIOBase
from numpy.typing import NDArray
import numpy as np


class ColourSpace(IntEnum):
    sRGB = 0
    Linear = 1


class FrameType(IntEnum):
    Key = 0
    Predicted = 1


@dataclass
class QovHeader:
    magic: str = "qoiv"
    width: int = 640
    height: int = 480
    colourspace: ColourSpace = ColourSpace.sRGB
    # There are 3 padding bytes after the colourspace field to align the structure to 16 bytes.

    @staticmethod
    def read(file: BufferedIOBase):
        header_packed: bytes = file.read(16)
        if len(header_packed) != 16:
            raise ValueError(f"Invalid header size, was {len(header_packed)}")
        magic, width, height, colourspace = struct.unpack("<4sIIBxxx", header_packed)
        if magic != b"qoiv":
            raise ValueError("Invalid magic number")
        if colourspace not in ColourSpace:
            raise ValueError("Invalid colourspace")
        return QovHeader(
            magic=magic.decode("utf-8"),
            width=width,
            height=height,
            colourspace=ColourSpace(colourspace),
        )

    def write(self, file: BufferedIOBase):
        if self.magic != "qoiv":
            raise ValueError("Invalid magic number")
        if self.colourspace not in ColourSpace:
            raise ValueError("Invalid colourspace")
        file.write(
            struct.pack(
                "<4sIIBxxx",
                self.magic.encode("utf-8"),
                self.width,
                self.height,
                self.colourspace,
            )
        )


class PixelHashMap:
    def __init__(self, size: int = 64):
        self.size = size
        self.pixels = np.array([[0, 0, 0]] * self.size)

    def push(self, pixel: NDArray[np.uint8]):
        r, g, b = pixel
        index = (r * 3 + g * 5 + b * 7) % self.size

        self.pixels[index] = pixel

    def __getitem__(self, index: int) -> NDArray[np.uint8]:
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        return self.pixels[index]

    def __contains__(self, pixel: NDArray[np.uint8]) -> bool:
        r, g, b = pixel
        index = (r * 3 + g * 5 + b * 7) % self.size
        return np.array_equal(self.pixels[index], pixel)

    def clear(self):
        self.pixels.fill(0)


@dataclass
class QovFrameHeader:
    frame_type: FrameType

    @staticmethod
    def read(file: BufferedIOBase):
        frame_type = FrameType(int.from_bytes(file.read(1)))
        if frame_type not in FrameType:
            raise ValueError("Invalid frame type")
        return QovFrameHeader(frame_type=frame_type)

    def write(self, file: BufferedIOBase):
        file.write(struct.pack("<B", self.frame_type))
