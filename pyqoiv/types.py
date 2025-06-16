import struct
from dataclasses import dataclass
from enum import IntEnum
from io import BufferedIOBase
from numpy.typing import NDArray
import numpy as np


class ColourSpace(IntEnum):
    """Enum to differentiate colour spaces."""

    sRGB = 0
    Linear = 1


class FrameType(IntEnum):
    """Enum to differentiate frame types."""

    # Key Frame types are independently decodeable or encodable.
    Key = 0
    # Predicted Frame types are encoded based on the previous key frame.
    Predicted = 1


@dataclass
class QovHeader:
    """Data type to facilitate reading and writing the QOV file header."""

    magic: str = "qoiv"
    width: int = 640
    height: int = 480
    colourspace: ColourSpace = ColourSpace.sRGB
    # There are 3 padding bytes after the colourspace field to align the structure to 16 bytes.

    @staticmethod
    def read(file: BufferedIOBase) -> "QovHeader":
        """Read the header from the provided file handle."""
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

    def write(self, file: BufferedIOBase) -> None:
        """Write the header to the provided file handle."""
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
    """Hash map for constant time lookup of previously used colours."""

    def __init__(self, size: int = 64):
        """Construct a fixed size hash map for colours."""
        self.size = size
        self.pixels = np.array([[0, 0, 0]] * self.size)

    def push(self, pixel: NDArray[np.uint8]) -> int:
        """Push a pixel into the hash map."""
        r, g, b = pixel

        index = self.index_of(r, g, b)
        self.pixels[index] = pixel
        return index

    def __getitem__(self, index: int) -> NDArray[np.uint8]:
        """Get an item from the hash map by index."""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of bounds")
        return self.pixels[index]

    def __contains__(self, pixel: NDArray[np.uint8]) -> bool:
        """Check the colour is in the map"""
        r, g, b = pixel
        return PixelHashMap.pixel_equal(self.pixels[self.index_of(r, g, b)], pixel)

    def clear(self):
        """Clear the hash map."""
        self.pixels.fill(0)

    def index_of(self, r: int | np.uint8, g: int | np.uint8, b: int | np.uint8) -> int:
        """Calculate the index of a pixel in the hash map."""
        return (r * 3 + g * 5 + b * 7) % self.size

    @staticmethod
    def pixel_equal(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> bool:
        """Check if two pixels are equal."""
        return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@dataclass
class QovFrameHeader:
    """Header for a single frame."""

    frame_type: FrameType

    @staticmethod
    def read(file: BufferedIOBase) -> "QovFrameHeader":
        """Read the frame header from the provided file handle."""
        frame_type = FrameType(int.from_bytes(file.read(1)))
        if frame_type not in FrameType:
            raise ValueError("Invalid frame type")
        return QovFrameHeader(frame_type=frame_type)

    def write(self, file: BufferedIOBase):
        """Write the frame header to the provided file handle."""
        file.write(struct.pack("<B", self.frame_type))
