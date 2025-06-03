import struct
import os
from collections.abc import Sized
from typing import Protocol
from io import BufferedIOBase
from dataclasses import dataclass


class Opcode(Sized, Protocol):
    """An interface for opcodes used in the QOV encoding."""

    def write(self, file: BufferedIOBase) -> None:
        """Write the opcode to file."""
        ...


@dataclass
class RgbOpcode(Opcode):
    """The QOI_OP_RGB opcode, encodes a single RGB pixel."""

    r: int
    g: int
    b: int

    def write(self, file: BufferedIOBase) -> None:
        """Write the RGB opcode to the provided file handle."""
        file.write(struct.pack("<B3B", 0xFE, self.r, self.g, self.b))

    def __len__(self) -> int:
        """Fixed size of 4"""
        return 4

    @staticmethod
    def is_next(file: BufferedIOBase) -> bool:
        """Read the next byte and determine if it is a RgbOpcode."""
        code = file.read(1)
        file.seek(-1, os.SEEK_CUR)
        return code == b"\xfe"
