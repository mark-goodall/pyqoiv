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

    @staticmethod
    def read(file: BufferedIOBase) -> "RgbOpcode":
        """Read an RGB opcode from the provided file handle."""
        code = file.read(4)
        if len(code) != 4 or code[0] != 0xFE:
            raise ValueError("Invalid RGB opcode")
        r, g, b = struct.unpack("<3B", code[1:])
        return RgbOpcode(r, g, b)


@dataclass
class IndexOpcode(Opcode):
    """The QOI_OP_INDEX opcode, encodes an index into the pixel hash map."""

    index: int

    def write(self, file: BufferedIOBase) -> None:
        """Write the Index opcode to the provided file handle."""
        if 0 > self.index or self.index > 63:
            raise ValueError("Index must be between 0 and 63")
        file.write(struct.pack("<B", self.index & 0x3F))

    def __len__(self) -> int:
        """Fixed size of 1"""
        return 1

    @staticmethod
    def is_next(file: BufferedIOBase) -> bool:
        """Read the next byte and determine if it is an IndexOpcode."""
        code = file.read(1)
        file.seek(-1, os.SEEK_CUR)
        return 0 <= code[0] <= 63

    @staticmethod
    def read(file: BufferedIOBase) -> "IndexOpcode":
        """Read an Index opcode from the provided file handle."""
        code = file.read(1)
        if len(code) != 1 or not (0 <= code[0] <= 63):
            raise ValueError("Invalid Index opcode")
        return IndexOpcode(index=code[0])


@dataclass
class DiffOpcode(Opcode):
    """The QOI_OP_DIFF opcode, encodes a difference between the last pixel and the current pixel."""

    dr: int
    dg: int
    db: int

    def write(self, file: BufferedIOBase) -> None:
        """Write the Diff opcode to the provided file handle."""
        if not (-2 <= self.dr < 2 and -2 <= self.dg < 2 and -2 <= self.db < 2):
            raise ValueError("Diff values must be between -2 and 1")
        file.write(
            struct.pack(
                "<B", 0x40 | (self.dr + 2) << 4 | (self.dg + 2) << 2 | (self.db + 2)
            )
        )

    def __len__(self) -> int:
        """Fixed size of 1"""
        return 1

    @staticmethod
    def is_next(file: BufferedIOBase) -> bool:
        """Determine if the next opcode is a DiffOpcode."""
        code = file.read(1)
        file.seek(-1, os.SEEK_CUR)
        return code[0] & 0xC0 == 0x40

    @staticmethod
    def read(file: BufferedIOBase) -> "DiffOpcode":
        """Read a Diff opcode from the provided file handle."""
        code = file.read(1)
        if len(code) != 1 or code[0] & 0xC0 != 0x40:
            raise ValueError("Invalid Diff opcode")
        dr = ((code[0] >> 4) & 0x03) - 2
        dg = ((code[0] >> 2) & 0x03) - 2
        db = (code[0] & 0x03) - 2
        return DiffOpcode(dr, dg, db)
