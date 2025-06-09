import struct
import os
from collections.abc import Sized
from typing import Protocol, Optional
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


@dataclass
class RunOpcode(Opcode):
    """The QOI_OP_RUN opcode, encodes a run of identical pixels."""

    run: int

    def write(self, file: BufferedIOBase) -> None:
        """Write the Run opcode to the provided file handle."""
        if not (1 <= self.run <= 62):
            raise ValueError("Run value must be between 1 and 62")
        file.write(struct.pack("<B", 0xC0 | (self.run - 1)))

    def __len__(self) -> int:
        """Fixed size of 1"""
        return 1

    @staticmethod
    def is_next(file: BufferedIOBase) -> bool:
        """Determine if the next opcode is a RunOpcode."""
        code = file.read(1)
        file.seek(-1, os.SEEK_CUR)
        return (code[0] & 0xC0) == 0xC0 and code[0] != 0xFF and code[0] != 0xFE

    @staticmethod
    def read(file: BufferedIOBase) -> "RunOpcode":
        """Read a Run opcode from the provided file handle."""
        code = file.read(1)
        if len(code) != 1 or code[0] & 0xC0 != 0xC0:
            raise ValueError("Invalid Run opcode")
        run = (code[0] & 0x3F) + 1
        return RunOpcode(run=run)


@dataclass
class FrameRunOpcode(Opcode):
    """This Opcode replaces the QOI_OP_RGBA opcode, and is used to represent a run of identical pixels from a previous frame."""

    is_keyframe: bool
    run: int

    def write(self, file: BufferedIOBase) -> None:
        """Write the FrameRunOpcode to the provided file handle."""
        if not (1 <= self.run <= 128):
            raise ValueError("Run value must be between 1 and 128")
        file.write(struct.pack("<BB", 0xFF, (self.run - 1) | self.is_keyframe << 7))

    def __len__(self) -> int:
        return 2

    @staticmethod
    def is_next(file: BufferedIOBase) -> bool:
        code = file.read(1)
        file.seek(-1, os.SEEK_CUR)
        return code[0] == 0xFF

    @staticmethod
    def read(file: BufferedIOBase) -> "FrameRunOpcode":
        """Read a FrameRunOpcode from the provided file handle."""
        code = file.read(2)
        if len(code) != 2 or code[0] != 0xFF:
            raise ValueError("Invalid FrameRun opcode")
        is_keyframe = (code[1] & 0x80) != 0
        run = (code[1] & 0x7F) + 1
        return FrameRunOpcode(is_keyframe, run)


class DiffFrameOpcode(Opcode):
    """This Opcode replaces the QOI_OP_LUMA opcode, and is used to query data from previous frames.

    If key_frame is True, then the value is fetched from the keyframe.
    If key_frame is False, then the value is fetched from the previous frame.
    If use_index is True, then the value is fetched from the PixelHashMap, for the keyframe, using the index.
    If use_index is False, then the value is fetched from the frame buffer.

    key_frame == False and use_index == True is an invalid state.

    dr, dg, and db are the differences in red, green, and blue channels respectively from the pixel fetched from the above location.
    If use_index is False, these differences are added to the value in index, which will be between -32..31.
    """

    def __init__(
        self,
        key_frame: bool,
        use_index: bool,
        dr: int,
        dg: int,
        db: int,
        index: Optional[int] = None,
        diff: Optional[int] = None,
    ):
        """Construct a new DiffFrameOpcode."""
        self.key_frame = key_frame
        self.use_index = use_index
        if diff and index:
            raise ValueError("Only one of index or diff can be provided")
        if index is not None:
            self.index = index
        elif diff is not None:
            self.index = diff + 32
        else:
            raise ValueError("Either index or diff must be provided")
        self.dr = dr
        self.dg = dg
        self.db = db

    def __len__(self) -> int:
        """Fixed size of 2"""
        return 2

    def __eq__(self, other: object) -> bool:
        """Check for equality with another DiffFrameOpcode."""
        if isinstance(other, DiffFrameOpcode):
            return (
                self.key_frame == other.key_frame
                and self.use_index == other.use_index
                and self.index == other.index
                and self.dr == other.dr
                and self.dg == other.dg
                and self.db == other.db
            )
        return False

    @staticmethod
    def is_next(file: BufferedIOBase) -> bool:
        """Determine if the next opcode is a DiffFrameOpcode."""
        code = file.read(1)
        file.seek(-1, os.SEEK_CUR)
        return (code[0] & 0xC0) == 0x80

    @staticmethod
    def read(file: BufferedIOBase) -> "DiffFrameOpcode":
        """Read a DiffFrameOpcode from the provided file handle."""
        b = file.read(2)
        if len(b) != 2 or (b[0] & 0xC0) != 0x80:
            raise ValueError("Invalid DiffFrame opcode")
        key_frame = (b[1] & 0x80) != 0
        use_index = (b[1] & 0x40) != 0
        index = b[0] & 0x3F
        dr = ((b[1] >> 4) & 0x03) - 2
        dg = ((b[1] >> 2) & 0x03) - 2
        db = (b[1] & 0x03) - 2
        return DiffFrameOpcode(key_frame, use_index, dr, dg, db, index)

    @property
    def index(self) -> int:
        """Getter for index."""
        return self._index

    @property
    def diff(self) -> int:
        """calculate the difference from the base index of 32."""
        return self.index - 32

    @index.setter
    def index(self, value: int):
        """Set the index."""
        if value < 0 or value > 63:
            raise ValueError("Index must be between 0 and 63")
        self._index = value

    @property
    def dr(self) -> int:
        """Getter for dr (red difference)."""
        return self._dr

    @dr.setter
    def dr(self, value: int):
        """Set the red difference."""
        if not (-2 <= value < 2):
            raise ValueError("dr must be between -2 and 1")
        self._dr = value

    @property
    def dg(self) -> int:
        """Getter for dg (green difference)."""
        return self._dg

    @dg.setter
    def dg(self, value: int):
        """Set the green difference."""
        if not (-2 <= value < 2):
            raise ValueError("dg must be between -2 and 1")
        self._dg = value

    @property
    def db(self) -> int:
        """Getter for db (blue difference)."""
        return self._db

    @db.setter
    def db(self, value: int):
        """Set the blue difference."""
        if not (-2 <= value < 2):
            raise ValueError("db must be between -2 and 1")
        self._db = value

    def write(self, file: BufferedIOBase) -> None:
        """Write the DiffFrameOpcode to the provided file handle."""
        file.write(
            struct.pack(
                "<BB",
                0x80 | self.index,
                int(self.key_frame) << 7
                | int(self.use_index) << 6
                | (self.dr + 2) << 4
                | (self.dg + 2) << 2
                | (self.db + 2),
            )
        )
