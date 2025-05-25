from .types import ColourSpace, QovHeader, PixelHashMap, QovFrameHeader, FrameType
from io import BytesIO
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from enum import Enum
from dataclasses import dataclass
from typing import List, Protocol
from collections.abc import Sized
from io import BufferedIOBase


class KeyFrameNow(Enum):
    No = 0
    Maybe = 1
    Yes = 2


class Opcode(Sized, Protocol):
    def write(self, file: BufferedIOBase) -> None: ...


@dataclass
class EncodedFrame:
    header: QovFrameHeader
    opcodes: List[Opcode]

    def __len__(self):
        return sum([len(opcode) for opcode in self.opcodes])

    def write(self, file: BufferedIOBase) -> None:
        self.header.write(file)
        for opcode in self.opcodes:
            opcode.write(file)


class Encoder:
    def __init__(
        self,
        file: BytesIO,
        width: int,
        height: int,
        colourspace: ColourSpace,
        keyframe_interval: Optional[int] = None,
        max_keyframe_interval: Optional[int] = 600,
    ):
        self.header = QovHeader(width=width, height=height, colourspace=colourspace)
        self.file = file
        self.keyframe_interval = keyframe_interval
        self.max_keyframe_interval = max_keyframe_interval
        self.last_keyframe: Optional[NDArray[np.uint8]] = None
        self.frames_since_last_keyframe: int = -1
        self.header.write(file)
        self.pixels = PixelHashMap()

    def trigger_keyframe(self) -> None:
        self.frames_since_last_keyframe = -1

    @property
    def is_next_frame_keyframe(self) -> KeyFrameNow:
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
        opcodes = []
        # TODO
        return EncodedFrame(
            header=QovFrameHeader(frame_type=FrameType.Key), opcodes=opcodes
        )

    def encode_predicted(
        self, frame: NDArray[np.uint8], pixels: PixelHashMap
    ) -> EncodedFrame:
        # TODO
        return self.encode_keyframe(frame, pixels)

    def push(self, frame: NDArray[np.uint8]) -> None:
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
