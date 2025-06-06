from .types import ColourSpace, QovHeader, PixelHashMap, QovFrameHeader, FrameType
from .opcodes import (
    Opcode,
    RgbOpcode,
    IndexOpcode,
    DiffOpcode,
    RunOpcode,
    DiffFrameOpcode,
)
from io import BytesIO
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import List
from io import BufferedIOBase


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
    ):
        """Construct a new encoder."""
        self.header = QovHeader(width=width, height=height, colourspace=colourspace)
        self.file = file
        self.keyframe_interval = keyframe_interval
        self.last_keyframe: Optional[NDArray[np.uint8]] = None
        self.frames_since_last_keyframe: int = -1
        self.header.write(file)
        self.pixels = PixelHashMap()

    def trigger_keyframe(self) -> None:
        """Ensure that the next frame is a keyframe."""
        self.frames_since_last_keyframe = -1

    @property
    def is_next_frame_keyframe(self) -> bool:
        """Determine if the next frame is a keyframe."""
        if self.frames_since_last_keyframe == -1:
            self.frames_since_last_keyframe = 0
            return True

        if self.keyframe_interval is None:
            return True

        if self.frames_since_last_keyframe >= self.keyframe_interval:
            self.frames_since_last_keyframe = 0
            return True

        return False

    def encode_keyframe(
        self, frame: NDArray[np.uint8], pixels: PixelHashMap
    ) -> EncodedFrame:
        """Encode a frame as a keyframe."""
        return self._encode_frame(frame, pixels, None, None)

    def _encode_frame(
        self,
        frame: NDArray[np.uint8],
        pixels: PixelHashMap,
        key_frame_flat: Optional[NDArray[np.uint8]],
        key_pixels: Optional[PixelHashMap],
        exhaustive: bool = False,
    ):
        opcodes: List[Opcode] = []

        last_pixel: Optional[NDArray[np.uint8]] = None
        last_pixel_count = 0
        for pixel_pos, pixel in enumerate(frame.reshape(-1, 3)):
            opcode = None

            if last_pixel is not None:
                # Handle runs
                if np.array_equal(pixel, last_pixel) and last_pixel_count < 62:
                    last_pixel_count += 1
                    continue
                elif np.array_equal(pixel, last_pixel):
                    opcodes.append(RunOpcode(run=last_pixel_count))
                    last_pixel_count = 1
                    continue
                elif last_pixel_count > 0:
                    opcodes.append(RunOpcode(run=last_pixel_count))
                    last_pixel_count = 0

                diff = pixel - last_pixel
                if -2 <= diff[0] < 2 and -2 <= diff[1] < 2 and -2 <= diff[2] < 2:
                    opcode = DiffOpcode(diff[0], diff[1], diff[2])

            if pixel in pixels:
                opcode = IndexOpcode(index=pixels.push(pixel))

            if key_frame_flat is not None and key_pixels is not None:
                if exhaustive:
                    raise NotImplementedError()
                else:
                    if pixel in key_pixels:
                        key_index = key_pixels.index_of(pixel[0], pixel[1], pixel[2])
                        opcode = DiffFrameOpcode(True, True, key_index, 0, 0, 0)
                    elif np.array_equal(pixel, key_frame_flat[pixel_pos]):
                        opcode = DiffFrameOpcode(True, False, 0, 0, 0, 0)
            pixels.push(pixel)
            last_pixel = pixel
            if opcode is None:
                opcode = RgbOpcode(r=pixel[0], g=pixel[1], b=pixel[2])
            opcodes.append(opcode)

        if last_pixel_count > 0:
            opcodes.append(RunOpcode(run=last_pixel_count))

        frame_type = FrameType.Key if key_pixels is None else FrameType.Predicted
        return EncodedFrame(
            header=QovFrameHeader(
                frame_type=frame_type,
            ),
            opcodes=opcodes,
        )

    def encode_predicted(
        self,
        frame: NDArray[np.uint8],
        pixels: PixelHashMap,
        key_frame_flat: NDArray[np.uint8],
        key_pixels: PixelHashMap,
    ) -> EncodedFrame:
        """Encode a predicted frame."""
        return self._encode_frame(frame, pixels, key_frame_flat, key_pixels)

    def push(self, frame: NDArray[np.uint8]) -> None:
        """Push a new frame into the encoder."""

        if self.is_next_frame_keyframe:
            self.pixels.clear()
            encoded = self.encode_keyframe(frame, self.pixels)
            self.key_frame_flat = frame.reshape(-1, 3)
            encoded.write(self.file)

        else:
            encoded = self.encode_predicted(
                frame, self.pixels, self.key_frame_flat, PixelHashMap()
            )
            encoded.write(self.file)
            self.frames_since_last_keyframe += 1

    def flush(self) -> None:
        """Flush the encoder to the file."""
        self.file.flush()
