from io import BytesIO
from numpy.typing import NDArray
import numpy as np
from .types import QovHeader, QovFrameHeader, FrameType, PixelHashMap
from .opcodes import (
    RgbOpcode,
    DiffOpcode,
    IndexOpcode,
    FrameRunOpcode,
    RunOpcode,
    DiffFrameOpcode,
)


class Decoder:
    """Decode a QOIV file into frames."""

    def __init__(self, file: BytesIO):
        """Construct a new decoder."""
        self.file = file
        self.header = QovHeader.read(file)
        self.first_frame_pos = file.tell()
        self.pixel_count = self.header.width * self.header.height
        self.key_pixels = PixelHashMap()

    def __iter__(self) -> "Decoder":
        """Setup the iterator"""
        self.frame_pos = self.first_frame_pos
        self.file.seek(self.frame_pos)
        self.key_pixels = PixelHashMap()
        self.key_frame_flat = None
        return self

    def __next__(self):
        """Get the next frame."""
        return self.read_frame()

    def read_frame(self) -> NDArray[np.uint8]:
        """Read the next frame from the file."""
        frame_header = QovFrameHeader.read(self.file)

        frame = np.zeros((self.header.height, self.header.width, 3), dtype=np.uint8)

        pixels = PixelHashMap()

        pixel_read = 0
        while pixel_read < self.pixel_count:
            cy = pixel_read // self.header.width
            cx = pixel_read % self.header.width
            if RgbOpcode.is_next(self.file):
                opcode = RgbOpcode.read(self.file)
                frame[
                    cy,
                    cx,
                ] = [opcode.r, opcode.g, opcode.b]
                pixels.push(frame[cy, cx])
                pixel_read += 1
            elif DiffOpcode.is_next(self.file):
                opcode = DiffOpcode.read(self.file)
                frame[cy, cx] = frame[
                    (pixel_read - 1) // self.header.width,
                    (pixel_read - 1) % self.header.width,
                ] + np.array([opcode.dr, opcode.dg, opcode.db])
                pixels.push(frame[cy, cx])
                pixel_read += 1
            elif RunOpcode.is_next(self.file):
                opcode = RunOpcode.read(self.file)
                last_pixel = frame[
                    (pixel_read - 1) // self.header.width,
                    (pixel_read - 1) % self.header.width,
                ]
                frame.reshape(-1, 3, copy=False)[
                    pixel_read : pixel_read + opcode.run
                ] = last_pixel
                pixel_read += opcode.run
            elif IndexOpcode.is_next(self.file):
                index_opcode = IndexOpcode.read(self.file)
                frame[cy, cx] = pixels[index_opcode.index]
                pixel_read += 1
            elif DiffFrameOpcode.is_next(self.file):
                if self.key_frame_flat is None:
                    raise ValueError("Unexpected DiffFrameOpcode without key frame.")
                if frame_header.frame_type == FrameType.Key:
                    raise ValueError("Unexpected DiffFrameOpcode in key frame.")

                diff_frame_opcode = DiffFrameOpcode.read(self.file)
                if diff_frame_opcode.key_frame:
                    if diff_frame_opcode.use_index:
                        pixel = self.key_pixels[diff_frame_opcode.index] + np.array(
                            [
                                diff_frame_opcode.dr,
                                diff_frame_opcode.dg,
                                diff_frame_opcode.db,
                            ]
                        )

                        frame[cy, cx] = pixel
                        pixels.push(pixel)
                    else:
                        pixel = self.key_frame_flat[pixel_read] + np.array(
                            [
                                diff_frame_opcode.diff + diff_frame_opcode.dr,
                                diff_frame_opcode.diff + diff_frame_opcode.dg,
                                diff_frame_opcode.diff + diff_frame_opcode.db,
                            ]
                        )

                        frame[cy, cx] = pixel
                        pixels.push(pixel)
                else:
                    raise NotImplementedError()
                pixel_read += 1
            elif FrameRunOpcode.is_next(self.file) and self.key_frame_flat is not None:
                opcode = FrameRunOpcode.read(self.file)
                if not opcode.is_keyframe:
                    raise NotImplementedError()

                frame.reshape(-1, 3, copy=False)[
                    pixel_read : pixel_read + opcode.run
                ] = self.key_frame_flat[pixel_read : pixel_read + opcode.run]
                pixel_read += opcode.run
            else:
                raise ValueError("Unexpected opcode in key frame.")

        if frame_header.frame_type == FrameType.Key:
            self.key_pixels = pixels
            self.key_frame_flat = frame.reshape(-1, 3)

        return frame
