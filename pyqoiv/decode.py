from io import BytesIO
from numpy.typing import NDArray
import numpy as np
from .types import QovHeader, QovFrameHeader, FrameType, PixelHashMap
from .opcodes import RgbOpcode, DiffOpcode, IndexOpcode, RunOpcode


class Decoder:
    """Decode a QOIV file into frames."""

    def __init__(self, file: BytesIO):
        """Construct a new decoder."""
        self.file = file
        self.header = QovHeader.read(file)
        self.first_frame_pos = file.tell()
        self.pixel_count = self.header.width * self.header.height
        self.pixels = PixelHashMap()

    def __iter__(self) -> "Decoder":
        """Setup the iterator"""
        self.frame_pos = self.first_frame_pos
        self.file.seek(self.frame_pos)
        self.pixels = PixelHashMap()
        return self

    def __next__(self):
        """Get the next frame."""
        return self.read_frame()

    def read_frame(self) -> NDArray[np.uint8]:
        """Read the next frame from the file."""
        frame_header = QovFrameHeader.read(self.file)

        frame = np.zeros((self.header.height, self.header.width, 3), dtype=np.uint8)

        if frame_header.frame_type == FrameType.Key:
            pixel_read = 0
            self.pixels.clear()
            while pixel_read < self.pixel_count:
                cy = pixel_read // self.header.width
                cx = pixel_read % self.header.width
                if RgbOpcode.is_next(self.file):
                    opcode = RgbOpcode.read(self.file)
                    frame[
                        cy,
                        cx,
                    ] = [opcode.r, opcode.g, opcode.b]
                    self.pixels.push(frame[cy, cx])
                    pixel_read += 1
                elif DiffOpcode.is_next(self.file):
                    opcode = DiffOpcode.read(self.file)
                    frame[cy, cx] = frame[
                        (pixel_read - 1) // self.header.width,
                        (pixel_read - 1) % self.header.width,
                    ] + np.array([opcode.dr, opcode.dg, opcode.db])
                    self.pixels.push(frame[cy, cx])
                    pixel_read += 1
                elif RunOpcode.is_next(self.file):
                    opcode = RunOpcode.read(self.file)
                    last_pixel = frame[
                        (pixel_read - 1) // self.header.width,
                        (pixel_read - 1) % self.header.width,
                    ]
                    for i in range(opcode.run):
                        frame[
                            (pixel_read + i) // self.header.width,
                            (pixel_read + i) % self.header.width,
                        ] = last_pixel
                    pixel_read += opcode.run
                elif IndexOpcode.is_next(self.file):
                    index_opcode = IndexOpcode.read(self.file)
                    frame[cy, cx] = self.pixels[index_opcode.index]
                    pixel_read += 1
                else:
                    raise ValueError("Unexpected opcode in key frame.")

        else:
            raise NotImplementedError("Frame reading not implemented yet.")
        return frame
