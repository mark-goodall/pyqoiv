from .types import ColourSpace, QovHeader, PixelHashMap, QovFrameHeader, FrameType
from .opcodes import (
    Opcode,
    RgbOpcode,
    IndexOpcode,
    DiffOpcode,
    RunOpcode,
    DiffFrameOpcode,
    FrameRunOpcode,
)
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
        file: BufferedIOBase,
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
        self.total_frames = 0
        self.header.write(file)
        self.pixels = PixelHashMap()

    def __repr__(self) -> str:
        """String representation of the encoder."""
        return f"Encoder(width={self.header.width}, height={self.header.height}, colourspace={self.header.colourspace}, keyframe_interval={self.keyframe_interval}, total_frames={self.total_frames})"

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

    @staticmethod
    def pixel_equal(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> bool:
        """Check if two pixels are equal."""
        return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]

    @staticmethod
    def pixels_equal(pixels: NDArray[np.uint8]) -> int:
        equal_rows = np.all(pixels[1:] == pixels[0], axis=1)
        for i in range(len(equal_rows)):
            if not equal_rows[i]:
                return i
        return len(pixels) - 1

    def _encode_frame(
        self,
        frame: NDArray[np.uint8],
        pixels: PixelHashMap,
        key_frame_flat: Optional[NDArray[np.uint8]],
        key_pixels: Optional[PixelHashMap],
    ) -> EncodedFrame:
        """Encode a single frame."""
        opcodes: List[Opcode] = []

        last_pixel: Optional[NDArray[np.uint8]] = None
        last_pixel_count: int = 0
        is_kf_flat = key_frame_flat is not None
        is_kf_pixels = key_pixels is not None
        frame_flat = frame.reshape(-1, 3, copy=False)
        pixel_pos = 0
        pixel_len = len(frame_flat)
        while pixel_pos < pixel_len:
            pixel = frame_flat[pixel_pos]
            if last_pixel is not None:
                # Handle runs
                count = Encoder.pixels_equal(frame_flat[pixel_pos - 1 : pixel_pos + 62])
                if count > 0:
                    opcodes.append(RunOpcode(run=count))
                    pixel_pos += count
                    continue

                dr, dg, db = pixel - last_pixel
                if -2 <= dr < 2 and -2 <= dg < 2 and -2 <= db < 2:
                    opcodes.append(DiffOpcode(dr, dg, db))
                    pixels.push(pixel)
                    last_pixel = pixel
                    pixel_pos += 1
                    continue

            if pixel in pixels:
                opcodes.append(
                    IndexOpcode(index=pixels.index_of(pixel[0], pixel[1], pixel[2]))
                )
                pixels.push(pixel)
                last_pixel = pixel
                pixel_pos += 1
                continue

            if is_kf_flat:
                count = Encoder.get_pixels_equal(
                    frame_flat[pixel_pos : pixel_pos + 128],
                    key_frame_flat[pixel_pos : pixel_pos + 128],
                )
                match count:
                    case 0:
                        pass
                    case 1:
                        opcodes.append(DiffFrameOpcode(True, False, 0, 0, 0, diff=0))
                        pixels.push(pixel)
                        last_pixel = pixel
                        pixel_pos += 1
                        continue
                    case _:
                        m = min(128, count)
                        opcodes.append(FrameRunOpcode(is_keyframe=True, run=m))
                        pixel_pos += m
                        last_pixel = frame_flat[pixel_pos - 1]
                        continue

            if is_kf_pixels and pixel in key_pixels:
                key_index = key_pixels.index_of(pixel[0], pixel[1], pixel[2])
                opcodes.append(DiffFrameOpcode(True, True, 0, 0, 0, index=key_index))
                pixels.push(pixel)
                last_pixel = pixel
                pixel_pos += 1
                continue

            # Fallback to RGB opcode
            pixels.push(pixel)
            last_pixel = pixel
            pixel_pos += 1
            opcodes.append(RgbOpcode(r=pixel[0], g=pixel[1], b=pixel[2]))

        if last_pixel_count > 0:
            opcodes.append(RunOpcode(run=last_pixel_count))

        frame_type = FrameType.Key if key_pixels is None else FrameType.Predicted
        return EncodedFrame(
            header=QovFrameHeader(
                frame_type=frame_type,
            ),
            opcodes=opcodes,
        )

    @staticmethod
    def get_pixels_equal(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> int:
        """Return the number of pixels that are equal in two arrays."""
        equal_rows = np.all(a == b, axis=1)
        return (
            int(np.argmax(~equal_rows)) if not np.all(equal_rows) else len(equal_rows)
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
                frame, PixelHashMap(), self.key_frame_flat, self.pixels
            )
            encoded.write(self.file)
            self.frames_since_last_keyframe += 1

        self.total_frames += 1

    def flush(self) -> None:
        """Flush the encoder to the file."""
        self.file.flush()
