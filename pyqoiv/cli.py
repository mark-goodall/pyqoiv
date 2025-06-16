import time
from numpy.typing import NDArray
import typer
from pathlib import Path
from pyqoiv.encode import Encoder
from pyqoiv.decode import Decoder
from pyqoiv.types import ColourSpace
import ffmpeg
import numpy as np
import tqdm as tqdm
from typing import Generator
import json

app = typer.Typer()


@app.command()
def encode(input_file: Path, output_file: Path) -> None:
    """Encode a qoiv formatted file from any video file ffmpeg supports."""
    probe = ffmpeg.probe(str(input_file))
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    if video_stream is None:
        raise typer.BadParameter("Input file does not contain a video stream.")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    frame_rate = int(video_stream["avg_frame_rate"].split("/")[0])
    duration = float(probe["format"]["duration"])
    approx_frames = int(frame_rate * duration)

    encoder = Encoder(
        output_file.open("wb"), width, height, ColourSpace.Linear, keyframe_interval=20
    )

    out = (
        ffmpeg.input(str(input_file))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, quiet=True)
    )

    def read_frames() -> Generator[NDArray[np.uint8], None, None]:
        while True:
            in_bytes = out.stdout.read(width * height * 3)
            if not in_bytes:
                break
            in_frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
            yield in_frame

    for frame in tqdm.tqdm(read_frames(), total=approx_frames, desc="Encoding"):
        encoder.push(frame)
    out.stdout.close()

    pass


@app.command()
def decode(input_file: Path, output_file: Path) -> None:
    """Decode qoiv formatted file into a ffv1 encoded video file."""

    decoder = Decoder(input_file.open("rb"))
    width, height = decoder.header.width, decoder.header.height
    out = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}")
        .output(str(output_file), vcodec="ffv1")
        .run_async(pipe_stdin=True, quiet=True)
    )

    for frame, _ in tqdm.tqdm(decoder, desc="Decoding"):
        out.stdin.write(frame.reshape(-1, 3, copy=False).tobytes())

    out.stdin.close()
    out.wait()


@app.command()
def frameinfo(input_file: Path) -> None:
    """Print the information about frames and opcodes in a qoiv file."""
    decoder = Decoder(input_file.open("rb"))
    print(f"Header: {decoder.header}")

    last_pos = decoder.file.tell()
    then = time.time()
    for count, (frame, details) in enumerate(decoder):
        now = time.time()
        frame_info = {
            "frame_number": count,
            "frame_position": decoder.file.tell(),
            "frame_size": decoder.file.tell() - last_pos,
            "opcodes": details,
            "time_since_last_frame": now - then,
        }
        last_pos = decoder.file.tell()
        then = now
        print(json.dumps(frame_info, indent=2))
