from io import BytesIO
from pyqoiv.decode import Decoder
import numpy as np
from pyqoiv.types import QovHeader, QovFrameHeader, FrameType, PixelHashMap
from pyqoiv.opcodes import (
    DiffOpcode,
    DiffFrameOpcode,
    RunOpcode,
    RgbOpcode,
    IndexOpcode,
)
from pyqoiv.encode import EncodedFrame


def test_decoder_decodes_flat_frame_as_expected():
    file = BytesIO()
    QovHeader(width=10, height=10).write(file)
    EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Key),
        opcodes=[RgbOpcode(1, 1, 1)] * 100,
    ).write(file)
    file.seek(0)
    decoder = Decoder(file)
    assert decoder.header.width == 10
    assert decoder.header.height == 10
    frame = next(decoder)
    assert np.array_equal(
        np.ones((decoder.header.height, decoder.header.width, 3), dtype=np.uint8),
        frame,
    )


def test_decoder_decodes_diff_frame_as_expected():
    file = BytesIO()
    QovHeader(width=3, height=1).write(file)
    EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Key),
        opcodes=[RgbOpcode(1, 1, 1), DiffOpcode(1, 1, 1), DiffOpcode(-1, 0, 1)],
    ).write(file)
    file.seek(0)
    decoder = Decoder(file)
    assert decoder.header.width == 3
    assert decoder.header.height == 1
    frame = next(decoder)
    assert np.array_equal(
        np.array([[[1, 1, 1], [2, 2, 2], [1, 2, 3]]]),
        frame,
    )


def test_decoder_decodes_index_frame_as_expected():
    file = BytesIO()
    pixels = PixelHashMap()
    QovHeader(width=4, height=1).write(file)
    EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Key),
        opcodes=[
            RgbOpcode(1, 1, 1),
            RgbOpcode(2, 2, 2),
            IndexOpcode(pixels.index_of(1, 1, 1)),
            IndexOpcode(pixels.index_of(2, 2, 2)),
        ],
    ).write(file)
    file.seek(0)
    decoder = Decoder(file)
    assert decoder.header.width == 4
    assert decoder.header.height == 1
    frame = next(decoder)
    assert np.array_equal(
        np.array([[[1, 1, 1], [2, 2, 2], [1, 1, 1], [2, 2, 2]]]),
        frame,
    )


def test_decoder_decodes_run_frame_as_expected():
    file = BytesIO()
    QovHeader(width=4, height=1).write(file)
    EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Key),
        opcodes=[
            RgbOpcode(1, 1, 1),
            RunOpcode(run=3),
        ],
    ).write(file)
    file.seek(0)
    decoder = Decoder(file)
    assert decoder.header.width == 4
    assert decoder.header.height == 1
    frame = next(decoder)
    assert np.array_equal(
        np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]]),
        frame,
    )


def test_decoder_decodes_diff_frame_frame_as_expected():
    file = BytesIO()
    QovHeader(width=4, height=1).write(file)
    EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Key),
        opcodes=[
            RgbOpcode(1, 1, 1),
            RgbOpcode(2, 2, 2),
            RgbOpcode(3, 3, 3),
            RgbOpcode(4, 4, 4),
        ],
    ).write(file)
    pixels = PixelHashMap()
    EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Predicted),
        opcodes=[
            DiffFrameOpcode(
                key_frame=True,
                use_index=True,
                index=pixels.index_of(1, 1, 1),
                dr=0,
                db=0,
                dg=0,
            ),
            DiffFrameOpcode(
                key_frame=True,
                use_index=True,
                index=pixels.index_of(2, 2, 2),
                dr=0,
                db=0,
                dg=0,
            ),
            DiffFrameOpcode(
                key_frame=True,
                use_index=True,
                index=pixels.index_of(3, 3, 3),
                dr=0,
                db=0,
                dg=0,
            ),
            DiffFrameOpcode(
                key_frame=True,
                use_index=True,
                index=pixels.index_of(3, 3, 3),
                dr=0,
                db=0,
                dg=0,
            ),
        ],
    ).write(file)
    EncodedFrame(
        header=QovFrameHeader(frame_type=FrameType.Predicted),
        opcodes=[
            DiffFrameOpcode(
                key_frame=True,
                use_index=True,
                index=pixels.index_of(4, 4, 4),
                dr=0,
                db=0,
                dg=0,
            ),
            DiffFrameOpcode(
                key_frame=True,
                use_index=True,
                index=pixels.index_of(2, 2, 2),
                dr=0,
                db=0,
                dg=0,
            ),
            DiffFrameOpcode(
                key_frame=True,
                use_index=True,
                index=pixels.index_of(3, 3, 3),
                dr=0,
                db=0,
                dg=0,
            ),
            DiffFrameOpcode(
                key_frame=True,
                use_index=False,
                diff=0,
                dr=0,
                db=0,
                dg=0,
            ),
        ],
    ).write(file)
    file.seek(0)
    decoder = Decoder(file)
    assert decoder.header.width == 4
    assert decoder.header.height == 1
    next(decoder)
    frame = next(decoder)
    assert np.array_equal(
        np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 3]]]),
        frame,
    )
    frame = next(decoder)
    assert np.array_equal(
        np.array([[[4, 4, 4], [2, 2, 2], [3, 3, 3], [4, 4, 4]]]),
        frame,
    )
