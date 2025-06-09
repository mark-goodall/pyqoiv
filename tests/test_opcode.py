from pyqoiv.opcodes import (
    DiffFrameOpcode,
    RgbOpcode,
    IndexOpcode,
    DiffOpcode,
    RunOpcode,
    FrameRunOpcode,
)
from io import BytesIO
import pytest


def test_rgb_opcode():
    rgb = RgbOpcode(255, 128, 64)
    assert len(rgb) == 4
    file = BytesIO()
    rgb.write(file)
    file.seek(0)
    assert RgbOpcode.is_next(file)
    assert not IndexOpcode.is_next(file)
    assert not DiffOpcode.is_next(file)
    new_rgb = RgbOpcode.read(file)
    assert rgb == new_rgb


def test_rgb_opcode_invalid():
    b = b"\xff\x80\x40"
    assert not RgbOpcode.is_next(BytesIO(b))
    with pytest.raises(ValueError):
        RgbOpcode.read(BytesIO(b))
    b = b"\xfe\x80\x40"
    with pytest.raises(ValueError):
        RgbOpcode.read(BytesIO(b))


def test_index_opcode():
    index = IndexOpcode(42)
    assert len(index) == 1
    file = BytesIO()
    index.write(file)
    file.seek(0)
    assert IndexOpcode.is_next(file)
    assert not DiffOpcode.is_next(file)
    assert not RgbOpcode.is_next(file)
    new_index = IndexOpcode.read(file)
    assert index == new_index


def test_index_opcode_invalid():
    with pytest.raises(ValueError):
        index = IndexOpcode(201)
        index.write(BytesIO())
    with pytest.raises(ValueError):
        index = IndexOpcode(-201)
        index.write(BytesIO())

    b = b"\xff\x80\x40"
    assert not IndexOpcode.is_next(BytesIO(b))
    with pytest.raises(ValueError):
        IndexOpcode.read(BytesIO(b))


def test_diff_opcode():
    diff = DiffOpcode(-1, 0, 1)
    assert len(diff) == 1
    file = BytesIO()
    diff.write(file)
    file.seek(0)
    assert DiffOpcode.is_next(file)
    assert not IndexOpcode.is_next(file)
    assert not RgbOpcode.is_next(file)
    new_diff = DiffOpcode.read(file)
    assert diff == new_diff


def test_diff_opcode_invalid():
    with pytest.raises(ValueError):
        diff = DiffOpcode(2, -4, 9)
        diff.write(BytesIO())
    b = b"\xff\x80\x40"
    assert not DiffOpcode.is_next(BytesIO(b))
    with pytest.raises(ValueError):
        DiffOpcode.read(BytesIO(b))


def test_run_opcode():
    run = RunOpcode(42)
    assert len(run) == 1
    file = BytesIO()
    run.write(file)
    file.seek(0)
    assert RunOpcode.is_next(file)
    new_run = RunOpcode.read(file)
    assert run == new_run


def test_run_opcode_invalid():
    file = BytesIO()
    with pytest.raises(ValueError):
        run = RunOpcode(63)
        run.write(file)
    with pytest.raises(ValueError):
        run = RunOpcode(0)
        run.write(file)
    b = b"\x0f\x80\x40"
    with pytest.raises(ValueError):
        RunOpcode.read(BytesIO(b))


def test_diff_frame_opcode():
    diff_frame = DiffFrameOpcode(
        key_frame=True, use_index=False, index=10, dr=-1, dg=0, db=1
    )
    assert len(diff_frame) == 2
    file = BytesIO()
    diff_frame.write(file)
    file.seek(0)
    assert DiffFrameOpcode.is_next(file)
    new_diff_frame = DiffFrameOpcode.read(file)
    assert diff_frame == new_diff_frame


def test_diff_frame_opcode_invalid():
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, index=64, dr=-1, dg=0, db=1)
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, index=-64, dr=-1, dg=0, db=1)
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, index=2, dr=-3, dg=0, db=1)
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, index=2, dr=0, dg=-3, db=1)
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, index=2, dr=0, dg=0, db=-3)
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, index=2, dr=3, dg=0, db=1)
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, index=2, dr=0, dg=3, db=1)
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, index=2, dr=0, dg=0, db=3)
    with pytest.raises(ValueError):
        DiffFrameOpcode(key_frame=True, use_index=False, dr=0, dg=0, db=3)
    b = b"\xc0\x40"
    assert not DiffFrameOpcode.is_next(BytesIO(b))
    with pytest.raises(ValueError):
        DiffFrameOpcode.read(BytesIO(b))


def test_frame_run_opcode():
    frame_run = FrameRunOpcode(is_keyframe=True, run=23)
    assert len(frame_run) == 2
    file = BytesIO()
    frame_run.write(file)
    file.seek(0)
    assert FrameRunOpcode.is_next(file)
    new_frame_run = FrameRunOpcode.read(file)
    assert frame_run == new_frame_run


def test_frame_run_opcode_invalid():
    with pytest.raises(ValueError):
        frame_run = FrameRunOpcode(is_keyframe=True, run=623)
        frame_run.write(BytesIO())

    b = b"\xc0\x40"
    assert not FrameRunOpcode.is_next(BytesIO(b))
    with pytest.raises(ValueError):
        FrameRunOpcode.read(BytesIO(b))


def test_comparisions():
    opcodes = [
        RgbOpcode(255, 128, 64),
        DiffOpcode(-1, 0, 1),
        IndexOpcode(42),
        RunOpcode(42),
        DiffFrameOpcode(True, False, -1, 0, 1, index=10),
        FrameRunOpcode(True, 23),
    ]

    for a in opcodes:
        for b in opcodes:
            file = BytesIO()
            a.write(file)
            file.seek(0)
            assert a.is_next(file)
            if a == b:
                assert b.is_next(file)
            else:
                assert not b.is_next(file), f"Expected {a} and {b} to not be equal"
