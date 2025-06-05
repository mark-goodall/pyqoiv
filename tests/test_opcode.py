from pyqoiv.opcodes import RgbOpcode, IndexOpcode, DiffOpcode, RunOpcode
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


def test_comparisions():
    opcodes = [
        RgbOpcode(255, 128, 64),
        DiffOpcode(-1, 0, 1),
        IndexOpcode(42),
        RunOpcode(42),
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
                assert not b.is_next(file)
