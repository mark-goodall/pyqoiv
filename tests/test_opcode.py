from pyqoiv.opcodes import RgbOpcode, IndexOpcode, DiffOpcode
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
