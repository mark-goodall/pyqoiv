import pytest
from pyqoiv.types import QovHeader, PixelHashMap, QovFrameHeader, FrameType
from io import BytesIO
import numpy as np


def test_header():
    h = QovHeader()
    assert h.magic == "qoiv"
    file = BytesIO()
    h.write(file)
    file.seek(0)
    h2 = QovHeader.read(file)
    assert h2.magic == h.magic
    assert h2.width == h.width
    assert h2.height == h.height
    assert h2.colourspace == h.colourspace


def test_pixel_hash_map():
    m = PixelHashMap()
    # 61
    red = np.array([255, 0, 0], dtype=np.uint8)
    green = np.array([0, 255, 0], dtype=np.uint8)
    blue = np.array([0, 0, 255], dtype=np.uint8)
    clash_red = np.array([17, 2, 0], dtype=np.uint8)

    assert red not in m
    assert green not in m
    assert blue not in m
    assert clash_red not in m

    m.push(red)
    assert red in m
    assert green not in m
    assert blue not in m
    assert clash_red not in m

    m.push(green)
    assert red in m
    assert green in m
    assert blue not in m
    assert clash_red not in m

    m.push(blue)
    assert red in m
    assert green in m
    assert blue in m
    assert clash_red not in m

    assert np.array_equal(m[61], red)
    m.push(clash_red)
    assert red not in m
    assert green in m
    assert blue in m
    assert clash_red in m
    assert np.array_equal(m[61], clash_red)

    m.clear()
    assert red not in m
    assert green not in m
    assert blue not in m
    assert clash_red not in m
    assert not np.array_equal(m[61], clash_red)


def test_frame_header():
    h = QovFrameHeader(FrameType.Key)
    file = BytesIO()
    h.write(file)
    file.seek(0)
    h2 = QovFrameHeader.read(file)
    assert h2.frame_type == h.frame_type

    with pytest.raises(ValueError):
        h = QovFrameHeader(10)  # type:ignore
        file = BytesIO()
        h.write(file)
        file.seek(0)
        h2 = QovFrameHeader.read(file)
