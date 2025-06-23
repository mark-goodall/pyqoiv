# pyqoiv

[![Test Code](https://github.com/mark-goodall/pyqoiv/actions/workflows/test.yml/badge.svg)](https://github.com/mark-goodall/pyqoiv/actions/workflows/test.yml)
[![Docs](https://app.readthedocs.org/projects/pyqoiv/badge/?version=latest)](https://pyqoiv.readthedocs.io/en/latest/)

A python implementation of the QOV video format, based on the [QOI image format](https://qoiformat.org/).
Contains a number of opcode modifications to allow for inter frame comparisons.
When post processed with zstd to further compress, the resulting file is
comparable to FFV1 in terms of compression ratio. It is not as good as H265
lossless.

The implementation is horribly slow, but it works.

## Opcodes

The [QOI format specification](https://qoiformat.org/qoi-specification.pdf) lists the opcodes supported by the QOI format.
The QOIV format supports most of these opcodes, with the exception of:

- QOI_OP_RGBA
- QOI_OP_LUMA

To replace these opcodes, the following 2 byte opcodes are used (described in
more detail in the code):

- FrameRunOpcode - This is used to represent a run of identical pixels from a
  reference frame.
- DiffFrameOpcode - This is used to represent a single pixel from a reference
  frame or hashmap, with some applied difference.
