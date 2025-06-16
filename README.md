# pyqoiv

A python implementation of the QOV video format, based on the QOI image format.
Contains a number of opcode modifications to allow for inter frame comparisons.
When post processed with zstd to further compress, the resulting file is
comparable to FFV1 in terms of compression ratio. It is not as good as H265
lossless.

The implementation is horribly slow, but it works.
