## 0.2.0 (2025-06-16)

### Feat

- added another opcode for comparing with previous frames
- initial encoding cli command
- implemented predicted frame encoding and decoding
- simplify encoder keyframe selection and add predicted frames (broken)
- add new frame diff opcode
- added initial decoder
- added run opcode + tests
- support and test DiffOpcode including reading back
- support index opcode from QOI
- Initial commit

### Fix

- address pre-commit issue seen only when shellcheck is installed
- Added missing import for tests

### Refactor

- renaming from pyqov to pyqoiv

### Perf

- reworked encode and decode for perf, compariable compression to ffv1 using zstd
- made the encoder slightly faster

## v0.3.0 (2025-06-16)

### Feat

- added another opcode for comparing with previous frames
- initial encoding cli command
- implemented predicted frame encoding and decoding
- simplify encoder keyframe selection and add predicted frames (broken)
- add new frame diff opcode
- added initial decoder
- added run opcode + tests
- support and test DiffOpcode including reading back
- support index opcode from QOI
- Initial commit

### Fix

- update typo in release process
- address pre-commit issue seen only when shellcheck is installed
- Added missing import for tests

### Refactor

- renaming from pyqov to pyqoiv

### Perf

- reworked encode and decode for perf, compariable compression to ffv1 using zstd
- made the encoder slightly faster
