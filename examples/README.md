# CyxWiz Examples

This directory contains example programs demonstrating different ways to use CyxWiz.

## C API Example

**File**: `c_api_example.c`

Demonstrates the pure C interface (`extern "C"`) to CyxWiz backend:
- Initialization and shutdown
- Device management
- Tensor creation and operations
- Optimizer usage
- Memory tracking

### Compile (after building CyxWiz):

**Windows (MSVC)**:
```cmd
cl c_api_example.c /I..\cyxwiz-backend\include ^
   /link ..\build\windows-release\lib\cyxwiz-backend.lib
```

**Windows (GCC)**:
```bash
gcc c_api_example.c -I../cyxwiz-backend/include \
    -L../build/windows-release/lib -lcyxwiz-backend -o example
```

**Linux**:
```bash
gcc c_api_example.c -I../cyxwiz-backend/include \
    -L../build/linux-release/lib -lcyxwiz-backend -o example \
    -Wl,-rpath,../build/linux-release/lib
```

**macOS**:
```bash
gcc c_api_example.c -I../cyxwiz-backend/include \
    -L../build/macos-release/lib -lcyxwiz-backend -o example \
    -Wl,-rpath,@loader_path/../build/macos-release/lib
```

### Run:
```bash
./example  # or example.exe on Windows
```

## C++ API Examples

Coming soon:
- C++ tensor operations
- Building and training a simple neural network
- Using the Python API
- Distributed training across network

## Python API Example

**File**: `python/ex1_pycyxwiz.py`

Demonstrates how to adapt the Coursera Machine Learning ex1 linear regression walkthrough
using the PyCyxWiz bindings:
- warm-up identity matrix via `pycyxwiz.linalg.eye`
- scatter plot of the original `ex1data1.txt` dataset (stored next to the script)
- computeCost and gradientDescent implemented with `pycyxwiz.linalg.matmul`
- predictions for new population values
- surface and contour plots of the cost function

### Usage

1. Build the CyxWiz backend with `CYXWIZ_BUILD_PYTHON=ON` (the example adds `build/windows-release/lib/Release` to `PATH` on Windows).
2. From the repo root run `python examples/python/ex1_pycyxwiz.py`.

The script automatically reads `examples/python/ex1data1.txt` and will show the plots before exiting.

## Integration Examples

Coming soon:
- Integrating CyxWiz with existing C projects
- Using CyxWiz from Rust (via FFI)
- Using CyxWiz from Go (via cgo)
- Android integration
