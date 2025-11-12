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

## Integration Examples

Coming soon:
- Integrating CyxWiz with existing C projects
- Using CyxWiz from Rust (via FFI)
- Using CyxWiz from Go (via cgo)
- Android integration
