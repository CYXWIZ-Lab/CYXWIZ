# Server Node Tests

This directory contains all testing infrastructure for the CyxWiz Server Node P2P service.

## Quick Start

### Build All Tests
```bash
cd build
cmake --build . --config Release -t test_job_execution_service -t standalone_p2p_server -t mock_engine_client
```

### Run Unit Tests
```bash
./build/bin/Release/test_job_execution_service.exe
```

### Manual Testing (Two Terminals)

**Terminal 1 - Start Server:**
```bash
./build/bin/Release/standalone_p2p_server.exe
```

**Terminal 2 - Run Client:**
```bash
./build/bin/Release/mock_engine_client.exe localhost:50052 my_job
```

## Files

### Test Implementation
- `test_job_execution_service.cpp` - Catch2 unit tests (463 lines)
- `standalone_p2p_server.cpp` - Manual test server (77 lines)
- `mock_engine_client.cpp` - Engine simulator (306 lines)

### Documentation
- `P2P_TESTING_GUIDE.md` - **Complete testing documentation** (read this first!)
- `README.md` - This file (quick reference)

## Test Executables

All executables are built to `build/bin/Release/`:

| Executable | Size | Purpose |
|------------|------|---------|
| `test_job_execution_service.exe` | 6.6 MB | Automated Catch2 tests |
| `standalone_p2p_server.exe` | 5.9 MB | Interactive P2P server |
| `mock_engine_client.exe` | 5.5 MB | Engine simulator |

## What Gets Tested

✅ Connection authentication
✅ Job submission & validation
✅ Real-time training metrics streaming
✅ Interactive control (pause/resume/stop)
✅ Checkpoint management
✅ Model weights download (chunked transfer)
✅ Concurrent job handling

## Need Help?

📖 **Full Documentation**: See [P2P_TESTING_GUIDE.md](./P2P_TESTING_GUIDE.md)

🐛 **Troubleshooting**: Check the Troubleshooting section in the guide

🔧 **Build Issues**: Verify vcpkg dependencies are installed

## Test Coverage

- **9 test suites** with 80+ assertions
- **~90% P2P RPC coverage**
- **Full workflow validation**: Connect → Submit → Train → Download

## Next Phase

After successful P2P testing, proceed to:
- **Phase 3**: Engine P2P Client Implementation
- **Phase 4**: Central Server Integration (JWT tokens, node assignment)
- **Phase 5**: End-to-End Testing

See `developers_docs/p2p_workflow/P2P_WORKFLOW_DESIGN.md` for the complete workflow design.
