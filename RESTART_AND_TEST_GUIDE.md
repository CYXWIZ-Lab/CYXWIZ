# Restart and GPU Testing Guide

**Purpose:** Resume Phase 6 Session 4 after system restart to test ArrayFire GPU acceleration

**Status:** ⏸️ Paused at GPU integration (DLL lock issue)

---

## 🔄 Step 1: Restart Your Computer

**Why?** Windows has locked `cyxwiz-backend.dll` and won't release it

**Action:**
1. Save all work
2. Restart Windows
3. Return to this guide

---

## 🛠️ Step 2: Rebuild Backend (After Restart)

**Location:** `D:\Dev\CyxWiz_Claude`

### 2.1 Open Terminal

```bash
cd D:\Dev\CyxWiz_Claude
```

### 2.2 Verify Git Status

```bash
git status
git log --oneline -3
```

**Expected:** You should see commit `38d9fa9` (ArrayFire GPU Integration)

### 2.3 Rebuild Backend

```bash
cmake --build build/windows-release --target cyxwiz-backend --config Release
```

**Expected Output:**
```
cyxwiz-backend.vcxproj -> D:\Dev\CyxWiz_Claude\build\windows-release\bin\Release\cyxwiz-backend.dll
```

**If This Fails:**
- Check that no Python scripts are running
- Close CyxWiz Engine if open
- Try again

### 2.4 Rebuild pycyxwiz

```bash
cmake --build build/windows-release --target pycyxwiz --config Release
```

**Expected Output:**
```
pycyxwiz.vcxproj -> D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release\pycyxwiz.cp314-win_amd64.pyd
```

### 2.5 Copy Updated DLL

```bash
cp build/windows-release/bin/Release/cyxwiz-backend.dll build/windows-release/lib/Release/
```

**Verification:**
```bash
ls -lh build/windows-release/lib/Release/*.dll | head -5
```

You should see the updated `cyxwiz-backend.dll` with current timestamp.

---

## ✅ Step 3: Test Correctness

### 3.1 Run Existing Tests

```bash
python test_arithmetic_verify.py
```

**Expected:** All 5 tests should PASS ✅

**What This Tests:**
- Addition, subtraction, multiplication, division
- Factory methods (zeros, ones)
- GPU implementation produces same results as CPU

### 3.2 Run NumPy Conversion Tests

```bash
python test_numpy_conversion.py
```

**Expected:** All 6 tests should PASS ✅

**If Tests Fail:**
- GPU computation might have issues
- Check for error messages in output
- Fallback to CPU should still work

---

## 🚀 Step 4: Benchmark GPU Performance

### 4.1 Create Benchmark Script

Save this as `benchmark_gpu.py`:

```python
#!/usr/bin/env python3
"""
GPU Performance Benchmark - CyxWiz vs NumPy
"""
import sys
import os
import time
import numpy as np

# Setup PATH
dll_dirs = [
    r"D:\Dev\CyxWiz_Claude\build\windows-release\bin\Release",
    r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release",
]
for dll_dir in dll_dirs:
    if dll_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = dll_dir + os.pathsep + os.environ.get('PATH', '')

build_dir = r"D:\Dev\CyxWiz_Claude\build\windows-release\lib\Release"
sys.path.insert(0, build_dir)

import pycyxwiz as cx

print("=" * 70)
print(" GPU Performance Benchmark: CyxWiz vs NumPy")
print("=" * 70)
print()

cx.initialize()
print(f"CyxWiz Version: {cx.get_version()}")
print(f"ArrayFire: {cx.get_version()}")  # Should show AF info if GPU active
print()

# Test different tensor sizes
sizes = [100, 1000, 10000, 100000, 1000000]
iterations = 100

print(f"Testing {iterations} iterations of element-wise addition")
print()
print(f"{'Size':<12} {'NumPy (ms)':<15} {'CyxWiz (ms)':<15} {'Speedup':<10}")
print("-" * 70)

for size in sizes:
    # Create test data
    a_np = np.ones(size, dtype=np.float32)
    b_np = np.ones(size, dtype=np.float32)

    # NumPy baseline
    start = time.perf_counter()
    for _ in range(iterations):
        c_np = a_np + b_np
    numpy_time = (time.perf_counter() - start) * 1000  # ms

    # CyxWiz (GPU)
    a_cx = cx.Tensor.from_numpy(a_np)
    b_cx = cx.Tensor.from_numpy(b_np)

    start = time.perf_counter()
    for _ in range(iterations):
        c_cx = a_cx + b_cx
    cyxwiz_time = (time.perf_counter() - start) * 1000  # ms

    # Calculate speedup
    speedup = numpy_time / cyxwiz_time if cyxwiz_time > 0 else 0
    speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"0.{int(1/speedup)}x slower"

    print(f"{size:<12} {numpy_time:<15.2f} {cyxwiz_time:<15.2f} {speedup_str:<10}")

    # Verify correctness
    c_result = c_cx.to_numpy()
    if not np.allclose(c_np, c_result):
        print(f"  WARNING: Results don't match for size {size}!")

print()
print("=" * 70)
print(" Benchmark Complete")
print("=" * 70)

cx.shutdown()
```

### 4.2 Run Benchmark

```bash
python benchmark_gpu.py
```

**Expected Results:**

**If GPU Acceleration is Working:**
```
Size         NumPy (ms)      CyxWiz (ms)     Speedup
----------------------------------------------------------------------
100          0.05            0.50            0.1x slower  (overhead)
1000         0.50            0.60            0.8x slower  (breakeven)
10000        5.00            1.00            5.00x        (GPU wins!)
100000       50.00           2.00            25.00x       (GPU wins!)
1000000      500.00          10.00           50.00x       (GPU wins!)
```

**If GPU Acceleration Failed (CPU fallback):**
```
Size         NumPy (ms)      CyxWiz (ms)     Speedup
----------------------------------------------------------------------
100          0.05            0.10            0.5x slower
1000         0.50            1.00            0.5x slower
10000        5.00            10.00           0.5x slower
100000       50.00           100.00          0.5x slower
1000000      500.00          1000.00         0.5x slower
```

### 4.3 Check ArrayFire Logs

Look for these messages in the output:

**GPU Active:**
```
[info] ArrayFire initialized successfully
[info] OpenCL backend available
```

**GPU Failed:**
```
[warning] ArrayFire operation failed, falling back to CPU: <error message>
```

---

## 📊 Step 5: Analyze Results

### 5.1 Interpreting Performance

**Scenario 1: GPU is 10-100x faster for large tensors** ✅
- **Status:** Success! GPU acceleration working
- **Next:** Implement GPU for other operators (-, *, /)

**Scenario 2: GPU is slower or same speed** ⚠️
- **Status:** GPU overhead too high or not used
- **Check:** Are ArrayFire logs showing GPU usage?
- **Possible causes:**
  - GPU not being used (check logs)
  - Overhead of CPU↔GPU copies dominates
  - ArrayFire using CPU backend

**Scenario 3: Results don't match NumPy** ❌
- **Status:** GPU computation has bugs
- **Action:** Check ArrayFire implementation
- **Fallback:** Should use CPU version (safe)

### 5.2 GPU Utilization Check

**On Windows:**
```bash
# Open Task Manager
# Go to Performance → GPU
# Should see GPU usage spike during benchmark
```

**Alternative:**
- Install GPU-Z or similar tool
- Monitor GPU usage during benchmark
- Should see ~50-100% GPU utilization

---

## 🔧 Step 6: If GPU Not Working

### Troubleshooting

#### Issue 1: No GPU Detected

**Check:**
```bash
cd build/windows-release/bin/Release
./cyxwiz-engine
# Look for "ArrayFire initialized" message
```

**Expected:**
```
[info] ArrayFire initialized successfully
[info] OpenCL backend available
[info] CUDA backend available  (if NVIDIA GPU)
```

**Fix:**
- Ensure ArrayFire is installed
- Check that GPU drivers are up-to-date
- Verify CUDA toolkit installed (for NVIDIA)

#### Issue 2: ArrayFire Exception

**Symptoms:** Benchmark shows CPU performance (slow)

**Check logs for:**
```
[warning] ArrayFire operation failed, falling back to CPU: <message>
```

**Common Causes:**
- Out of GPU memory
- Unsupported operation
- ArrayFire backend not initialized

**Fix:**
- Check available GPU memory
- Reduce tensor size
- Verify ArrayFire installation

#### Issue 3: Build Issues

**If rebuild fails after restart:**

```bash
# Clean build
cmake --build build/windows-release --target clean --config Release

# Full rebuild
cmake --build build/windows-release --config Release
```

---

## 📝 Step 7: Document Results

### 7.1 Create Performance Report

Save your benchmark results to a file:

```bash
python benchmark_gpu.py > GPU_PERFORMANCE_RESULTS.txt
```

### 7.2 Take Screenshots

- Task Manager GPU usage during benchmark
- Terminal output showing speedup numbers
- Any error messages if GPU failed

### 7.3 Update Session 4 Progress

Create `developer_docs/phase6/PHASE6_SESSION4_PROGRESS.md` with:
- Benchmark results
- GPU utilization stats
- Screenshots
- Any issues encountered
- Next steps

---

## 🎯 Step 8: Next Development Tasks

### If GPU Acceleration Working:

**1. Implement GPU for Remaining Operators** (30 min)
```bash
# Edit: cyxwiz-backend/src/core/tensor.cpp
# Copy pattern from operator+ to:
# - operator-
# - operator*
# - operator/
```

**2. Test All Operators** (10 min)
```bash
python test_arithmetic_verify.py
```

**3. Benchmark All Operators** (10 min)
- Test subtraction, multiplication, division
- Verify speedups

**4. Optimize (Optional)** (1-2 hours)
- Keep af::array alive between operations
- Lazy CPU sync (only on to_numpy())
- Expected: Additional 2-5x speedup

### If GPU Not Working:

**1. Debug ArrayFire Integration**
- Check logs carefully
- Verify GPU is detected
- Test simple ArrayFire example

**2. Consider CPU Optimization**
- Add SIMD vectorization
- Use OpenMP parallelization
- ~2-4x speedup possible

**3. Document Issues**
- Create bug report
- Note error messages
- Ask for help if needed

---

## 📋 Quick Reference Commands

### Rebuild Everything
```bash
cd D:\Dev\CyxWiz_Claude
cmake --build build/windows-release --target cyxwiz-backend --config Release
cmake --build build/windows-release --target pycyxwiz --config Release
cp build/windows-release/bin/Release/cyxwiz-backend.dll build/windows-release/lib/Release/
```

### Run All Tests
```bash
python test_arithmetic_verify.py
python test_numpy_conversion.py
python test_pycyxwiz.py
```

### Benchmark
```bash
python benchmark_gpu.py
```

### Check Git Status
```bash
git status
git log --oneline -5
```

---

## ⏭️ What to Tell Me When You Return

Please report:

1. **Did rebuild succeed?** (Yes/No + any errors)
2. **Did tests pass?** (All/Some/None)
3. **Benchmark results:** (Copy speedup numbers)
4. **GPU utilization:** (Task Manager stats)
5. **Any error messages:** (Copy exact text)

This will help me determine next steps!

---

## 🎉 Expected Success Outcome

After following this guide, you should have:

✅ Successfully rebuilt backend with GPU support
✅ All tests passing (correctness maintained)
✅ **10-100x speedup** for large tensors on GPU
✅ GPU utilization visible in Task Manager
✅ Ready to implement GPU for other operators

---

**Current Status:** ⏸️ Waiting for system restart

**Next Status:** 🚀 Testing GPU acceleration

**Good luck, and see you after the restart!** 🎯
