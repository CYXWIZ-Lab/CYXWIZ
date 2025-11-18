# CUDA Integration - Quick Start

## TL;DR - 3 Steps

### 1️⃣ Install CUDA Toolkit
```
URL: https://developer.nvidia.com/cuda-downloads
Select: Windows → x86_64 → 10/11 → exe (network)
Choose: Express installation
Time: ~25 minutes
```

### 2️⃣ Enable CUDA in CyxWiz
```powershell
.\enable_cuda.bat
```

### 3️⃣ Test It
```powershell
# Terminal 1
cd cyxwiz-central-server && cargo run --release

# Terminal 2
cd build\windows-release\bin\Release && .\cyxwiz-server-node.exe

# Look for: "CUDA device 0: X.X GB total, Y.Y GB free"
```

---

## What This Does

**Before:**
```
Using estimated memory for GTX 1050 Ti: 4 GB  ← Guess
```

**After:**
```
CUDA device 0: 4.00 GB total, 3.85 GB free    ← Real-time accurate
```

---

## Files Created

| File | Purpose |
|------|---------|
| `CUDA_SETUP_GUIDE.md` | 📖 Complete guide with troubleshooting |
| `CUDA_INTEGRATION_SUMMARY.md` | 📊 Technical details and rationale |
| `CUDA_QUICKSTART.md` | ⚡ This file - quick reference |
| `enable_cuda.bat` | 🔧 Automated build script |
| `verify_cuda.bat` | ✅ Verification script |

---

## Verification Checklist

After running `enable_cuda.bat`, check:

- [ ] `nvcc --version` shows CUDA version
- [ ] Build completes without errors
- [ ] Server Node logs show "CUDA device X: ..." (not "estimated")
- [ ] Memory values change as you use GPU

---

## Troubleshooting

**Q: "nvcc not found"**
A: Open a NEW terminal after CUDA installation

**Q: "CMake can't find CUDA"**
A: Run with explicit path:
```powershell
cmake -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x" ...
```

**Q: "Still shows estimated memory"**
A: Run `verify_cuda.bat` to diagnose

---

## Why Do This?

✅ **Accurate job scheduling** - Server knows real available VRAM
✅ **Prevent OOM errors** - Don't assign jobs when GPU is full
✅ **Better debugging** - See actual memory usage
✅ **Native NVIDIA support** - Use CUDA backend instead of OpenCL

---

## Current Status

✅ Code ready (device.cpp has CUDA implementation)
✅ Build system ready (CMakeLists.txt updated)
✅ Scripts ready (enable_cuda.bat created)
⏳ **Waiting: CUDA Toolkit installation**

---

## Next Steps

After CUDA is working:

**Option A:** Continue to Phase 5 (Job Execution)
**Option B:** Test multi-node distributed training
**Option C:** Integrate plotting system (ImPlot)

See `PHASE5_PLAN.md` for details.

---

**Time Investment:**
- CUDA installation: 25 min (one-time)
- Rebuild with CUDA: 5 min
- **Total: 30 minutes for accurate GPU memory reporting**

---

**Questions?** See `CUDA_SETUP_GUIDE.md` for detailed instructions.
