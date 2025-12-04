# GPU/CUDA Issue - Troubleshooting Notes

## Problem
CUDA initialization fails with error:
```
CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment
```

## System Info
- **GPU**: NVIDIA RTX 2000 Ada (8GB VRAM)
- **Driver Version**: 580.95.05
- **System CUDA**: 13.0
- **PyTorch CUDA**: 12.4 (bundled with PyTorch 2.6.0)
- **nvidia-smi**: Works correctly, GPU detected
- **Kernel modules**: Loaded (nvidia, nvidia_uvm, nvidia_drm)

## What We Tried

### ✗ Freed GPU memory
- Killed Jupyter kernel holding 4GB VRAM
- GPU shows only 13MB usage (display server)
- **Result**: Still fails

### ✗ Reinstalled PyTorch
- Reinstalled with CUDA 12.4 from official wheels
- **Result**: Still fails

### ✗ Environment isolation
- Tested without LD_LIBRARY_PATH
- Tested in completely clean environment
- Tested with system Python
- **Result**: Fails everywhere

### ✗ Library path configuration
- System has CUDA 13.0 libraries (`libcudart.so.13`)
- PyTorch has bundled CUDA 12.4 libraries
- **Result**: Conflict not resolved

## Root Cause
Likely a **driver/runtime version mismatch** or corrupted CUDA state:
- System CUDA 13.0 drivers
- PyTorch CUDA 12.4 runtime
- Some incompatibility preventing initialization

## Potential Solutions (To Try)

### 1. System Reboot (Recommended First)
```bash
sudo reboot
```
Sometimes CUDA gets into a bad state that only a reboot can fix.

### 2. Install PyTorch with System CUDA Version
This likely won't work as PyTorch 2.6.0 doesn't officially support CUDA 13.0, but worth trying:
```bash
source .venv/bin/activate
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu130
```

### 3. Downgrade System CUDA to 12.x
If you don't need CUDA 13.0 for other projects:
```bash
sudo apt remove cuda-13-0
sudo apt install cuda-12-4
# Then reboot
```

### 4. Use Docker (Isolated Environment)
Create a containerized environment with matching CUDA versions:
```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
# Install Point-E dependencies
```

### 5. Use Google Colab / Cloud GPU
If local GPU can't be fixed, use cloud resources:
- Google Colab (free GPU)
- Lambda Labs
- Paperspace

## Current Workaround: CPU Mode

The framework **works perfectly on CPU**, just slower:

**CPU Performance**:
- Without upsampler: ~2 minutes → 1,024 points
- With upsampler: ~4-5 minutes → 4,096 points

**Expected GPU Performance** (when fixed):
- Without upsampler: ~15-30 seconds → 1,024 points
- With upsampler: ~30-60 seconds → 4,096 points

**Speedup**: 4-10x faster on GPU

## Usage

### Current (CPU only)
```bash
source .venv/bin/activate
python pretrained_inference/examples/quick_test_pointe.py \
    --prompt "a red car" \
    --no-upsampler
```

### After GPU Fix
Same command, but it will automatically use GPU when available.

## Testing GPU After Changes

Run the diagnostic script:
```bash
source .venv/bin/activate
python pretrained_inference/test_gpu.py
```

Should see:
```
✓ CUDA available: True
✓ GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
✓ GPU tensor operations work!
```

## Related Links
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues)

## Status
🔴 **GPU Currently Not Working** - Using CPU fallback
⏱️ **CPU Mode Functional** - All features work, just slower

---

**Last Updated**: 2025-12-01
**System**: NVIDIA RTX 2000 Ada with CUDA 13.0 drivers
