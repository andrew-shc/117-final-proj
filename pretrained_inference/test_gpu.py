#!/usr/bin/env python3
"""
GPU/CUDA diagnostic script.
Run this in a fresh terminal to test GPU availability.
"""

import sys

print("="*60)
print("GPU/CUDA Diagnostic Test")
print("="*60)

# Test 1: Import torch
try:
    import torch
    print("\n✓ PyTorch imported successfully")
    print(f"  Version: {torch.__version__}")
except ImportError as e:
    print(f"\n✗ Failed to import PyTorch: {e}")
    sys.exit(1)

# Test 2: CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (compiled): {torch.version.cuda}")

# Test 3: Device count
device_count = torch.cuda.device_count()
print(f"CUDA device count: {device_count}")

# Test 4: Get device name
if torch.cuda.is_available():
    print("\n" + "="*60)
    print("GPU Information:")
    print("="*60)
    for i in range(device_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Compute capability: {props.major}.{props.minor}")

    # Test 5: Simple tensor operation on GPU
    print("\n" + "="*60)
    print("Testing GPU tensor operations...")
    print("="*60)
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = x @ y
        print("✓ GPU tensor operations work!")
        print(f"  Device: {z.device}")
    except Exception as e:
        print(f"✗ GPU tensor operations failed: {e}")
else:
    print("\n" + "="*60)
    print("CUDA NOT AVAILABLE")
    print("="*60)
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU detected")
    print("  2. CUDA drivers not installed")
    print("  3. PyTorch compiled without CUDA support")
    print("  4. GPU already in use by another process")
    print("  5. Environment variable issues (CUDA_VISIBLE_DEVICES)")
    print("\nTo fix:")
    print("  1. Check nvidia-smi output")
    print("  2. Kill processes using GPU")
    print("  3. Reinstall PyTorch with CUDA:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

print("\n" + "="*60)
