#!/usr/bin/env python3
"""
Setup verification script.

Run this to check if all dependencies are installed correctly.
"""

import sys


def check_imports():
    """Check if all required packages can be imported."""
    print("Checking dependencies...\n")

    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'plotly': 'Plotly',
    }

    optional = {
        'point_e': 'Point-E',
        'shap_e': 'Shap-E',
        'open3d': 'Open3D',
        'PIL': 'Pillow',
    }

    all_good = True

    # Check required packages
    print("Required packages:")
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            all_good = False

    print("\nOptional packages (for full functionality):")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - Not installed (some features may not work)")

    # Check CUDA
    print("\nCUDA availability:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available - Will use CPU (slower)")
    except:
        print("  ? Could not check CUDA status")

    print("\n" + "=" * 60)
    if all_good:
        print("✓ All required dependencies installed!")
        print("\nYou're ready to go! Try:")
        print("  python pretrained_inference/examples/quick_test_pointe.py")
    else:
        print("✗ Some required dependencies are missing.")
        print("\nPlease run:")
        print("  pip install -r pretrained_inference/requirements.txt")
    print("=" * 60)

    return all_good


def check_folder_structure():
    """Check if folder structure is correct."""
    from pathlib import Path

    print("\n\nChecking folder structure...\n")

    base = Path(__file__).parent
    expected = [
        'models/pointe_inference.py',
        'models/shape_inference.py',
        'utils/visualization.py',
        'examples/quick_test_pointe.py',
        'examples/quick_test_shape.py',
        'examples/batch_generate.py',
        'README.md',
        'requirements.txt',
    ]

    all_present = True
    for path in expected:
        full_path = base / path
        if full_path.exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} - MISSING")
            all_present = False

    if all_present:
        print("\n✓ Folder structure is correct!")
    else:
        print("\n✗ Some files are missing!")

    return all_present


if __name__ == '__main__':
    print("=" * 60)
    print("Pretrained 3D Point Cloud Inference - Setup Verification")
    print("=" * 60)

    deps_ok = check_imports()
    structure_ok = check_folder_structure()

    if deps_ok and structure_ok:
        print("\n🎉 Everything looks good!")
        sys.exit(0)
    else:
        print("\n⚠️  Please fix the issues above before proceeding.")
        sys.exit(1)
