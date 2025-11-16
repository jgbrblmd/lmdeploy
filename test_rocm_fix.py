#!/usr/bin/env python3
"""Test script to verify ROCm fixes."""
import os
import sys

# Try setting environment variable if needed
if 'HSA_OVERRIDE_GFX_VERSION' not in os.environ:
    print("Note: HSA_OVERRIDE_GFX_VERSION is not set. You may need to set it if issues persist.")
    print("For gfx906, try: export HSA_OVERRIDE_GFX_VERSION=9.0.6")

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if hasattr(torch.version, 'hip') and torch.version.hip is not None:
    print(f"ROCm version: {torch.version.hip}")
    print("ROCm detected!")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"Device: {props.name}")
    print(f"Architecture: {getattr(props, 'gcnArchName', 'unknown')}")
    print(f"Major: {props.major}, Minor: {props.minor}")
    
    # Test tensor operations
    try:
        print("\nTesting tensor operations...")
        a = torch.tensor([1, 2], device='cuda')
        print("✓ Created tensor")
        
        b = a.new_tensor([3, 4], device='cuda')
        print("✓ Created new_tensor")
        
        c = a + b
        print("✓ Added tensors")
        
        expected = a.new_tensor([4, 6])
        torch.testing.assert_close(c, expected)
        print("✓ assert_close passed")
        
        print("\nAll tests passed! PyTorch is working correctly on ROCm.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error during tensor operations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("CUDA/HIP is not available!")
    sys.exit(1)

