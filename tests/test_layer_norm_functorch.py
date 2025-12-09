"""Test script for LayerNormFn (full) functorch compatibility."""

import sys
import os

# Set up path before torch import - go up one level from tests/ to find src/
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
from torch.func import vmap

# Bypass the mamba2_torch __init__.py and import directly from the ops module file
import importlib.util
ops_module_path = os.path.join(project_root, 'src', 'mamba2_torch', 'ops')
sys.path.insert(0, ops_module_path)

# We need to handle the relative import issue. Let's manually set up the module.
# First set up the parent package namespace
import types
mamba2_torch = types.ModuleType('mamba2_torch')
mamba2_torch.ops = types.ModuleType('mamba2_torch.ops')
sys.modules['mamba2_torch'] = mamba2_torch
sys.modules['mamba2_torch.ops'] = mamba2_torch.ops

# Load custom_fwd_bwd first (required by layer_norm)
custom_fwd_bwd_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.custom_fwd_bwd",
    os.path.join(ops_module_path, "custom_fwd_bwd.py")
)
custom_fwd_bwd_module = importlib.util.module_from_spec(custom_fwd_bwd_spec)
sys.modules['mamba2_torch.ops.custom_fwd_bwd'] = custom_fwd_bwd_module
mamba2_torch.ops.custom_fwd_bwd = custom_fwd_bwd_module
# Add relative import path
sys.modules['..ops.custom_fwd_bwd'] = custom_fwd_bwd_module
custom_fwd_bwd_spec.loader.exec_module(custom_fwd_bwd_module)

# Now load the layer_norm module
layer_norm_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.layer_norm",
    os.path.join(ops_module_path, "layer_norm.py")
)
layer_norm_module = importlib.util.module_from_spec(layer_norm_spec)
sys.modules['mamba2_torch.ops.layer_norm'] = layer_norm_module
mamba2_torch.ops.layer_norm = layer_norm_module
layer_norm_spec.loader.exec_module(layer_norm_module)

layer_norm_fn = layer_norm_module.layer_norm_fn
rms_norm_fn = layer_norm_module.rms_norm_fn
layer_norm_ref = layer_norm_module.layer_norm_ref
rms_norm_ref = layer_norm_module.rms_norm_ref


def test_basic_forward_backward():
    """Test basic forward/backward to ensure no regression."""
    print("=" * 60)
    print("Testing basic forward/backward (no regression)")
    print("=" * 60)

    # Set up test parameters
    batch = 2
    seqlen = 64
    hidden_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Create test inputs with requires_grad
    x = torch.randn(batch, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)

    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  weight: {weight.shape}")
    print(f"  bias: {bias.shape}")

    # Forward pass
    y = layer_norm_fn(x, weight, bias)
    print(f"\nOutput shape:")
    print(f"  y: {y.shape}")

    # Backward pass
    loss = y.sum()
    loss.backward()

    print(f"\nGradient shapes:")
    print(f"  x.grad: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"  weight.grad: {weight.grad.shape if weight.grad is not None else 'None'}")
    print(f"  bias.grad: {bias.grad.shape if bias.grad is not None else 'None'}")

    # Verify all gradients are computed
    assert x.grad is not None, "x.grad should not be None"
    assert weight.grad is not None, "weight.grad should not be None"
    assert bias.grad is not None, "bias.grad should not be None"

    print("\n✓ Basic forward/backward test PASSED")
    return True


def test_with_residual():
    """Test forward/backward with residual connection."""
    print("\n" + "=" * 60)
    print("Testing with residual connection")
    print("=" * 60)

    batch = 2
    seqlen = 64
    hidden_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Create test inputs with requires_grad
    x = torch.randn(batch, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True)
    residual = torch.randn(batch, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)

    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  residual: {residual.shape}")
    print(f"  weight: {weight.shape}")

    # Forward pass with residual
    y = layer_norm_fn(x, weight, bias, residual=residual)
    print(f"\nOutput shape:")
    print(f"  y: {y.shape}")

    # Backward pass
    loss = y.sum()
    loss.backward()

    print(f"\nGradient shapes:")
    print(f"  x.grad: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"  residual.grad: {residual.grad.shape if residual.grad is not None else 'None'}")
    print(f"  weight.grad: {weight.grad.shape if weight.grad is not None else 'None'}")

    assert x.grad is not None, "x.grad should not be None"
    assert residual.grad is not None, "residual.grad should not be None"
    assert weight.grad is not None, "weight.grad should not be None"

    print("✓ Residual test PASSED")
    return True


def test_with_prenorm():
    """Test forward/backward with prenorm=True."""
    print("\n" + "=" * 60)
    print("Testing with prenorm=True")
    print("=" * 60)

    batch = 2
    seqlen = 64
    hidden_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Create test inputs
    x = torch.randn(batch, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True)
    residual = torch.randn(batch, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)

    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  residual: {residual.shape}")

    # Forward pass with prenorm
    result = layer_norm_fn(x, weight, bias, residual=residual, prenorm=True)
    y, residual_out = result
    print(f"\nOutput shapes:")
    print(f"  y: {y.shape}")
    print(f"  residual_out: {residual_out.shape}")

    # Backward pass
    loss = y.sum() + residual_out.sum()
    loss.backward()

    assert x.grad is not None, "x.grad should not be None"
    assert residual.grad is not None, "residual.grad should not be None"
    assert weight.grad is not None, "weight.grad should not be None"

    print("✓ Prenorm test PASSED")
    return True


def test_rmsnorm():
    """Test RMSNorm variant."""
    print("\n" + "=" * 60)
    print("Testing RMSNorm variant")
    print("=" * 60)

    batch = 2
    seqlen = 64
    hidden_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    x = torch.randn(batch, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)

    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  weight: {weight.shape}")

    # Forward pass
    y = rms_norm_fn(x, weight, bias)
    print(f"\nOutput shape:")
    print(f"  y: {y.shape}")

    # Backward pass
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "x.grad should not be None"
    assert weight.grad is not None, "weight.grad should not be None"

    print("✓ RMSNorm test PASSED")
    return True


def test_compare_with_reference():
    """Compare with reference implementation to ensure correctness."""
    print("\n" + "=" * 60)
    print("Testing comparison with reference implementation")
    print("=" * 60)

    batch = 2
    seqlen = 64
    hidden_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    x = torch.randn(batch, seqlen, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    bias = torch.randn(hidden_size, device=device, dtype=dtype)

    # Run optimized version
    y_opt = layer_norm_fn(x, weight, bias)

    # Run reference version
    y_ref = layer_norm_ref(x, weight, bias)

    # Compare
    max_diff = (y_opt - y_ref).abs().max().item()
    print(f"Max difference between optimized and reference: {max_diff:.2e}")

    # Tolerance for float32
    assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    print("✓ Reference comparison test PASSED")
    return True


def test_compare_rmsnorm_with_reference():
    """Compare RMSNorm with reference implementation."""
    print("\n" + "=" * 60)
    print("Testing RMSNorm comparison with reference")
    print("=" * 60)

    batch = 2
    seqlen = 64
    hidden_size = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    x = torch.randn(batch, seqlen, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    bias = torch.randn(hidden_size, device=device, dtype=dtype)

    # Run optimized version (RMSNorm)
    y_opt = rms_norm_fn(x, weight, bias)

    # Run reference version
    y_ref = rms_norm_ref(x, weight, bias)

    # Compare
    max_diff = (y_opt - y_ref).abs().max().item()
    print(f"Max difference between optimized and reference: {max_diff:.2e}")

    # Tolerance for float32
    assert max_diff < 1e-4, f"Max diff {max_diff} exceeds tolerance"

    print("✓ RMSNorm reference comparison test PASSED")
    return True


def test_vmap():
    """Test vmap compatibility."""
    print("\n" + "=" * 60)
    print("Testing vmap compatibility")
    print("=" * 60)

    batch = 2
    seqlen = 64
    hidden_size = 128
    vmap_batch = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Create batched inputs for vmap
    x = torch.randn(vmap_batch, batch, seqlen, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    bias = torch.randn(hidden_size, device=device, dtype=dtype)

    print(f"vmap batch size: {vmap_batch}")
    print(f"Input shapes (before vmap):")
    print(f"  x: {x.shape}")
    print(f"  weight: {weight.shape}")
    print(f"  bias: {bias.shape}")

    # Apply vmap over the first dimension of x only
    # weight and bias are shared across the vmap batch
    vmapped_layernorm = vmap(lambda xi: layer_norm_fn(xi, weight, bias))
    y = vmapped_layernorm(x)

    print(f"\nOutput shape (after vmap):")
    print(f"  y: {y.shape}")

    expected_shape = (vmap_batch, batch, seqlen, hidden_size)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"

    # Verify by comparing with manual loop
    print("\nComparing vmap result with manual loop...")
    for i in range(vmap_batch):
        y_manual = layer_norm_fn(x[i], weight, bias)
        max_diff = (y[i] - y_manual).abs().max().item()
        assert max_diff < 1e-5, f"Batch {i}: max diff {max_diff} too large"

    print("✓ vmap test PASSED")
    return True


def test_vmap_with_residual():
    """Test vmap compatibility with residual."""
    print("\n" + "=" * 60)
    print("Testing vmap compatibility with residual")
    print("=" * 60)

    batch = 2
    seqlen = 64
    hidden_size = 128
    vmap_batch = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Create batched inputs for vmap
    x = torch.randn(vmap_batch, batch, seqlen, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(vmap_batch, batch, seqlen, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    bias = torch.randn(hidden_size, device=device, dtype=dtype)

    print(f"vmap batch size: {vmap_batch}")
    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  residual: {residual.shape}")

    # Apply vmap over x and residual
    vmapped_layernorm = vmap(lambda xi, ri: layer_norm_fn(xi, weight, bias, residual=ri))
    y = vmapped_layernorm(x, residual)

    print(f"\nOutput shape (after vmap):")
    print(f"  y: {y.shape}")

    expected_shape = (vmap_batch, batch, seqlen, hidden_size)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"

    # Verify by comparing with manual loop
    print("\nComparing vmap result with manual loop...")
    for i in range(vmap_batch):
        y_manual = layer_norm_fn(x[i], weight, bias, residual=residual[i])
        max_diff = (y[i] - y_manual).abs().max().item()
        assert max_diff < 1e-5, f"Batch {i}: max diff {max_diff} too large"

    print("✓ vmap with residual test PASSED")
    return True


def test_vmap_with_grad():
    """Test vmap with gradient computation."""
    print("\n" + "=" * 60)
    print("Testing vmap with gradient computation")
    print("=" * 60)

    batch = 2
    seqlen = 64
    hidden_size = 128
    vmap_batch = 3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Create batched inputs with requires_grad
    x = torch.randn(vmap_batch, batch, seqlen, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)

    # Apply vmap
    vmapped_layernorm = vmap(lambda xi: layer_norm_fn(xi, weight, bias))
    y = vmapped_layernorm(x)

    # Backward
    loss = y.sum()
    loss.backward()

    print(f"Gradient shapes:")
    print(f"  x.grad: {x.grad.shape if x.grad is not None else 'None'}")
    print(f"  weight.grad: {weight.grad.shape if weight.grad is not None else 'None'}")
    print(f"  bias.grad: {bias.grad.shape if bias.grad is not None else 'None'}")

    assert x.grad is not None, "x.grad should not be None"
    assert weight.grad is not None, "weight.grad should not be None"
    assert bias.grad is not None, "bias.grad should not be None"

    print("✓ vmap with gradient test PASSED")
    return True


if __name__ == "__main__":
    all_passed = True

    try:
        all_passed &= test_basic_forward_backward()
    except Exception as e:
        print(f"\n✗ Basic forward/backward test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    try:
        all_passed &= test_with_residual()
    except Exception as e:
        print(f"\n✗ Residual test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    try:
        all_passed &= test_with_prenorm()
    except Exception as e:
        print(f"\n✗ Prenorm test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    try:
        all_passed &= test_rmsnorm()
    except Exception as e:
        print(f"\n✗ RMSNorm test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    try:
        all_passed &= test_compare_with_reference()
    except Exception as e:
        print(f"\n✗ Reference comparison test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    try:
        all_passed &= test_compare_rmsnorm_with_reference()
    except Exception as e:
        print(f"\n✗ RMSNorm reference comparison test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    try:
        all_passed &= test_vmap()
    except Exception as e:
        print(f"\n✗ vmap test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    try:
        all_passed &= test_vmap_with_residual()
    except Exception as e:
        print(f"\n✗ vmap with residual test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    try:
        all_passed &= test_vmap_with_grad()
    except Exception as e:
        print(f"\n✗ vmap with gradient test FAILED: {e}")
        all_passed = False
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
