"""Test script for MambaSplitConv1dScanCombinedFn functorch compatibility."""

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

# Load the softplus module first (dependency)
softplus_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.softplus",
    os.path.join(ops_module_path, "softplus.py")
)
softplus_module = importlib.util.module_from_spec(softplus_spec)
sys.modules['mamba2_torch.ops.softplus'] = softplus_module
mamba2_torch.ops.softplus = softplus_module
softplus_spec.loader.exec_module(softplus_module)

# Load the ssd_bmm module (dependency)
ssd_bmm_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.ssd_bmm",
    os.path.join(ops_module_path, "ssd_bmm.py")
)
ssd_bmm_module = importlib.util.module_from_spec(ssd_bmm_spec)
sys.modules['mamba2_torch.ops.ssd_bmm'] = ssd_bmm_module
mamba2_torch.ops.ssd_bmm = ssd_bmm_module
ssd_bmm_spec.loader.exec_module(ssd_bmm_module)

# Load the ssd_chunk_state module (dependency)
ssd_chunk_state_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.ssd_chunk_state",
    os.path.join(ops_module_path, "ssd_chunk_state.py")
)
ssd_chunk_state_module = importlib.util.module_from_spec(ssd_chunk_state_spec)
sys.modules['mamba2_torch.ops.ssd_chunk_state'] = ssd_chunk_state_module
mamba2_torch.ops.ssd_chunk_state = ssd_chunk_state_module
ssd_chunk_state_spec.loader.exec_module(ssd_chunk_state_module)

# Load the ssd_state_passing module (dependency)
ssd_state_passing_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.ssd_state_passing",
    os.path.join(ops_module_path, "ssd_state_passing.py")
)
ssd_state_passing_module = importlib.util.module_from_spec(ssd_state_passing_spec)
sys.modules['mamba2_torch.ops.ssd_state_passing'] = ssd_state_passing_module
mamba2_torch.ops.ssd_state_passing = ssd_state_passing_module
ssd_state_passing_spec.loader.exec_module(ssd_state_passing_module)

# Load the ssd_chunk_scan module (dependency)
ssd_chunk_scan_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.ssd_chunk_scan",
    os.path.join(ops_module_path, "ssd_chunk_scan.py")
)
ssd_chunk_scan_module = importlib.util.module_from_spec(ssd_chunk_scan_spec)
sys.modules['mamba2_torch.ops.ssd_chunk_scan'] = ssd_chunk_scan_module
mamba2_torch.ops.ssd_chunk_scan = ssd_chunk_scan_module
ssd_chunk_scan_spec.loader.exec_module(ssd_chunk_scan_module)

# Load the layernorm_gated module (dependency)
layernorm_gated_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.layernorm_gated",
    os.path.join(ops_module_path, "layernorm_gated.py")
)
layernorm_gated_module = importlib.util.module_from_spec(layernorm_gated_spec)
sys.modules['mamba2_torch.ops.layernorm_gated'] = layernorm_gated_module
mamba2_torch.ops.layernorm_gated = layernorm_gated_module
layernorm_gated_spec.loader.exec_module(layernorm_gated_module)

# Load the k_activations module (dependency)
k_activations_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.k_activations",
    os.path.join(ops_module_path, "k_activations.py")
)
k_activations_module = importlib.util.module_from_spec(k_activations_spec)
sys.modules['mamba2_torch.ops.k_activations'] = k_activations_module
mamba2_torch.ops.k_activations = k_activations_module
k_activations_spec.loader.exec_module(k_activations_module)

# Load the custom_fwd_bwd module (dependency)
custom_fwd_bwd_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.custom_fwd_bwd",
    os.path.join(ops_module_path, "custom_fwd_bwd.py")
)
custom_fwd_bwd_module = importlib.util.module_from_spec(custom_fwd_bwd_spec)
sys.modules['mamba2_torch.ops.custom_fwd_bwd'] = custom_fwd_bwd_module
mamba2_torch.ops.custom_fwd_bwd = custom_fwd_bwd_module
custom_fwd_bwd_spec.loader.exec_module(custom_fwd_bwd_module)

# Now load the ssd_combined module
ssd_combined_spec = importlib.util.spec_from_file_location(
    "mamba2_torch.ops.ssd_combined",
    os.path.join(ops_module_path, "ssd_combined.py")
)
ssd_combined_module = importlib.util.module_from_spec(ssd_combined_spec)
sys.modules['mamba2_torch.ops.ssd_combined'] = ssd_combined_module
mamba2_torch.ops.ssd_combined = ssd_combined_module
ssd_combined_spec.loader.exec_module(ssd_combined_module)

mamba_split_conv1d_scan_combined = ssd_combined_module.mamba_split_conv1d_scan_combined
mamba_split_conv1d_scan_ref = ssd_combined_module.mamba_split_conv1d_scan_ref


def create_test_inputs(batch, seqlen, nheads, headdim, ngroups, dstate, conv_width=4, d_nonssm=0, device='cuda', dtype=torch.float32, requires_grad=True):
    """Create test inputs for MambaSplitConv1dScanCombinedFn."""
    dim = nheads * headdim
    # zxbcdt: (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
    input_dim = 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads
    zxbcdt = torch.randn(batch, seqlen, input_dim, device=device, dtype=dtype, requires_grad=requires_grad)
    # conv1d_weight: (dim + 2 * ngroups * dstate, width)
    conv1d_weight = torch.randn(dim + 2 * ngroups * dstate, conv_width, device=device, dtype=dtype, requires_grad=requires_grad)
    # conv1d_bias: (dim + 2 * ngroups * dstate,)
    conv1d_bias = torch.randn(dim + 2 * ngroups * dstate, device=device, dtype=dtype, requires_grad=requires_grad)
    # dt_bias: (nheads,)
    dt_bias = torch.randn(nheads, device=device, dtype=dtype, requires_grad=requires_grad)
    # A: (nheads,)
    A = torch.randn(nheads, device=device, dtype=dtype, requires_grad=requires_grad)
    # D: (nheads, headdim)
    D = torch.randn(nheads, headdim, device=device, dtype=dtype, requires_grad=requires_grad)

    return zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D


def test_basic_forward_backward():
    """Test basic forward/backward to ensure no regression."""
    print("=" * 60)
    print("Testing basic forward/backward (no regression)")
    print("=" * 60)

    # Set up test parameters
    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 16
    conv_width = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test (requires causal_conv1d_cuda)")
        return True
    dtype = torch.float32

    # Create test inputs
    zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D = create_test_inputs(
        batch, seqlen, nheads, headdim, ngroups, dstate, conv_width, device=device, dtype=dtype
    )

    print(f"Input shapes:")
    print(f"  zxbcdt: {zxbcdt.shape}")
    print(f"  conv1d_weight: {conv1d_weight.shape}")
    print(f"  conv1d_bias: {conv1d_bias.shape}")
    print(f"  dt_bias: {dt_bias.shape}")
    print(f"  A: {A.shape}")
    print(f"  D: {D.shape}")
    print(f"  chunk_size: {chunk_size}")

    # Forward pass
    out = mamba_split_conv1d_scan_combined(
        zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
        headdim=headdim, ngroups=ngroups
    )
    print(f"\nOutput shape:")
    print(f"  out: {out.shape}")

    # Backward pass
    loss = out.sum()
    loss.backward()

    print(f"\nGradient shapes:")
    print(f"  zxbcdt.grad: {zxbcdt.grad.shape if zxbcdt.grad is not None else 'None'}")
    print(f"  conv1d_weight.grad: {conv1d_weight.grad.shape if conv1d_weight.grad is not None else 'None'}")
    print(f"  conv1d_bias.grad: {conv1d_bias.grad.shape if conv1d_bias.grad is not None else 'None'}")
    print(f"  dt_bias.grad: {dt_bias.grad.shape if dt_bias.grad is not None else 'None'}")
    print(f"  A.grad: {A.grad.shape if A.grad is not None else 'None'}")
    print(f"  D.grad: {D.grad.shape if D.grad is not None else 'None'}")

    # Verify all gradients are computed
    assert zxbcdt.grad is not None, "zxbcdt.grad should not be None"
    assert conv1d_weight.grad is not None, "conv1d_weight.grad should not be None"
    assert conv1d_bias.grad is not None, "conv1d_bias.grad should not be None"
    assert dt_bias.grad is not None, "dt_bias.grad should not be None"
    assert A.grad is not None, "A.grad should not be None"
    assert D.grad is not None, "D.grad should not be None"

    print("\n✓ Basic forward/backward test PASSED")
    return True


def test_forward_backward_with_return_final_states():
    """Test forward/backward with return_final_states=True."""
    print("\n" + "=" * 60)
    print("Testing forward/backward with return_final_states=True")
    print("=" * 60)

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 16
    conv_width = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test (requires causal_conv1d_cuda)")
        return True
    dtype = torch.float32

    zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D = create_test_inputs(
        batch, seqlen, nheads, headdim, ngroups, dstate, conv_width, device=device, dtype=dtype
    )

    out, final_states = mamba_split_conv1d_scan_combined(
        zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
        return_final_states=True, headdim=headdim, ngroups=ngroups
    )
    print(f"Output shape: {out.shape}")
    print(f"Final states shape: {final_states.shape}")

    # Backward on output only
    loss = out.sum()
    loss.backward()

    assert zxbcdt.grad is not None, "zxbcdt.grad should not be None"
    print(f"zxbcdt.grad shape: {zxbcdt.grad.shape}")

    print("\n✓ Forward/backward with return_final_states test PASSED")
    return True


def test_forward_backward_with_initial_states():
    """Test forward/backward with initial_states parameter."""
    print("\n" + "=" * 60)
    print("Testing forward/backward with initial_states parameter")
    print("=" * 60)

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 16
    conv_width = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test (requires causal_conv1d_cuda)")
        return True
    dtype = torch.float32

    zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D = create_test_inputs(
        batch, seqlen, nheads, headdim, ngroups, dstate, conv_width, device=device, dtype=dtype
    )
    initial_states = torch.randn(batch, nheads, headdim, dstate, device=device, dtype=dtype, requires_grad=True)

    print(f"initial_states shape: {initial_states.shape}")

    out = mamba_split_conv1d_scan_combined(
        zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
        initial_states=initial_states, headdim=headdim, ngroups=ngroups
    )
    print(f"Output shape: {out.shape}")

    loss = out.sum()
    loss.backward()

    assert initial_states.grad is not None, "initial_states.grad should not be None"
    print(f"initial_states.grad shape: {initial_states.grad.shape}")

    print("\n✓ Forward/backward with initial_states test PASSED")
    return True


def test_forward_backward_with_rmsnorm():
    """Test forward/backward with rmsnorm_weight parameter."""
    print("\n" + "=" * 60)
    print("Testing forward/backward with rmsnorm_weight parameter")
    print("=" * 60)

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 16
    conv_width = 4
    dim = nheads * headdim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test (requires causal_conv1d_cuda)")
        return True
    dtype = torch.float32

    zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D = create_test_inputs(
        batch, seqlen, nheads, headdim, ngroups, dstate, conv_width, device=device, dtype=dtype
    )
    rmsnorm_weight = torch.randn(dim, device=device, dtype=dtype, requires_grad=True)

    print(f"rmsnorm_weight shape: {rmsnorm_weight.shape}")

    out = mamba_split_conv1d_scan_combined(
        zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
        rmsnorm_weight=rmsnorm_weight, headdim=headdim, ngroups=ngroups
    )
    print(f"Output shape: {out.shape}")

    loss = out.sum()
    loss.backward()

    assert rmsnorm_weight.grad is not None, "rmsnorm_weight.grad should not be None"
    print(f"rmsnorm_weight.grad shape: {rmsnorm_weight.grad.shape}")

    print("\n✓ Forward/backward with rmsnorm_weight test PASSED")
    return True


def test_forward_backward_with_outproj():
    """Test forward/backward with output projection parameters."""
    print("\n" + "=" * 60)
    print("Testing forward/backward with output projection")
    print("=" * 60)

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 16
    conv_width = 4
    dim = nheads * headdim
    out_dim = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test (requires causal_conv1d_cuda)")
        return True
    dtype = torch.float32

    zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D = create_test_inputs(
        batch, seqlen, nheads, headdim, ngroups, dstate, conv_width, device=device, dtype=dtype
    )
    outproj_weight = torch.randn(out_dim, dim, device=device, dtype=dtype, requires_grad=True)
    outproj_bias = torch.randn(out_dim, device=device, dtype=dtype, requires_grad=True)

    print(f"outproj_weight shape: {outproj_weight.shape}")
    print(f"outproj_bias shape: {outproj_bias.shape}")

    out = mamba_split_conv1d_scan_combined(
        zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
        outproj_weight=outproj_weight, outproj_bias=outproj_bias,
        headdim=headdim, ngroups=ngroups
    )
    print(f"Output shape: {out.shape}")

    loss = out.sum()
    loss.backward()

    assert outproj_weight.grad is not None, "outproj_weight.grad should not be None"
    assert outproj_bias.grad is not None, "outproj_bias.grad should not be None"
    print(f"outproj_weight.grad shape: {outproj_weight.grad.shape}")
    print(f"outproj_bias.grad shape: {outproj_bias.grad.shape}")

    print("\n✓ Forward/backward with output projection test PASSED")
    return True


def test_forward_backward_with_amp():
    """Test forward/backward with AMP (automatic mixed precision)."""
    print("\n" + "=" * 60)
    print("Testing forward/backward with AMP")
    print("=" * 60)

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 16
    conv_width = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test (requires causal_conv1d_cuda)")
        return True
    dtype = torch.float32

    zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D = create_test_inputs(
        batch, seqlen, nheads, headdim, ngroups, dstate, conv_width, device=device, dtype=dtype
    )

    print("Running with AMP autocast enabled...")

    with torch.cuda.amp.autocast():
        out = mamba_split_conv1d_scan_combined(
            zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
            headdim=headdim, ngroups=ngroups
        )
        print(f"Output shape: {out.shape}")
        print(f"Output dtype: {out.dtype}")

        loss = out.sum()

    loss.backward()

    assert zxbcdt.grad is not None, "zxbcdt.grad should not be None"
    print(f"zxbcdt.grad dtype: {zxbcdt.grad.dtype}")

    print("\n✓ Forward/backward with AMP test PASSED")
    return True


def test_vmap():
    """Test vmap compatibility."""
    print("\n" + "=" * 60)
    print("Testing vmap compatibility")
    print("=" * 60)

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 16
    conv_width = 4
    vmap_batch = 3
    dim = nheads * headdim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test (requires causal_conv1d_cuda)")
        return True
    dtype = torch.float32

    # Create batched inputs for vmap
    input_dim = 2 * dim + 2 * ngroups * dstate + nheads
    zxbcdt = torch.randn(vmap_batch, batch, seqlen, input_dim, device=device, dtype=dtype)
    # Shared parameters
    conv1d_weight = torch.randn(dim + 2 * ngroups * dstate, conv_width, device=device, dtype=dtype)
    conv1d_bias = torch.randn(dim + 2 * ngroups * dstate, device=device, dtype=dtype)
    dt_bias = torch.randn(nheads, device=device, dtype=dtype)
    A = torch.randn(nheads, device=device, dtype=dtype)
    D = torch.randn(nheads, headdim, device=device, dtype=dtype)

    print(f"vmap batch size: {vmap_batch}")
    print(f"Input shapes (before vmap):")
    print(f"  zxbcdt: {zxbcdt.shape}")
    print(f"  conv1d_weight: {conv1d_weight.shape} (shared)")
    print(f"  conv1d_bias: {conv1d_bias.shape} (shared)")
    print(f"  dt_bias: {dt_bias.shape} (shared)")
    print(f"  A: {A.shape} (shared)")
    print(f"  D: {D.shape} (shared)")

    # Apply vmap
    vmapped_fn = vmap(
        lambda z: mamba_split_conv1d_scan_combined(
            z, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
            headdim=headdim, ngroups=ngroups
        )
    )
    out = vmapped_fn(zxbcdt)

    print(f"\nOutput shape (after vmap):")
    print(f"  out: {out.shape}")

    expected_shape = (vmap_batch, batch, seqlen, dim)
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

    # Verify by comparing with manual loop
    print("\nComparing vmap result with manual loop...")
    for i in range(vmap_batch):
        out_manual = mamba_split_conv1d_scan_combined(
            zxbcdt[i], conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
            headdim=headdim, ngroups=ngroups
        )
        max_diff = (out[i] - out_manual).abs().max().item()
        assert max_diff < 1e-5, f"Batch {i}: max diff {max_diff} too large"

    print("✓ vmap test PASSED")
    return True


def test_vmap_with_grad():
    """Test vmap with gradient computation."""
    print("\n" + "=" * 60)
    print("Testing vmap with gradient computation")
    print("=" * 60)

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 32
    ngroups = 2
    dstate = 16
    chunk_size = 16
    conv_width = 4
    vmap_batch = 3
    dim = nheads * headdim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("CUDA not available, skipping test (requires causal_conv1d_cuda)")
        return True
    dtype = torch.float32

    # Create batched inputs for vmap
    input_dim = 2 * dim + 2 * ngroups * dstate + nheads
    zxbcdt = torch.randn(vmap_batch, batch, seqlen, input_dim, device=device, dtype=dtype, requires_grad=True)
    # Shared parameters
    conv1d_weight = torch.randn(dim + 2 * ngroups * dstate, conv_width, device=device, dtype=dtype, requires_grad=True)
    conv1d_bias = torch.randn(dim + 2 * ngroups * dstate, device=device, dtype=dtype, requires_grad=True)
    dt_bias = torch.randn(nheads, device=device, dtype=dtype, requires_grad=True)
    A = torch.randn(nheads, device=device, dtype=dtype, requires_grad=True)
    D = torch.randn(nheads, headdim, device=device, dtype=dtype, requires_grad=True)

    # Apply vmap
    vmapped_fn = vmap(
        lambda z: mamba_split_conv1d_scan_combined(
            z, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
            headdim=headdim, ngroups=ngroups
        )
    )
    out = vmapped_fn(zxbcdt)

    # Backward
    loss = out.sum()
    loss.backward()

    print(f"Gradient shapes:")
    print(f"  zxbcdt.grad: {zxbcdt.grad.shape if zxbcdt.grad is not None else 'None'}")
    print(f"  conv1d_weight.grad: {conv1d_weight.grad.shape if conv1d_weight.grad is not None else 'None'}")
    print(f"  conv1d_bias.grad: {conv1d_bias.grad.shape if conv1d_bias.grad is not None else 'None'}")
    print(f"  dt_bias.grad: {dt_bias.grad.shape if dt_bias.grad is not None else 'None'}")
    print(f"  A.grad: {A.grad.shape if A.grad is not None else 'None'}")
    print(f"  D.grad: {D.grad.shape if D.grad is not None else 'None'}")

    assert zxbcdt.grad is not None, "zxbcdt.grad should not be None"
    assert conv1d_weight.grad is not None, "conv1d_weight.grad should not be None"
    assert conv1d_bias.grad is not None, "conv1d_bias.grad should not be None"
    assert dt_bias.grad is not None, "dt_bias.grad should not be None"
    assert A.grad is not None, "A.grad should not be None"
    assert D.grad is not None, "D.grad should not be None"

    print("✓ vmap with gradient test PASSED")
    return True


if __name__ == "__main__":
    all_passed = True

    tests = [
        ("Basic forward/backward", test_basic_forward_backward),
        ("Forward/backward with return_final_states", test_forward_backward_with_return_final_states),
        ("Forward/backward with initial_states", test_forward_backward_with_initial_states),
        ("Forward/backward with rmsnorm", test_forward_backward_with_rmsnorm),
        ("Forward/backward with outproj", test_forward_backward_with_outproj),
        ("Forward/backward with AMP", test_forward_backward_with_amp),
        ("vmap", test_vmap),
        ("vmap with gradient", test_vmap_with_grad),
    ]

    for name, test_fn in tests:
        try:
            all_passed &= test_fn()
        except Exception as e:
            print(f"\n✗ {name} test FAILED: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
