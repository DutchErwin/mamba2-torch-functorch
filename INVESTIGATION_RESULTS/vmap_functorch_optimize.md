# Enabling vmap functorch parallelization for mamba2-torch

Making the mamba2-torch library compatible with `torch.vmap` requires significant modifications to all `torch.autograd.Function` classes. **The core blocker is that Triton kernels cannot process BatchedTensor objects created by vmap**, meaning full compatibility requires both the new `setup_context` pattern AND custom `vmap()` staticmethod implementations for each autograd.Function class.

This is a known unsolved problem—**Issue #239 on state-spaces/mamba** has been open since March 2024 with no official response or solution. The good news: the required changes are mechanical and well-documented, but the implementation effort is substantial.

## Four autograd.Function classes require modification

The mamba2-torch repository mirrors the state-spaces/mamba structure. All autograd.Function classes live under `src/mamba2_torch/ops/` and `src/mamba2_torch/ops/triton/`:

| Class | File | Role | Triton Kernels |
|-------|------|------|----------------|
| **MambaSplitConv1dScanCombinedFn** | `ops/triton/ssd_combined.py` | Top-level entry point for Mamba2.forward() | Multiple via helpers |
| **MambaChunkScanCombinedFn** | `ops/triton/ssd_combined.py` | Core chunk scan combining all operations | _bmm_chunk_fwd, _chunk_state_fwd, _state_passing_fwd, _chunk_scan_fwd |
| **SelectiveScanFn** | `ops/selective_scan_interface.py` | Mamba1 selective scan (CUDA kernels) | selective_scan_cuda |
| **MambaInnerFn** | `ops/selective_scan_interface.py` | Mamba1 inner function | selective_scan_cuda, _layer_norm_fwd |

Additional lower-level autograd.Function classes exist in separate triton files: `ssd_bmm.py` (ChunkBMMFn), `ssd_chunk_state.py` (ChunkStateFn), `ssd_state_passing.py` (StatePassingFn), and `ssd_chunk_scan.py` (ChunkScanFn). These are called internally by the combined functions and also require modification if you want nested vmap composability.

## The functorch compatibility pattern explained

The transformation from old to new pattern follows a strict recipe. Here's a complete before/after example:

**BEFORE (Current incompatible pattern):**
```python
class MambaChunkScanCombinedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
                initial_states=None, seq_idx=None, dt_softplus=False,
                dt_limit=(0.0, float("inf")), return_final_states=False):
        # ctx is first argument - triggers the error
        out, out_x, dt_out, dA_cumsum, states, final_states = (
            _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z,
                                           dt_bias, initial_states, seq_idx,
                                           dt_softplus, dt_limit)
        )
        
        # Saving tensors happens IN forward()
        ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum,
                              A, B, C, D, z, dt_bias, initial_states, seq_idx)
        ctx.dt_softplus = dt_softplus
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.chunk_size = chunk_size
        
        return out if not return_final_states else (out, final_states)

    @staticmethod
    def backward(ctx, dout, *args):
        saved = ctx.saved_tensors
        # ... backward implementation
```

**AFTER (Functorch-compatible pattern):**
```python
class MambaChunkScanCombinedFn(torch.autograd.Function):
    @staticmethod
    def forward(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None,
                initial_states=None, seq_idx=None, dt_softplus=False,
                dt_limit=(0.0, float("inf")), return_final_states=False):
        # NO ctx argument - key change
        out, out_x, dt_out, dA_cumsum, states, final_states = (
            _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z,
                                           dt_bias, initial_states, seq_idx,
                                           dt_softplus, dt_limit)
        )
        
        # Return intermediates needed for backward
        # These become additional outputs that setup_context can access
        return (out, out_x, dA_cumsum, final_states)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # inputs = tuple of ALL forward() arguments
        (x, dt, A, B, C, chunk_size, D, z, dt_bias,
         initial_states, seq_idx, dt_softplus, dt_limit, return_final_states) = inputs
        
        # output = return value of forward()
        out, out_x, dA_cumsum, final_states = output
        
        # Mark non-differentiable outputs (if any)
        # ctx.mark_non_differentiable(final_states)  # if applicable
        
        # Save tensors for backward via save_for_backward
        ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum,
                              A, B, C, D, z, dt_bias, initial_states, seq_idx)
        
        # Non-tensor values as direct attributes
        ctx.dt_softplus = dt_softplus
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.chunk_size = chunk_size

    @staticmethod
    def backward(ctx, grad_out, grad_out_x, grad_dA_cumsum, grad_final_states):
        # MUST accept one gradient per forward() output
        saved = ctx.saved_tensors
        # ... backward implementation unchanged
        return (grad_x, grad_dt, grad_A, grad_B, grad_C, None,  # chunk_size
                grad_D, grad_z, grad_dt_bias, grad_initial_states,
                None, None, None, None)  # seq_idx, dt_softplus, dt_limit, return_final_states

    @staticmethod
    def vmap(info, in_dims, x, dt, A, B, C, chunk_size, D, z, dt_bias,
             initial_states, seq_idx, dt_softplus, dt_limit, return_final_states):
        # REQUIRED for Triton kernels - cannot use generate_vmap_rule=True
        x_bdim, dt_bdim, A_bdim, B_bdim, C_bdim, *rest_bdims = in_dims
        
        # Move batch dimension to front for all batched inputs
        if x_bdim is not None:
            x = x.movedim(x_bdim, 0)
        if dt_bdim is not None:
            dt = dt.movedim(dt_bdim, 0)
        # ... handle all batched dimensions
        
        # Call function with adjusted tensors
        result = MambaChunkScanCombinedFn.apply(
            x, dt, A, B, C, chunk_size, D, z, dt_bias,
            initial_states, seq_idx, dt_softplus, dt_limit, return_final_states
        )
        
        # Return (outputs, out_dims) - batch dim at position 0 for all outputs
        return result, (0, 0, 0, 0)  # One dim per output
```

## Triton kernels require manual vmap rules

The critical limitation: **`generate_vmap_rule = True` does NOT work with Triton kernels**. When PyTorch's vmap encounters a BatchedTensor and passes it to a Triton kernel, you get:

```
Cannot access data pointer of Tensor that doesn't have storage
```

This happens because `BatchedTensor` is a wrapper without direct storage, and Triton kernels require `tensor.data_ptr()` to access raw memory. The solution is implementing a manual `vmap()` staticmethod for each autograd.Function that:

1. Extracts batch dimensions from input tensors using `in_dims`
2. Moves batch dimensions to consistent positions (typically index 0)
3. Calls the underlying function with properly shaped tensors
4. Returns outputs with explicit batch dimension positions

## Step-by-step modification plan

**Phase 1: Modify ssd_combined.py (highest impact)**

| Order | Class | Lines (approx) | Complexity | Dependencies |
|-------|-------|----------------|------------|--------------|
| 1 | MambaChunkScanCombinedFn | 500-580 | High | Lower-level triton ops |
| 2 | MambaSplitConv1dScanCombinedFn | 700-930 | Very High | MambaChunkScanCombinedFn, causal_conv1d |

**Phase 2: Modify selective_scan_interface.py (Mamba1 support)**

| Order | Class | Lines (approx) | Complexity | Dependencies |
|-------|-------|----------------|------------|--------------|
| 3 | SelectiveScanFn | 19-75 | Medium | selective_scan_cuda |
| 4 | MambaInnerFn | 160-320 | High | SelectiveScanFn, causal_conv1d |

**Phase 3: Modify lower-level triton ops (for nested composability)**

| Order | File | Classes | Complexity |
|-------|------|---------|------------|
| 5 | ssd_bmm.py | ChunkBMMFn | Medium |
| 6 | ssd_chunk_state.py | ChunkStateFn | Medium |
| 7 | ssd_state_passing.py | StatePassingFn | Medium |
| 8 | ssd_chunk_scan.py | ChunkScanFn | High |

**Key modifications per file:**
- Remove `ctx` from `forward()` signature
- Add `setup_context(ctx, inputs, output)` staticmethod
- Return any intermediates from `forward()` that backward needs
- Update `backward()` to accept gradients for ALL outputs
- Implement manual `vmap()` staticmethod with batch dimension handling
- Handle `@custom_fwd`/`@custom_bwd` decorators (may need removal or update)

## Memory analysis for vmap on your hardware

Your setup: **RTX PRO 6000 Blackwell (96GB)**, current usage **23GB**, target **<40GB**

| vmap Configuration | Peak Memory Estimate | Risk Level |
|--------------------|---------------------|------------|
| `chunk_size=None` (full 16384 batch) | ~92GB (4× current) | **DANGEROUS** - near OOM |
| `chunk_size=8192` | ~46GB | Moderate risk |
| `chunk_size=4096` (recommended) | ~30-35GB | **Safe** - within budget |
| `chunk_size=2048` | ~26-28GB | Very safe |
| `chunk_size=1` (sequential) | ~23GB | No parallelism benefit |

**Recommended configuration:**
```python
from torch.func import vmap

# Safe approach matching your existing chunk structure
batched_mamba = vmap(
    mamba_forward_fn,
    chunk_size=4096,      # Match existing chunk processing
    randomness='error'    # Fail on random ops (safe default)
)
```

The `chunk_size` parameter processes samples in batches, limiting peak memory. With **17GB headroom** (40GB target - 23GB current), `chunk_size=4096` provides parallelism while staying safely within budget.

## Testing strategy for correctness verification

**Unit tests for each modified class:**

```python
import torch
from torch.func import vmap, grad, jacrev

def test_vmap_compatibility(fn, sample_inputs):
    """Test that function works with vmap transforms"""
    # 1. Test basic vmap
    batched_fn = vmap(fn)
    batched_inputs = [x.unsqueeze(0).expand(4, *x.shape) for x in sample_inputs]
    result = batched_fn(*batched_inputs)
    assert result.shape[0] == 4
    
    # 2. Test vmap + grad composition
    def loss_fn(x):
        return fn(x).sum()
    grad_fn = vmap(grad(loss_fn))
    grads = grad_fn(batched_inputs[0])
    assert grads.shape == batched_inputs[0].shape
    
    # 3. Compare against sequential baseline
    sequential_results = torch.stack([fn(*[inp[i] for inp in sample_inputs]) 
                                       for i in range(4)])
    torch.testing.assert_close(result, sequential_results, rtol=1e-4, atol=1e-4)

def test_numerical_gradients(fn, sample_inputs):
    """Verify gradients match finite differences"""
    from torch.autograd import gradcheck
    inputs = [x.double().requires_grad_(True) for x in sample_inputs]
    assert gradcheck(fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-3)
```

**Integration test with HMS-Net:**

```python
def test_chunked_scan_vmap():
    """Test that _chunked_scan works with vmap parallelization"""
    model = HMSNet(...)  # Your model
    x = torch.randn(16384, dim)  # Full batch
    chunks = x.chunk(4, dim=0)   # 4 chunks of 4096
    
    # Sequential baseline
    sequential_out = torch.cat([model._chunked_scan(c) for c in chunks])
    
    # vmap parallelized
    batched_scan = vmap(model._chunked_scan, chunk_size=4096)
    stacked_chunks = torch.stack(chunks)
    vmap_out = batched_scan(stacked_chunks).reshape(-1, dim)
    
    torch.testing.assert_close(sequential_out, vmap_out, rtol=1e-4, atol=1e-4)
```

## Alternative approaches if direct modification fails

**Option 1: Use pure PyTorch fallback**
The mamba2-torch repo supports disabling Triton kernels:
```python
config = Mamba2Config.from_pretrained(path)
config.use_triton_kernels = False  # Pure PyTorch mode
model = Mamba2ForCausalLM.from_pretrained(path, config=config)
```
This enables vmap compatibility at the cost of ~3-5× slower inference.

**Option 2: Manual parallel scan with torch.compile**
Instead of vmap, use batched Triton kernels directly:
```python
@torch.compile(mode="reduce-overhead")
def parallel_chunked_scan(chunks):
    # Process all chunks as a batched operation
    # Requires modifying Triton kernels to handle batch dimension
    pass
```

**Option 3: Use JAX implementation**
The `CosmoNaught/mamba2-jax` implementation has native vmap support via JAX's functional paradigm. Consider JAX for research/prototyping if PyTorch vmap compatibility proves too complex.

**Option 4: CUDA graph capture**
For fixed batch sizes, CUDA graphs can parallelize the sequential loop:
```python
# Capture the sequential processing as a CUDA graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for chunk in chunks:
        outputs.append(mamba_scan(chunk))
# Replay captured graph - parallel execution
g.replay()
```

## Rollback plan

If modifications introduce correctness issues:

1. **Git-based rollback**: All changes should be on a feature branch
   ```bash
   git checkout main -- src/mamba2_torch/ops/
   ```

2. **Runtime fallback**: Add environment variable toggle
   ```python
   USE_VMAP_COMPATIBLE = os.environ.get("MAMBA_VMAP_COMPAT", "0") == "1"
   
   if USE_VMAP_COMPATIBLE:
       from .ops_vmap import MambaChunkScanCombinedFn
   else:
       from .ops import MambaChunkScanCombinedFn
   ```

3. **Gradual rollout**: Modify one class at a time, test thoroughly before proceeding

## Expected performance impact

| Metric | Current | With vmap (chunk_size=4096) | Target |
|--------|---------|----------------------------|--------|
| Latency | 6.61s | ~3.5-4.5s | <3.3s |
| VRAM | 23.2GB | 30-35GB | <40GB |
| Throughput | 0.61 samples/sec | ~0.9-1.1 samples/sec | >1.2 samples/sec |

The **2× speedup target may be achievable** but depends heavily on how much of the sequential chunk processing is actually parallelizable. The vmap overhead for Triton kernels with manual batching rules adds latency compared to native batched operations, so the actual speedup may be closer to **1.5-1.8×**.

For the full 2× improvement, consider combining vmap with:
- `torch.compile` for the non-Triton portions
- Increasing the Triton kernel's internal parallelism
- Profile-guided optimization of the chunk size

The fundamental architectural limitation is that Mamba's state-space recurrence requires sequential state passing between chunks, which cannot be fully parallelized regardless of vmap support.