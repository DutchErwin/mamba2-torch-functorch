# Plan v001: Functorch Compatibility for mamba2-torch

## Goal
Add `setup_context` AND `vmap` staticmethods to all `torch.autograd.Function` classes to enable functorch transforms (vmap, grad, jacrev, etc.).

## Error Being Fixed
```
RuntimeError: In order to use an autograd.Function with functorch transforms
(vmap, grad, jvp, jacrev, ...), it must override the setup_context staticmethod.
```

## Classes to Modify (8 total, ordered by complexity)

| # | Class | File | Lines | Complexity | Notes |
|---|-------|------|-------|------------|-------|
| 1 | ChunkStateFn | ssd_chunk_state.py | 793-824 | Simple | 4 saved tensors, no conditionals |
| 2 | StatePassingFn | ssd_state_passing.py | 284-309 | Simple | Returns tuple, has boolean ctx attr |
| 3 | LayerNormFn | layernorm_gated.py | 338-378 | Simple | 8 parameters |
| 4 | ChunkScanFn | ssd_chunk_scan.py | 1706-1767 | Medium | Conditional `out`/`out_x` saves |
| 5 | LayerNormFn | layer_norm.py | 726-883 | Medium | Many parameters, variable returns |
| 6 | MambaChunkScanCombinedFn | ssd_combined.py | 525-543 | High | Conditional saves/returns |
| 7 | LayerNormLinearFn | layer_norm.py | 983-1087 | High | Uses @custom_fwd/@custom_bwd |
| 8 | MambaSplitConv1dScanCombinedFn | ssd_combined.py | 740-894 | Very High | Decorators + complex logic |

## Transformation Pattern

### Before (incompatible):
```python
class ExampleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, flag=False):
        result = compute(x, y)
        ctx.save_for_backward(x, y)
        ctx.flag = flag
        return result

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        return grad_x, grad_y, None
```

### After (functorch-compatible):
```python
class ExampleFn(torch.autograd.Function):
    @staticmethod
    def forward(x, y, flag=False):  # NO ctx parameter
        result = compute(x, y)
        return result, x, y  # Return tensors needed for backward

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, flag = inputs
        result, x_saved, y_saved = output
        ctx.save_for_backward(x_saved, y_saved)
        ctx.flag = flag

    @staticmethod
    def backward(ctx, grad_out, *_):  # Accept gradients for ALL outputs
        x, y = ctx.saved_tensors
        return grad_x, grad_y, None
```

### Wrapper Function Update:
```python
def example_fn(x, y, flag=False):
    result = ExampleFn.apply(x, y, flag)
    return result[0]  # Extract only user-facing output
```

## vmap Staticmethod Pattern (Required for Triton Kernels)

Triton kernels cannot process `BatchedTensor` objects. Error:
```
Cannot access data pointer of Tensor that doesn't have storage
```

**Solution**: Implement manual `vmap()` staticmethod that:
1. Receives `in_dims` telling which dimension is batched for each input
2. Moves batch dimensions to consistent position (index 0)
3. Reshapes tensors to merge batch dim with existing batch dim
4. Calls the function on properly shaped tensors
5. Returns `(outputs, out_dims)` specifying batch dim position in outputs

### vmap Staticmethod Template:
```python
@staticmethod
def vmap(info, in_dims, *args):
    # Move batch dims to position 0
    def move_bdim_to_front(x, bdim):
        if bdim is None:
            return x
        return x.movedim(bdim, 0)

    processed_args = [move_bdim_to_front(arg, dim) for arg, dim in zip(args, in_dims)]

    # Get batch size from first batched input
    batch_size = None
    for arg, dim in zip(args, in_dims):
        if dim is not None:
            batch_size = arg.shape[dim]
            break

    # Merge vmap batch with tensor batch for Triton kernels
    # Call function, then unmerge batch in outputs

    result = ExampleFn.apply(*processed_args)

    # Return (outputs, out_dims)
    if isinstance(result, tuple):
        return result, (0,) * len(result)
    return result, 0
```

## Implementation Order Per Class

1. `setup_context` staticmethod (enables grad, jacrev)
2. Update `backward` to accept extra gradient args
3. Update wrapper function
4. Test basic forward/backward
5. `vmap` staticmethod (enables torch.vmap)
6. Test vmap functionality

## Special Considerations

### Conditional Tensor Saving (ChunkScanFn, MambaChunkScanCombinedFn)
Pattern: `ctx.save_for_backward(out if z is None else out_x, ...)`

Solution: Return BOTH `out` and `out_x`, do conditional in setup_context.

### @custom_fwd/@custom_bwd Decorators
- Keep decorators on `forward` and `backward`
- Do NOT add decorators to `setup_context`

### Variable Return Values (return_final_states flag)
Always return consistent tuple structure from forward; handle flag logic in wrapper.

## Testing Strategy

After each class modification:
```python
import torch
from torch.func import vmap

# 1. Test basic forward/backward (no regression)
result = fn(*inputs)
result.sum().backward()

# 2. Test vmap compatibility
batched_fn = vmap(lambda *args: fn(*[a.unsqueeze(0) for a in args]).squeeze(0))
batched_result = batched_fn(*inputs)
```

## Critical Files
- `src/mamba2_torch/ops/ssd_chunk_state.py` - ChunkStateFn
- `src/mamba2_torch/ops/ssd_state_passing.py` - StatePassingFn
- `src/mamba2_torch/ops/layernorm_gated.py` - LayerNormFn (gated)
- `src/mamba2_torch/ops/ssd_chunk_scan.py` - ChunkScanFn
- `src/mamba2_torch/ops/layer_norm.py` - LayerNormFn, LayerNormLinearFn
- `src/mamba2_torch/ops/ssd_combined.py` - MambaChunkScanCombinedFn, MambaSplitConv1dScanCombinedFn

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Breaking existing API | Update wrapper functions to preserve return types |
| @custom_fwd incompatibility | Test with AMP enabled after each decorator class |
| Incorrect gradient flow | Verify with `torch.autograd.gradcheck` |
| Memory increase from extra returns | Only return what's needed; tensors are references |
| vmap batch merging errors | Carefully track tensor shapes; add asserts |
| Non-batched inputs in vmap | Handle `in_dims=None` case by broadcasting |

---

Created: 2025-12-09 02:07:12 UTC
