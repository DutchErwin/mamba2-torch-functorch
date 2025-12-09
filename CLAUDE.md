# Claude Code Project Notes

## Project: mamba2-torch-functorch

This project adds functorch compatibility to the mamba2-torch library.

## Recent Changes

### Class 1: ChunkStateFn (COMPLETED)
**File:** `src/mamba2_torch/ops/ssd_chunk_state.py`

Converted `ChunkStateFn` to be functorch-compatible by:

1. **Removed `ctx` from `forward()` signature** - The new functorch-compatible pattern doesn't pass ctx to forward.

2. **Added `setup_context()` staticmethod** - Handles saving tensors for backward pass and setting context attributes.

3. **Updated `backward()` signature** - Now accepts extra gradient args with `*_` to handle gradients for all returned tensors.

4. **Updated wrapper function** - `chunk_state()` now returns `result[0]` to extract only the user-facing output.

5. **Used `.view_as()` for returned tensors** - PyTorch requires views (not original tensors) when returning inputs from forward() with setup_context. Returns `B.view_as(B)` instead of `B`.

6. **Added `vmap()` staticmethod** - Enables `torch.vmap` compatibility by:
   - Moving batch dims to front position
   - Broadcasting non-batched inputs
   - Merging vmap batch with tensor batch for Triton kernel compatibility
   - Unmerging batch dimensions in outputs

### Class 2: StatePassingFn (COMPLETED)
**File:** `src/mamba2_torch/ops/ssd_state_passing.py`

Converted `StatePassingFn` to be functorch-compatible by:

1. **Removed `ctx` from `forward()` signature** - The new functorch-compatible pattern doesn't pass ctx to forward.

2. **Returns multiple outputs** - Returns `(out, final_states, out_saved, dA_chunk_cumsum_saved, has_initial_states)`:
   - `out` and `final_states` are user-facing outputs
   - `out_saved` and `dA_chunk_cumsum_saved` are views for backward (using `.view_as()`)
   - `has_initial_states` is a boolean flag

3. **Added `setup_context()` staticmethod** - Unpacks inputs and outputs, saves tensors for backward, sets `ctx.has_initial_states`.

4. **Updated `backward()` signature** - Now accepts `(ctx, dout, dfinal_states, *_)` to handle gradients for all returned outputs.

5. **Updated wrapper function** - `state_passing()` now returns `(result[0], result[1])` to extract only user-facing outputs.

6. **Added `vmap()` staticmethod** - Enables `torch.vmap` compatibility with support for optional `initial_states` parameter.

### Class 3: LayerNormFn - Gated (COMPLETED)
**File:** `src/mamba2_torch/ops/layernorm_gated.py`

Converted `LayerNormFn` to be functorch-compatible by:

1. **Removed `ctx` from `forward()` signature** - The new functorch-compatible pattern doesn't pass ctx to forward.

2. **Returns multiple outputs** - Returns `(y, x_saved, weight_saved, bias_saved, mean, rstd, z_saved, x_shape_og)`:
   - `y` is the user-facing output
   - Saved tensors use `.view_as()` for PyTorch compatibility
   - `x_shape_og` is passed for shape restoration in backward

3. **Added `setup_context()` staticmethod** - Handles conditional saving based on whether `bias` and `z` are None. Sets context attributes: `has_bias`, `has_z`, `x_shape_og`, `eps`, `group_size`, `norm_before_gate`, `is_rms_norm`.

4. **Updated `backward()` signature** - Now accepts `(ctx, dy, *_)` to handle gradients for all returned outputs. Unpacks saved tensors based on `ctx.has_bias` and `ctx.has_z` flags.

5. **Updated wrapper functions** - Both `layernorm_fn()` and `rmsnorm_fn()` now return `result[0]` to extract only the user-facing output.

6. **Added `vmap()` staticmethod** - Enables `torch.vmap` compatibility by:
   - Moving batch dims to front position
   - Broadcasting non-batched inputs (weight, bias typically shared)
   - Merging vmap batch with tensor batch for Triton kernel compatibility
   - Handling optional `z` gating parameter
   - Unmerging batch dimensions in outputs

### Class 4: ChunkScanFn (COMPLETED)
**File:** `src/mamba2_torch/ops/ssd_chunk_scan.py`

Converted `ChunkScanFn` to be functorch-compatible by:

1. **Removed `ctx` from `forward()` signature** - The new functorch-compatible pattern doesn't pass ctx to forward.

2. **Returns multiple outputs** - Returns `(out, out_x, B, C, CB, x, dt, dA_cumsum, prev_states, D, z)`:
   - `out` is the user-facing output
   - `out_x` is returned for conditional save logic (used when `z is not None`)
   - All saved tensors use `.view_as()` for PyTorch compatibility
   - Optional tensors (`D`, `z`, `out_x`) return `None` when not provided

3. **Added `setup_context()` staticmethod** - Handles conditional saving: saves `out` if `z is None`, else saves `out_x`. Sets context attributes: `has_z`, `has_D`.

4. **Updated `backward()` signature** - Now accepts `(ctx, dout, *_)` to handle gradients for all returned outputs. Uses `ctx.has_z` instead of checking `z is not None`.

5. **Updated wrapper function** - `chunk_scan()` now returns `result[0]` to extract only the user-facing output.

6. **Added `vmap()` staticmethod** - Enables `torch.vmap` compatibility by:
   - Moving batch dims to front position
   - Broadcasting non-batched inputs
   - Merging vmap batch with tensor batch for Triton kernel compatibility
   - Handling optional `D` and `z` parameters
   - Unmerging batch dimensions in outputs

### Class 5: LayerNormFn - Full (COMPLETED)
**File:** `src/mamba2_torch/ops/layer_norm.py`

Converted `LayerNormFn` to be functorch-compatible by:

1. **Removed `ctx` from `forward()` signature** - The new functorch-compatible pattern doesn't pass ctx to forward.

2. **Returns consistent tuple structure** - Returns 17 items:
   - User-facing outputs: `(y, y1, residual_out, dropout_mask, dropout_mask1)`
   - Saved tensors for backward: `(residual_out_saved, weight_saved, bias_saved, weight1_saved, bias1_saved, rowscale_saved, seeds_saved, mean_saved, rstd_saved)`
   - Context attributes: `(x_shape_og, has_residual, has_x1)`
   - All saved tensors use `.view_as()` for PyTorch compatibility
   - Optional tensors return `None` when not provided

3. **Added `setup_context()` staticmethod** - Conditionally saves tensors based on which are present. Sets context attributes: `x_shape_og`, `eps`, `dropout_p`, `is_rms_norm`, `has_residual`, `has_x1`, `prenorm`, `x_dtype`, plus flags for optional tensors (`has_residual_out`, `has_bias`, `has_weight1`, `has_bias1`, `has_rowscale`, `has_seeds`, `has_mean`).

4. **Updated `backward()` signature** - Now accepts `(ctx, dy, dy1, dresidual_grad, ddropout_mask, ddropout_mask1, *_)`. Unpacks saved tensors based on `ctx.has_*` flags using an index counter.

5. **Updated wrapper functions** - Both `layer_norm_fn()` and `rms_norm_fn()` extract user-facing outputs and reconstruct the original variable return structure based on `prenorm`, `weight1`, and `return_dropout_mask` flags.

6. **Added `vmap()` staticmethod** - Enables `torch.vmap` compatibility by:
   - Moving batch dims to front position
   - Broadcasting non-batched inputs (weight, bias typically shared)
   - Merging vmap batch with tensor batch for Triton kernel compatibility
   - Handling many optional parameters (`residual`, `x1`, `weight1`, `bias1`, `rowscale`)
   - Unmerging batch dimensions in all 17 outputs

### Class 6: MambaChunkScanCombinedFn (COMPLETED)
**File:** `src/mamba2_torch/ops/ssd_combined.py`

Converted `MambaChunkScanCombinedFn` to be functorch-compatible by:

1. **Removed `ctx` from `forward()` signature** - The new functorch-compatible pattern doesn't pass ctx to forward.

2. **Returns 19-item tuple** - Returns `(out, final_states, out_x, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx, dt_dtype, dt_softplus, chunk_size, dt_limit, return_final_states)`:
   - User-facing outputs: `out` (always), `final_states` (when `return_final_states=True`)
   - `out_x` is returned for conditional save logic (used when `z is not None`)
   - All saved tensors use `.view_as()` for PyTorch compatibility
   - Optional tensors (`D`, `z`, `dt_bias`, `initial_states`, `seq_idx`) return `None` when not provided
   - Non-tensor values (`dt_dtype`, `dt_softplus`, `chunk_size`, `dt_limit`, `return_final_states`) passed through for setup_context

3. **Added `setup_context()` staticmethod** - Handles conditional saving: saves `out` if `z is None`, else saves `out_x`. Sets context attributes: `dt_dtype`, `dt_softplus`, `chunk_size`, `dt_limit`, `return_final_states`, `has_D`, `has_z`, `has_dt_bias`, `has_initial_states`, `has_seq_idx`.

4. **Updated `backward()` signature** - Now accepts `(ctx, dout, dfinal_states, *_)` to handle gradients for all returned outputs. Uses `ctx.return_final_states` to determine if `dfinal_states` should be passed to the backward function.

5. **Updated wrapper function** - `mamba_chunk_scan_combined()` now extracts user-facing outputs and returns based on `return_final_states` flag: either just `out` or `(out, final_states)`.

6. **Added `vmap()` staticmethod** - Enables `torch.vmap` compatibility by:
   - Moving batch dims to front position
   - Broadcasting non-batched inputs (`x`, `dt`, `B`, `C`, `z`, `initial_states`, `seq_idx`)
   - Keeping shared parameters unbroadcast (`A`, `D`, `dt_bias`)
   - Merging vmap batch with tensor batch for Triton kernel compatibility
   - Handling all optional parameters
   - Unmerging batch dimensions in outputs (except shared parameters)

### Class 7: LayerNormLinearFn (COMPLETED)
**File:** `src/mamba2_torch/ops/layer_norm.py`

Converted `LayerNormLinearFn` to be functorch-compatible by:

1. **Removed `ctx` from `forward()` signature** - The new functorch-compatible pattern doesn't pass ctx to forward. Kept `@custom_fwd` decorator.

2. **Returns 15-item tuple** - Returns `(out, residual_out_reshaped, residual_out_saved, norm_weight_saved, norm_bias_saved, linear_weight_saved, mean_saved, rstd_saved, x_shape_og, eps, is_rms_norm, has_residual, prenorm, x_dtype, linear_bias_is_none)`:
   - User-facing outputs: `out` (always), `residual_out_reshaped` (when `prenorm=True`)
   - All saved tensors use `.view_as()` for PyTorch compatibility
   - Optional tensors (`norm_bias`, `mean`) return `None` when not provided
   - Non-tensor values passed through for setup_context

3. **Added `setup_context()` staticmethod** - Conditionally saves tensors based on which are present. Sets context attributes including `has_norm_bias`, `has_mean`. **Critical for @custom_bwd compatibility**: Sets `ctx._fwd_used_autocast` and `ctx._dtype` which are required by the `@custom_bwd` decorator when using `setup_context`.

4. **Kept `@custom_bwd` on `backward()`** - Updated signature to `(ctx, dout, dresidual_out_grad, *_)` to handle gradients for all returned outputs. Unpacks saved tensors based on `ctx.has_*` flags.

5. **Updated wrapper function** - `layer_norm_linear_fn()` extracts user-facing outputs and returns based on `prenorm` flag.

6. **Added `vmap()` staticmethod** - Enables `torch.vmap` compatibility by:
   - Moving batch dims to front position
   - Broadcasting non-batched inputs (norm_weight, norm_bias, linear_weight, linear_bias typically shared)
   - Merging vmap batch with tensor batch for Triton kernel compatibility
   - Handling optional residual parameter
   - Unmerging batch dimensions in outputs

**Key Discovery**: When using `@custom_fwd`/`@custom_bwd` decorators with `setup_context`, you must manually set `ctx._fwd_used_autocast = torch.is_autocast_enabled()` and `ctx._dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None` in `setup_context` because the decorator can't set these when `ctx` is not passed to `forward()`.

### Class 8: MambaSplitConv1dScanCombinedFn (COMPLETED)
**File:** `src/mamba2_torch/ops/ssd_combined.py`

Converted `MambaSplitConv1dScanCombinedFn` to be functorch-compatible by:

1. **Removed `ctx` from `forward()` signature** - Kept `@custom_fwd` decorator.

2. **Returns 24-item tuple** - Returns `(out, final_states, out_x, zxbcdt, conv1d_weight, conv1d_bias, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias, outproj_weight_dtype, dt_limit, return_final_states, activation, rmsnorm_eps, norm_before_gate, chunk_size, headdim, ngroups)`:
   - User-facing outputs: `out` (always), `final_states` (when `return_final_states=True`)
   - All saved tensors use `.view_as()` for PyTorch compatibility
   - Optional tensors return `None` when not provided
   - Non-tensor values passed through for setup_context

3. **Added `setup_context()` staticmethod** - Saves tensors for backward. Sets context attributes: `outproj_weight_dtype`, `dt_limit`, `return_final_states`, `activation`, `rmsnorm_eps`, `norm_before_gate`, `chunk_size`, `headdim`, `ngroups`. **Critical for @custom_bwd compatibility**: Sets `ctx._fwd_used_autocast` and `ctx._dtype`.

4. **Updated `backward()` signature** - Now accepts `(ctx, dout, dfinal_states, *_)` to handle gradients for all returned outputs. Uses `dfinal_states_for_bwd` based on `ctx.return_final_states`.

5. **Updated wrapper function** - `mamba_split_conv1d_scan_combined()` now extracts user-facing outputs and returns based on `return_final_states` flag.

6. **Added `vmap()` staticmethod** - Enables `torch.vmap` compatibility by:
   - Moving batch dims to front position
   - Broadcasting batched inputs (`zxbcdt`, `initial_states`, `seq_idx`)
   - Keeping shared parameters unbroadcast (`conv1d_weight`, `conv1d_bias`, `dt_bias`, `A`, `D`, `rmsnorm_weight`, `outproj_weight`, `outproj_bias`)
   - Merging vmap batch with tensor batch for Triton kernel compatibility
   - Unmerging batch dimensions in outputs

**Note:** Testing requires `causal-conv1d` package. If tests fail with API mismatch errors, it may indicate incompatibility between the causal-conv1d version and the expected API in mamba2-torch.

## Testing Notes

- Tests require a CUDA-compatible environment (Triton kernels only run on CUDA)
- Test files:
  - `tests/test_chunk_state_functorch.py` (Class 1)
  - `tests/test_state_passing_functorch.py` (Class 2)
  - `tests/test_layernorm_gated_functorch.py` (Class 3)
  - `tests/test_chunk_scan_functorch.py` (Class 4)
  - `tests/test_layer_norm_functorch.py` (Class 5)
  - `tests/test_mamba_chunk_scan_combined_functorch.py` (Class 6)
  - `tests/test_layer_norm_linear_functorch.py` (Class 7)
  - `tests/test_mamba_split_conv1d_scan_combined_functorch.py` (Class 8) - requires compatible causal-conv1d
- Tests for Classes 1-7 pass with hmsnet2025 conda environment (PyTorch 2.9.1+cu129)
- Class 8 test requires causal-conv1d with compatible API

## Implementation Pattern

The functorch-compatible pattern for `torch.autograd.Function`:

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

    @staticmethod
    def vmap(info, in_dims, *args):
        # Handle vmap batch dimension merging
        # Return (outputs, out_dims)
        pass
```

## Status

**All 8 classes have been converted to functorch-compatible pattern.**

See `tasks.md` for the complete task list with all items checked off.
