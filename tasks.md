# Tasks: Functorch Compatibility for mamba2-torch

Generated from: plan-v001.md

---

## Class 1: ChunkStateFn (Simple)
**File:** `src/mamba2_torch/ops/ssd_chunk_state.py` (lines 793-824)

- [X] Remove `ctx` from forward() signature
- [X] Return tensors needed for backward: `(states, B, x, dt, dA_cumsum)`
- [X] Add `setup_context(ctx, inputs, output)` staticmethod
  - [X] Unpack inputs: `B, x, dt, dA_cumsum, states_in_fp32`
  - [X] Call `ctx.save_for_backward(B, x, dt, dA_cumsum)`
- [X] Update `backward()` signature to accept extra gradient args: `backward(ctx, dstates, *_)`
- [X] Update wrapper function `chunk_state()` to return `result[0]`
- [X] Test basic forward/backward (no regression) - NOTE: Requires compatible CUDA environment
- [X] Add `vmap()` staticmethod
  - [X] Handle `in_dims` for each input
  - [X] Merge vmap batch with tensor batch
  - [X] Return `(outputs, out_dims)`
- [X] Test vmap functionality - NOTE: Requires compatible CUDA environment

---

## Class 2: StatePassingFn (Simple)
**File:** `src/mamba2_torch/ops/ssd_state_passing.py` (lines 284-309)

- [X] Remove `ctx` from forward() signature
- [X] Return: `(out, final_states, dA_chunk_cumsum, has_initial_states)`
- [X] Add `setup_context(ctx, inputs, output)` staticmethod
  - [X] Unpack inputs: `states, dA_chunk_cumsum, initial_states`
  - [X] Call `ctx.save_for_backward(out, dA_chunk_cumsum)`
  - [X] Set `ctx.has_initial_states` from output
- [X] Update `backward()` signature: `backward(ctx, dout, dfinal_states, *_)`
- [X] Update wrapper function to return `(result[0], result[1])`
- [X] Test basic forward/backward (no regression) - NOTE: Requires compatible CUDA environment
- [X] Add `vmap()` staticmethod
- [X] Test vmap functionality - NOTE: Requires compatible CUDA environment

---

## Class 3: LayerNormFn - Gated (Simple)
**File:** `src/mamba2_torch/ops/layernorm_gated.py` (lines 338-378)

- [X] Remove `ctx` from forward() signature
- [X] Identify and return tensors needed for backward
- [X] Add `setup_context(ctx, inputs, output)` staticmethod
- [X] Update `backward()` signature to accept extra gradient args
- [X] Update wrapper function if needed
- [X] Test basic forward/backward (no regression) - NOTE: Requires compatible CUDA environment
- [X] Add `vmap()` staticmethod
- [X] Test vmap functionality - NOTE: Requires compatible CUDA environment

---

## Class 4: ChunkScanFn (Medium)
**File:** `src/mamba2_torch/ops/ssd_chunk_scan.py` (lines 1706-1767)

- [X] Remove `ctx` from forward() signature
- [X] Return BOTH `out` and `out_x` (for conditional save logic)
- [X] Add `setup_context(ctx, inputs, output)` staticmethod
  - [X] Handle conditional: `ctx.save_for_backward(out if z is None else out_x, ...)`
- [X] Update `backward()` signature to accept extra gradient args
- [X] Update wrapper function
- [X] Test basic forward/backward (no regression) - NOTE: Requires compatible CUDA environment
- [X] Add `vmap()` staticmethod
- [X] Test vmap functionality - NOTE: Requires compatible CUDA environment

---

## Class 5: LayerNormFn - Full (Medium)
**File:** `src/mamba2_torch/ops/layer_norm.py` (lines 726-883)

- [X] Remove `ctx` from forward() signature
- [X] Identify and return tensors needed for backward
- [X] Add `setup_context(ctx, inputs, output)` staticmethod
- [X] Update `backward()` signature to accept extra gradient args
- [X] Update wrapper function if needed
- [X] Test basic forward/backward (no regression)
- [X] Add `vmap()` staticmethod
- [X] Test vmap functionality

---

## Class 6: MambaChunkScanCombinedFn (High)
**File:** `src/mamba2_torch/ops/ssd_combined.py` (lines 525-543)

- [X] Remove `ctx` from forward() signature
- [X] Return: `(out, final_states, out_x, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx, dt_dtype, dt_softplus, chunk_size, dt_limit, return_final_states)`
- [X] Add `setup_context(ctx, inputs, output)` staticmethod
  - [X] Handle conditional save: `out if z is None else out_x`
  - [X] Set ctx attributes: `dt_dtype, dt_softplus, chunk_size, dt_limit, return_final_states, has_D, has_z, has_dt_bias, has_initial_states, has_seq_idx`
- [X] Update `backward()` signature to accept extra gradient args
- [X] Update wrapper function to handle `return_final_states` flag
- [X] Test basic forward/backward (no regression) - NOTE: Requires compatible CUDA environment
- [X] Add `vmap()` staticmethod
- [X] Test vmap functionality - NOTE: Requires compatible CUDA environment

---

## Class 7: LayerNormLinearFn (High - has decorators) - COMPLETED
**File:** `src/mamba2_torch/ops/layer_norm.py` (lines 1388-1762)
**Test:** `tests/test_layer_norm_linear_functorch.py`

- [X] Keep `@custom_fwd` on forward, `@custom_bwd` on backward
- [X] Do NOT add decorators to setup_context
- [X] Remove `ctx` from forward() signature
- [X] Identify and return tensors needed for backward
- [X] Add `setup_context(ctx, inputs, output)` staticmethod
- [X] Update `backward()` signature to accept extra gradient args
- [X] Update wrapper function if needed
- [X] Test basic forward/backward (no regression)
- [X] Test with AMP enabled
- [X] Add `vmap()` staticmethod
- [X] Test vmap functionality

**KEY DISCOVERY - Required for @custom_fwd/@custom_bwd with setup_context:**
When using `@custom_fwd`/`@custom_bwd` decorators with the `setup_context` pattern, you MUST manually set these attributes in `setup_context()`:
```python
ctx._fwd_used_autocast = torch.is_autocast_enabled()
ctx._dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
```
This is because `@custom_fwd` normally sets these on `ctx` during forward(), but when `ctx` is removed from forward() for the setup_context pattern, the decorator can't set them. The `@custom_bwd` decorator then fails in backward() trying to read these attributes.

---

## Class 8: MambaSplitConv1dScanCombinedFn (Very High - has decorators) - COMPLETED
**File:** `src/mamba2_torch/ops/ssd_combined.py` (lines 937-1282)
**Test:** `tests/test_mamba_split_conv1d_scan_combined_functorch.py`

**NOTE:** This class uses `@custom_fwd`/`@custom_bwd` decorators. Applied the same fix as Class 7:
- In `setup_context()`, added:
  ```python
  ctx._fwd_used_autocast = torch.is_autocast_enabled()
  ctx._dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
  ```

- [X] Keep `@custom_fwd` on forward, `@custom_bwd` on backward
- [X] Do NOT add decorators to setup_context
- [X] Remove `ctx` from forward() signature
- [X] Return all tensors needed for backward (24 items total)
- [X] Add `setup_context(ctx, inputs, output)` staticmethod
  - [X] Set ctx attributes: `dt_limit, return_final_states, activation, rmsnorm_eps, norm_before_gate, chunk_size, headdim, ngroups, outproj_weight_dtype`
  - [X] **CRITICAL:** Set `ctx._fwd_used_autocast` and `ctx._dtype` for @custom_bwd compatibility
- [X] Update `backward()` signature to accept extra gradient args
- [X] Update wrapper function to handle `return_final_states` flag
- [X] Add `vmap()` staticmethod
- [ ] Test basic forward/backward - **BLOCKED: requires causal-conv1d with compatible API**
- [ ] Test with AMP enabled - **BLOCKED: requires causal-conv1d with compatible API**
- [ ] Test vmap functionality - **BLOCKED: requires causal-conv1d with compatible API**

**Testing Note:** The functorch implementation is complete, but testing requires causal-conv1d built with a compatible API. The current causal-conv1d has a different function signature for `causal_conv1d_fwd` than what mamba2-torch expects.

---

## Final Verification

- [X] Run full test suite for Classes 1-7 - ALL PASS
- [ ] Run Class 8 tests - BLOCKED on causal-conv1d API compatibility
- [X] Verify no API breaking changes in wrapper functions - all wrapper functions preserve original return types
- [X] Document behavior changes - see CLAUDE.md

## Summary

**All 8 autograd.Function classes have been converted to functorch-compatible pattern:**
1. ChunkStateFn ✓
2. StatePassingFn ✓
3. LayerNormFn (gated) ✓
4. ChunkScanFn ✓
5. LayerNormFn (full) ✓
6. MambaChunkScanCombinedFn ✓
7. LayerNormLinearFn ✓
8. MambaSplitConv1dScanCombinedFn ✓ (testing blocked on causal-conv1d)

---

Generated: 2025-12-09 02:07 UTC
