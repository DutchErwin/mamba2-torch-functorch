# Mamba2-torch Functorch Compatibility Investigation

## Objective
Enable `torch.vmap` parallelization for mamba2-torch to eliminate sequential chunk processing in HMS-Net's `_chunked_scan` function.

**Expected Outcome:** 2-5x throughput improvement by parallelizing SSM operations across spatial locations.

---

## Environment Specifications

| Component | Version | Notes |
|-----------|---------|-------|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell | 96GB VRAM, sm_120 compute |
| **PyTorch** | 2.9.1+cu129 | CUDA 12.9, required for Blackwell |
| **Triton** | 3.5.1 | Blackwell kernel support |
| **Python** | 3.12+ | Required for PyTorch 2.9 |
| **mamba2-torch** | latest (pip) | Triton-based Mamba-2 implementation |
| **OS** | Linux (WSL2) | Ubuntu on Windows |

**Repository:** https://github.com/vasqu/mamba2-torch

---

## Current Problem

When attempting to use `torch.vmap` on mamba2-torch SSM operations, we get:

```
RuntimeError: In order to use an autograd.Function with functorch transforms
(vmap, grad, jvp, jacrev, ...), it must override the setup_context staticmethod.
```

**Why This Matters:**
Our `_chunked_scan` function processes SSM operations in a sequential loop:
```python
# Current: 8 sequential kernel launches (after Phase 5 optimization)
for i in range(0, B_total, chunk_size):  # chunk_size=2048
    chunk_x = x[i:i+chunk_size]
    out = ssm(chunk_x, mask[i:i+chunk_size])  # Sequential kernel launch
    outputs.append(out)
```

With vmap, this could become a single parallelized operation:
```python
# Goal: Single parallel kernel launch
out = torch.vmap(ssm)(x, mask)  # All chunks processed in parallel
```

---

## Research Questions

### 1. Locate autograd.Function Classes in mamba2-torch

**Question:** Where are the `torch.autograd.Function` classes that need modification?

**Expected locations in mamba2-torch repository:**
```
mamba2_torch/
├── ops/
│   ├── ssd_bmm.py           # State Space Duality BMM operations
│   ├── ssd_chunk_scan.py    # Chunked scan implementation
│   └── ssd_state_passing.py # State passing between chunks
├── modules/
│   └── mamba2.py            # Main Mamba2 module
└── triton/
    └── *.py                 # Triton kernel implementations
```

**Research needed:**
- List all classes inheriting from `torch.autograd.Function`
- Identify which ones are in the forward/backward path for Mamba2
- Determine dependencies between these classes

---

### 2. Functorch Compatibility Pattern

**Question:** What is the exact pattern for adding functorch support?

**PyTorch Documentation:** https://pytorch.org/docs/main/notes/extending.func.html

**Required Changes (from PyTorch docs):**

**BEFORE (Legacy pattern - incompatible with vmap):**
```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x
```

**AFTER (Functorch-compatible pattern):**
```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, y):  # No ctx parameter!
        return x * y

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        return grad_output * y, grad_output * x
```

**Key changes:**
1. Remove `ctx` from `forward()` signature
2. Add `setup_context(ctx, inputs, output)` staticmethod
3. Move `ctx.save_for_backward()` from `forward` to `setup_context`

**Research needed:**
- Verify this pattern works with Triton kernel calls
- Check if `ctx.mark_non_differentiable()` needs special handling
- Understand memory implications of the new pattern

---

### 3. Existing Work

**Question:** Are there existing PRs, forks, or implementations with functorch support?

**Check these sources:**
1. **mamba2-torch GitHub:** https://github.com/vasqu/mamba2-torch/issues
2. **mamba-ssm GitHub:** https://github.com/state-spaces/mamba/issues
3. **PyTorch Forums:** Search for "mamba vmap" or "ssm functorch"
4. **Academic implementations:** Check papers citing Mamba-2

**Research needed:**
- Any open issues about vmap compatibility
- Any PRs attempting to add functorch support
- Alternative Mamba implementations with vmap support

---

### 4. Memory Implications

**Question:** What are the memory implications of using vmap?

**Our context:**
- 96GB VRAM available
- Current usage: ~23GB (24% utilization)
- 72GB headroom for vmap overhead

**Concerns:**
- vmap may replicate certain buffers across the batch dimension
- Need to understand vmap's memory overhead model
- May need to vmap over smaller groups instead of full batch

**Research needed:**
- vmap memory overhead documentation
- Best practices for memory-efficient vmap usage
- Whether to vmap over chunks (e.g., 4096) vs full batch (16384)

---

### 5. Step-by-Step Modification Plan

**Question:** What is the complete modification plan for mamba2-torch?

**Deliverable requested:**
1. List of files requiring changes (with line numbers if possible)
2. Estimated lines of code per file
3. Order of changes (dependencies)
4. Testing strategy for each change
5. Rollback plan if issues arise

---

## Our Codebase Integration

Once mamba2-torch is functorch-compatible, we would modify:

**File:** `hmsnet/core/mask_gated_ssm.py`

```python
# BEFORE: Sequential chunked processing
@torch.compiler.disable
def _chunked_scan(self, ssm, x, mask, reverse=False):
    outputs = []
    for i in range(0, B_total, chunk_size):
        chunk_x = x[i:i+chunk_size]
        out = ssm(chunk_x, mask[i:i+chunk_size])
        outputs.append(out)
    return torch.cat(outputs, dim=0)

# AFTER: Vectorized with vmap
def _vectorized_scan(self, ssm, x, mask, reverse=False):
    # vmap over the batch dimension
    batched_ssm = torch.vmap(lambda xi, mi: ssm(xi.unsqueeze(0), mi.unsqueeze(0)).squeeze(0))
    return batched_ssm(x, mask)
```

---

## Benchmark Requirements

Any solution must be validated with:

```python
python -c "
import torch, time
from hmsnet.core.factorized_block import HMSNetBackbone

model = HMSNetBackbone(in_channels=5, embed_dim=512, num_layers=12,
                       d_state=64, use_checkpoint=True).cuda().bfloat16()
x = torch.randn(4, 5, 16, 256, 256, device='cuda', dtype=torch.bfloat16)
mask = torch.zeros(4, 1, 16, 256, 256, device='cuda', dtype=torch.bfloat16)

# Warmup
for _ in range(2):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, _ = model(x, mask)
        out.mean().backward()
    model.zero_grad()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

times = []
for _ in range(5):
    start = time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, _ = model(x, mask)
        out.mean().backward()
    torch.cuda.synchronize()
    times.append(time.time() - start)
    model.zero_grad()

print(f'Time: {sum(times)/len(times):.2f}s')
print(f'VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f}GB')
"
```

**Current baseline (after Phase 5):**
- Time: 6.61s
- VRAM: 23.2GB
- Throughput: 0.61 samples/sec

**Target (with vmap):**
- Time: <3.3s (2x improvement)
- VRAM: <40GB (acceptable on 96GB)
- Throughput: >1.2 samples/sec

---

## Deliverables Requested

From deep research, please provide:

1. **File list:** All mamba2-torch files containing `torch.autograd.Function`
2. **Code examples:** Before/after for each class needing modification
3. **Dependency analysis:** Order of changes and potential breaking changes
4. **Testing strategy:** How to verify correctness after modifications
5. **Memory analysis:** Expected vmap memory overhead
6. **Alternative approaches:** If direct modification isn't feasible

---

## Why This is Achievable

1. **Clear error message:** PyTorch tells us exactly what's missing (`setup_context`)
2. **Well-documented pattern:** PyTorch docs provide the exact transformation
3. **Limited scope:** Only `autograd.Function` classes need changes
4. **Testable incrementally:** Can verify each class independently
5. **Fallback available:** Can keep chunked scan as backup

---

*Created: December 8, 2024*
*For: Deep Research Investigation*
*Priority: HIGH - Primary optimization track for Claude Code implementation*
