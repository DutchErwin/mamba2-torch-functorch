# mamba2-torch-functorch

**Mamba2 with `torch.vmap` / functorch compatibility for batched operations**

> A fork of [vasqu/mamba2-torch](https://github.com/vasqu/mamba2-torch) adding full `torch.func.vmap` support for all Triton-accelerated operations.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Key Features

- **Full `torch.vmap` compatibility** - All 8 custom `torch.autograd.Function` classes converted to functorch-compatible pattern
- **Gradient support with vmap** - Compute per-sample gradients, Jacobians, and Hessians
- **Batched parameter sweeps** - Run multiple model configurations in parallel
- **PyTorch 2.x native** - Uses modern `torch.func` APIs (replaces deprecated `functorch` package)
- **Preserves all optimizations** - Triton kernels, causal-conv1d CUDA, pure PyTorch fallback

### Use Cases

- **Neural Architecture Search** - Evaluate multiple Mamba configurations simultaneously
- **Ensemble Models** - Batch process across ensemble members
- **Per-sample Gradients** - Required for differential privacy, influence functions
- **Meta-learning** - MAML-style algorithms with batched inner loops
- **Bayesian Inference** - Batched posterior sampling

---

## What's Different From Original mamba2-torch?

The original [mamba2-torch](https://github.com/vasqu/mamba2-torch) uses `torch.autograd.Function` with the traditional `ctx`-in-forward pattern, which is **incompatible with `torch.vmap`**. This fork converts all autograd functions to use:

1. **`setup_context()` pattern** - Separates context saving from forward computation
2. **`vmap()` staticmethod** - Handles batch dimension merging for Triton kernels
3. **Proper tensor views** - Uses `.view_as()` for saved tensors as required by PyTorch

### Converted Classes

| Class | File | Description |
|-------|------|-------------|
| `ChunkStateFn` | `ssd_chunk_state.py` | Chunk state computation |
| `StatePassingFn` | `ssd_state_passing.py` | State passing between chunks |
| `LayerNormFn` (gated) | `layernorm_gated.py` | Gated layer normalization |
| `ChunkScanFn` | `ssd_chunk_scan.py` | Chunk-wise selective scan |
| `LayerNormFn` (full) | `layer_norm.py` | Full layer normalization |
| `MambaChunkScanCombinedFn` | `ssd_combined.py` | Combined Mamba scan |
| `LayerNormLinearFn` | `layer_norm.py` | Fused layer norm + linear |
| `MambaSplitConv1dScanCombinedFn` | `ssd_combined.py` | Full Mamba2 block with conv1d |

---

## Installation

### Basic Installation

```bash
git clone https://github.com/DutchErwin/mamba2-torch-functorch.git
cd mamba2-torch-functorch
pip install -e .
```

### With causal-conv1d (Fastest Path)

For maximum performance, install [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d):

```bash
pip install causal-conv1d
```

#### Building causal-conv1d for Newer GPUs (Blackwell/SM 12.0+)

If you have a newer GPU (e.g., RTX 5090, Blackwell architecture), you may need to build from source:

```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d

# Set CUDA architecture for your GPU
export TORCH_CUDA_ARCH_LIST="12.0"  # For Blackwell
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

# If using a specific CUDA toolkit
export CUDA_HOME=/path/to/cuda-12.x

pip install -e .
```

### Environment Setup

For reproducible CUDA paths, create a `setup_env.sh`:

```bash
#!/bin/bash
export CUDA_HOME=/usr/local/cuda  # Adjust to your CUDA installation
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Source before running:
```bash
source setup_env.sh
```

---

## Usage

### Basic vmap Example

```python
import torch
from torch.func import vmap
from mamba2_torch.ops import mamba_chunk_scan_combined

# Model parameters
batch, seqlen, nheads, headdim = 2, 64, 4, 32
ngroups, dstate, chunk_size = 2, 16, 16
dim = nheads * headdim

# Create inputs
x = torch.randn(batch, seqlen, nheads, headdim, device='cuda')
dt = torch.randn(batch, seqlen, nheads, device='cuda')
A = torch.randn(nheads, device='cuda')
B = torch.randn(batch, seqlen, ngroups, dstate, device='cuda')
C = torch.randn(batch, seqlen, ngroups, dstate, device='cuda')

# Standard forward
out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size)
print(f"Standard output: {out.shape}")  # (2, 64, 4, 32)

# vmap over a batch of different inputs
vmap_batch = 3
x_batched = torch.randn(vmap_batch, batch, seqlen, nheads, headdim, device='cuda')

vmapped_fn = vmap(
    lambda x_: mamba_chunk_scan_combined(x_, dt, A, B, C, chunk_size),
    in_dims=0
)
out_vmapped = vmapped_fn(x_batched)
print(f"Vmapped output: {out_vmapped.shape}")  # (3, 2, 64, 4, 32)
```

### Per-sample Gradients

```python
import torch
from torch.func import vmap, grad

def loss_fn(x, dt, A, B, C, chunk_size, target):
    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size)
    return ((out - target) ** 2).mean()

# Compute gradient for each sample in batch
per_sample_grads = vmap(
    grad(loss_fn),
    in_dims=(0, 0, None, 0, 0, None, 0)
)(x_batched, dt_batched, A, B_batched, C_batched, chunk_size, targets)
```

### Using with HuggingFace Models

The original mamba2-torch HuggingFace integration still works:

```python
from transformers import AutoTokenizer
from mamba2_torch import Mamba2ForCausalLM, Mamba2Config

model = Mamba2ForCausalLM.from_pretrained("path/to/model").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"].to("cuda")
out = model.generate(input_ids, max_new_tokens=10)
```

See the original [mamba2-torch README](https://github.com/vasqu/mamba2-torch) for model conversion scripts and advanced usage.

---

## Troubleshooting

### CUDA/Triton Compilation Errors

**Problem:** Triton kernel compilation fails or produces incorrect results.

**Solutions:**
1. Ensure CUDA toolkit is properly installed and `nvcc` is in PATH
2. Set `CUDA_HOME` environment variable
3. For newer GPUs, ensure Triton supports your architecture

```bash
# Check CUDA version
nvcc --version

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### causal-conv1d API Errors

**Problem:** `TypeError` or `RuntimeError` when calling `causal_conv1d_cuda` functions.

**Cause:** The causal-conv1d API changed to require pre-allocated output tensors.

**Solution:** This fork already includes the updated API calls. If you're seeing errors, ensure you have the latest code:

```bash
git pull origin main
pip install -e .
```

### Tensor Stride Assertions

**Problem:** `AssertionError: assert dy.stride(-1) == 1`

**Cause:** Non-contiguous tensors passed to Triton kernels.

**Solution:** This is fixed in the current version. The fix adds `.contiguous()` calls before Triton kernel operations.

### vmap Returns NaN

**Problem:** vmap output contains NaN while regular forward works.

**Cause:** Tensor memory layout issues after reshape operations in vmap.

**Solution:** Fixed by adding `.contiguous()` after reshaping merged batch tensors in the `vmap()` staticmethod.

### Blackwell GPU (SM 12.0) Support

**Problem:** CUDA compilation errors on RTX 5090 or other Blackwell GPUs.

**Solution:**
```bash
# For causal-conv1d
export TORCH_CUDA_ARCH_LIST="12.0"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install -e /path/to/causal-conv1d

# For mamba2-torch (Triton handles this automatically)
# Just ensure you have CUDA 12.x toolkit
```

---

## Running Tests

```bash
# Set up environment
source setup_env.sh  # If you created one
conda activate your_env

# Run all functorch tests
python tests/test_chunk_state_functorch.py
python tests/test_state_passing_functorch.py
python tests/test_layernorm_gated_functorch.py
python tests/test_chunk_scan_functorch.py
python tests/test_layer_norm_functorch.py
python tests/test_mamba_chunk_scan_combined_functorch.py
python tests/test_layer_norm_linear_functorch.py
python tests/test_mamba_split_conv1d_scan_combined_functorch.py  # Requires causal-conv1d
```

All tests verify:
- Forward/backward correctness
- Gradient computation
- vmap consistency (comparing with manual loop)
- vmap with gradient computation
- AMP (automatic mixed precision) compatibility

---

## Technical Details

### The functorch-Compatible Pattern

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(x, y, flag=False):  # NO ctx parameter
        result = compute(x, y)
        return result, x.view_as(x), y.view_as(y)  # Return saved tensors

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
    def vmap(info, in_dims, x, y, flag):
        # Merge vmap batch with tensor batch for kernel compatibility
        # Call forward with merged tensor
        # Unmerge outputs
        return outputs, out_dims
```

### Key Implementation Notes

1. **`@custom_fwd`/`@custom_bwd` with `setup_context`**: When using AMP decorators, manually set `ctx._fwd_used_autocast` and `ctx._dtype` in `setup_context`.

2. **Tensor contiguity**: Always call `.contiguous()` after `rearrange()` or `reshape()` before passing to CUDA/Triton kernels.

3. **vmap batch merging**: Triton kernels don't understand vmap batch dimensions. Merge vmap batch with tensor batch, call kernel, then unmerge.

---

## Performance Notes

- **vmap overhead**: Small overhead for batch dimension handling, but enables operations not otherwise possible
- **Triton kernels**: First call compiles the kernel (may take a few seconds), subsequent calls are fast
- **Memory**: vmap doesn't create copies for shared parameters (weights, biases)

---

## Credits

This fork builds on the excellent work of:

- **[mamba2-torch](https://github.com/vasqu/mamba2-torch)** by vasqu - HuggingFace-compatible Mamba2
- **[Mamba](https://github.com/state-spaces/mamba)** by Tri Dao and Albert Gu - Original Mamba implementation
- **[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)** by Dao-AILab - Optimized causal convolution

## Citations

```bibtex
@inproceedings{mamba2,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}

@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and others",
    booktitle = "EMNLP: System Demonstrations",
    year = "2020"
}
```

---

## License

Same license as the original [mamba2-torch](https://github.com/vasqu/mamba2-torch).

---

**Keywords:** mamba2 functorch, torch.vmap mamba, SSM vmap, state space model batched, mamba per-sample gradients, functorch autograd.Function, vmap Triton kernels, torch.func.vmap compatibility, mamba neural architecture search, batched mamba inference
