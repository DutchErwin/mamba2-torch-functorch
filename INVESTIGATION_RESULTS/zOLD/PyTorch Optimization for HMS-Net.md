# Optimizing HMS-Net Training on NVIDIA Blackwell: A Deep Technical Analysis

## 1. Introduction and Architectural Context

The training of high-resolution video inpainting models, specifically those leveraging State Space Models (SSMs) like Mamba-2, presents a distinct class of computational challenges that differ fundamentally from standard Transformer workloads. The HMS-Net architecture, with its four-directional scan mechanism (temporal forward/backward, spatial forward/backward) and massive intermediate activation tensors, places extreme pressure on GPU memory subsystems and scheduling logic. This report provides an exhaustive technical analysis and optimization strategy for training HMS-Net on the NVIDIA RTX PRO 6000 (Blackwell architecture, compute capability `sm_120`).

The primary objective is to resolve the memory "Out of Memory" (OOM) bottlenecks and throughput latency issues currently hindering the scaling of the model to 411M parameters and beyond. The analysis is grounded in the specific architectural characteristics of the Blackwell GPU, the internal mechanics of the PyTorch 2.9+ compiler stack, and the mathematical properties of the Mamba-2 recurrence.

### 1.1 The Hardware Substrate: NVIDIA Blackwell (`sm_120`)

To optimize effectively, one must characterize the target hardware. The RTX PRO 6000, built on the Blackwell architecture, introduces several features critical to this workload that are not present in previous Ada Lovelace or Ampere generations.

*   **Tensor Memory Accelerator (TMA):** Blackwell enhances the TMA, a specialized hardware unit responsible for asynchronous data movement between global memory and shared memory. In the context of HMS-Net, where the "chunked sequential processing bottleneck" implies memory-bandwidth-bound operations, properly utilizing TMA can hide the latency of loading the next chunk of video tokens while the Streaming Multiprocessors (SMs) compute the current scan.
*   **Micro-Scaling Formats (MXFP8):** Unlike the standard FP8 (E4M3/E5M2) found in Hopper/Ada, Blackwell introduces native support for Micro-scaling (MX) formats. Standard FP8 applies a single scaling factor to an entire tensor (or row). This is often insufficient for SSMs, where the recurrent state $h_t$ can exhibit significant outliers that carry long-term dependencies. MXFP8 applies scaling factors to blocks of 32 elements. This block-level granularity allows for the aggressive quantization of the massive linear projection layers in HMS-Net—reducing the memory footprint of weights and activations by nearly 50% compared to BFloat16—without the catastrophic signal degradation associated with standard quantization of recurrent networks.
*   **L2 Cache Residency:** The Blackwell architecture features a significantly larger L2 cache compared to its predecessors. For models like Mamba-2, which perform repeated element-wise operations (scans) over large sequences, maximizing L2 residency is key to preventing global memory bandwidth saturation.

### 1.2 The Software Stack: PyTorch 2.9 and Triton 3.5

The software environment—PyTorch 2.9.1 and Triton 3.5.1—provides the necessary primitives to exploit this hardware. PyTorch 2.9 introduces refined control over the `torch.compile` backend (Inductor), allowing for more granular memory planning and kernel fusion strategies. Triton 3.5 includes specific optimizations for `sm_120`, including support for the new warp-level instructions and improved pipelining for scan operations.

This report will systematically address the four key technical challenges identified: `torch.compile` memory explosion, sequential chunk processing latency, massive intermediate activations, and the fusion layer bottleneck.

---

## 2. Advanced `torch.compile` Optimization for Blackwell

The reported issue—`torch.compile` with `mode="reduce-overhead"` causing OOM on a 96GB GPU—is a deterministic consequence of how CUDA Graphs interact with dynamic memory allocation in high-dimensional video models. Understanding this mechanism is the first step toward a solution.

### 2.1 The CUDA Graph Memory Pool Mechanism

When `mode="reduce-overhead"` is selected, PyTorch Inductor attempts to capture the entire model execution graph into a CUDA Graph. The primary goal of CUDA Graphs is to eliminate the CPU launch overhead (the time the CPU takes to tell the GPU what to do) by recording a sequence of kernel launches and replaying them as a single node.

However, CUDA Graphs require a **static memory address space**. To ensure safe execution without the overhead of dynamic `cudaMalloc` calls during the graph replay, the allocator must pre-allocate a monolithic memory pool. This pool must be large enough to handle the *worst-case* peak memory usage of the graph. In a standard eager execution, the PyTorch caching allocator aggressively reuses memory blocks. For example, the memory used for the output of Layer $N$ might be freed and immediately reused for the input of Layer $N+2$.

In a captured CUDA Graph for a model as large as HMS-Net, the graph allocator effectively "freezes" the life-cycle of every tensor required for the replay. For video inputs of shape `[B, C, T, H, W]`, the intermediate activations for the 4-directional scans are massive. Even if the "live" memory at any single nanosecond is only 40GB, the *union* of all memory addresses required over the lifetime of the forward pass might exceed 150GB. The graph allocator cannot perform the same dynamic defragmentation and opportunistic reuse as the eager allocator during the capture phase, leading to the observed OOM.

### 2.2 Recommended Compilation Strategies for HMS-Net

For Blackwell GPUs running memory-critical workloads, the standard "reduce-overhead" mode is often counter-productive. The following strategies provide a graduated path to optimization.

#### 2.2.1 Strategy A: `max-autotune-no-cudagraphs`

The most immediate fix is to decouple the kernel optimization from the graph capture mechanism.

```python
model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

*   **Mechanism:** This mode enables the Inductor compiler to perform aggressive kernel fusion (fusing pointwise operations into the Mamba scan, fusing element-wise gates, etc.) and uses Triton autotuning to select the best block sizes and warp configurations for the `sm_120` architecture. Crucially, it **disables** the CUDA Graph capture.
*   **Benefit:** This allows the standard PyTorch caching allocator to manage memory dynamically. It can reclaim memory from the "Temporal Forward" scan before allocating memory for the "Spatial Backward" scan, significantly reducing peak VRAM usage compared to the pre-allocated pool of `reduce-overhead`.
*   **Performance Trade-off:** There will be slightly higher CPU overhead compared to full graph capture, but on a massive model like HMS-Net, the GPU execution time likely dominates the CPU dispatch time, making this trade-off acceptable.

#### 2.2.2 Strategy B: Fine-Grained Inductor Configuration

If CPU overhead remains a bottleneck (e.g., if the batch size is small, like 4), one can attempt to re-enable CUDA Graphs but with strict memory planning controls. This requires manipulating `torch._inductor.config`.

```python
import torch._inductor.config as config

# 1. Enable aggressive memory freeing in generated kernels
# This forces the compiler to generate code that releases buffers
# as early as possible, potentially allowing the graph allocator to pack tighter.
config.triton.memory_planning = True

# 2. Disable the monolithic CUDA graph capture if it still OOMs,
# but keep other optimizations.
config.triton.cudagraphs = False

# 3. Tuning for Blackwell (sm_120)
# Blackwell has high coordinate descent capabilities for autotuning.
config.coordinate_descent_tuning = True
config.coordinate_descent_check_all = True
```

The `memory_planning` flag is particularly relevant. It instructs the Inductor backend to perform a liveness analysis of all intermediate tensors and generate a static allocation plan that minimizes the peak footprint. While `reduce-overhead` does this implicitly, setting it explicitly while disabling cudagraphs ensures you get the memory-efficient kernel schedule without the rigid memory pool requirement.

### 2.3 Handling Dynamic Control Flow and Graph Breaks

HMS-Net processes SSM scans in chunks. If this chunking is implemented via a Python loop (e.g., `for i in range(0, B_total, chunk_size)`), `torch.compile` will trigger a "Graph Break" at every iteration if it cannot statically determine the loop bound or unroll it.

*   **The Problem:** Graph breaks cause the compiler to exit the optimized graph, return to Python, and then re-enter a new graph for the next chunk. This fragmentation destroys the benefits of compilation and can lead to excessive memory usage as multiple compiled sub-graphs might hold onto their own memory contexts.
*   **Optimization with `torch.compiler.disable`:** For the specific chunking loop that is causing bottlenecks, it is often better to disable compilation for the outer loop driver while compiling the inner computation unit.

```python
class ChunkedSSM(torch.nn.Module):
    def __init__(self, ssm_block):
        super().__init__()
        self.ssm_block = torch.compile(ssm_block, mode="max-autotune-no-cudagraphs")

    @torch.compiler.disable
    def forward(self, x):
        # This loop runs in eager Python, avoiding graph breaks
        # The heavy lifting inside self.ssm_block is still compiled
        outputs = []
        for i in range(0, x.shape[0], 512):
            chunk = x[i:i+512]
            outputs.append(self.ssm_block(chunk))
        return torch.cat(outputs)
```

This "surgical compilation" ensures that the heavy compute (the SSM) is optimized, while the dynamic control flow (the chunking) does not confuse the compiler's graph capture logic.

### 2.4 Partial CUDA Graphs

PyTorch 2.9 allows for capturing partial CUDA Graphs for static subgraphs. If the `s_model` dimensions are fixed (e.g., 512 chunks), but the total number of chunks varies, the inner processing of a single chunk is a perfect candidate for a CUDA Graph.

By wrapping the inner `process_chunk` function with `torch.compile(mode="reduce-overhead", fullgraph=True)`, you force PyTorch to capture *just that chunk's execution* into a graph. The outer loop essentially launches the same CUDA Graph $N$ times. This is much more memory efficient than trying to capture the loop of $N$ iterations into one giant graph.

---

## 3. Memory Optimization Beyond Gradient Checkpointing

While gradient checkpointing is enabled, the 96GB limit is still being breached by the sheer volume of activations. We must move beyond standard checkpointing to advanced memory management techniques available in the PyTorch/CUDA ecosystem.

### 3.1 Advanced Memory Allocator Configurations

The PyTorch caching allocator is designed for speed, not minimal fragmentation. For workloads with variable-sized tensors (like chunked video processing where the last chunk might be smaller), fragmentation can cause OOM even when free memory exists.

**Recommendation: `expandable_segments`**
The `expandable_segments` setting is critical for Blackwell GPUs running large-model training. It allows the allocator to create memory segments that can be expanded virtually without requiring a contiguous physical allocation for the full expanded size initially.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
```

*   `expandable_segments:True`: This enables the CUDA driver's virtual memory management features to resize allocations. It prevents the scenario where a 2GB hole exists, but a 1.5GB tensor cannot be allocated because the hole is split into two non-contiguous 1GB segments.
*   `garbage_collection_threshold:0.8`: This instructs the allocator to start reclaiming unused segments when GPU memory usage hits 80%, rather than waiting for an OOM event. This proactive cleanup is essential for the long-running iterations of video training.

### 3.2 Selective and Checkpoint Sequential Strategies

Standard `torch.utils.checkpoint.checkpoint` (recomputing the forward pass during backward) saves memory but increases compute.

*   **`checkpoint_sequential`:** This utility is designed for `nn.Sequential` models. It divides the model into segments and only stores the activations at the boundaries of these segments. For a 16-layer HMS-Net, splitting it into 4 segments of 4 layers each can be a sweet spot.

```python
# Example usage for HMS-Net layers
from torch.utils.checkpoint import checkpoint_sequential
# layers is a nn.Sequential of 16 Mamba blocks
output = checkpoint_sequential(self.layers, segments=4, input=x)
```

*   **Selective Checkpointing:** Instead of checkpointing the entire Mamba block (which includes lightweight convolutions and projections), checkpoint *only* the SSM scan operation. The scan operation generates the largest activation footprint (the recurrent states). Checkpointing just this op saves the most memory with the least compute penalty compared to recomputing the linear projections.

### 3.3 Activation Offloading (Unified Memory)

On the RTX PRO 6000, we can leverage Unified Memory (system RAM backing GPU memory) for activation checkpointing if performance allows. Using hooks to offload saved tensors to CPU and prefetch them before the backward pass can effectively expand memory capacity to the size of system RAM.

```python
# Experimental feature in torch.distributed.fsdp or manual hooks
# Save to CPU
torch.autograd.graph.save_on_cpu(activations=True)
```

While slower, the high bandwidth of PCIe Gen 4/5 on Blackwell workstations makes this a viable fallback for extreme-resolution training.

### 3.4 In-Place Operations

Mamba-2's fusion layer involves adding results from four branches.

```python
# Current Pattern (High Memory)
out = fusion(torch.cat([t_fwd, t_bwd, s_fwd, s_bwd], dim=-1))

# Optimized Pattern (Low Memory - Accumulation)
out = proj_t_fwd(t_fwd)      # Allocate Output
out.add_(proj_t_bwd(t_bwd))  # In-place add
out.add_(proj_s_fwd(s_fwd))  # In-place add
out.add_(proj_s_bwd(s_bwd))  # In-place add
```

Using `add_` (in-place addition) avoids allocating memory for the intermediate sum and the concatenated tensor. This specific pattern is detailed further in Section 6.

### 3.5 Memory Profiling with Snapshots

To blindly optimize memory is inefficient. The user should utilize the PyTorch Memory Snapshot tool to visualize the exact tensor causing the spike.

```python
# Start recording
torch.cuda.memory._record_memory_history(max_entries=100000)

# Run a single training step
train_step()

# Dump snapshot
torch.cuda.memory._dump_snapshot("hms_net_oom.pickle")
```

The resulting `.pickle` file can be uploaded to `pytorch.org/memory_viz`. This visualization will likely show a massive allocation bar corresponding to the `torch.cat` operation in the fusion layer or the graph pool allocation, confirming the diagnosis.

---

## 4. Optimizing Sequential Chunk Processing

The user reports that increasing batch size *decreases* per-sample throughput. This is a classic signature of **CPU overhead dominance** (kernel launch latency). The Python loop `for i in range(0, B_total, chunk_size)` dispatches kernels sequentially. As the batch size grows, `B_total` grows, and the loop executes more times. The GPU (Blackwell) consumes the small chunks faster than the Python interpreter can feed them, leading to GPU starvation gaps.

### 4.1 Parallelization with `torch.vmap`

The most robust solution is to vectorize the chunk processing loop using `torch.vmap` (vectorizing map). `vmap` pushes the loop down into the C++ dispatcher, allowing PyTorch to construct a single kernel launch (or a small set of launches) that processes all chunks in parallel.

**Mathematical Transformation:**
Instead of $N$ sequential calls of $f(x_{chunk})$, we execute one call of $F(X_{batch})$.

**Code Implementation:**

```python
def _optimized_chunked_scan_vmap(self, ssm, x, mask):
    """
    Optimized chunk processing using torch.vmap to parallelize execution.
    x: [B_total, L, D]
    """
    chunk_size = self.chunk_size
    B_total, L, D = x.shape

    # 1. Padding: Ensure B_total is divisible by chunk_size
    pad_len = (chunk_size - (B_total % chunk_size)) % chunk_size
    if pad_len > 0:
        # Pad the batch dimension
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len))
        mask = torch.nn.functional.pad(mask, (0, pad_len))

    # 2. Reshape to [Num_Chunks, chunk_size, L, D]
    # This effectively creates a "batch" of chunks
    x_reshaped = x.view(-1, chunk_size, L, D)
    mask_reshaped = mask.view(-1, chunk_size)

    # 3. Define the per-chunk function
    # ssm is the state space model layer
    def chunk_fn(x_c, m_c):
        return ssm(x_c, m_c)

    # 4. Use vmap to parallelize over the first dimension (Num_Chunks)
    # randomness='different' is crucial if the SSM block contains dropout
    # enabling independent dropout masks per chunk.
    out_reshaped = torch.vmap(chunk_fn, randomness='different')(x_reshaped, mask_reshaped)

    # 5. Restore original shape
    out = out_reshaped.view(-1, L, D)

    # 6. Remove padding
    if pad_len > 0:
        out = out[:-pad_len]

    return out
```

**Why this works on Blackwell:** The `sm_120` GPU has massive parallelism (over 100 SMs). Processing a single chunk (batch size 4, chunk size 512) barely scratches the surface of the GPU's capacity. Vectorizing thousands of chunks allows the GPU scheduler to fill all SMs, converting the problem from latency-bound to throughput-bound.

### 4.2 CUDA Streams for Manual Concurrency

If `vmap` is not feasible (e.g., due to complex state dependencies between chunks that `vmap` cannot express), explicit CUDA Stream management is the fallback.

```python
def _chunked_scan_streams(self, ssm, x, mask):
    # Create a pool of streams
    num_streams = 4
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    results = [None] * num_chunks

    # Iterate and dispatch
    for i in range(0, B_total, chunk_size):
        chunk_idx = i // chunk_size
        stream = streams[chunk_idx % num_streams]

        with torch.cuda.stream(stream):
            # Async execution
            chunk_x = x[i:i+chunk_size]
            chunk_mask = mask[i:i+chunk_size]
            results[chunk_idx] = ssm(chunk_x, chunk_mask)

    # Synchronize all streams to ensure completion
    torch.cuda.synchronize()
    return torch.cat(results, dim=0)
```

This allows the CPU to queue up work on Stream 1 while Stream 0 is executing, hiding the launch latency.

---

## 5. Tensor Parallelism and Model Parallelism on Single GPU

The user asks: "Can tensor parallelism help even on a single GPU?"
Conventionally, Tensor Parallelism (TP) splits tensors across multiple GPUs to aggregate memory and compute. On a single GPU, standard TP (like `torch.distributed.tensor.parallel`) adds communication overhead without increasing compute or memory capacity. Therefore, standard TP is not recommended for a single GPU.
However, the *concept* of model parallelism can be applied to the Fusion Layer to solve the memory explosion.

### 5.1 The Fusion Layer Bottleneck

The fusion layer performs:

1.  Concatenation: $[T_{fwd}, T_{bwd}, S_{fwd}, S_{bwd}] \rightarrow \text{BigTensor}$
2.  Linear Projection: $\text{BigTensor} \times W \rightarrow \text{Output}$

The intermediate `BigTensor` is huge ($262144 \times 2048$ floats).

### 5.2 Algorithmic Fusion (Split-Accumulate)

We can apply "Logical Model Parallelism" by splitting the Linear layer itself.
Mathematically:

$$Y = W \cdot [X_1, X_2, X_3, X_4]^T + b$$

$$Y = W_1 \cdot X_1 + W_2 \cdot X_2 + W_3 \cdot X_3 + W_4 \cdot X_4 + b$$

Where $W$ is split into 4 chunks $[W_1, W_2, W_3, W_4]$.
By computing $W_i \cdot X_i$ sequentially and accumulating the result, we never materialize the `BigTensor` input tensor.

**Optimized Code Pattern:**

```python
class FusedSSMOutput(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # Split the large Linear(4*embed_dim, embed_dim) into 4 smaller ones
        # This uses the exact same number of parameters
        self.proj_t_fwd = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_t_bwd = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_s_fwd = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        # Only the last one typically keeps the bias for the final sum
        self.proj_s_bwd = torch.nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, t_fwd, t_bwd, s_fwd, s_bwd):
        # 1. Project Temporal Forward
        # Memory Alloc: [B, L, D]
        out = self.proj_t_fwd(t_fwd)

        # 2. Accumulate Temporal Backward (In-place)
        # Memory Alloc: No new large tensor, just the buffer for matmul result
        out.add_(self.proj_t_bwd(t_bwd))

        # 3. Accumulate Spatial Forward
        out.add_(self.proj_s_fwd(s_fwd))

        # 4. Accumulate Spatial Backward
        out.add_(self.proj_s_bwd(s_bwd))

        return out
```

**Impact:**

*   **Peak Memory:** Reduced by ~75% at this layer.
*   **Performance:** Slightly higher kernel launch count (4 GEMMs vs 1 GEMM), but avoids the memory allocation latency and potential paging/thrashing associated with the 16GB concatenated tensor.

---

## 6. Custom Kernels and Triton Optimization

While the algorithmic fix (Split-Accumulate) is often sufficient, Blackwell's architecture shines when using custom kernels that exploit its specific memory hierarchy.

### 6.1 Triton for Fused Reshape + Linear

The user specifically asks about fusing reshape + linear + activation. This is a prime candidate for Triton. A standard PyTorch `rearrange` followed by `Linear` usually forces a memory copy to make the tensor contiguous before the GEMM. A Triton kernel can load non-contiguous blocks directly from memory and compute the dot product, skipping the reshape copy entirely.

**Triton Kernel Concept:**

1.  **Grid:** One program instance per output block.
2.  **Load:** Load blocks of $X$ using stride arithmetic that mimics the rearrange logic. For `(B, C, T, H, W) -> (B, T, H, W, C)`, the kernel calculates pointers based on the strided layout of the source tensor.
3.  **Compute:** Perform the dot product with the weight matrix $W$ (loaded into SRAM).
4.  **Store:** Write the result.

**Triton Implementation Strategy:**
Writing a full GEMM (General Matrix Multiply) in Triton that beats cuBLAS on Blackwell is non-trivial. A better approach for this specific Concat -> Linear bottleneck is a Fused Concatenation Kernel.
Instead of writing the full GEMM, write a kernel that takes the 4 input pointers and produces the contiguous `BigTensor` block tile-by-tile in SRAM, feeding it directly to a `tl.dot` operation if possible, or simply optimizing the cat operation to be stream-aware.
However, given the complexity, the **Split-Accumulate** method (Section 5.2) is preferred for maintainability. The performance gain of a custom Triton GEMM over 4 cuBLAS calls is marginal compared to the engineering effort, whereas the memory gain is identical.

### 6.2 Mamba-2 Parallel Scan in Triton

The core of Mamba-2 is the SSD (Structured State Space Duality) scan. The default PyTorch implementation might be sequential. The "Mamba-2" paper introduces a chunked scan algorithm that is parallelizable.

**Optimized Pattern:** Use the `mamba-ssm` library's Triton backend, which already implements the parallel scan. If implementing manually:

*   **Chunk-wise Parallelism:** Compute the "local" scan for all chunks in parallel.
*   **State Passing:** Compute the "carry" state for each chunk.
*   **Scan Carry:** Perform a scan over the carry states (very small sequence).
*   **Distribute:** Apply the carry to the local chunks.

This "scan-of-scans" approach is what enables Mamba-2 to scale on GPUs. Ensure you are using `mamba_ssm.ops.triton.ssd_combined` if available, or porting the 1-SS scan algorithm.

---

## 7. Precision Engineering: BF16 and MXFP8

Blackwell's "killer feature" for this workload is MXFP8.

### 7.1 Leveraging MXFP8 with `torchao`

The `torchao` (PyTorch Architecture Optimization) library is the gateway to Blackwell's MXFP8 capabilities. Standard BFloat16 uses 16 bits per weight. MXFP8 uses 8 bits, effectively doubling model capacity in memory.

**Recipe for HMS-Net:**

1.  **Keep Recurrence in Float32:** The SSM recurrence ($h_t = A h_{t-1} + B x_t$) is sensitive to precision. Accumulating errors over long video sequences in FP8 will lead to divergence. Keep the scan state in `float32`.
2.  **Convert Projections to MXFP8:** The input/output projections (Linear layers) surrounding the SSM are statistically robust. Convert these to MXFP8.

**Implementation:**

```python
import torchao
from torchao.float8 import convert_to_float8_training

# Apply Micro-Scaling FP8 conversion
# This targets the Linear layers compatible with Blackwell
convert_to_float8_training(model)
```

This utilizes the `torch._scaled_mm` intrinsics that map directly to Blackwell's Tensor Cores.

### 7.2 `torch.set_float32_matmul_precision`

For the parts of the model remaining in Float32 (or TF32), ensure the Tensor Cores are engaged.

```python
# 'high' or 'medium' enables TF32 (10-bit mantissa)
# providing near-FP32 accuracy with near-FP16 speed
torch.set_float32_matmul_precision('high')
```

### 7.3 Autocast Strategies

Use `torch.autocast` to manage the mixed precision boundaries automatically.

```python
# Blackwell specific context
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # Linear layers (converted to FP8) will execute in FP8
    # Unconverted layers run in BF16
    # SSM scan (if marked fp32) runs in FP32
    output = model(input)
```

---

## 8. Data Loading and Pipeline Optimization

With the GPU computation optimized, data loading becomes the next bottleneck, especially for video.

### 8.1 Optimizing Video Loading

Loading 16 frames of $256 \times 256$ video requires significant I/O.

*   **Decord / DALI:** Use NVIDIA DALI (Data Loading Library) or `decord` for GPU-accelerated video decoding. Standard OpenCV or TorchVision video readers often decode on CPU, creating a bottleneck.
*   **`pin_memory=True`:** Essential. It allocates page-locked memory on the host, allowing the DMA controller to transfer data to the GPU without CPU involvement.
*   **`non_blocking=True`:** When moving data to GPU (`input.to("cuda", non_blocking=True)`), this allows the transfer to overlap with GPU computation *if* the transfer is on a separate stream.

### 8.2 Overlapping Data Transfer

PyTorch's DataLoader prefetching attempts to hide latency, but explicit stream management ensures it.

```python
# Data Prefetcher Pattern
class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
```

---

## 9. Profiling and Debugging

To verify these optimizations, use the following tools:

### 9.1 PyTorch Profiler

The profiler helps visualize the execution trace. Look for "gaps" between GPU kernels—these indicate CPU overhead (Python loops, data loading, or graph breaks).

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/hms_net_prof'),
    record_shapes=True,
    profile_memory=True,  # Crucial for finding OOM culprits
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

### 9.2 Memory Statistics

Use `torch.cuda.memory_stats()` to monitor fragmentation. If `reserved_bytes` is much larger than `allocated_bytes`, fragmentation is the issue, confirming the need for `expandable_segments`.

---

## 10. Conclusion and Final Implementation Plan

To successfully train HMS-Net on the NVIDIA RTX PRO 6000 (Blackwell), you must transition from a naive implementation to a hardware-aware architecture.

**The Solution Roadmap:**

1.  **Fix OOM:** Switch `torch.compile` to `mode="max-autotune-no-cudagraphs"` and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
2.  **Fix Fusion Explosion:** Refactor the fusion layer to use the "Split-Accumulate" pattern (Section 5.2).
3.  **Fix Sequential Latency:** Replace the chunking loop with `torch.vmap` (Section 4.1) or implement the parallel SSD scan.
4.  **Boost Performance:** Enable MXFP8 training via `torchao` to double throughput and halving weight memory.

### Optimized Code Snippet Summary

```python
# 1. Environment Configuration
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

import torch
import torchao
from torchao.float8 import convert_to_float8_training

# 2. Model Definition
model = MyHMSNet().cuda()

# 3. Apply Blackwell MXFP8 Optimization
convert_to_float8_training(model)

# 4. Compilation Strategy (Avoiding OOM)
model = torch.compile(model, mode="max-autotune-no-cudagraphs")

# 5. Optimized Training Loop
def train_step(input, target):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Forward pass with vmap-optimized chunking
        output = model(input)
        loss = criterion(output, target)

    # Standard backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

This configuration maximizes the capabilities of the `sm_120` architecture while directly addressing the memory and latency constraints of the HMS-Net model.

#### Works cited

1.  Faster Training Throughput in FP8 Precision with NVIDIA NeMo | NVIDIA Technical Blog, accessed December 8, 2025, [https://developer.nvidia.com/blog/faster-training-throughput-in-fp8-precision-with-nvidia-nemo/](https://developer.nvidia.com/blog/faster-training-throughput-in-fp8-precision-with-nvidia-nemo/)
2.  Accelerating 2K scale pre-training up to 1.28x with TorchAO, MXFP8 and TorchTitan on Crusoe B200 Cluster - PyTorch, accessed December 8, 2025, [https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/](https://pytorch.org/blog/accelerating-2k-scale-pre-training-up-to-1-28x-with-torchao-mxfp8-and-torchtitan-on-crusoe-b200-cluster/)
3.  torch.compile — PyTorch 2.9 documentation, accessed December 8, 2025, [https://docs.pytorch.org/docs/stable/generated/torch.compile.html](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
4.  Everything You Need to Know About PyTorch Compile | by LambdaFlux - Medium, accessed December 8, 2025, [https://medium.com/@lambdafluxofficial/everything-you-need-to-know-about-pytorch-compile-3d7fd94ce701](https://medium.com/@lambdafluxofficial/everything-you-need-to-know-about-pytorch-compile-3d7fd94ce701)
5.  .venv/lib/python3.11/site-packages/torch/_inductor/config.py · koichi12/llm_tutorial at 70fbf20b4ddc5dc677bf521e5e39c22461fa7256 - Hugging Face, accessed December 8, 2025, [https://huggingface.co/koichi12/llm_tutorial/blob/70fbf20b4ddc5dc677bf521e5e39c22461fa7256/.venv/lib/python3.11/site-packages/torch/_inductor/config.py](https://huggingface.co/koichi12/llm_tutorial/blob/70fbf20b4ddc5dc677bf521e5e39c22461fa7256/.venv/lib/python3.11/site-packages/torch/_inductor/config.py)
6.  CUDA semantics — PyTorch 2.9 documentation, accessed December 8, 2025, [https://docs.pytorch.org/docs/stable/notes/cuda.html](https://docs.pytorch.org/docs/stable/notes/cuda.html)
7.  Understanding CUDA Memory Usage — PyTorch 2.9 documentation, accessed December 8, 2025, [https://docs.pytorch.org/docs/stable/torch_cuda_memory.html](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
8.  Understanding GPU Memory 1: Visualizing All Allocations over Time - PyTorch, accessed December 8, 2025, [https://pytorch.org/blog/understanding-gpu-memory-1/](https://pytorch.org/blog/understanding-gpu-memory-1/)
9.  State Space Duality (Mamba-2) Part III - The Algorithm | Tri Dao, accessed December 8, 2025, [https://tridao.me/blog/2024/mamba2-part3-algorithm/](https://tridao.me/blog/2024/mamba2-part3-algorithm/)
10. Mamba-2: Algorithms and Systems | Princeton Language and Intelligence, accessed December 8, 2025, [https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems](https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems)
