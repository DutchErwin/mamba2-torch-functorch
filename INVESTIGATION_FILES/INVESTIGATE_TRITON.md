# Custom Triton Kernels for Parallel SSM Investigation

## Objective
Write custom Triton kernels that process all SSM chunks in parallel within a single kernel launch, eliminating sequential Python loop overhead entirely.

**Expected Outcome:** 5-10x throughput improvement by fusing all chunk processing into GPU-parallel operations.

**Priority:** LONG-TERM - This is significantly more complex than Track 1 (Functorch). Pursue after Track 1 is complete or blocked.

---

## Environment Specifications

| Component | Version | Notes |
|-----------|---------|-------|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell | 96GB VRAM, sm_120 compute |
| **PyTorch** | 2.9.1+cu129 | CUDA 12.9, required for Blackwell |
| **Triton** | 3.5.1 | Blackwell sm_120 kernel support |
| **Python** | 3.12+ | Required for PyTorch 2.9 |
| **mamba2-torch** | latest (pip) | Contains existing Triton SSM kernels |

**Blackwell-Specific Features (sm_120):**
- 192 Streaming Multiprocessors (SMs)
- Thread Block Clusters
- Distributed Shared Memory (DSM)
- TMA (Tensor Memory Accelerator)
- FP4 tensor core support

---

## The Core Challenge

### Why SSMs Are Hard to Parallelize

State Space Models have a fundamental sequential dependency:
```
state_{t+1} = A * state_t + B * input_t
output_t = C * state_t + D * input_t
```

Each state depends on the previous state, creating a chain:
```
state_0 → state_1 → state_2 → ... → state_T
```

**Naive parallelization is impossible** because computing `state_5` requires `state_4`, which requires `state_3`, etc.

### How Mamba-2 Solves This

Mamba-2 uses **State Space Duality (SSD)** which reformulates the recurrence as a structured matrix operation that CAN be parallelized using:

1. **Parallel Scan Algorithms** (e.g., Blelloch scan)
2. **Chunked computation** with inter-chunk state passing
3. **Matrix decomposition** that exposes parallelism

**Key Paper:** "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Mamba-2 paper)

---

## Research Questions

### 1. Understand Existing mamba2-torch Triton Kernels

**Question:** What does the current Triton implementation look like?

**Files to analyze in mamba2-torch:**
```
mamba2_torch/triton/
├── ssd_bmm_triton.py      # Batched matrix multiply kernels
├── ssd_chunk_scan.py      # Chunk-wise scan kernels
└── ssd_state_passing.py   # State passing between chunks
```

**Research needed:**
- Current kernel structure and grid configuration
- How chunks are processed (serial vs parallel)
- Where sequential dependencies exist
- What optimizations are already applied

---

### 2. Parallel Scan Algorithms

**Question:** What parallel scan algorithms work for SSMs?

**Algorithms to research:**

**Blelloch Scan (Prefix Sum):**
- Work-efficient parallel prefix algorithm
- O(n) work, O(log n) depth
- Applicable when operation is associative

**Hillis-Steele Scan:**
- Simpler but not work-efficient
- O(n log n) work, O(log n) depth

**Chunked Parallel Scan:**
- Process chunks in parallel
- Sequential state passing between chunks
- Hybrid approach used by Mamba-2

**Research needed:**
- How SSD reformulation enables parallel scan
- Memory requirements for parallel state computation
- Trade-offs between different algorithms

---

### 3. Grid Parallelism Over Chunks

**Question:** Can we write a Triton kernel with grid parallelism over chunks?

**Current situation:**
```python
# Python loop - sequential
for chunk_idx in range(num_chunks):
    process_chunk(chunk_idx)
```

**Goal:**
```python
# Single Triton kernel - parallel
@triton.jit
def parallel_chunk_kernel(chunks, num_chunks):
    chunk_idx = tl.program_id(0)  # Parallel over chunks
    # Process chunk_idx in parallel with all other chunks
```

**Challenges:**
- State passing between chunks requires synchronization
- How to handle inter-chunk dependencies within single kernel
- Shared memory vs global memory for state

**Research needed:**
- Triton grid configuration for chunk parallelism
- State passing patterns in Triton
- Memory coalescing for chunk access

---

### 4. Blackwell-Specific Optimizations

**Question:** What sm_120 features can we leverage?

**Thread Block Clusters:**
- Groups of thread blocks that can cooperate
- Enables cross-block communication without global memory
- Could help with inter-chunk state passing

**Distributed Shared Memory (DSM):**
- Shared memory visible across cluster
- Lower latency than global memory
- Potential for efficient state passing

**TMA (Tensor Memory Accelerator):**
- Hardware-accelerated tensor loads
- Async copy with explicit sync
- Could overlap compute with memory

**FP4 Tensor Cores:**
- 4-bit floating point for inference
- 8x compute density vs FP16
- May require quantization-aware training

**Research needed:**
- Triton 3.5 support for sm_120 features
- Example kernels using Thread Block Clusters
- TMA usage patterns in Triton

---

### 5. Example Kernel Structure

**Question:** What would a chunk-parallel SSM kernel look like?

**Pseudocode concept:**
```python
@triton.jit
def parallel_ssm_kernel(
    x_ptr, mask_ptr, out_ptr,
    A_ptr, B_ptr, C_ptr, D_ptr,
    seq_len, d_model, d_state,
    num_chunks, chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Get chunk index from grid
    chunk_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # Load chunk data
    chunk_start = chunk_idx * chunk_size
    x_chunk = tl.load(x_ptr + batch_idx * seq_len + chunk_start, ...)

    # Initialize state (from previous chunk or zero)
    if chunk_idx == 0:
        state = tl.zeros([d_state], dtype=tl.float32)
    else:
        state = load_state_from_previous_chunk(...)  # The hard part!

    # Process sequence within chunk
    for t in range(chunk_size):
        # SSM recurrence
        state = A * state + B * x_chunk[t]
        out[t] = C * state + D * x_chunk[t]

    # Store output
    tl.store(out_ptr + batch_idx * seq_len + chunk_start, out, ...)

    # Store state for next chunk
    store_state_for_next_chunk(state, ...)  # Also hard!
```

**Key challenges in pseudocode:**
1. `load_state_from_previous_chunk` - requires synchronization
2. `store_state_for_next_chunk` - inter-chunk communication
3. These create implicit sequential dependency

**Research needed:**
- How Mamba-2 handles inter-chunk state in their kernels
- Techniques for implicit parallelism despite state dependency
- Whether SSD reformulation eliminates this dependency

---

### 6. Existing Implementations to Study

**Question:** What existing implementations can we learn from?

**FlashAttention:**
- Pioneered chunked GPU kernels for transformers
- Tiling and recomputation strategies
- Memory-efficient backward pass

**Mamba-2 Official Implementation:**
- https://github.com/state-spaces/mamba
- CUDA kernels (not Triton, but algorithmic reference)
- SSD parallel formulation

**mamba2-torch:**
- https://github.com/vasqu/mamba2-torch
- Triton implementation we're using
- Starting point for modifications

**Linear Attention Implementations:**
- Similar recurrence structure to SSMs
- Some have parallel implementations

**Research needed:**
- FlashAttention tiling strategy applicability
- Mamba-2 CUDA kernel analysis
- Any academic parallel SSM implementations

---

## Benchmark Requirements

Any solution must be validated:

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

**Target (with custom Triton):**
- Time: <1.5s (5x improvement)
- VRAM: <50GB (acceptable on 96GB)
- Throughput: >2.5 samples/sec

---

## Deliverables Requested

From deep research, please provide:

1. **Algorithm analysis:** How SSD enables parallelism, step-by-step
2. **Kernel pseudocode:** Detailed structure of chunk-parallel kernel
3. **Memory analysis:** Expected memory requirements for parallel approach
4. **Blackwell feasibility:** Which sm_120 features are usable in Triton 3.5
5. **Implementation roadmap:** Step-by-step plan if feasible
6. **Complexity assessment:** Realistic effort estimate

---

## Why This is Difficult

1. **Sequential state dependency:** Fundamental to SSM formulation
2. **Algorithmic complexity:** Requires understanding parallel scan theory
3. **Triton expertise:** Need deep knowledge of GPU programming
4. **No existing example:** Would be novel implementation
5. **Testing complexity:** Hard to verify correctness of parallel algorithm

**Recommendation:** Pursue Track 1 (Functorch) first. If that's blocked or insufficient, return to this track with deep research findings.

---

*Created: December 8, 2024*
*For: Deep Research Investigation*
*Priority: LOW - Long-term optimization track, pursue after Track 1*
