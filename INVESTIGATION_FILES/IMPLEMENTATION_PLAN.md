# Role & Objective
You are a Principal Research Engineer. You are implementing "HMS-Net" on a **local NVIDIA RTX 6000 (Blackwell) workstation with 92GB VRAM**.

**CRITICAL ENVIRONMENT CONTEXT:**
- **Hardware:** Blackwell Architecture (sm_100+). 92GB VRAM.
- **Pre-Compiled Library 1:** `mamba_ssm` (Mamba 2 version) is ALREADY installed. Do not add it to requirements.
- **Pre-Compiled Library 2:** `sageattention` (v3) is ALREADY installed. Do not add it to requirements.
- **Goal:** Build a "Hyper-Modern" version of HMS-Net that leverages these specific kernels for maximum throughput.

# The Implementation Strategy
Create `IMPLEMENTATION_PLAN.md` with these specific "God-Tier" adjustments:

## Phase 1: Infrastructure (Custom Binary Verification)
- **Goal:** Verify custom binaries before coding logic.
- **Task:** Create `check_environment.py`. It must:
    1. Import `from mamba_ssm import Mamba2` (Verify v2 API).
    2. Import `sageattention` (Verify import name and version).
    3. Run a dummy forward pass on CUDA to ensure no symbol errors occur.
- **Data:** `MockDataEngine` should yield **4K Tensors**: `[B=1, C=5, T=100, H=2160, W=3840]`. We have 92GB VRAM; target 4K immediately.

## Phase 2: The Mamba 2 Backbone
- **Goal:** Upgrade Section 3.2 to use Mamba 2 (State Space Duality).
- **Task:** Implement `MaskGatedSSM` using `mamba_ssm.modules.mamba2.Mamba2`.
- **Constraint:** Mamba 2 requires `d_model` to be divisible by `headdim` (usually 64 or 128). You must architect the channel dimensions to respect this constraint.
- **Logic:** The "Learned Null Token" gating (Section 3.2.2) happens *before* the input enters the Mamba 2 kernel (as `Mamba2` is a monolithic block).
- **Verification:** `test_state_persistence.py` must pass using the Mamba 2 kernel.

## Phase 3: Semantic & Geometric Logic
- **Goal:** Full Fidelity Scene Understanding.
- **Task:** `SceneGraph` (Section 3.0). The paper uses a Transformer. **SWAP:** Replace the internal Attention with `sageattention`.
- **Task:** `DepthWrapper`. Use **Depth Anything v2 (Large)**. Keep it resident in VRAM.

## Phase 4: Sage-Powered Refinement
- **Goal:** Accelerate the Wavelet LCM Refiner (Section 3.3).
- **Modification:** The Refiner uses Cross-Attention. You must implement a `SageAttentionWrapper` that replaces `torch.nn.functional.scaled_dot_product_attention` with the compiled `sageattention` kernel.
- **Benefit:** This enables real-time refinement steps on the Blackwell Tensor Cores.

## Phase 5: Integration
- **Task:** `inference.py`.
- **Feature:** Add a `--scrub-speed` benchmark. With Mamba 2 and SageAttention, we expect >100 FPS on 4K scrubbing.

# Instructions
1. Analyze the paper.
2. Generate the `IMPLEMENTATION_PLAN.md` tailored for this Mamba2/Sage/Blackwell stack.
3. **STOP.** Wait for approval.