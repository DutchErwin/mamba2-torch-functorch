# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

from typing import Optional

import math
from packaging import version

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn, causal_conv1d_cuda = None, None

from ..ops.ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd
from ..ops.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd
from ..ops.ssd_chunk_state import _chunk_state_fwd, _chunk_state_bwd_db
from ..ops.ssd_chunk_state import _chunk_state_bwd_ddAcs_stable
from ..ops.ssd_chunk_state import chunk_state, chunk_state_ref
from ..ops.ssd_state_passing import _state_passing_fwd, _state_passing_bwd
from ..ops.ssd_state_passing import state_passing, state_passing_ref
from ..ops.ssd_chunk_scan import _chunk_scan_fwd, _chunk_scan_bwd_dz, _chunk_scan_bwd_dstates
from ..ops.ssd_chunk_scan import _chunk_scan_bwd_dC, _chunk_scan_bwd_dcb
from ..ops.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_stable
from ..ops.ssd_chunk_scan import chunk_scan, chunk_scan_ref
from ..ops.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_prev
from ..ops.layernorm_gated import rmsnorm_fn, _layer_norm_fwd, _layer_norm_bwd
from ..ops.k_activations import _swiglu_fwd, _swiglu_bwd
from ..ops.custom_fwd_bwd import custom_bwd, custom_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
        # Pointers to matrices
        x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, D_ptr,
        b_ptr, dstates_ptr,
        dx_ptr, ddt_ptr, dD_ptr,
        # Matrix dimensions
        chunk_size, hdim, dstate,
        batch, seqlen, nheads_ngroups_ratio,
        # Strides
        stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
        stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
        stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
        stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
        stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
        stride_seq_idx_batch, stride_seq_idx_seqlen,
        stride_D_head,
        stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
        stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
        stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
        stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
        stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
        # Meta-parameters
        HAS_D: tl.constexpr,
        D_HAS_HDIM: tl.constexpr,
        HAS_SEQ_IDX: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_DSTATE: tl.constexpr,
        IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(dA_cs_last - dA_cs_m)
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
    # Might be faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    # However, we're getting error with the Triton compiler 2.1.0 for that code path:
    # Unexpected mma -> mma layout conversion
    # Triton 2.2.0 fixes this
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate)
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    # x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    # x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    # dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    # dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    # ddt = tl.sum(acc * x, axis=1) * dt_m
    # ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    # tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        # For some reason setting mask to (offs_m[:, None] < chunk_size_limit) is much slower
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0).to(tl.float32)
        cb *= tl.exp(dA_cs_k[None, :] - dA_cs_m[:, None])
        # If we don't have the (k + offs_k[None, :] < K_MAX) mask, for indices outside this range,
        # we might have dA_cs_m = 0.0 and dA_cs_k very negative, and tl.exp will return inf.
        # Multiplying with cb, which is 0.0 outside the range, will make the result NaN.
        # This will cause NaN in acc, and hence NaN in dx and ddt.
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


def _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                  if D is not None else (0, 0, 0, 0, 0))
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                            batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x, CB, dout, dt, dA_cumsum, seq_idx, D, B, dstates, dx, ddt, dD,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(-1), CB.stride(-2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            D.stride(0) if D is not None else 0,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
                           D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return dx, ddt.to(dtype=dt.dtype), dD


def _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    # # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
    # dA_cumsum_tmp0, dt_tmp0 = _chunk_cumsum_fwd(dt[:, :147], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp1, dt_tmp1 = _chunk_cumsum_fwd(dt[:, 147:], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp2, dt_tmp2 = _chunk_cumsum_fwd(dt[:, 147:256], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    # states_tmp0 = _chunk_state_fwd(B[:, :147], x[:, :147], dt_tmp0, dA_cumsum_tmp0, states_in_fp32=True)
    # states_tmp1 = _chunk_state_fwd(B[:, 147:], x[:, 147:], dt_tmp1, dA_cumsum_tmp1, states_in_fp32=True)
    # states_tmp2 = _chunk_state_fwd(B[:, 147:256], x[:, 147:256], dt_tmp2, dA_cumsum_tmp2, states_in_fp32=True)
    states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                              initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                              seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    # states_tmp0 = rearrange(_state_passing_fwd(rearrange(states_tmp0, "... p n -> ... (p n)"), dA_cumsum_tmp0[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    # states_tmp1 = rearrange(_state_passing_fwd(rearrange(states_tmp1, "... p n -> ... (p n)"), dA_cumsum_tmp1[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)
    return out, out_x, dt, dA_cumsum, states, final_states


def _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D=None, z=None,
                                   dt_bias=None, initial_states=None, dfinal_states=None, seq_idx=None, dt_softplus=False,
                                   dt_limit=(0.0, float("inf")),
                                   dx=None, ddt=None, dB=None, dC=None, dz=None, recompute_output=False):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)
    # TD: For some reason Triton (2.1.0 and 2.2.0) errors with
    # "[CUDA]: invalid device context" (e.g. during varlne test), and cloning makes it work. Idk why.
    dt_in = dt.clone()
    dA_cumsum, dt = _chunk_cumsum_fwd(dt_in, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, _ = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                   initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                   seq_idx=seq_idx, chunk_size=chunk_size)
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, has_ddAcs=False, D=D, dz=dz, recompute_output=recompute_output)
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out
    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype)
    # dstates has length nchunks, containing the gradient to initial states at index 0 and
    # gradient to the states of chunk (nchunks - 2) at index (nchunks - 1)
    # Do computation in fp32 but convert dstates and states to fp16/bf16 since dstates and states
    # will be used in matmul in the next kernels.
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=rearrange(dfinal_states, "... p n -> ... (p n)") if dfinal_states is not None else None,
        seq_idx=seq_idx,
        has_initial_states=initial_states is not None,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    # dstates has length nchunks, containing the gradient to states of chunk 0 at index 0 and
    # gradient to the final states at index (nchunks - 1)
    # states has length nchunks, containing the initial states at index 0 and the state for chunk (nchunks - 2) at index (nchunks - 1)
    # The final states is not stored.
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
    dinitial_states = rearrange(dinitial_states, "... (p n) -> ... p n", n=dstate) if dinitial_states is not None else None
    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=D, seq_idx=seq_idx, dx=dx)
    # dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, ngroups=ngroups)
    dB, ddA_next = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups)
    # dC = _chunk_scan_bwd_dC(states[:, :-1].to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups)
    # Computing ddA with the dcb kernel is much slower, so we're not using it for now
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    # dCB, ddA_tmp = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, CB=CB, ngroups=ngroups)
    dCB = dCB.to(CB.dtype)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given)
    # If we have z, then dout_x is recomputed in fp32 so dD = (dout_x * x).sum() is more accurate
    # than dD_from_x = (dout_x * x).sum() where dout_x is in fp16/bf16
    if z is None:
        dD = dD_from_x
    # Formula for ddA_cumsum, assuming out is the output of the forward pass before adding x * D.
    # ddA_cumsum = torch.einsum("bclhp,bclhp->bhcl", out.float(), dout.float()) - ddt * dt
    # However, this is numerically unstable: when we do the reverse cumsum on ddA_cumsum, there might
    # be a lot of underflow.

    # This is already done as part of bwd_dC kernel
    # ddA_cumsum_prev = _chunk_scan_bwd_ddAcs_prev(states[:, :-1], C, dout, dA_cumsum, seq_idx=seq_idx)
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    # This is already done as part of bwd_dB kernel
    # ddA_next = _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates, seq_idx=seq_idx)
    # We don't need to pass in seq_idx because CB also zeros out entries where seq_idx[i] != seq_idx[j]
    ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)
    ddA += ddA_next + ddA_prev

    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt, dt_in, A, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit, ddt=ddt_given)

    # These 2 lines are just to test ddt and dA being computed by old code
    # _, dA = selective_scan_bwd(dout, x, dt, A, B, C, D=D.float(), z=z)
    # ddt_given.copy_(ddt)

    return_vals = (dx, ddt_given, dA, dB_given, dC_given, dD, dz, ddt_bias, dinitial_states)
    return return_vals if not recompute_output else (*return_vals, outz)


def selective_scan_bwd(dout, x, dt, A, B, C, D=None, z=None):
    """
    Argument:
        dout: (batch, seqlen, nheads, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size) or (batch, nheads, headdim, nchunks, chunk_size)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    try:
        import selective_scan
    except ImportError:
        selective_scan = None

    batch, seqlen, nheads, headdim = x.shape
    chunk_size = dt.shape[-1]
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    x = rearrange(x, "b l h p -> b (h p) l")
    squeeze_dt = dt.dim() == 4
    if dt.dim() == 4:
        dt = repeat(dt, "b h c l -> b h p c l", p=headdim)
    dt = rearrange(dt, "b h p c l -> b (h p) (c l)", p=headdim)
    squeeze_A = A.dim() == 1
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    B = rearrange(B, "b l g n -> b g n l")
    C = rearrange(C, "b l g n -> b g n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")

    if x.stride(-1) != 1:
        x = x.contiguous()
    if dt.stride(-1) != 1:
        dt = dt.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    _, intermediate, *rest = selective_scan.fwd(x, dt.to(dtype=x.dtype), A, B, C, D, z, None, False)
    if z is not None:
        out = rest[0]
    else:
        out = None

    dout = rearrange(dout, "b l h p -> b (h p) l")

    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
    # backward of selective_scan with the backward of chunk).
    # Here we just pass in None and dz will be allocated in the C++ code.
    _, ddt, dA, *rest = selective_scan.bwd(
        x, dt.to(dtype=x.dtype), A, B, C, D, z, None, dout, intermediate, out, None, False,
        False  # option to recompute out_z, not used here
    )
    ddt = rearrange(ddt, "b (h p) (c l) -> b h p c l", p=headdim, l=chunk_size)
    if squeeze_dt:
        ddt = ddt.float().sum(dim=2)
    if squeeze_A:
        dA = rearrange(dA, "(h p) n -> h p n", p=headdim).sum(dim=(1, 2))
    return ddt, dA


class MambaChunkScanCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False):
        dt_dtype = dt.dtype
        out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=dt_softplus, dt_limit=dt_limit)
        # Return user-facing outputs first, then views of tensors needed for backward
        # out_x is returned for conditional save logic in setup_context (use out if z is None, else out_x)
        # (PyTorch requires views, not original tensors, when using setup_context)
        return (out, final_states,
                out_x.view_as(out_x) if out_x is not None else None,
                x.view_as(x), dt.view_as(dt), dA_cumsum.view_as(dA_cumsum),
                A.view_as(A), B.view_as(B), C.view_as(C),
                D.view_as(D) if D is not None else None,
                z.view_as(z) if z is not None else None,
                dt_bias.view_as(dt_bias) if dt_bias is not None else None,
                initial_states.view_as(initial_states) if initial_states is not None else None,
                seq_idx.view_as(seq_idx) if seq_idx is not None else None,
                dt_dtype, dt_softplus, chunk_size, dt_limit, return_final_states)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, dt_softplus, dt_limit, return_final_states = inputs
        (out, final_states, out_x_saved, x_saved, dt_saved, dA_cumsum_saved,
         A_saved, B_saved, C_saved, D_saved, z_saved, dt_bias_saved,
         initial_states_saved, seq_idx_saved,
         dt_dtype, _dt_softplus, _chunk_size, _dt_limit, _return_final_states) = output
        # Conditional save: use out if z is None, else use out_x
        out_for_backward = out.view_as(out) if z is None else out_x_saved
        ctx.save_for_backward(out_for_backward, x_saved, dt_saved, dA_cumsum_saved,
                              A_saved, B_saved, C_saved, D_saved, z_saved,
                              dt_bias_saved, initial_states_saved, seq_idx_saved)
        ctx.dt_dtype = dt_dtype
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.has_D = D is not None
        ctx.has_z = z is not None
        ctx.has_dt_bias = dt_bias is not None
        ctx.has_initial_states = initial_states is not None
        ctx.has_seq_idx = seq_idx is not None

    @staticmethod
    def backward(ctx, dout, dfinal_states, *_):
        out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors
        dfinal_states_for_bwd = dfinal_states if ctx.return_final_states else None
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states_for_bwd, seq_idx=seq_idx, dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit)
        return dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, dinitial_states, None, None, None, None

    @staticmethod
    def vmap(info, in_dims, x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, dt_softplus, dt_limit, return_final_states):
        # Handle vmap by moving batch dim to front and merging with existing batch
        def move_bdim_to_front(tensor, bdim):
            if tensor is None or bdim is None:
                return tensor
            return tensor.movedim(bdim, 0)

        # in_dims order: x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, dt_softplus, dt_limit, return_final_states
        # Tensor inputs: x(0), dt(1), A(2), B(3), C(4), D(6), z(7), dt_bias(8), initial_states(9), seq_idx(10)
        # Non-tensor inputs: chunk_size(5), dt_softplus(11), dt_limit(12), return_final_states(13)

        # Get vmap batch size from first batched tensor input
        vmap_batch_size = None
        tensor_indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]  # indices of tensor inputs
        tensors = [x, dt, A, B, C, D, z, dt_bias, initial_states, seq_idx]
        for i, tensor_idx in enumerate(tensor_indices):
            bdim = in_dims[tensor_idx]
            tensor = tensors[i]
            if bdim is not None and tensor is not None:
                vmap_batch_size = tensor.shape[bdim]
                break

        if vmap_batch_size is None:
            # No batched inputs, just call directly
            result = MambaChunkScanCombinedFn.apply(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, dt_softplus, dt_limit, return_final_states)
            # Return None for all output dims since no batching
            return result, (None,) * len(result)

        # Move batch dims to front
        x_batched = move_bdim_to_front(x, in_dims[0])
        dt_batched = move_bdim_to_front(dt, in_dims[1])
        A_batched = move_bdim_to_front(A, in_dims[2])
        B_batched = move_bdim_to_front(B, in_dims[3])
        C_batched = move_bdim_to_front(C, in_dims[4])
        D_batched = move_bdim_to_front(D, in_dims[6])
        z_batched = move_bdim_to_front(z, in_dims[7])
        dt_bias_batched = move_bdim_to_front(dt_bias, in_dims[8])
        initial_states_batched = move_bdim_to_front(initial_states, in_dims[9])
        seq_idx_batched = move_bdim_to_front(seq_idx, in_dims[10])

        # Broadcast non-batched inputs (expand to vmap_batch_size)
        # x: (batch, seqlen, nheads, headdim)
        if in_dims[0] is None:
            x_batched = x_batched.unsqueeze(0).expand(vmap_batch_size, *x_batched.shape)
        # dt: (batch, seqlen, nheads)
        if in_dims[1] is None:
            dt_batched = dt_batched.unsqueeze(0).expand(vmap_batch_size, *dt_batched.shape)
        # A: (nheads,) - typically shared, don't broadcast
        # B: (batch, seqlen, ngroups, dstate)
        if in_dims[3] is None:
            B_batched = B_batched.unsqueeze(0).expand(vmap_batch_size, *B_batched.shape)
        # C: (batch, seqlen, ngroups, dstate)
        if in_dims[4] is None:
            C_batched = C_batched.unsqueeze(0).expand(vmap_batch_size, *C_batched.shape)
        # D: (nheads, headdim) or (nheads,) - typically shared
        # z: optional, (batch, seqlen, nheads, headdim)
        if z is not None and in_dims[7] is None:
            z_batched = z_batched.unsqueeze(0).expand(vmap_batch_size, *z_batched.shape)
        # dt_bias: (nheads,) - typically shared
        # initial_states: optional, (batch, nheads, headdim, dstate)
        if initial_states is not None and in_dims[9] is None:
            initial_states_batched = initial_states_batched.unsqueeze(0).expand(vmap_batch_size, *initial_states_batched.shape)
        # seq_idx: optional, (batch, seqlen)
        if seq_idx is not None and in_dims[10] is None:
            seq_idx_batched = seq_idx_batched.unsqueeze(0).expand(vmap_batch_size, *seq_idx_batched.shape)

        # Merge vmap batch dim with tensor batch dim
        # x: (vmap_batch, batch, seqlen, nheads, headdim) -> (vmap_batch * batch, seqlen, nheads, headdim)
        x_shape = x_batched.shape
        x_merged = x_batched.reshape(x_shape[0] * x_shape[1], *x_shape[2:])

        # dt: (vmap_batch, batch, seqlen, nheads) -> (vmap_batch * batch, seqlen, nheads)
        dt_shape = dt_batched.shape
        dt_merged = dt_batched.reshape(dt_shape[0] * dt_shape[1], *dt_shape[2:])

        # A: (nheads,) - shared, use directly or take first if batched
        if in_dims[2] is not None:
            A_merged = A_batched[0]
        else:
            A_merged = A

        # B: (vmap_batch, batch, seqlen, ngroups, dstate) -> (vmap_batch * batch, seqlen, ngroups, dstate)
        B_shape = B_batched.shape
        B_merged = B_batched.reshape(B_shape[0] * B_shape[1], *B_shape[2:])

        # C: (vmap_batch, batch, seqlen, ngroups, dstate) -> (vmap_batch * batch, seqlen, ngroups, dstate)
        C_shape = C_batched.shape
        C_merged = C_batched.reshape(C_shape[0] * C_shape[1], *C_shape[2:])

        # D: (nheads, headdim) or (nheads,) - shared, use directly or take first if batched
        D_merged = None
        if D is not None:
            if in_dims[6] is not None:
                D_merged = D_batched[0]
            else:
                D_merged = D

        # z: optional, (vmap_batch, batch, seqlen, nheads, headdim) -> (vmap_batch * batch, seqlen, nheads, headdim)
        z_merged = None
        if z_batched is not None:
            z_shape = z_batched.shape
            z_merged = z_batched.reshape(z_shape[0] * z_shape[1], *z_shape[2:])

        # dt_bias: (nheads,) - shared, use directly or take first if batched
        dt_bias_merged = None
        if dt_bias is not None:
            if in_dims[8] is not None:
                dt_bias_merged = dt_bias_batched[0]
            else:
                dt_bias_merged = dt_bias

        # initial_states: optional, (vmap_batch, batch, nheads, headdim, dstate) -> (vmap_batch * batch, nheads, headdim, dstate)
        initial_states_merged = None
        if initial_states_batched is not None:
            is_shape = initial_states_batched.shape
            initial_states_merged = initial_states_batched.reshape(is_shape[0] * is_shape[1], *is_shape[2:])

        # seq_idx: optional, (vmap_batch, batch, seqlen) -> (vmap_batch * batch, seqlen)
        seq_idx_merged = None
        if seq_idx_batched is not None:
            si_shape = seq_idx_batched.shape
            seq_idx_merged = seq_idx_batched.reshape(si_shape[0] * si_shape[1], *si_shape[2:])

        # Call the function with merged batches
        result = MambaChunkScanCombinedFn.apply(
            x_merged, dt_merged, A_merged, B_merged, C_merged, chunk_size,
            D_merged, z_merged, dt_bias_merged, initial_states_merged, seq_idx_merged,
            dt_softplus, dt_limit, return_final_states
        )

        # Unmerge outputs
        # Output structure: (out, final_states, out_x, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx, dt_dtype, dt_softplus, chunk_size, dt_limit, return_final_states)
        merged_batch = result[0].shape[0]
        original_batch = merged_batch // vmap_batch_size

        outputs_unmerged = []
        out_dims_list = []

        for idx, out in enumerate(result):
            if out is None:
                outputs_unmerged.append(None)
                out_dims_list.append(None)
            elif idx in [6, 9, 11]:  # A, D, dt_bias are shared (not batched)
                outputs_unmerged.append(out)
                out_dims_list.append(None)
            elif idx >= 14:  # Non-tensor outputs: dt_dtype, dt_softplus, chunk_size, dt_limit, return_final_states
                outputs_unmerged.append(out)
                out_dims_list.append(None)
            elif isinstance(out, torch.Tensor) and out.dim() > 0:
                # Tensor outputs that need unmerging
                out_unmerged = out.reshape(vmap_batch_size, original_batch, *out.shape[1:])
                outputs_unmerged.append(out_unmerged)
                out_dims_list.append(0)
            else:
                outputs_unmerged.append(out)
                out_dims_list.append(None)

        return tuple(outputs_unmerged), tuple(out_dims_list)


def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
        or (out, final_states) if return_final_states is True
    """
    result = MambaChunkScanCombinedFn.apply(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, dt_softplus, dt_limit, return_final_states)
    # Extract user-facing outputs: result[0] is out, result[1] is final_states
    out, final_states = result[0], result[1]
    if return_final_states:
        return out, final_states
    else:
        return out


def mamba_chunk_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True)
    # 2. Pass the state to all the chunks by weighted cumsum.
    states = rearrange(state_passing(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                       "... (p n) -> ... p n", n=dstate)
    # 3. Compute the output for each chunk
    out = chunk_scan(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out


def ssd_chunk_scan_combined_ref(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state_ref(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
    # 2. Pass the state to all the chunks by weighted cumsum.
    # state_passing_ref is much less numerically stable
    states = rearrange(state_passing_ref(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])[0],
                       "... (p n) -> ... p n", n=dstate)
    states = states.to(states_dtype)
    # 3. Compute the output for each chunk
    out = chunk_scan_ref(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    return out


def ssd_selective_scan(x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,) or (nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except ImportError:
        selective_scan_fn = None

    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    x = rearrange(x, "b l h p -> b (h p) l")
    if dt.dim() == 3:
        dt = repeat(dt, "b l h -> b l h p", p=headdim)
    dt = rearrange(dt, "b l h p -> b (h p) l")
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    B = rearrange(B, "b l g n -> b g n l")
    C = rearrange(C, "b l g n -> b g n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")
    if dt_bias is not None:
        if dt_bias.dim() == 1:
            dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
        dt_bias = rearrange(dt_bias, "h p -> (h p)")
    if dt_limit != (0.0, float("inf")):
        if dt_bias is not None:
            dt = dt + rearrange(dt_bias, "d -> d 1")
        if dt_softplus:
            dt = F.softplus(dt)
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1]).to(x.dtype)
        dt_bias = None
        dt_softplus = None
    out = selective_scan_fn(x, dt, A, B, C, D=D, z=z, delta_bias=dt_bias, delta_softplus=dt_softplus)
    return rearrange(out, "b (h p) l -> b l h p", p=headdim)


def mamba_conv1d_scan_ref(xBC, conv1d_weight, conv1d_bias, dt, A, chunk_size, D=None, z=None,
                          dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")),
                          activation="silu", headdim=None, ngroups=1):
    """
    Argument:
        xBC: (batch, seqlen, dim + 2 * ngroups * dstate) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, dim)
        dt_bias: (nheads) or (nheads, headdim)
        headdim: if D is 1D and z is None, headdim must be passed in
    Return:
        out: (batch, seqlen, dim)
    """
    batch, seqlen, nheads = dt.shape[:3]
    assert nheads % ngroups == 0
    if z is not None:
        dim = z.shape[-1]
        assert dim % nheads == 0
        headdim = dim // nheads
    else:
        if D.dim() == 1:
            assert headdim is not None
        else:
            headdim = D.shape[1]
        dim = nheads * headdim
    xBC = rearrange(causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation),
                    "b d s -> b s d")
    dstate = (xBC.shape[-1] - dim) // ngroups // 2
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(), z=z, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    return rearrange(out, "b s h p -> b s (h p)")


class MambaSplitConv1dScanCombinedFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
                rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
                ngroups=1, norm_before_gate=True):
        assert activation in [None, "silu", "swish"]
        if D.dim() == 1:
            assert headdim is not None
            nheads, = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_t = rearrange(xBC, "b s d -> b d s").contiguous()
        xBC_conv_t = torch.empty_like(xBC_t)
        causal_conv1d_cuda.causal_conv1d_fwd(xBC_t, conv1d_weight, conv1d_bias, seq_idx, None, xBC_conv_t, None, activation in ["silu", "swish"])
        xBC_conv = rearrange(xBC_conv_t, "b d s -> b s d")
        x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        if rmsnorm_weight is None:
            out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit)
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
            if d_nonssm > 0:
                out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
        else:
            out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit)
            # reshape input data into 2D tensor
            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            if d_nonssm == 0:
                out = None
            else:
                out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
                out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
                _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
            out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
                                           group_size=dim // ngroups,
                                           norm_before_gate=norm_before_gate, is_rms_norm=True)
            if d_nonssm == 0:
                out = rearrange(out, "(b s) d -> b s d", b=batch)
            else:
                out = out01
        outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
        if outproj_weight is not None:
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
            out = F.linear(out, outproj_weight, outproj_bias)
        else:
            assert outproj_bias is None
        # Return user-facing outputs first, then views of tensors needed for backward
        # (PyTorch requires views, not original tensors, when using setup_context)
        return (out, final_states,
                out_x.view_as(out_x),
                zxbcdt.view_as(zxbcdt), conv1d_weight.view_as(conv1d_weight), conv1d_bias.view_as(conv1d_bias),
                A.view_as(A), D.view_as(D), dt_bias.view_as(dt_bias),
                initial_states.view_as(initial_states) if initial_states is not None else None,
                seq_idx.view_as(seq_idx) if seq_idx is not None else None,
                rmsnorm_weight.view_as(rmsnorm_weight) if rmsnorm_weight is not None else None,
                rstd.view_as(rstd) if rstd is not None else None,
                outproj_weight.view_as(outproj_weight) if outproj_weight is not None else None,
                outproj_bias.view_as(outproj_bias) if outproj_bias is not None else None,
                # Non-tensor context attributes
                outproj_weight_dtype, dt_limit, return_final_states, activation,
                rmsnorm_eps, norm_before_gate, chunk_size, headdim, ngroups)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx,
         dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps,
         outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate) = inputs
        (out, final_states, out_x_saved, zxbcdt_saved, conv1d_weight_saved, conv1d_bias_saved,
         A_saved, D_saved, dt_bias_saved, initial_states_saved, seq_idx_saved,
         rmsnorm_weight_saved, rstd_saved, outproj_weight_saved, outproj_bias_saved,
         outproj_weight_dtype, _dt_limit, _return_final_states, _activation,
         _rmsnorm_eps, _norm_before_gate, _chunk_size, _headdim, _ngroups) = output

        ctx.save_for_backward(zxbcdt_saved, conv1d_weight_saved, conv1d_bias_saved,
                              out_x_saved, A_saved, D_saved, dt_bias_saved,
                              initial_states_saved, seq_idx_saved,
                              rmsnorm_weight_saved, rstd_saved,
                              outproj_weight_saved, outproj_bias_saved)
        ctx.outproj_weight_dtype = outproj_weight_dtype
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        # Required for @custom_bwd compatibility when using setup_context pattern
        ctx._fwd_used_autocast = torch.is_autocast_enabled()
        ctx._dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
        # Track which optional tensors are present
        ctx.has_initial_states = initial_states is not None
        ctx.has_seq_idx = seq_idx is not None
        ctx.has_rmsnorm_weight = rmsnorm_weight is not None
        ctx.has_outproj_weight = outproj_weight is not None
        ctx.has_outproj_bias = outproj_bias is not None

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, dfinal_states, *_):
        zxbcdt, conv1d_weight, conv1d_bias, out, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias = ctx.saved_tensors
        dfinal_states_for_bwd = dfinal_states if ctx.return_final_states else None
        headdim = ctx.headdim
        nheads = D.shape[0]
        dim = nheads * headdim
        assert nheads % ctx.ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ctx.ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ctx.ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        recompute_output = outproj_weight is not None
        if recompute_output:
            out_recompute = torch.empty(*out.shape[:2], d_nonssm + dim, device=out.device, dtype=out.dtype)
            out0_recompute, out1_recompute = out_recompute.split([d_nonssm, dim], dim=-1)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        # Recompute x, B, C
        xBC_t = rearrange(xBC, "b s d -> b d s").contiguous()
        xBC_conv_t = torch.empty_like(xBC_t)
        causal_conv1d_cuda.causal_conv1d_fwd(xBC_t, conv1d_weight, conv1d_bias, seq_idx, None, xBC_conv_t, None, ctx.activation in ["silu", "swish"])
        xBC_conv = rearrange(xBC_conv_t, "b d s -> b s d")
        x, B, C = torch.split(xBC_conv, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ctx.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ctx.ngroups)
        dzxbcdt = torch.empty_like(zxbcdt)
        dzx0, dz, dxBC_given, ddt_given = torch.split(dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)
        dxBC = torch.empty_like(xBC)
        dx, dB, dC = torch.split(dxBC, [dim, ctx.ngroups * dstate, ctx.ngroups * dstate], dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
        dx = rearrange(dx, "b l (h p) -> b l h p", h=nheads)
        dB = rearrange(dB, "b l (g n) -> b l g n", g=ctx.ngroups)
        dC = rearrange(dC, "b l (g n) -> b l g n", g=ctx.ngroups)
        if outproj_weight is not None:
            dout_og = dout
            dout = F.linear(dout, outproj_weight.t())
        if d_nonssm > 0:
            dout0, dout = dout.split([d_nonssm, dim], dim=-1)
            _swiglu_bwd(zx0, dout0, dxy=dzx0, recompute_output=True, out=out0_recompute)
        dout = rearrange(dout, "b s (h p) -> b s h p", p=headdim)
        if rmsnorm_weight is None:
            dz = rearrange(dz, "b l (h p) -> b l h p", h=nheads)
            dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states, *rest = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states_for_bwd, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC, dz=dz, recompute_output=recompute_output
            )
            out_for_linear = rearrange(rest[0], "b s h p -> b s (h p)") if recompute_output else None
            drmsnorm_weight = None
        else:
            batch = dout.shape[0]
            dout = dout.contiguous()  # Ensure contiguous before 2D rearrange for Triton kernel
            dy_rms = rearrange(dout, "b s h p -> (b s) (h p)")
            dz = rearrange(dz, "b l d -> (b l) d")
            x_rms = rearrange(out, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            out1_recompute = rearrange(out1_recompute, "b s d -> (b s) d") if recompute_output else None
            dout, drmsnorm_weight, _, dz, *rest = _layer_norm_bwd(dy_rms, x_rms, rmsnorm_weight, None, ctx.rmsnorm_eps, None, rstd, z_rms, group_size=dim // ctx.ngroups, norm_before_gate=ctx.norm_before_gate, is_rms_norm=True, recompute_output=recompute_output, dz=dz, out=out1_recompute if recompute_output else None)
            out_for_linear = out_recompute if recompute_output else None
            dout = rearrange(dout, "(b s) (h p) -> b s h p", b=batch, p=headdim)
            dx, ddt, dA, dB, dC, dD, _, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(
                dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states_for_bwd, seq_idx=seq_idx, dt_softplus=True, dt_limit=ctx.dt_limit, dx=dx, ddt=ddt_given, dB=dB, dC=dC
            )

        if outproj_weight is not None:
            doutproj_weight = torch.einsum("bso,bsd->od", dout_og, out_for_linear)
            doutproj_bias = dout_og.sum(dim=(0, 1)) if outproj_bias is not None else None
        else:
            doutproj_weight, doutproj_bias = None, None
        dxBC_given = rearrange(dxBC_given, "b s d -> b d s").contiguous()
        dweight = torch.zeros_like(conv1d_weight, dtype=torch.float32)
        dbias = torch.zeros_like(conv1d_bias, dtype=torch.float32) if conv1d_bias is not None else None
        causal_conv1d_cuda.causal_conv1d_bwd(
            rearrange(xBC, "b s d -> b d s").contiguous(), conv1d_weight, conv1d_bias,
            rearrange(dxBC, "b s d -> b d s").contiguous(), seq_idx, None, None, dxBC_given, dweight, dbias, None, ctx.activation in ["silu", "swish"]
        )
        # Cast gradients back to original dtype
        dweight = dweight.to(conv1d_weight.dtype)
        if dbias is not None:
            dbias = dbias.to(conv1d_bias.dtype)
        dxBC_given = rearrange(dxBC_given, "b d s -> b s d")
        return dzxbcdt, dweight, dbias, ddt_bias, dA, dD, None, dinitial_states, None, None, None, None, drmsnorm_weight, None, doutproj_weight, doutproj_bias, None, None, None

    @staticmethod
    def vmap(info, in_dims, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
             initial_states, seq_idx, dt_limit, return_final_states, activation,
             rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate):
        # Handle vmap by moving batch dim to front and merging with existing batch
        def move_bdim_to_front(tensor, bdim):
            if tensor is None or bdim is None:
                return tensor
            return tensor.movedim(bdim, 0)

        # in_dims order: zxbcdt(0), conv1d_weight(1), conv1d_bias(2), dt_bias(3), A(4), D(5), chunk_size(6),
        #                initial_states(7), seq_idx(8), dt_limit(9), return_final_states(10), activation(11),
        #                rmsnorm_weight(12), rmsnorm_eps(13), outproj_weight(14), outproj_bias(15),
        #                headdim(16), ngroups(17), norm_before_gate(18)
        # Tensor inputs: zxbcdt(0), conv1d_weight(1), conv1d_bias(2), dt_bias(3), A(4), D(5),
        #                initial_states(7), seq_idx(8), rmsnorm_weight(12), outproj_weight(14), outproj_bias(15)

        # Get vmap batch size from first batched tensor input
        vmap_batch_size = None
        tensor_indices = [0, 1, 2, 3, 4, 5, 7, 8, 12, 14, 15]
        tensors = [zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D,
                   initial_states, seq_idx, rmsnorm_weight, outproj_weight, outproj_bias]
        for i, tensor_idx in enumerate(tensor_indices):
            bdim = in_dims[tensor_idx]
            tensor = tensors[i]
            if bdim is not None and tensor is not None:
                vmap_batch_size = tensor.shape[bdim]
                break

        if vmap_batch_size is None:
            # No batched inputs, just call directly
            result = MambaSplitConv1dScanCombinedFn.apply(
                zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size,
                initial_states, seq_idx, dt_limit, return_final_states, activation,
                rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate
            )
            # Return None for all output dims since no batching
            return result, (None,) * len(result)

        # Move batch dims to front
        zxbcdt_batched = move_bdim_to_front(zxbcdt, in_dims[0])
        conv1d_weight_batched = move_bdim_to_front(conv1d_weight, in_dims[1])
        conv1d_bias_batched = move_bdim_to_front(conv1d_bias, in_dims[2])
        dt_bias_batched = move_bdim_to_front(dt_bias, in_dims[3])
        A_batched = move_bdim_to_front(A, in_dims[4])
        D_batched = move_bdim_to_front(D, in_dims[5])
        initial_states_batched = move_bdim_to_front(initial_states, in_dims[7])
        seq_idx_batched = move_bdim_to_front(seq_idx, in_dims[8])
        rmsnorm_weight_batched = move_bdim_to_front(rmsnorm_weight, in_dims[12])
        outproj_weight_batched = move_bdim_to_front(outproj_weight, in_dims[14])
        outproj_bias_batched = move_bdim_to_front(outproj_bias, in_dims[15])

        # Broadcast non-batched inputs (expand to vmap_batch_size)
        # zxbcdt: (batch, seqlen, input_dim)
        if in_dims[0] is None:
            zxbcdt_batched = zxbcdt_batched.unsqueeze(0).expand(vmap_batch_size, *zxbcdt_batched.shape)
        # initial_states: optional, (batch, nheads, headdim, dstate)
        if initial_states is not None and in_dims[7] is None:
            initial_states_batched = initial_states_batched.unsqueeze(0).expand(vmap_batch_size, *initial_states_batched.shape)
        # seq_idx: optional, (batch, seqlen)
        if seq_idx is not None and in_dims[8] is None:
            seq_idx_batched = seq_idx_batched.unsqueeze(0).expand(vmap_batch_size, *seq_idx_batched.shape)

        # Merge vmap batch dim with tensor batch dim
        # zxbcdt: (vmap_batch, batch, seqlen, input_dim) -> (vmap_batch * batch, seqlen, input_dim)
        zxbcdt_shape = zxbcdt_batched.shape
        zxbcdt_merged = zxbcdt_batched.reshape(zxbcdt_shape[0] * zxbcdt_shape[1], *zxbcdt_shape[2:]).contiguous()

        # Shared parameters - use directly or take first if batched
        # conv1d_weight: (dim + 2 * ngroups * dstate, width) - typically shared
        conv1d_weight_merged = conv1d_weight_batched[0] if in_dims[1] is not None else conv1d_weight
        # conv1d_bias: (dim + 2 * ngroups * dstate,) - typically shared
        conv1d_bias_merged = conv1d_bias_batched[0] if in_dims[2] is not None else conv1d_bias
        # dt_bias: (nheads,) - typically shared
        dt_bias_merged = dt_bias_batched[0] if in_dims[3] is not None else dt_bias
        # A: (nheads,) - typically shared
        A_merged = A_batched[0] if in_dims[4] is not None else A
        # D: (nheads, headdim) or (nheads,) - typically shared
        D_merged = D_batched[0] if in_dims[5] is not None else D

        # initial_states: optional, (vmap_batch, batch, nheads, headdim, dstate) -> (vmap_batch * batch, nheads, headdim, dstate)
        initial_states_merged = None
        if initial_states_batched is not None:
            is_shape = initial_states_batched.shape
            initial_states_merged = initial_states_batched.reshape(is_shape[0] * is_shape[1], *is_shape[2:]).contiguous()

        # seq_idx: optional, (vmap_batch, batch, seqlen) -> (vmap_batch * batch, seqlen)
        seq_idx_merged = None
        if seq_idx_batched is not None:
            si_shape = seq_idx_batched.shape
            seq_idx_merged = seq_idx_batched.reshape(si_shape[0] * si_shape[1], *si_shape[2:]).contiguous()

        # rmsnorm_weight: optional, (dim,) - typically shared
        rmsnorm_weight_merged = None
        if rmsnorm_weight is not None:
            rmsnorm_weight_merged = rmsnorm_weight_batched[0] if in_dims[12] is not None else rmsnorm_weight

        # outproj_weight: optional, (out_dim, dim) - typically shared
        outproj_weight_merged = None
        if outproj_weight is not None:
            outproj_weight_merged = outproj_weight_batched[0] if in_dims[14] is not None else outproj_weight

        # outproj_bias: optional, (out_dim,) - typically shared
        outproj_bias_merged = None
        if outproj_bias is not None:
            outproj_bias_merged = outproj_bias_batched[0] if in_dims[15] is not None else outproj_bias

        # Call the function with merged batches
        result = MambaSplitConv1dScanCombinedFn.apply(
            zxbcdt_merged, conv1d_weight_merged, conv1d_bias_merged, dt_bias_merged,
            A_merged, D_merged, chunk_size,
            initial_states_merged, seq_idx_merged, dt_limit, return_final_states, activation,
            rmsnorm_weight_merged, rmsnorm_eps, outproj_weight_merged, outproj_bias_merged,
            headdim, ngroups, norm_before_gate
        )

        # Unmerge outputs
        # Output structure: (out, final_states, out_x, zxbcdt, conv1d_weight, conv1d_bias,
        #                    A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd,
        #                    outproj_weight, outproj_bias,
        #                    outproj_weight_dtype, dt_limit, return_final_states, activation,
        #                    rmsnorm_eps, norm_before_gate, chunk_size, headdim, ngroups)
        merged_batch = result[0].shape[0]
        original_batch = merged_batch // vmap_batch_size

        outputs_unmerged = []
        out_dims_list = []

        # Indices of shared parameters that shouldn't be unmerged
        shared_indices = [4, 5, 6, 7, 8, 11, 13, 14]  # conv1d_weight, conv1d_bias, A, D, dt_bias, rmsnorm_weight, outproj_weight, outproj_bias

        for idx, out in enumerate(result):
            if out is None:
                outputs_unmerged.append(None)
                out_dims_list.append(None)
            elif idx in shared_indices:  # Shared parameters (not batched)
                outputs_unmerged.append(out)
                out_dims_list.append(None)
            elif idx >= 15:  # Non-tensor outputs: outproj_weight_dtype, dt_limit, return_final_states, etc.
                outputs_unmerged.append(out)
                out_dims_list.append(None)
            elif isinstance(out, torch.Tensor) and out.dim() > 0:
                # Tensor outputs that need unmerging
                out_unmerged = out.reshape(vmap_batch_size, original_batch, *out.shape[1:])
                outputs_unmerged.append(out_unmerged)
                out_dims_list.append(0)
            else:
                outputs_unmerged.append(out)
                out_dims_list.append(None)

        return tuple(outputs_unmerged), tuple(out_dims_list)


def mamba_split_conv1d_scan_combined(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
        or (out, final_states) if return_final_states is True
    """
    result = MambaSplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate)
    # Extract user-facing outputs: result[0] is out, result[1] is final_states
    out, final_states = result[0], result[1]
    if return_final_states:
        return out, final_states
    else:
        return out


def mamba_split_conv1d_scan_ref(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, dt_limit=(0.0, float("inf")), activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    if D.dim() == 1:
        assert headdim is not None
        nheads, = D.shape
    else:
        nheads, headdim = D.shape
    assert nheads % ngroups == 0
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // ngroups // 2
    assert zxbcdt.shape == (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads)
    assert dt_bias.shape == (nheads,)
    assert A.shape == (nheads,)
    if rmsnorm_weight is not None:
        assert rmsnorm_weight.shape == (dim,)
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * ngroups * dstate, nheads], dim=-1)
    xBC = rearrange(causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation),
                    "b d s -> b s d")
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(),
                             z=z if rmsnorm_weight is None else None, dt_bias=dt_bias, dt_softplus=True, dt_limit=dt_limit)
    out = rearrange(out, "b s h p -> b s (h p)")
    if rmsnorm_weight is not None:
        out = rmsnorm_fn(out, rmsnorm_weight, None, z=rearrange(z, "b l h p -> b l (h p)"), eps=rmsnorm_eps,
                         norm_before_gate=norm_before_gate)
    if outproj_weight is not None:
        out = F.linear(out, outproj_weight, outproj_bias)
    return out
