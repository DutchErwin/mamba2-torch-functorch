# Copyright (c) 2024, Tri Dao.
# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This backward pass is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

import math

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange


def rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        B,  # pointer to the biases
        Z,  # pointer to the other branch
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        stride_x_row,  # how much to increase the pointer when moving by 1 row
        stride_y_row,
        stride_z_row,
        M,  # number of rows in X
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_Z: tl.constexpr,
        NORM_BEFORE_GATE: tl.constexpr,
        IS_RMS_NORM: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x *= z * tl.sigmoid(z)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask).to(tl.float32)
        y *= z * tl.sigmoid(z)
    # Write output
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(x, weight, bias, eps, z=None, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1
    mean = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[grid](x, out, weight, bias, z, mean, rstd,
                                           x.stride(0), out.stride(0), z.stride(0) if z is not None else 0,
                                           M, group_size, eps,
                                           BLOCK_N=BLOCK_N,
                                           NORM_BEFORE_GATE=norm_before_gate,
                                           IS_RMS_NORM=is_rms_norm,
                                           num_warps=num_warps)
    return out, mean, rstd



@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
        X,   # pointer to the input
        W,   # pointer to the weights
        B,   # pointer to the biases
        Z,   # pointer to the other branch
        Y,   # pointer to the output to be recomputed
        DY,  # pointer to the output gradient
        DX,  # pointer to the input gradient
        DW,  # pointer to the partial sum of weights gradient
        DB,  # pointer to the partial sum of biases gradient
        DZ,  # pointer to the other branch
        Mean,   # pointer to the mean
        Rstd,   # pointer to the 1/std
        stride_x_row,  # how much to increase the pointer when moving by 1 row
        stride_z_row,
        stride_y_row,
        stride_dy_row,
        stride_dx_row,
        stride_dz_row,
        stride_dw_row,
        stride_db_row,
        M,  # number of rows in X
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        rows_per_program,
        NORM_BEFORE_GATE: tl.constexpr,
        IS_RMS_NORM: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_Z: tl.constexpr,
        RECOMPUTE_OUTPUT: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    group = tl.program_id(1)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row + group * N
    if HAS_Z:
        Z += row_start * stride_z_row + group * N
        DZ += row_start * stride_dz_row + group * N
    DY += row_start * stride_dy_row + group * N
    DX += row_start * stride_dx_row + group * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if (RECOMPUTE_OUTPUT or HAS_Z) and HAS_BIAS:
        B += group * N
        b = tl.load(B + cols, mask=mask, other=0.).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.).to(tl.float32)
            x_og = x
            x = x_og * z * tl.sigmoid(z)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.)
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.).to(tl.float32)
            z_sigmoid = tl.sigmoid(z)
            y = xhat * w + b if HAS_BIAS else xhat * w
            if RECOMPUTE_OUTPUT:
                tl.store(Y + cols, y * z * z_sigmoid, mask=mask)
            dz = dy * y * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dy *= z * z_sigmoid
        else:
            if RECOMPUTE_OUTPUT:
                y = xhat * w + b if HAS_BIAS else xhat * w
                tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=0) / N
        if not IS_RMS_NORM:
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            dx = (wdy - xhat * c1) * rstd
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_Z and not NORM_BEFORE_GATE:
            z_sigmoid = tl.sigmoid(z)
            dz = dx * x_og * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dx *= z * z_sigmoid
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_Z:
            Z += stride_z_row
            DZ += stride_dz_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * stride_dw_row + group * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * stride_db_row + group * N + cols, db, mask=mask)


def _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, z=None, group_size=None,
                    norm_before_gate=True, is_rms_norm=False, recompute_output=False, dz=None, out=None):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    dx = torch.empty_like(x)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
        assert dz.stride(-1) == 1
    else:
        dz = torch.empty_like(z) if z is not None else None
    if recompute_output:
        if out is None:
            out = torch.empty_like(x)
        assert out.shape == x.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    # If group size is small (e.g., 64), we're only using 1 warp. So having just 108 programs
    # would limit the occupancy.
    nrow_groups = math.ceil(sm_count * math.ceil(4 / num_warps) / ngroups)
    _dw = torch.empty((nrow_groups, N), dtype=torch.float32, device=weight.device)
    _db = torch.empty((nrow_groups, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / nrow_groups)
    grid = (nrow_groups, ngroups)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](x, weight, bias, z, out if recompute_output else None,
                                     dy, dx, _dw, _db, dz, mean, rstd,
                                     x.stride(0),
                                     z.stride(0) if z is not None else 0,
                                     0 if not recompute_output else out.stride(0),
                                     dy.stride(0), dx.stride(0),
                                     dz.stride(0) if dz is not None else 0,
                                     _dw.stride(0),
                                     _db.stride(0) if _db is not None else 0,
                                     M, group_size, eps,
                                     rows_per_program,
                                     BLOCK_N=BLOCK_N,
                                     NORM_BEFORE_GATE=norm_before_gate,
                                     IS_RMS_NORM=is_rms_norm,
                                     num_warps=num_warps)
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    return (dx, dw, db, dz) if not recompute_output else (dx, dw, db, dz, out)


class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True,
                is_rms_norm=False):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = _layer_norm_fwd(x, weight, bias, eps, z=z, group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=is_rms_norm)
        # Return output and views of tensors needed for backward
        # (PyTorch requires views, not the original tensors, when using setup_context)
        return (
            y.reshape(x_shape_og),
            x.view_as(x),
            weight.view_as(weight),
            bias.view_as(bias) if bias is not None else None,
            mean,  # mean is newly created, no need for view_as
            rstd,  # rstd is newly created, no need for view_as
            z.view_as(z) if z is not None else None,
            x_shape_og,  # Pass shape as part of output (will be unpacked in setup_context)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm = inputs
        y, x_saved, weight_saved, bias_saved, mean, rstd, z_saved, x_shape_og = output
        # Save tensors (handle None values)
        if bias_saved is not None and z_saved is not None:
            ctx.save_for_backward(x_saved, weight_saved, bias_saved, mean, rstd, z_saved)
        elif bias_saved is not None:
            ctx.save_for_backward(x_saved, weight_saved, bias_saved, mean, rstd)
        elif z_saved is not None:
            ctx.save_for_backward(x_saved, weight_saved, mean, rstd, z_saved)
        else:
            ctx.save_for_backward(x_saved, weight_saved, mean, rstd)
        ctx.has_bias = bias_saved is not None
        ctx.has_z = z_saved is not None
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm

    @staticmethod
    def backward(ctx, dy, *_):
        # Unpack saved tensors based on what was saved
        saved = ctx.saved_tensors
        if ctx.has_bias and ctx.has_z:
            x, weight, bias, mean, rstd, z = saved
        elif ctx.has_bias:
            x, weight, bias, mean, rstd = saved
            z = None
        elif ctx.has_z:
            x, weight, mean, rstd, z = saved
            bias = None
        else:
            x, weight, mean, rstd = saved
            bias = None
            z = None

        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        dx, dw, db, dz = _layer_norm_bwd(dy, x, weight, bias, ctx.eps, mean, rstd, z, ctx.group_size,
                                         ctx.norm_before_gate, ctx.is_rms_norm)
        return dx.reshape(ctx.x_shape_og), dw, db, dz.reshape(ctx.x_shape_og) if dz is not None else None, None, None, None, None

    @staticmethod
    def vmap(info, in_dims, x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm):
        # Handle vmap by moving batch dim to front and merging with existing batch
        def move_bdim_to_front(tensor, bdim):
            if bdim is None or tensor is None:
                return tensor
            return tensor.movedim(bdim, 0)

        # Get vmap batch size from first batched input
        vmap_batch_size = None
        for tensor, bdim in zip([x, weight, bias, z], in_dims[:4]):
            if bdim is not None and tensor is not None:
                vmap_batch_size = tensor.shape[bdim]
                break

        # Move batch dims to front
        x_batched = move_bdim_to_front(x, in_dims[0])
        weight_batched = move_bdim_to_front(weight, in_dims[1])
        bias_batched = move_bdim_to_front(bias, in_dims[2]) if bias is not None else None
        z_batched = move_bdim_to_front(z, in_dims[3]) if z is not None else None

        # For x, merge vmap batch with tensor's leading dims
        # x: (..., hidden_size) - can have any leading dims
        # After vmap: (vmap_batch, ..., hidden_size)
        if in_dims[0] is None:
            # x not batched, broadcast by adding vmap_batch dim
            x_batched = x_batched.unsqueeze(0).expand(vmap_batch_size, *x_batched.shape)

        # Merge vmap batch into first dimension
        # x: (vmap_batch, batch, ..., hidden_size) -> (vmap_batch * batch, ..., hidden_size)
        x_shape = x_batched.shape
        x_merged = x_batched.reshape(x_shape[0] * x_shape[1], *x_shape[2:])

        # Handle z similarly if present
        z_merged = None
        if z_batched is not None:
            if in_dims[3] is None:
                z_batched = z_batched.unsqueeze(0).expand(vmap_batch_size, *z_batched.shape)
            z_shape = z_batched.shape
            z_merged = z_batched.reshape(z_shape[0] * z_shape[1], *z_shape[2:])

        # weight and bias are typically not batched (shared across batch)
        # but handle the case if they are
        if in_dims[1] is not None:
            # weight is batched - this is unusual but handle it
            weight_shape = weight_batched.shape
            weight_merged = weight_batched.reshape(weight_shape[0] * weight_shape[1], *weight_shape[2:])
        else:
            weight_merged = weight_batched

        if bias_batched is not None:
            if in_dims[2] is not None:
                bias_shape = bias_batched.shape
                bias_merged = bias_batched.reshape(bias_shape[0] * bias_shape[1], *bias_shape[2:])
            else:
                bias_merged = bias_batched
        else:
            bias_merged = None

        # Call the function with merged batches
        result = LayerNormFn.apply(
            x_merged, weight_merged, bias_merged, z_merged, eps, group_size, norm_before_gate, is_rms_norm
        )

        # Unpack result
        y, x_out, weight_out, bias_out, mean_out, rstd_out, z_out, x_shape_og = result

        # Unmerge y: shape is (vmap_batch * batch, ..., hidden_size)
        # -> (vmap_batch, batch, ..., hidden_size)
        merged_batch = y.shape[0]
        original_batch = merged_batch // vmap_batch_size
        y_unmerged = y.reshape(vmap_batch_size, original_batch, *y.shape[1:])

        # x_out is 2D (M, N) where M = total_elements / N, N = hidden_size
        # It was flattened inside forward(). We need to unmerge along the M dimension.
        # x_out shape: (vmap_batch * original_M, hidden_size)
        # -> (vmap_batch, original_M, hidden_size)
        original_M = x_out.shape[0] // vmap_batch_size
        x_unmerged = x_out.reshape(vmap_batch_size, original_M, x_out.shape[-1])

        # weight and bias outputs - these are not reshaped internally
        if in_dims[1] is not None:
            weight_unmerged = weight_out.reshape(vmap_batch_size, -1)
        else:
            weight_unmerged = weight_out

        if bias_out is not None:
            if in_dims[2] is not None:
                bias_unmerged = bias_out.reshape(vmap_batch_size, -1)
            else:
                bias_unmerged = bias_out
        else:
            bias_unmerged = None

        # mean and rstd - these are 1D tensors of shape (ngroups * M,)
        # For simplicity, unmerge along the first dimension
        if mean_out is not None:
            original_mean_size = mean_out.shape[0] // vmap_batch_size
            mean_unmerged = mean_out.reshape(vmap_batch_size, original_mean_size)
        else:
            mean_unmerged = None
        original_rstd_size = rstd_out.shape[0] // vmap_batch_size
        rstd_unmerged = rstd_out.reshape(vmap_batch_size, original_rstd_size)

        # z_out is also 2D if present, same treatment as x_out
        if z_out is not None:
            original_z_M = z_out.shape[0] // vmap_batch_size
            z_unmerged = z_out.reshape(vmap_batch_size, original_z_M, z_out.shape[-1])
        else:
            z_unmerged = None

        # Return (outputs, out_dims) - all outputs have vmap batch dim at position 0
        # Outputs: (y, x, weight, bias, mean, rstd, z, x_shape_og)
        out_dims = (
            0,  # y
            0,  # x
            0 if in_dims[1] is not None else None,  # weight
            0 if bias_out is not None and in_dims[2] is not None else None,  # bias
            0 if mean_unmerged is not None else None,  # mean
            0,  # rstd
            0 if z_out is not None else None,  # z
            None,  # x_shape_og (not a tensor)
        )

        return (y_unmerged, x_unmerged, weight_unmerged, bias_unmerged, mean_unmerged, rstd_unmerged, z_unmerged, x_shape_og), out_dims


def layernorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=False):
    result = LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)
    return result[0]  # Extract only user-facing output (y)


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    result = LayerNormFn.apply(x, weight, bias, z, eps, group_size, norm_before_gate, True)
    return result[0]  # Extract only user-facing output (y)


class LayerNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return layernorm_fn(x, self.weight, self.bias, z=z, group_size=self.group_size, eps=self.eps,
                            norm_before_gate=self.norm_before_gate)


class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        return rmsnorm_fn(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                          norm_before_gate=self.norm_before_gate)