# Copyright (c) 2024, Tri Dao.
# Implement dropout + residual + layer_norm / rms_norm.

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

import math
import warnings

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ..ops.custom_fwd_bwd import custom_fwd, custom_bwd


def layer_norm_ref(
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        dropout_mask=None,
        dropout_mask1=None,
        upcast=False,
):
    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else residual
        x1 = x1.float() if x1 is not None else None
        weight1 = weight1.float() if weight1 is not None else None
        bias1 = bias1.float() if bias1 is not None else None
    if x1 is not None:
        assert rowscale is None, "rowscale is not supported with parallel LayerNorm"
    if rowscale is not None:
        x = x * rowscale[..., None]
    if dropout_p > 0.0:
        if dropout_mask is not None:
            x = x.masked_fill(~dropout_mask, 0.0) / (1.0 - dropout_p)
        else:
            x = F.dropout(x, p=dropout_p)
        if x1 is not None:
            if dropout_mask1 is not None:
                x1 = x1.masked_fill(~dropout_mask1, 0.0) / (1.0 - dropout_p)
            else:
                x1 = F.dropout(x1, p=dropout_p)
    if x1 is not None:
        x = x + x1
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(
        dtype
    )
    if weight1 is None:
        return out if not prenorm else (out, x)
    else:
        out1 = F.layer_norm(
            x.to(weight1.dtype), x.shape[-1:], weight=weight1, bias=bias1, eps=eps
        ).to(dtype)
        return (out, out1) if not prenorm else (out, out1, x)


def rms_norm_ref(
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        dropout_mask=None,
        dropout_mask1=None,
        upcast=False,
):
    dtype = x.dtype
    if upcast:
        x = x.float()
        weight = weight.float()
        bias = bias.float() if bias is not None else None
        residual = residual.float() if residual is not None else residual
        x1 = x1.float() if x1 is not None else None
        weight1 = weight1.float() if weight1 is not None else None
        bias1 = bias1.float() if bias1 is not None else None
    if x1 is not None:
        assert rowscale is None, "rowscale is not supported with parallel LayerNorm"
    if rowscale is not None:
        x = x * rowscale[..., None]
    if dropout_p > 0.0:
        if dropout_mask is not None:
            x = x.masked_fill(~dropout_mask, 0.0) / (1.0 - dropout_p)
        else:
            x = F.dropout(x, p=dropout_p)
        if x1 is not None:
            if dropout_mask1 is not None:
                x1 = x1.masked_fill(~dropout_mask1, 0.0) / (1.0 - dropout_p)
            else:
                x1 = F.dropout(x1, p=dropout_p)
    if x1 is not None:
        x = x + x1
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = ((x * rstd * weight) + bias if bias is not None else (x * rstd * weight)).to(dtype)
    if weight1 is None:
        return out if not prenorm else (out, x)
    else:
        out1 = ((x * rstd * weight1) + bias1 if bias1 is not None else (x * rstd * weight1)).to(
            dtype
        )
        return (out, out1) if not prenorm else (out, out1, x)

def config_prune(configs):

    if torch.version.hip:
        try:
            # set warp size based on gcn architecure
            gcn_arch_name = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx10" in gcn_arch_name or "gfx11" in gcn_arch_name:
                # radeon
                warp_size = 32
            else:
                # instinct
                warp_size = 64
        except AttributeError as e:
            # fall back to crude method to set warp size
            device_name = torch.cuda.get_device_properties(0).name
            if 'instinct' in device_name.lower():
                warp_size = 64
            else:
                warp_size = 32
            warnings.warn(f"{e}, warp size set to {warp_size} based on device name: {device_name}", UserWarning)

    else:
        # cuda
        warp_size = 32

    max_block_sz = 1024
    max_num_warps = max_block_sz // warp_size
    pruned_configs = [config for config in configs if config.num_warps <= max_num_warps]
    return pruned_configs

configs_autotune = [
    triton.Config({}, num_warps=1),
    triton.Config({}, num_warps=2),
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
    triton.Config({}, num_warps=32),
]

pruned_configs_autotune = config_prune(configs_autotune)

@triton.autotune(
    configs = pruned_configs_autotune,
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None})
@triton.heuristics({"HAS_X1": lambda args: args["X1"] is not None})
@triton.heuristics({"HAS_W1": lambda args: args["W1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["B1"] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        B,  # pointer to the biases
        RESIDUAL,  # pointer to the residual
        X1,
        W1,
        B1,
        Y1,
        RESIDUAL_OUT,  # pointer to the residual
        ROWSCALE,
        SEEDS,  # Dropout seeds for each row
        DROPOUT_MASK,
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        stride_x_row,  # how much to increase the pointer when moving by 1 row
        stride_y_row,
        stride_res_row,
        stride_res_out_row,
        stride_x1_row,
        stride_y1_row,
        M,  # number of rows in X
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        dropout_p,  # Dropout probability
        IS_RMS_NORM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        STORE_RESIDUAL_OUT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_DROPOUT: tl.constexpr,
        STORE_DROPOUT_MASK: tl.constexpr,
        HAS_ROWSCALE: tl.constexpr,
        HAS_X1: tl.constexpr,
        HAS_W1: tl.constexpr,
        HAS_B1: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        # Compute dropout mask
        # 7 rounds is good enough, and reduces register pressure
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            # Compute dropout mask
            # 7 rounds is good enough, and reduces register pressure
            keep_mask = (
                    tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            )
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        x += x1
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
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
    # Write output
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)


def _layer_norm_fwd(
        x,
        weight,
        bias,
        eps,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        dropout_p=0.0,
        rowscale=None,
        out_dtype=None,
        residual_dtype=None,
        is_rms_norm=False,
        return_dropout_mask=False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if weight1 is not None:
        y1 = torch.empty_like(y)
        assert y1.stride(-1) == 1
    else:
        y1 = None
    if (
            residual is not None
            or (residual_dtype is not None and residual_dtype != x.dtype)
            or dropout_p > 0.0
            or rowscale is not None
            or x1 is not None
    ):
        residual_out = torch.empty(
            M, N, device=x.device, dtype=residual_dtype if residual_dtype is not None else x.dtype
        )
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    if dropout_p > 0.0:
        seeds = torch.randint(
            2**32, (M if x1 is None else 2 * M,), device=x.device, dtype=torch.int64
        )
    else:
        seeds = None
    if return_dropout_mask and dropout_p > 0.0:
        dropout_mask = torch.empty(M if x1 is None else 2 * M, N, device=x.device, dtype=torch.bool)
    else:
        dropout_mask = None
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x,
            y,
            weight,
            bias,
            residual,
            x1,
            weight1,
            bias1,
            y1,
            residual_out,
            rowscale,
            seeds,
            dropout_mask,
            mean,
            rstd,
            x.stride(0),
            y.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            x1.stride(0) if x1 is not None else 0,
            y1.stride(0) if y1 is not None else 0,
            M,
            N,
            eps,
            dropout_p,
            is_rms_norm,
            BLOCK_N,
            residual is not None,
            residual_out is not None,
            bias is not None,
            dropout_p > 0.0,
            dropout_mask is not None,
            rowscale is not None,
            )
    # residual_out is None if residual is None and residual_dtype == input_dtype and dropout_p == 0.0
    if dropout_mask is not None and x1 is not None:
        dropout_mask, dropout_mask1 = dropout_mask.tensor_split(2, dim=0)
    else:
        dropout_mask1 = None
    return (
        y,
        y1,
        mean,
        rstd,
        residual_out if residual_out is not None else x,
        seeds,
        dropout_mask,
        dropout_mask1,
    )


@triton.autotune(
    configs=pruned_configs_autotune,
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS", "HAS_DROPOUT"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
@triton.heuristics({"HAS_ROWSCALE": lambda args: args["ROWSCALE"] is not None})
@triton.heuristics({"HAS_DY1": lambda args: args["DY1"] is not None})
@triton.heuristics({"HAS_DX1": lambda args: args["DX1"] is not None})
@triton.heuristics({"HAS_B1": lambda args: args["DB1"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
        X,  # pointer to the input
        W,  # pointer to the weights
        B,  # pointer to the biases
        Y,  # pointer to the output to be recomputed
        DY,  # pointer to the output gradient
        DX,  # pointer to the input gradient
        DW,  # pointer to the partial sum of weights gradient
        DB,  # pointer to the partial sum of biases gradient
        DRESIDUAL,
        W1,
        DY1,
        DX1,
        DW1,
        DB1,
        DRESIDUAL_IN,
        ROWSCALE,
        SEEDS,
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        stride_x_row,  # how much to increase the pointer when moving by 1 row
        stride_y_row,
        stride_dy_row,
        stride_dx_row,
        stride_dres_row,
        stride_dy1_row,
        stride_dx1_row,
        stride_dres_in_row,
        M,  # number of rows in X
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        dropout_p,
        rows_per_program,
        IS_RMS_NORM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_DRESIDUAL: tl.constexpr,
        STORE_DRESIDUAL: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_DROPOUT: tl.constexpr,
        HAS_ROWSCALE: tl.constexpr,
        HAS_DY1: tl.constexpr,
        HAS_DX1: tl.constexpr,
        HAS_B1: tl.constexpr,
        RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    # Do not early exit if row_start >= M, because we need to write DW and DB
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_DY1:
        DY1 += row_start * stride_dy1_row
    if HAS_DX1:
        DX1 += row_start * stride_dx1_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DY1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_DY1:
        dw1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if HAS_B1:
            db1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if HAS_DY1:
            dy1 = tl.load(DY1 + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_DY1:
            wdy += w1 * dy1
            dw1 += dy1 * xhat
            if HAS_B1:
                db1 += dy1
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        if HAS_DX1:
            if HAS_DROPOUT:
                keep_mask = (
                        tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
                )
                dx1 = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
            else:
                dx1 = dx
            tl.store(DX1 + cols, dx1, mask=mask)
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            dx = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + row).to(tl.float32)
            dx *= rowscale
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
        if HAS_DY1:
            DY1 += stride_dy1_row
        if HAS_DX1:
            DX1 += stride_dx1_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)
    if HAS_DY1:
        tl.store(DW1 + row_block_id * N + cols, dw1, mask=mask)
        if HAS_B1:
            tl.store(DB1 + row_block_id * N + cols, db1, mask=mask)


def _layer_norm_bwd(
        dy,
        x,
        weight,
        bias,
        eps,
        mean,
        rstd,
        dresidual=None,
        dy1=None,
        weight1=None,
        bias1=None,
        seeds=None,
        dropout_p=0.0,
        rowscale=None,
        has_residual=False,
        has_x1=False,
        is_rms_norm=False,
        x_dtype=None,
        recompute_output=False,
):
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if dy1 is not None:
        assert weight1 is not None
        assert dy1.shape == dy.shape
        assert dy1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if seeds is not None:
        assert seeds.is_contiguous()
        assert seeds.shape == (M if not has_x1 else M * 2,)
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    # allocate output
    dx = (
        torch.empty_like(x)
        if x_dtype is None
        else torch.empty(M, N, dtype=x_dtype, device=x.device)
    )
    dresidual_in = (
        torch.empty_like(x)
        if has_residual
           and (dx.dtype != x.dtype or dropout_p > 0.0 or rowscale is not None or has_x1)
        else None
    )
    dx1 = torch.empty_like(dx) if (has_x1 and dropout_p > 0.0) else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None
    if recompute_output:
        assert weight1 is None, "recompute_output is not supported with parallel LayerNorm"

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    _db = (
        torch.empty((sm_count, N), dtype=torch.float32, device=bias.device)
        if bias is not None
        else None
    )
    _dw1 = torch.empty_like(_dw) if weight1 is not None else None
    _db1 = torch.empty_like(_db) if bias1 is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x,
            weight,
            bias,
            y,
            dy,
            dx,
            _dw,
            _db,
            dresidual,
            weight1,
            dy1,
            dx1,
            _dw1,
            _db1,
            dresidual_in,
            rowscale,
            seeds,
            mean,
            rstd,
            x.stride(0),
            0 if not recompute_output else y.stride(0),
            dy.stride(0),
            dx.stride(0),
            dresidual.stride(0) if dresidual is not None else 0,
            dy1.stride(0) if dy1 is not None else 0,
            dx1.stride(0) if dx1 is not None else 0,
            dresidual_in.stride(0) if dresidual_in is not None else 0,
            M,
            N,
            eps,
            dropout_p,
            rows_per_program,
            is_rms_norm,
            BLOCK_N,
            dresidual is not None,
            dresidual_in is not None,
            bias is not None,
            dropout_p > 0.0,
            )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    dw1 = _dw1.sum(0).to(weight1.dtype) if weight1 is not None else None
    db1 = _db1.sum(0).to(bias1.dtype) if bias1 is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype and dropout_p == 0.0 and rowscale is None:
        dresidual_in = dx
    if has_x1 and dropout_p == 0.0:
        dx1 = dx
    return (
        (dx, dw, db, dresidual_in, dx1, dw1, db1)
        if not recompute_output
        else (dx, dw, db, dresidual_in, dx1, dw1, db1, y)
    )


class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
            x,
            weight,
            bias,
            residual=None,
            x1=None,
            weight1=None,
            bias1=None,
            eps=1e-6,
            dropout_p=0.0,
            rowscale=None,
            prenorm=False,
            residual_in_fp32=False,
            is_rms_norm=False,
            return_dropout_mask=False,
    ):
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        if x1 is not None:
            assert x1.shape == x_shape_og
            assert rowscale is None, "rowscale is not supported with parallel LayerNorm"
            x1 = x1.reshape(-1, x1.shape[-1])
            if x1.stride(-1) != 1:
                x1 = x1.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        if weight1 is not None:
            weight1 = weight1.contiguous()
        if bias1 is not None:
            bias1 = bias1.contiguous()
        if rowscale is not None:
            rowscale = rowscale.reshape(-1).contiguous()
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, y1_out, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1 = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            residual,
            x1,
            weight1,
            bias1,
            dropout_p=dropout_p,
            rowscale=rowscale,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            return_dropout_mask=return_dropout_mask,
        )
        # Track which inputs were provided
        has_residual = residual is not None
        has_x1 = x1 is not None

        # Reshape outputs to original shape
        y = y.reshape(x_shape_og)
        y1_out = y1_out.reshape(x_shape_og) if y1_out is not None else None
        residual_out_reshaped = residual_out.reshape(x_shape_og) if residual_out is not None else None
        dropout_mask = dropout_mask.reshape(x_shape_og) if dropout_mask is not None else None
        dropout_mask1 = dropout_mask1.reshape(x_shape_og) if dropout_mask1 is not None else None

        # Return a consistent tuple structure:
        # (y, y1, residual_out, dropout_mask, dropout_mask1,
        #  residual_out_saved, weight_saved, bias_saved, weight1_saved, bias1_saved,
        #  rowscale_saved, seeds_saved, mean_saved, rstd_saved,
        #  x_shape_og, has_residual, has_x1)
        return (
            y,
            y1_out,
            residual_out_reshaped,
            dropout_mask,
            dropout_mask1,
            # Saved tensors for backward (use view_as for tensors that came from input)
            residual_out.view_as(residual_out) if residual_out is not None else None,
            weight.view_as(weight),
            bias.view_as(bias) if bias is not None else None,
            weight1.view_as(weight1) if weight1 is not None else None,
            bias1.view_as(bias1) if bias1 is not None else None,
            rowscale.view_as(rowscale) if rowscale is not None else None,
            seeds,  # seeds is newly created, no need for view_as
            mean,   # mean is newly created, no need for view_as
            rstd,   # rstd is newly created, no need for view_as
            # Context attributes passed as part of output
            x_shape_og,
            has_residual,
            has_x1,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x, weight, bias, residual, x1, weight1, bias1, eps, dropout_p,
         rowscale, prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask) = inputs

        (y, y1_out, residual_out_reshaped, dropout_mask, dropout_mask1,
         residual_out_saved, weight_saved, bias_saved, weight1_saved, bias1_saved,
         rowscale_saved, seeds_saved, mean_saved, rstd_saved,
         x_shape_og, has_residual, has_x1) = output

        # Build list of tensors to save (handle None values)
        # Order: residual_out, weight, bias, weight1, bias1, rowscale, seeds, mean, rstd
        # We need to track which ones are present
        tensors_to_save = []
        if residual_out_saved is not None:
            tensors_to_save.append(residual_out_saved)
        tensors_to_save.append(weight_saved)  # weight is always present
        if bias_saved is not None:
            tensors_to_save.append(bias_saved)
        if weight1_saved is not None:
            tensors_to_save.append(weight1_saved)
        if bias1_saved is not None:
            tensors_to_save.append(bias1_saved)
        if rowscale_saved is not None:
            tensors_to_save.append(rowscale_saved)
        if seeds_saved is not None:
            tensors_to_save.append(seeds_saved)
        if mean_saved is not None:
            tensors_to_save.append(mean_saved)
        tensors_to_save.append(rstd_saved)  # rstd is always present

        ctx.save_for_backward(*tensors_to_save)

        # Set context attributes
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.dropout_p = dropout_p
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = has_residual
        ctx.has_x1 = has_x1
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        # Track which optional tensors were saved
        ctx.has_residual_out = residual_out_saved is not None
        ctx.has_bias = bias_saved is not None
        ctx.has_weight1 = weight1_saved is not None
        ctx.has_bias1 = bias1_saved is not None
        ctx.has_rowscale = rowscale_saved is not None
        ctx.has_seeds = seeds_saved is not None
        ctx.has_mean = mean_saved is not None

    @staticmethod
    def backward(ctx, dy, dy1, dresidual_grad, ddropout_mask, ddropout_mask1, *_):
        # Unpack saved tensors based on which ones were saved
        saved = list(ctx.saved_tensors)
        idx = 0

        # residual_out (x) - always first if present
        if ctx.has_residual_out:
            x = saved[idx]
            idx += 1
        else:
            x = None

        # weight - always present
        weight = saved[idx]
        idx += 1

        # bias - optional
        if ctx.has_bias:
            bias = saved[idx]
            idx += 1
        else:
            bias = None

        # weight1 - optional
        if ctx.has_weight1:
            weight1 = saved[idx]
            idx += 1
        else:
            weight1 = None

        # bias1 - optional
        if ctx.has_bias1:
            bias1 = saved[idx]
            idx += 1
        else:
            bias1 = None

        # rowscale - optional
        if ctx.has_rowscale:
            rowscale = saved[idx]
            idx += 1
        else:
            rowscale = None

        # seeds - optional
        if ctx.has_seeds:
            seeds = saved[idx]
            idx += 1
        else:
            seeds = None

        # mean - optional (not present for RMS norm)
        if ctx.has_mean:
            mean = saved[idx]
            idx += 1
        else:
            mean = None

        # rstd - always present
        rstd = saved[idx]

        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape

        # Handle dy1 gradient (for weight1)
        if ctx.has_weight1 and dy1 is not None:
            dy1 = dy1.reshape(-1, dy1.shape[-1])
            if dy1.stride(-1) != 1:
                dy1 = dy1.contiguous()
            assert dy1.shape == x.shape
        else:
            dy1 = None

        # Handle dresidual gradient (for prenorm)
        if ctx.prenorm and dresidual_grad is not None:
            dresidual = dresidual_grad.reshape(-1, dresidual_grad.shape[-1])
            if dresidual.stride(-1) != 1:
                dresidual = dresidual.contiguous()
            assert dresidual.shape == x.shape
        else:
            dresidual = None

        dx, dw, db, dresidual_in, dx1, dw1, db1 = _layer_norm_bwd(
            dy,
            x,
            weight,
            bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            dy1,
            weight1,
            bias1,
            seeds,
            ctx.dropout_p,
            rowscale,
            ctx.has_residual,
            ctx.has_x1,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            dx1.reshape(ctx.x_shape_og) if dx1 is not None else None,
            dw1,
            db1,
            None,  # eps
            None,  # dropout_p
            None,  # rowscale
            None,  # prenorm
            None,  # residual_in_fp32
            None,  # is_rms_norm
            None,  # return_dropout_mask
        )

    @staticmethod
    def vmap(info, in_dims, x, weight, bias, residual, x1, weight1, bias1, eps, dropout_p,
             rowscale, prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask):
        # Handle vmap by moving batch dim to front and merging with existing batch
        def move_bdim_to_front(tensor, bdim):
            if bdim is None or tensor is None:
                return tensor
            return tensor.movedim(bdim, 0)

        # in_dims order matches inputs: x, weight, bias, residual, x1, weight1, bias1,
        # eps, dropout_p, rowscale, prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask
        x_bdim = in_dims[0]
        weight_bdim = in_dims[1]
        bias_bdim = in_dims[2]
        residual_bdim = in_dims[3]
        x1_bdim = in_dims[4]
        weight1_bdim = in_dims[5]
        bias1_bdim = in_dims[6]
        rowscale_bdim = in_dims[9]

        # Get vmap batch size from first batched input
        vmap_batch_size = None
        for tensor, bdim in [(x, x_bdim), (weight, weight_bdim), (bias, bias_bdim),
                             (residual, residual_bdim), (x1, x1_bdim)]:
            if bdim is not None and tensor is not None:
                vmap_batch_size = tensor.shape[bdim]
                break

        if vmap_batch_size is None:
            # No batched inputs, just call normally
            result = LayerNormFn.apply(x, weight, bias, residual, x1, weight1, bias1, eps, dropout_p,
                                       rowscale, prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask)
            # All outputs have no batch dim
            out_dims = (None,) * len(result)
            return result, out_dims

        # Move batch dims to front
        x_batched = move_bdim_to_front(x, x_bdim)
        weight_batched = move_bdim_to_front(weight, weight_bdim)
        bias_batched = move_bdim_to_front(bias, bias_bdim) if bias is not None else None
        residual_batched = move_bdim_to_front(residual, residual_bdim) if residual is not None else None
        x1_batched = move_bdim_to_front(x1, x1_bdim) if x1 is not None else None
        weight1_batched = move_bdim_to_front(weight1, weight1_bdim) if weight1 is not None else None
        bias1_batched = move_bdim_to_front(bias1, bias1_bdim) if bias1 is not None else None
        rowscale_batched = move_bdim_to_front(rowscale, rowscale_bdim) if rowscale is not None else None

        # For x, merge vmap batch with tensor's leading dims
        # x: (..., hidden_size) - can have any leading dims
        if x_bdim is None:
            # x not batched, broadcast by adding vmap_batch dim
            x_batched = x_batched.unsqueeze(0).expand(vmap_batch_size, *x_batched.shape)

        # Merge vmap batch into first dimension
        # x: (vmap_batch, batch, ..., hidden_size) -> (vmap_batch * batch, ..., hidden_size)
        x_shape = x_batched.shape
        x_merged = x_batched.reshape(x_shape[0] * x_shape[1], *x_shape[2:])

        # Handle residual similarly if present
        residual_merged = None
        if residual_batched is not None:
            if residual_bdim is None:
                residual_batched = residual_batched.unsqueeze(0).expand(vmap_batch_size, *residual_batched.shape)
            res_shape = residual_batched.shape
            residual_merged = residual_batched.reshape(res_shape[0] * res_shape[1], *res_shape[2:])

        # Handle x1 similarly if present
        x1_merged = None
        if x1_batched is not None:
            if x1_bdim is None:
                x1_batched = x1_batched.unsqueeze(0).expand(vmap_batch_size, *x1_batched.shape)
            x1_shape = x1_batched.shape
            x1_merged = x1_batched.reshape(x1_shape[0] * x1_shape[1], *x1_shape[2:])

        # Handle rowscale similarly if present
        rowscale_merged = None
        if rowscale_batched is not None:
            if rowscale_bdim is None:
                rowscale_batched = rowscale_batched.unsqueeze(0).expand(vmap_batch_size, *rowscale_batched.shape)
            rs_shape = rowscale_batched.shape
            rowscale_merged = rowscale_batched.reshape(rs_shape[0] * rs_shape[1], *rs_shape[2:])

        # weight and bias are typically not batched (shared across batch)
        if weight_bdim is not None:
            weight_shape = weight_batched.shape
            weight_merged = weight_batched.reshape(weight_shape[0] * weight_shape[1], *weight_shape[2:])
        else:
            weight_merged = weight_batched

        if bias_batched is not None:
            if bias_bdim is not None:
                bias_shape = bias_batched.shape
                bias_merged = bias_batched.reshape(bias_shape[0] * bias_shape[1], *bias_shape[2:])
            else:
                bias_merged = bias_batched
        else:
            bias_merged = None

        # Handle weight1 and bias1
        if weight1_batched is not None:
            if weight1_bdim is not None:
                w1_shape = weight1_batched.shape
                weight1_merged = weight1_batched.reshape(w1_shape[0] * w1_shape[1], *w1_shape[2:])
            else:
                weight1_merged = weight1_batched
        else:
            weight1_merged = None

        if bias1_batched is not None:
            if bias1_bdim is not None:
                b1_shape = bias1_batched.shape
                bias1_merged = bias1_batched.reshape(b1_shape[0] * b1_shape[1], *b1_shape[2:])
            else:
                bias1_merged = bias1_batched
        else:
            bias1_merged = None

        # Call the function with merged batches
        result = LayerNormFn.apply(
            x_merged, weight_merged, bias_merged, residual_merged, x1_merged,
            weight1_merged, bias1_merged, eps, dropout_p, rowscale_merged,
            prenorm, residual_in_fp32, is_rms_norm, return_dropout_mask
        )

        # Unpack result - 17 items
        (y, y1_out, residual_out, dropout_mask, dropout_mask1,
         residual_out_saved, weight_saved, bias_saved, weight1_saved, bias1_saved,
         rowscale_saved, seeds_saved, mean_saved, rstd_saved,
         x_shape_og, has_residual, has_x1) = result

        # Helper to unmerge batch dimension
        def unmerge_batch(tensor, is_batched_input):
            if tensor is None:
                return None
            # tensor shape: (vmap_batch * batch, ...) -> (vmap_batch, batch, ...)
            merged_batch = tensor.shape[0]
            original_batch = merged_batch // vmap_batch_size
            return tensor.reshape(vmap_batch_size, original_batch, *tensor.shape[1:])

        # Unmerge user-facing outputs
        y_unmerged = unmerge_batch(y, True)
        y1_unmerged = unmerge_batch(y1_out, True) if y1_out is not None else None
        residual_out_unmerged = unmerge_batch(residual_out, True) if residual_out is not None else None
        dropout_mask_unmerged = unmerge_batch(dropout_mask, True) if dropout_mask is not None else None
        dropout_mask1_unmerged = unmerge_batch(dropout_mask1, True) if dropout_mask1 is not None else None

        # Unmerge saved tensors
        residual_out_saved_unmerged = unmerge_batch(residual_out_saved, True) if residual_out_saved is not None else None

        # For weight and bias, only unmerge if they were batched
        if weight_bdim is not None:
            weight_saved_unmerged = weight_saved.reshape(vmap_batch_size, -1)
        else:
            weight_saved_unmerged = weight_saved

        if bias_saved is not None:
            if bias_bdim is not None:
                bias_saved_unmerged = bias_saved.reshape(vmap_batch_size, -1)
            else:
                bias_saved_unmerged = bias_saved
        else:
            bias_saved_unmerged = None

        if weight1_saved is not None:
            if weight1_bdim is not None:
                weight1_saved_unmerged = weight1_saved.reshape(vmap_batch_size, -1)
            else:
                weight1_saved_unmerged = weight1_saved
        else:
            weight1_saved_unmerged = None

        if bias1_saved is not None:
            if bias1_bdim is not None:
                bias1_saved_unmerged = bias1_saved.reshape(vmap_batch_size, -1)
            else:
                bias1_saved_unmerged = bias1_saved
        else:
            bias1_saved_unmerged = None

        if rowscale_saved is not None:
            if rowscale_bdim is not None:
                rowscale_saved_unmerged = rowscale_saved.reshape(vmap_batch_size, -1)
            else:
                rowscale_saved_unmerged = rowscale_saved
        else:
            rowscale_saved_unmerged = None

        # seeds - 1D tensor, unmerge along first dimension
        if seeds_saved is not None:
            original_seeds_size = seeds_saved.shape[0] // vmap_batch_size
            seeds_saved_unmerged = seeds_saved.reshape(vmap_batch_size, original_seeds_size)
        else:
            seeds_saved_unmerged = None

        # mean and rstd - 1D tensors
        if mean_saved is not None:
            original_mean_size = mean_saved.shape[0] // vmap_batch_size
            mean_saved_unmerged = mean_saved.reshape(vmap_batch_size, original_mean_size)
        else:
            mean_saved_unmerged = None

        original_rstd_size = rstd_saved.shape[0] // vmap_batch_size
        rstd_saved_unmerged = rstd_saved.reshape(vmap_batch_size, original_rstd_size)

        # Build output tuple
        result_unmerged = (
            y_unmerged,
            y1_unmerged,
            residual_out_unmerged,
            dropout_mask_unmerged,
            dropout_mask1_unmerged,
            residual_out_saved_unmerged,
            weight_saved_unmerged,
            bias_saved_unmerged,
            weight1_saved_unmerged,
            bias1_saved_unmerged,
            rowscale_saved_unmerged,
            seeds_saved_unmerged,
            mean_saved_unmerged,
            rstd_saved_unmerged,
            x_shape_og,
            has_residual,
            has_x1,
        )

        # Build out_dims - all batched outputs have vmap batch at position 0
        out_dims = (
            0,  # y
            0 if y1_out is not None else None,  # y1
            0 if residual_out is not None else None,  # residual_out
            0 if dropout_mask is not None else None,  # dropout_mask
            0 if dropout_mask1 is not None else None,  # dropout_mask1
            0 if residual_out_saved is not None else None,  # residual_out_saved
            0 if weight_bdim is not None else None,  # weight_saved
            0 if bias_saved is not None and bias_bdim is not None else None,  # bias_saved
            0 if weight1_saved is not None and weight1_bdim is not None else None,  # weight1_saved
            0 if bias1_saved is not None and bias1_bdim is not None else None,  # bias1_saved
            0 if rowscale_saved is not None and rowscale_bdim is not None else None,  # rowscale_saved
            0 if seeds_saved is not None else None,  # seeds_saved
            0 if mean_saved is not None else None,  # mean_saved
            0,  # rstd_saved
            None,  # x_shape_og (not a tensor)
            None,  # has_residual (not a tensor)
            None,  # has_x1 (not a tensor)
        )

        return result_unmerged, out_dims


def layer_norm_fn(
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
        return_dropout_mask=False,
):
    result = LayerNormFn.apply(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        return_dropout_mask,
    )
    # Extract user-facing outputs from consistent tuple structure
    # result = (y, y1, residual_out, dropout_mask, dropout_mask1, ...saved tensors...)
    y, y1, residual_out, dropout_mask, dropout_mask1 = result[0], result[1], result[2], result[3], result[4]

    if not return_dropout_mask:
        if weight1 is None:
            return y if not prenorm else (y, residual_out)
        else:
            return (y, y1) if not prenorm else (y, y1, residual_out)
    else:
        if weight1 is None:
            return (
                (y, dropout_mask, dropout_mask1)
                if not prenorm
                else (y, residual_out, dropout_mask, dropout_mask1)
            )
        else:
            return (
                (y, y1, dropout_mask, dropout_mask1)
                if not prenorm
                else (y, y1, residual_out, dropout_mask, dropout_mask1)
            )


def rms_norm_fn(
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        return_dropout_mask=False,
):
    result = LayerNormFn.apply(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        True,  # is_rms_norm=True
        return_dropout_mask,
    )
    # Extract user-facing outputs from consistent tuple structure
    y, y1, residual_out, dropout_mask, dropout_mask1 = result[0], result[1], result[2], result[3], result[4]

    if not return_dropout_mask:
        if weight1 is None:
            return y if not prenorm else (y, residual_out)
        else:
            return (y, y1) if not prenorm else (y, y1, residual_out)
    else:
        if weight1 is None:
            return (
                (y, dropout_mask, dropout_mask1)
                if not prenorm
                else (y, residual_out, dropout_mask, dropout_mask1)
            )
        else:
            return (
                (y, y1, dropout_mask, dropout_mask1)
                if not prenorm
                else (y, y1, residual_out, dropout_mask, dropout_mask1)
            )


class RMSNorm(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, dropout_p=0.0, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if dropout_p > 0.0:
            self.drop = torch.nn.Dropout(dropout_p)
        else:
            self.drop = None
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=self.drop.p if self.drop is not None and self.training else 0.0,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class LayerNormLinearFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
            x,
            norm_weight,
            norm_bias,
            linear_weight,
            linear_bias,
            residual=None,
            eps=1e-6,
            prenorm=False,
            residual_in_fp32=False,
            is_rms_norm=False,
    ):
        x_shape_og = x.shape
        x_dtype = x.dtype
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        norm_weight = norm_weight.contiguous()
        if norm_bias is not None:
            norm_bias = norm_bias.contiguous()
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, _, mean, rstd, residual_out, *rest = _layer_norm_fwd(
            x,
            norm_weight,
            norm_bias,
            eps,
            residual,
            out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(),
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)

        # Track which inputs were provided
        has_residual = residual is not None
        linear_bias_is_none = linear_bias is None

        # Return a consistent tuple structure:
        # (out, residual_out_reshaped,  # User-facing outputs
        #  residual_out_saved, norm_weight_saved, norm_bias_saved,  # Saved tensors
        #  linear_weight_saved, mean_saved, rstd_saved,
        #  x_shape_og, eps, is_rms_norm, has_residual, prenorm, x_dtype, linear_bias_is_none)
        return (
            out,
            residual_out.reshape(x_shape_og) if prenorm else None,
            # Saved tensors for backward (use view_as for tensors)
            residual_out.view_as(residual_out),
            norm_weight.view_as(norm_weight),
            norm_bias.view_as(norm_bias) if norm_bias is not None else None,
            linear_weight.view_as(linear_weight),
            mean,  # mean is newly created, no need for view_as
            rstd,  # rstd is newly created, no need for view_as
            # Context attributes passed as part of output
            x_shape_og,
            eps,
            is_rms_norm,
            has_residual,
            prenorm,
            x_dtype,
            linear_bias_is_none,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x, norm_weight, norm_bias, linear_weight, linear_bias, residual,
         eps, prenorm, residual_in_fp32, is_rms_norm) = inputs

        (out, residual_out_reshaped,
         residual_out_saved, norm_weight_saved, norm_bias_saved,
         linear_weight_saved, mean_saved, rstd_saved,
         x_shape_og, eps_out, is_rms_norm_out, has_residual, prenorm_out, x_dtype, linear_bias_is_none) = output

        # Build list of tensors to save (handle None values)
        tensors_to_save = [residual_out_saved, norm_weight_saved]
        if norm_bias_saved is not None:
            tensors_to_save.append(norm_bias_saved)
        tensors_to_save.append(linear_weight_saved)
        if mean_saved is not None:
            tensors_to_save.append(mean_saved)
        tensors_to_save.append(rstd_saved)

        ctx.save_for_backward(*tensors_to_save)

        # Set context attributes
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = has_residual
        ctx.prenorm = prenorm
        ctx.x_dtype = x_dtype
        ctx.linear_bias_is_none = linear_bias_is_none
        ctx.has_norm_bias = norm_bias_saved is not None
        ctx.has_mean = mean_saved is not None
        # Required for @custom_bwd decorator to work with setup_context pattern
        ctx._fwd_used_autocast = torch.is_autocast_enabled()
        ctx._dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, dresidual_out_grad, *_):
        # Unpack saved tensors based on which ones were saved
        saved = list(ctx.saved_tensors)
        idx = 0

        # residual_out (x) - always present
        x = saved[idx]
        idx += 1

        # norm_weight - always present
        norm_weight = saved[idx]
        idx += 1

        # norm_bias - optional
        if ctx.has_norm_bias:
            norm_bias = saved[idx]
            idx += 1
        else:
            norm_bias = None

        # linear_weight - always present
        linear_weight = saved[idx]
        idx += 1

        # mean - optional (not present for RMS norm)
        if ctx.has_mean:
            mean = saved[idx]
            idx += 1
        else:
            mean = None

        # rstd - always present
        rstd = saved[idx]

        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        if ctx.prenorm and dresidual_out_grad is not None:
            dresidual = dresidual_out_grad.reshape(-1, dresidual_out_grad.shape[-1])
            if dresidual.stride(-1) != 1:
                dresidual = dresidual.contiguous()
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, _, _, _, y = _layer_norm_bwd(
            dy,
            x,
            norm_weight,
            norm_bias,
            ctx.eps,
            mean,
            rstd,
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            is_rms_norm=ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True,
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
        return (
            dx.reshape(ctx.x_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,  # eps
            None,  # prenorm
            None,  # residual_in_fp32
            None,  # is_rms_norm
        )

    @staticmethod
    def vmap(info, in_dims, x, norm_weight, norm_bias, linear_weight, linear_bias,
             residual, eps, prenorm, residual_in_fp32, is_rms_norm):
        # Handle vmap by moving batch dim to front and merging with existing batch
        def move_bdim_to_front(tensor, bdim):
            if bdim is None or tensor is None:
                return tensor
            return tensor.movedim(bdim, 0)

        # in_dims order: x, norm_weight, norm_bias, linear_weight, linear_bias,
        #                residual, eps, prenorm, residual_in_fp32, is_rms_norm
        x_bdim = in_dims[0]
        norm_weight_bdim = in_dims[1]
        norm_bias_bdim = in_dims[2]
        linear_weight_bdim = in_dims[3]
        linear_bias_bdim = in_dims[4]
        residual_bdim = in_dims[5]

        # Get vmap batch size from first batched input
        vmap_batch_size = None
        for tensor, bdim in [(x, x_bdim), (norm_weight, norm_weight_bdim),
                             (residual, residual_bdim)]:
            if bdim is not None and tensor is not None:
                vmap_batch_size = tensor.shape[bdim]
                break

        if vmap_batch_size is None:
            # No batched inputs, just call normally
            result = LayerNormLinearFn.apply(x, norm_weight, norm_bias, linear_weight, linear_bias,
                                             residual, eps, prenorm, residual_in_fp32, is_rms_norm)
            out_dims = (None,) * len(result)
            return result, out_dims

        # Move batch dims to front
        x_batched = move_bdim_to_front(x, x_bdim)
        norm_weight_batched = move_bdim_to_front(norm_weight, norm_weight_bdim)
        norm_bias_batched = move_bdim_to_front(norm_bias, norm_bias_bdim) if norm_bias is not None else None
        linear_weight_batched = move_bdim_to_front(linear_weight, linear_weight_bdim)
        linear_bias_batched = move_bdim_to_front(linear_bias, linear_bias_bdim) if linear_bias is not None else None
        residual_batched = move_bdim_to_front(residual, residual_bdim) if residual is not None else None

        # For x, merge vmap batch with tensor's leading dims
        if x_bdim is None:
            x_batched = x_batched.unsqueeze(0).expand(vmap_batch_size, *x_batched.shape)

        # Merge vmap batch into first dimension
        x_shape = x_batched.shape
        x_merged = x_batched.reshape(x_shape[0] * x_shape[1], *x_shape[2:])

        # Handle residual similarly if present
        residual_merged = None
        if residual_batched is not None:
            if residual_bdim is None:
                residual_batched = residual_batched.unsqueeze(0).expand(vmap_batch_size, *residual_batched.shape)
            res_shape = residual_batched.shape
            residual_merged = residual_batched.reshape(res_shape[0] * res_shape[1], *res_shape[2:])

        # norm_weight and norm_bias are typically not batched (shared)
        if norm_weight_bdim is not None:
            nw_shape = norm_weight_batched.shape
            norm_weight_merged = norm_weight_batched.reshape(nw_shape[0] * nw_shape[1], *nw_shape[2:])
        else:
            norm_weight_merged = norm_weight_batched

        if norm_bias_batched is not None:
            if norm_bias_bdim is not None:
                nb_shape = norm_bias_batched.shape
                norm_bias_merged = norm_bias_batched.reshape(nb_shape[0] * nb_shape[1], *nb_shape[2:])
            else:
                norm_bias_merged = norm_bias_batched
        else:
            norm_bias_merged = None

        # linear_weight and linear_bias are typically not batched (shared)
        if linear_weight_bdim is not None:
            lw_shape = linear_weight_batched.shape
            linear_weight_merged = linear_weight_batched.reshape(lw_shape[0] * lw_shape[1], *lw_shape[2:])
        else:
            linear_weight_merged = linear_weight_batched

        if linear_bias_batched is not None:
            if linear_bias_bdim is not None:
                lb_shape = linear_bias_batched.shape
                linear_bias_merged = linear_bias_batched.reshape(lb_shape[0] * lb_shape[1], *lb_shape[2:])
            else:
                linear_bias_merged = linear_bias_batched
        else:
            linear_bias_merged = None

        # Call the function with merged batches
        result = LayerNormLinearFn.apply(
            x_merged, norm_weight_merged, norm_bias_merged, linear_weight_merged, linear_bias_merged,
            residual_merged, eps, prenorm, residual_in_fp32, is_rms_norm
        )

        # Unpack result - 15 items
        (out, residual_out_reshaped,
         residual_out_saved, norm_weight_saved, norm_bias_saved,
         linear_weight_saved, mean_saved, rstd_saved,
         x_shape_og, eps_out, is_rms_norm_out, has_residual, prenorm_out, x_dtype, linear_bias_is_none) = result

        # Helper to unmerge batch dimension
        def unmerge_batch(tensor):
            if tensor is None:
                return None
            merged_batch = tensor.shape[0]
            original_batch = merged_batch // vmap_batch_size
            return tensor.reshape(vmap_batch_size, original_batch, *tensor.shape[1:])

        # Unmerge user-facing outputs
        out_unmerged = unmerge_batch(out)
        residual_out_reshaped_unmerged = unmerge_batch(residual_out_reshaped) if residual_out_reshaped is not None else None

        # Unmerge saved tensors
        residual_out_saved_unmerged = unmerge_batch(residual_out_saved)

        # For norm_weight and norm_bias, only unmerge if they were batched
        if norm_weight_bdim is not None:
            norm_weight_saved_unmerged = norm_weight_saved.reshape(vmap_batch_size, -1)
        else:
            norm_weight_saved_unmerged = norm_weight_saved

        if norm_bias_saved is not None:
            if norm_bias_bdim is not None:
                norm_bias_saved_unmerged = norm_bias_saved.reshape(vmap_batch_size, -1)
            else:
                norm_bias_saved_unmerged = norm_bias_saved
        else:
            norm_bias_saved_unmerged = None

        # For linear_weight, only unmerge if batched
        if linear_weight_bdim is not None:
            linear_weight_saved_unmerged = linear_weight_saved.reshape(vmap_batch_size, linear_weight_saved.shape[0] // vmap_batch_size, -1)
        else:
            linear_weight_saved_unmerged = linear_weight_saved

        # mean and rstd - 1D tensors
        if mean_saved is not None:
            original_mean_size = mean_saved.shape[0] // vmap_batch_size
            mean_saved_unmerged = mean_saved.reshape(vmap_batch_size, original_mean_size)
        else:
            mean_saved_unmerged = None

        original_rstd_size = rstd_saved.shape[0] // vmap_batch_size
        rstd_saved_unmerged = rstd_saved.reshape(vmap_batch_size, original_rstd_size)

        # Build output tuple
        result_unmerged = (
            out_unmerged,
            residual_out_reshaped_unmerged,
            residual_out_saved_unmerged,
            norm_weight_saved_unmerged,
            norm_bias_saved_unmerged,
            linear_weight_saved_unmerged,
            mean_saved_unmerged,
            rstd_saved_unmerged,
            x_shape_og,
            eps_out,
            is_rms_norm_out,
            has_residual,
            prenorm_out,
            x_dtype,
            linear_bias_is_none,
        )

        # Build out_dims - all batched outputs have vmap batch at position 0
        out_dims = (
            0,  # out
            0 if residual_out_reshaped is not None else None,  # residual_out_reshaped
            0,  # residual_out_saved
            0 if norm_weight_bdim is not None else None,  # norm_weight_saved
            0 if norm_bias_saved is not None and norm_bias_bdim is not None else None,  # norm_bias_saved
            0 if linear_weight_bdim is not None else None,  # linear_weight_saved
            0 if mean_saved is not None else None,  # mean_saved
            0,  # rstd_saved
            None,  # x_shape_og (not a tensor)
            None,  # eps (not a tensor)
            None,  # is_rms_norm (not a tensor)
            None,  # has_residual (not a tensor)
            None,  # prenorm (not a tensor)
            None,  # x_dtype (not a tensor)
            None,  # linear_bias_is_none (not a tensor)
        )

        return result_unmerged, out_dims


def layer_norm_linear_fn(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
):
    result = LayerNormLinearFn.apply(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
    )
    # Extract user-facing outputs from consistent tuple structure
    # result = (out, residual_out_reshaped, ...saved tensors...)
    out, residual_out_reshaped = result[0], result[1]
    return out if not prenorm else (out, residual_out_reshaped)