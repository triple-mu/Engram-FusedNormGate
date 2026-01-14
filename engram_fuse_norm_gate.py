import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def _norm_gate_kernel(
        gate_ptr,
        q_ptr,
        k_ptr,
        q_weight_ptr,
        k_weight_ptr,
        qk_stride_b: int,
        qk_stride_s: int,
        gate_stride_b: int,
        gate_stride_s: int,
        N: int,
        D: int,
        scale: float,
        eps: float,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
):
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    batch_seq_offset = b_id * qk_stride_b + s_id * qk_stride_s
    q_ptr += batch_seq_offset
    k_ptr += batch_seq_offset

    gate_ptr += b_id * gate_stride_b + s_id * gate_stride_s

    n_offset = tl.arange(0, BLOCK_N)
    n_mask = n_offset < N

    q_sq_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    k_sq_sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    dot_sum = tl.zeros([BLOCK_N], dtype=tl.float32)

    for off_d in range(0, D, BLOCK_D):
        d_offset = off_d + tl.arange(0, BLOCK_D)
        d_mask = d_offset < D

        cols = n_offset[:, None] * D + d_offset[None, :]
        mask = n_mask[:, None] & d_mask[None, :]

        q = tl.load(q_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        q_w = tl.load(q_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        k_w = tl.load(k_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        q_sq_sum += tl.sum(q * q, axis=1)
        k_sq_sum += tl.sum(k * k, axis=1)
        dot_sum += tl.sum(q * q_w * k * k_w, axis=1)

    q_rstd = tl.math.rsqrt(q_sq_sum / D + eps)
    k_rstd = tl.math.rsqrt(k_sq_sum / D + eps)

    gate_val = dot_sum * q_rstd * k_rstd * scale
    gate_val = gate_val * tl.math.rsqrt(tl.abs(gate_val) + 1e-6)

    gate_val = tl.sigmoid(gate_val)
    tl.store(gate_ptr + n_offset, gate_val, mask=n_mask)


def fuse_norm_gate(
        q: torch.Tensor,
        k: torch.Tensor,
        q_rmsnorm_weight: torch.Tensor,
        k_rmsnorm_weight: torch.Tensor,
        hidden_size: int,
        eps: float = 1e-6,
) -> torch.Tensor:
    q = q.contiguous()
    k = k.contiguous()
    q_rmsnorm_weight = q_rmsnorm_weight.contiguous()
    k_rmsnorm_weight = k_rmsnorm_weight.contiguous()
    B, S, N, D = q.shape
    scale = 1 / math.sqrt(hidden_size)

    gates = torch.empty(B, S, N, 1, dtype=q.dtype, device=q.device)
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_D = triton.next_power_of_2(min(8192 // BLOCK_N, D))
    with torch.cuda.device(q.device):
        _norm_gate_kernel[(S, B)](
            gates,
            q,
            k,
            q_rmsnorm_weight,
            k_rmsnorm_weight,
            q.stride(0),
            q.stride(1),
            gates.stride(0),
            gates.stride(1),
            N,
            D,
            scale,
            eps,
            BLOCK_N,
            BLOCK_D,
            num_warps=min(max(BLOCK_D // 256, 1), 8),
        )
    return gates


def norm_gate(
        q: torch.Tensor,
        k: torch.Tensor,
        q_rmsnorm_weight: torch.Tensor,
        k_rmsnorm_weight: torch.Tensor,
        hidden_size: int,
        eps: float = 1e-6,
) -> torch.Tensor:
    gates = []
    for i, (query, key) in enumerate(
            zip(
                k.unbind(dim=2),
                q.unbind(dim=2)
            )
    ):
        normed_key = F.rms_norm(key, (key.size(-1),), k_rmsnorm_weight[i], eps)
        normed_query = F.rms_norm(query, (query.size(-1),), q_rmsnorm_weight[i], eps)
        gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)
        gates.append(gate)
    gates = torch.stack(gates, dim=2)
    return gates


@torch.inference_mode()
def main():
    dtype = torch.float32
    device = torch.device('cuda:0')

    seq_len = 65536
    hc_mult = 4
    hidden_size = 1024

    q = torch.randn(1, seq_len, hc_mult, hidden_size, dtype=dtype, device=device)
    k = torch.randn(1, seq_len, hc_mult, hidden_size, dtype=dtype, device=device)

    q_w = torch.randn(hc_mult, hidden_size, dtype=dtype, device=device)
    k_w = torch.randn(hc_mult, hidden_size, dtype=dtype, device=device)

    gate_gt = norm_gate(q, k, q_w, k_w, hidden_size, 1e-6)
    gate_fuse = fuse_norm_gate(q, k, q_w, k_w, hidden_size, 1e-6)

    torch.testing.assert_close(gate_gt, gate_fuse, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    main()
