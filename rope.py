import torch
import triton
import triton.language as tl


ROPE_GROUP_SIZE = 4


@triton.jit
def rope_fw_bw_kernel(
    t_ptr, t_stride,
    cos_ptr, cos_stride,
    sin_ptr, sin_stride,
    seq_len,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)

    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    cos = tl.load(
        cos_ptr + (row_position % seq_len) * cos_stride + col_offsets, 
        mask = mask, 
        other = 0.
    )
    sin = tl.load(
        sin_ptr + (row_position % seq_len) * sin_stride + col_offsets, 
        mask = mask, 
        other = 0.
    )

    if BACKWARD_PASS:
        sin = -sin

    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    for k in range(head_start, head_end):
        t_offsets = row_position * t_stride + k * head_dim + col_offsets
        t_half_offsets = row_position * t_stride + k * head_dim + col_offsets + half_head_dim

        t = tl.load(t_ptr + t_offsets, mask = mask, other = 0).to(cos.dtype)
        t_half = tl.load(t_ptr + t_half_offsets, mask = mask, other = 0).to(cos.dtype)

        tl.store(t_ptr + t_offsets, t * cos - t_half * sin, mask = mask)
        tl.store(t_ptr + t_half_offsets, t_half * cos + t * sin, mask = mask)


class TritonRoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        t, 
        freqs
    ):
        batch, seq_len, n_heads, head_dim = t.shape
        freqs = freqs.squeeze()

        t = t.reshape(batch * seq_len, n_heads * head_dim)
        n_rows, _ = t.shape

        assert(seq_len <= freqs.shape[0])

        BLOCK_SIZE = triton.next_power_of_2(head_dim//2)

        num_warps = 4        
        if BLOCK_SIZE >=  8192: num_warps = 16
        elif BLOCK_SIZE >=  2048: num_warps = 8
    
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        cos, sin = freqs.cos(), freqs.sin()
        
        with torch.cuda.device(t.device.index):
            rope_fw_bw_kernel[(n_rows, n_groups, )](
                t, t.stride(0),
                cos, cos.stride(0),
                sin, sin.stride(0),
                seq_len, head_dim, n_heads,
                BACKWARD_PASS = False,
                BLOCK_SIZE = BLOCK_SIZE,
                num_warps = num_warps,
            )

        ctx.cos = cos
        ctx.sin = sin
        
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        
        return t.reshape(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dw):
        batch, seq_len, n_heads, head_dim = dw.shape
        dw = dw.reshape(batch * seq_len, n_heads * head_dim)
        n_rows, _ = dw.shape

        cos, sin = ctx.cos, ctx.sin

        with torch.cuda.device(dw.device.index):
            rope_fw_bw_kernel[(n_rows, ctx.n_groups, )](
                dw,  dw .stride(0),
                cos, cos.stride(0),
                sin, sin.stride(0),
                seq_len, head_dim, n_heads,
                BACKWARD_PASS = True,
                BLOCK_SIZE = ctx.BLOCK_SIZE,
                num_warps  = ctx.num_warps,
            )
        dw = dw.view(batch, seq_len, n_heads, head_dim)

        return dw, None, None,


rope = TritonRoPE.apply