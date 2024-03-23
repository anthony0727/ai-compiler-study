import os
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import triton

from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from rope import rope


DEVICE = 'cuda'


def sample_data(batch, seq_len, n_heads, head_dim):
    t = torch.randn([batch, seq_len, n_heads, head_dim], device=DEVICE)
    t.requires_grad_(True)
    freqs = torch.randn([seq_len, 1, 1, head_dim//2], device=DEVICE)
    dx = torch.randn_like(t)

    return t, freqs, dx


def test_rope(t_shape):
    t_triton, freqs, dx_triton = sample_data(*t_shape)

    t_torch = t_triton.clone()
    dx_torch = dx_triton.clone()

    t_triton = rope(t_triton, freqs)
    t_triton.backward(dx_triton, retain_graph=True)

    t_torch = apply_rotary_pos_emb(t_torch, torch.cat((freqs, freqs), dim=-1), tensor_format="bshd")
    t_torch.backward(dx_torch, retain_graph=True)

    torch.testing.assert_close(t_triton, t_torch, atol=1e-3, rtol=0)
    torch.testing.assert_close(t_triton.grad, t_torch.grad, atol=1e-5, rtol=0)

def get_benchmark(name, mode):
    return triton.testing.Benchmark(
        x_names=["seq_len"],  
        x_vals=[32 * i for i in range(1, 65)],
        line_arg="provider",  
        line_vals=["torch", "triton"],
        line_names=["Pytorch", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name=name,
        args={'batch': 32, 'n_heads': 8, 'head_dim': 256, 'mode': mode},
    )

@triton.testing.perf_report([
    get_benchmark("rope_fw", "fw"),
    get_benchmark("rope_bw", "bw"),
])
def bench_rope(batch, seq_len, n_heads, head_dim, provider, mode):
    t, freqs, dx = sample_data(batch, seq_len, n_heads, head_dim)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        fw = lambda: apply_rotary_pos_emb(t, torch.cat((freqs, freqs)), tensor_format="bshd")
    elif provider == "triton":
        fw = lambda: rope(t, freqs)

    if mode == "fw":
        gbps = lambda ms: 2 * t.numel() * t.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(fw, quantiles=quantiles)
    elif mode == "bw":
        gbps = lambda ms: 3 * t.numel() * t.element_size() / ms * 1e-6
        x = fw()
        bw = lambda: x.backward(dx, retain_graph=True)
        ms, min_ms, max_ms = triton.testing.do_bench(bw, quantiles=quantiles)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


def profile_rope(f, name):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
    ) as prof:
        f()

    print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))
    prof.export_stacks(f"./results/{name}_profiler_stacks.txt", "self_cuda_time_total")


if __name__ == "__main__":
    test_rope([1, 16, 32, 256])
    test_rope([1, 64, 32, 512])

    Path('./results').mkdir(parents=True, exist_ok=True)
    bench_rope.run(show_plots=True, print_data=True, save_path="./results")


    # TODO: empty file
    t, freqs, dx = sample_data(1, 64, 32, 512)
    profile_rope(lambda: apply_rotary_pos_emb(t, torch.cat((freqs, freqs)), tensor_format="bshd"), "torch")
    profile_rope(lambda: rope(t, freqs), "triton")