"""
Part 4: W8A8. Turn the trained float model into an int8 model, no retraining.

    python 04_quantization.py [out/nanoglm.pt]
"""

import sys

import torch
import torch.nn as nn
from nanoglm import evaluate, load
from quant import QLinear, calibrate, dynamic_quant, quantize_model, weight_quant

torch.manual_seed(0)
model, data = load(sys.argv[1] if len(sys.argv) > 1 else "out/nanoglm.pt")
cfg = model.cfg

# --- 1. one layer, by hand ---
layer = model.blocks[1].attn.qkv
captured = []
hook = layer.register_forward_hook(lambda m, inp, out: captured.append(inp[0]))
x_batch, _, _, _ = data.batch("val", 8, cfg.block_size)
with torch.no_grad():
    model(x_batch)
hook.remove()
x = captured[0].reshape(-1, cfg.n_embd)  # [tokens, C] the real inputs this layer sees
W = layer.weight.detach()  # [out, in]
y = x @ W.T

W_q, w_scale = weight_quant(W)
x_q, x_scale = dynamic_quant(x)
print("one row of W, float :", [f"{v:+.4f}" for v in W[0, :6].tolist()])
print("the same row, int8  :", W_q[0, :6].tolist(), f"x scale {w_scale[0].item():.6f}")
acc_int = x_q.to(torch.int32) @ W_q.to(torch.int32).T  # the real thing: integers in, integers out
acc_flt = x_q @ W_q.float().T  # what QLinear does
print(f"int32 accumulator: max |value| = {acc_int.abs().max().item()} (int32 holds 2147483647), "
      f"float emulation identical: {bool((acc_int == acc_flt).all())}")  # fmt: skip
y_w8a8 = acc_int * x_scale * w_scale
y_w8 = x @ (W_q * w_scale[:, None]).T


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


print(f"relative error of the layer output: W8A16 {rel_err(y_w8, y):.4f} | W8A8 dynamic {rel_err(y_w8a8, y):.4f}")

# why per-token scales: tokens differ a lot in magnitude
peak = x.abs().amax(-1)
print(f"max|x| per token at this layer: median {peak.median().item():.2f}, largest {peak.max().item():.2f} "
      f"({(peak.max() / peak.median()).item():.1f}x). One static scale must cover the largest, so the "
      f"median token only uses ~{127 * (peak.median() / peak.max()).item():.0f} of 127 levels.")  # fmt: skip

# --- 2. the whole model ---
calib_peaks = calibrate(model, [data.batch("train", 16, cfg.block_size)[0] for _ in range(4)])


def report(name, m):
    loss_d, acc_d = evaluate(m, data, sparse=False)
    loss_s, acc_s = evaluate(m, data, sparse="gather")
    size = sum(b.numel() * b.element_size() for b in list(m.parameters()) + list(m.buffers()))
    print(f"{name:30} {loss_d:>8.4f} {acc_d:>8.3f} | {loss_s:>8.4f} {acc_s:>8.3f} | {size / 1e6:>6.2f} MB")


print(f"\n{'':30} {'dense attention':^17} | {'sparse attention':^17} |")
print(f"{'':30} {'loss':>8} {'q. acc':>8} | {'loss':>8} {'q. acc':>8} | {'size':>9}")
report("float32", model)
report("W8A16 (weights only)", quantize_model(model, "float"))
report("W8A8 dynamic (per token)", quantize_model(model, "dynamic"))
report("W8A8 static (per layer)", quantize_model(model, "static", input_peaks=calib_peaks))
for bits in (6, 4):
    report(f"W{bits}A8 dynamic", quantize_model(model, "dynamic", w_bits=bits))
    report(f"W8A{bits} dynamic", quantize_model(model, "dynamic", a_bits=bits))
print("(size counts every int8 weight as 1 byte, so it is the size you would ship; W4 would pack to half of W8)")

# --- 3. which layers are fragile? quantize ONE group at a time, hard (W4A4), and watch the loss ---
groups = {
    "attn.qkv": lambda n: n.endswith("attn.qkv"),
    "attn.proj": lambda n: n.endswith("attn.proj"),
    "indexer (wq, wk_weights)": lambda n: "indexer" in n,
    "MoE router": lambda n: n.endswith("router"),
    "MoE experts": lambda n: ".experts." in n,
    "MoE shared expert": lambda n: ".shared." in n,
    "dense FFN (layer 0)": lambda n: ".ffn.gate_up" in n or ".ffn.down" in n,
    "lm_head": lambda n: n == "lm_head",
}
base_loss, base_acc = evaluate(model, data, sparse="gather")
print(
    f"\nsensitivity: only one group at W4A4 dynamic, sparse attention (float: loss {base_loss:.4f}, acc {base_acc:.3f})"
)
print(f"{'group':28} {'#weights':>9} {'loss':>8} {'d loss':>8} {'q. acc':>7}")
for name, select in groups.items():
    m = quantize_model(model, "dynamic", select, w_bits=4, a_bits=4)
    n = sum(mod.weight_q.numel() for mod in m.modules() if isinstance(mod, QLinear))
    loss, acc = evaluate(m, data, sparse="gather")
    print(f"{name:28} {n:>9} {loss:>8.4f} {loss - base_loss:>+8.4f} {acc:>7.3f}")
n_linear = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
print(f"({n_linear} weights in Linear layers in total; the embedding table is left in float)")
