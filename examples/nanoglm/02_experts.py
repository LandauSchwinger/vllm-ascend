"""
Part 2: what did the experts learn, and how does an engine run them?

    python 02_experts.py [out/nanoglm.pt]
"""

import sys
import time

import torch
from nanoglm import MoE, load

torch.manual_seed(0)
model, data = load(sys.argv[1] if len(sys.argv) > 1 else "out/nanoglm.pt")
cfg = model.cfg
x, y, qmask, _ = data.batch("val", 32, cfg.block_size)
model(x)

# --- 1. who goes where? Classify every input character and count expert picks per class ---
classes = {
    "prose letter": lambda c, rec: c.isalpha() and not rec,
    "prose space/punct": lambda c, rec: not c.isalnum() and c not in "[]=" and not rec,
    "record bracket [ ]": lambda c, rec: c in "[]",
    "record key letter": lambda c, rec: c.isalpha() and rec,
    "record = or ?": lambda c, rec: c in "=?" and rec,
    "record digit": lambda c, rec: c.isdigit(),
}
labels = torch.full(x.shape, -1)
for b in range(x.shape[0]):
    in_record = False
    for t, c in enumerate(data.decode(x[b])):
        if c == "[":
            in_record = True
        for i, test in enumerate(classes.values()):
            if test(c, in_record):
                labels[b, t] = i
                break
        if c == "]":
            in_record = False

for li, block in enumerate(model.blocks):
    if not isinstance(block.ffn, MoE):
        print(f"layer {li}: dense FFN (no routing)")
        continue
    topi = block.ffn.last_topi  # [B, T, K]
    print(f"\nlayer {li}: share of expert picks (%), rows sum to 100")
    print(f"{'':20}" + "".join(f"  e{e}" for e in range(cfg.n_experts)))
    for i, name in enumerate(classes):
        picks = torch.bincount(topi[labels == i].flatten(), minlength=cfg.n_experts).float()
        print(f"{name:20}" + "".join(f"{100 * p:4.0f}" for p in picks / picks.sum()))
    load = torch.bincount(topi.flatten(), minlength=cfg.n_experts).float()
    print(f"{'ALL TOKENS (load)':20}" + "".join(f"{100 * p:4.0f}" for p in load / load.sum()))

# --- 2. two ways to compute the same MoE layer ---
moe = next(b.ffn for b in model.blocks if isinstance(b.ffn, MoE))
h = torch.randn(64 * cfg.block_size, cfg.n_embd)


def moe_naive(m, x):
    """Run EVERY expert on EVERY token, then throw away all but top-k. Simple and wasteful."""
    scores = torch.sigmoid(m.router(x))
    topv, topi = scores.topk(m.top_k, dim=-1)
    gates = torch.zeros_like(scores).scatter_(1, topi, topv / topv.sum(-1, keepdim=True))
    return m.shared(x) + sum(gates[:, e : e + 1] * m.experts[e](x) for e in range(m.n_experts))


with torch.no_grad():
    t0 = time.time()
    a = moe_naive(moe, h)
    t1 = time.time()
    b = moe(h)  # the grouped version in nanoglm.py
    t2 = time.time()
print(f"\nnaive (all experts on all tokens): {t1 - t0:.3f}s | grouped (sort tokens by expert): {t2 - t1:.3f}s "
      f"| max diff {(a - b).abs().max().item():.1e}")  # fmt: skip
print(f"naive costs E={cfg.n_experts} expert passes per token, grouped costs K={cfg.moe_top_k}. "
      "For GLM-5.x that is 256 vs 8.")  # fmt: skip
total = sum(p.numel() for p in moe.parameters())
active = sum(p.numel() for p in moe.shared.parameters()) + cfg.moe_top_k * sum(
    p.numel() for p in moe.experts[0].parameters()
)
print(f"MoE layer parameters: {total / 1e3:.0f}K total, {active / 1e3:.0f}K touched per token "
      "(all of them must sit in memory, few of them cost compute)")  # fmt: skip
