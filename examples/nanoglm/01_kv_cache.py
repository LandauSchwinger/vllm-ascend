"""
Part 1: the KV cache, and what "sparse" buys at decode time.

    python 01_kv_cache.py [out/nanoglm.pt]
"""

import sys
import time

import torch
from nanoglm import load

torch.manual_seed(0)
torch.set_num_threads(1)  # timings below are about the algorithm, not about thread pools
model, data = load(sys.argv[1] if len(sys.argv) > 1 else "out/nanoglm.pt")
cfg = model.cfg
x, _, _, _ = data.batch("val", 1, cfg.block_size)

# --- 1. correctness: feeding tokens one by one with a cache == feeding the whole sequence ---
print("max |logits(full forward) - logits(prefill + cached decode)|")
for sparse in (False, "mask", "gather"):
    model.sparse = sparse
    full, _ = model(x)
    cache = model.new_cache()
    prefill = cfg.block_size // 2
    outs = [model(x[:, :prefill], cache=cache)[0]]  # prefill: many tokens at once
    for t in range(prefill, cfg.block_size):  # decode: one token at a time
        outs.append(model(x[:, t : t + 1], cache=cache)[0])
    print(f"  sparse={sparse!s:7}: {(torch.cat(outs, 1) - full).abs().max().item():.2e}")

# --- 2. what is in the cache? ---
layer = cache["layers"][0]
print("\ncache of layer 0 after", cache["pos"], "tokens:")
for name, t in layer.items():
    print(f"  {name:3} {tuple(t.shape)!s:20} {t.numel() * t.element_size() / 1024:.0f} KiB")
print("  ('ik' = lightning indexer keys: a second, much smaller cache next to K and V)")

# --- 3. speed: generate with and without the cache ---
prompt, n_new = x[:, :16], cfg.block_size - 16
for label, sparse, use_cache in (
    ("no cache, dense", False, False),
    ("cache, dense", False, True),
    ("cache, sparse (gather)", "gather", True),
):
    model.sparse = sparse
    t0 = time.time()
    out = model.generate(prompt, n_new, temperature=0, use_cache=use_cache)
    dt = time.time() - t0
    print(f"{label:24}: {n_new / dt:7.1f} tokens/s")

# --- 4. count the work of ONE decode step at context length S ---
# multiply-adds spent on dot products with previous tokens (the part that grows
# with context), for one indexer and the `share` attention layers it serves
share = cfg.n_layer // len(cfg.indexer_layers)
print(f"\nwork per decoded token, for {share} attention layer(s) sharing one indexer (multiply-adds):")
print(f"{'context S':>10} {'dense attn':>12} {'indexer':>10} {'sparse attn':>12} {'DSA total':>10} {'saving':>7}")
for name, H, hd, Hi, di, k in (
    ("nanoGLM", cfg.n_head, cfg.n_embd // cfg.n_head, cfg.index_n_head, cfg.index_head_dim, cfg.index_topk),
    ("a GLM-5.x sized layer (64 heads x 256, indexer 32 x 128, top-2048)", 64, 256, 32, 128, 2048),
):
    print(name)
    for S in (2048, 16384, 131072, 1048576) if k > 1000 else (cfg.block_size, 1024, 8192, 65536):
        dense = share * S * H * hd
        indexer = S * Hi * di
        sparse = share * min(k, S) * H * hd
        print(f"{S:>10} {dense:>12.2e} {indexer:>10.2e} {sparse:>12.2e} {indexer + sparse:>10.2e} "
              f"{dense / (indexer + sparse):>6.1f}x")  # fmt: skip
print("The indexer still looks at every token: DSA is O(S) per token with a small constant, not O(k).")
print("(the GLM-5.x head sizes above are illustrative: read the real ones from the model's config.json)")
