"""
Part 5: quantizing the sparse attention itself ("C8": 8-bit caches).

Two things live in the cache and are read for every decoded token:
  - the lightning indexer keys  (read for ALL previous tokens: the O(S) part)
  - the K/V of the real attention (read for the top-k tokens)
Both can be stored as int8. For the indexer the question is not "is the score
accurate" but "does the top-k set stay the same".

    python 05_quant_sparse.py [out/nanoglm.pt]
"""

import sys

import torch
from nanoglm import evaluate, load, stable_topk
from quant import fake_quant, hadamard, quantize_model

torch.manual_seed(0)
model, data = load(sys.argv[1] if len(sys.argv) > 1 else "out/nanoglm.pt")
cfg = model.cfg
T, d = cfg.block_size, cfg.index_head_dim
H = hadamard(d)


def set_indexer(fn, m=None):
    for block in (m or model).blocks:
        if block.attn.indexer is not None:
            block.attn.indexer.qk_transform = fn


def set_kv(fn, m=None):
    for block in (m or model).blocks:
        block.attn.kv_transform = fn


def make_transform(bits=None, rotate=False, outlier=1.0):
    """Build a function applied to indexer (q, k) right before their dot product."""
    stretch = torch.ones(d)
    stretch[:2] = outlier

    def transform(q, k):
        # "outlier": make two channels of k huge and the same channels of q tiny.
        # q.k is unchanged, so in float this is the SAME model.
        q, k = q / stretch, k * stretch
        if rotate:  # (qH).(kH) = q H H^T k = q.k : also changes nothing in float
            q, k = q @ H, k @ H
        if bits is not None:
            q, k = fake_quant(q, bits), fake_quant(k, bits)
        return q, k

    return transform


@torch.no_grad()
def selections(batches=2):
    """The top-k sets every indexer picks, on fixed validation data. Also needle hits."""
    data.rng.seed(0)
    model.sparse = "mask"
    out, hits, total = [], 0, 0
    for _ in range(batches):
        x, y, qmask, src = data.batch("val", 16, T)
        model(x, y)
        for i in cfg.indexer_layers:
            scores = model.blocks[i].attn.last["scores"]
            values, idx = stable_topk(scores, cfg.index_topk)
            chosen = torch.zeros_like(scores, dtype=torch.bool).scatter_(2, idx, torch.isfinite(values))
            out.append(chosen)
            hits += chosen.gather(2, src.clamp(min=0)[..., None]).squeeze(-1)[qmask].sum().item()
            total += qmask.sum().item()
    return out, hits / total


set_indexer(None)
reference, _ = selections()

print(f"A. quantizing the lightning indexer (top-{cfg.index_topk} of {T})\n")
print(f"{'indexer q, k':38} {'same picks':>10} {'needle recall':>14} {'loss':>8} {'q. acc':>7}")
rows = [("float", make_transform())]
for bits in (8, 6, 4, 3):
    rows.append((f"int{bits}", make_transform(bits)))
    rows.append((f"int{bits} + Hadamard", make_transform(bits, rotate=True)))
rows += [
    ("float, outlier channels x100", make_transform(outlier=100.0)),
    ("int8, outlier channels x100", make_transform(8, outlier=100.0)),
    ("int8 + Hadamard, outlier channels x100", make_transform(8, rotate=True, outlier=100.0)),
]
for name, transform in rows:
    set_indexer(transform)
    picks, recall = selections()
    same = sum((a & b).sum().item() for a, b in zip(picks, reference)) / sum(b.sum().item() for b in reference)
    loss, acc = evaluate(model, data, sparse="gather")
    print(f"{name:38} {same:>10.3f} {recall:>14.3f} {loss:>8.4f} {acc:>7.3f}")
set_indexer(None)

print("\nB. quantizing the K/V cache of the real attention (one scale per token per head)\n")
print(f"{'K, V in cache':38} {'loss':>8} {'q. acc':>7}")
for name, fn in [("float", None)] + [(f"int{b}", lambda t, b=b: fake_quant(t, b)) for b in (8, 6, 4)]:
    set_kv(fn)
    loss, acc = evaluate(model, data, sparse="gather")
    print(f"{name:38} {loss:>8.4f} {acc:>7.3f}")
set_kv(None)

print("\nC. everything at once: the 'w8a8c8' deployment\n")
print(f"{'configuration':60} {'loss':>8} {'q. acc':>7}")
loss, acc = evaluate(model, data, sparse=False)
print(f"{'float, dense attention':60} {loss:>8.4f} {acc:>7.3f}")
loss, acc = evaluate(model, data, sparse="gather")
print(f"{'float, sparse attention':60} {loss:>8.4f} {acc:>7.3f}")
q = quantize_model(model, "dynamic")
loss, acc = evaluate(q, data, sparse="gather")
print(f"{'W8A8 linears, sparse attention':60} {loss:>8.4f} {acc:>7.3f}")
set_indexer(make_transform(8, rotate=True), q)
set_kv(fake_quant, q)
loss, acc = evaluate(q, data, sparse="gather")
print(f"{'W8A8 linears + int8 indexer (Hadamard) + int8 K/V cache':60} {loss:>8.4f} {acc:>7.3f}")
