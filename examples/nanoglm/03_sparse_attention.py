"""
Part 3: sparse attention. Does the lightning indexer find the needle?

    python 03_sparse_attention.py [out/nanoglm.pt]
"""

import sys

import torch
from nanoglm import evaluate, load

torch.manual_seed(0)
model, data = load(sys.argv[1] if len(sys.argv) > 1 else "out/nanoglm.pt")
cfg = model.cfg
T = cfg.block_size


def set_selector(fn):
    for block in model.blocks:
        block.attn.select_scores = fn


@torch.no_grad()
def needle_recall(k, batches=4):
    """At query digits: how often is the needle (where the digit was SET) inside the top-k?"""
    data.rng.seed(0)
    model.sparse, model.top_k = "mask", k
    hits = {i: 0 for i in cfg.indexer_layers}
    total = 0
    for _ in range(batches):
        x, y, qmask, src = data.batch("val", 32, T)
        model(x, y)
        for i in hits:
            scores = model.blocks[i].attn.last["scores"]  # [B, T, S], already causal-masked
            kth = scores.sort(dim=-1, descending=True).values[..., min(k, T) - 1]
            needle_score = scores.gather(2, src.clamp(min=0)[..., None]).squeeze(-1)
            hits[i] += (needle_score >= kth)[qmask].sum().item()
        total += qmask.sum().item()
    model.top_k = cfg.index_topk
    return {i: h / total for i, h in hits.items()}


# --- 1. accuracy vs k, against two baselines that also keep only k tokens ---
selectors = {
    "lightning indexer": None,
    "last k tokens": lambda s: torch.arange(s.shape[-1], device=s.device).float().expand_as(s),
    "random k tokens": lambda s: torch.rand_like(s),
}
dense_loss, dense_acc = evaluate(model, data, sparse=False)
print(f"dense attention (all {T} tokens): loss {dense_loss:.3f}  query acc {dense_acc:.3f}\n")
print(f"{'k':>4} | " + " | ".join(f"{name:^24}" for name in selectors))
print(f"{'':>4} | " + " | ".join(f"{'loss':>10} {'query acc':>13}" for _ in selectors))
for k in (4, 8, 16, 32, 64, 128):
    if k > T:
        continue
    row = []
    model.top_k = k
    for fn in selectors.values():
        set_selector(fn)
        loss, acc = evaluate(model, data, sparse="gather")
        row.append(f"{loss:>10.3f} {acc:>13.3f}")
    print(f"{k:>4} | " + " | ".join(row) + ("   <- trained with this k" if k == cfg.index_topk else ""))
set_selector(None)
model.top_k = cfg.index_topk

# --- 2. is the needle in the selected set? ---
print("\nneedle recall at query digits (per layer that owns an indexer):")
for k in (4, 8, 16, 32, 64):
    recall = needle_recall(k)
    print(f"  k={k:3}: " + "  ".join(f"layer {i}: {r:.3f}" for i, r in recall.items()))

# --- 3. look at one query ---
data.rng.seed(3)
x, y, qmask, src = data.batch("val", 1, T)
model.sparse = False
model(x, y)
t = qmask[0].nonzero()[0].item()
needle = src[0, t].item()
text = data.decode(x[0])
print(f"\none query: ...{text[t - 12 : t + 1]!r} must be followed by {data.itos[y[0, t].item()]!r}")
print(f"the needle is {t - needle} tokens back: {text[needle - 3 : needle + 3]!r}")
print(" where does the model itself look? strongest head's attention on the needle, per layer:")
print(
    "   "
    + "  ".join(
        f"layer {i}: {b.attn.last['att'][0, :, t, needle].max().item():.2f}" for i, b in enumerate(model.blocks)
    )
)
for i in cfg.indexer_layers:
    scores = model.blocks[i].attn.last["scores"][0, t]  # [S]
    top = scores.topk(8).indices.tolist()
    print(f" indexer of layer {i}: needle is rank {(scores > scores[needle]).sum().item() + 1} of {t + 1} | "
          f"its top-8 are {[text[s] for s in top]} at distances {[t - s for s in top]}")  # fmt: skip

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(cfg.n_layer, 1, figsize=(11, 2.2 * cfg.n_layer), sharex=True, squeeze=False)
    for i, (ax, block) in enumerate(zip(axes[:, 0], model.blocks)):
        last = block.attn.last
        if last["scores"] is not None:  # this layer owns an indexer; the following layers reuse its picks
            owner, scores = i, last["scores"][0, t, : t + 1]
            chosen = scores.topk(min(cfg.index_topk, t + 1)).indices
        ax.bar(range(t + 1), last["att"][0, :, t, : t + 1].amax(0), width=1.0, color="#3b6fb6",
               label="real attention (strongest head)")  # fmt: skip
        ax.scatter(chosen, torch.full((len(chosen),), -0.06), s=9, color="black", zorder=3,
                   label=f"top-{cfg.index_topk} picked by the indexer of layer {owner}")  # fmt: skip
        ax.axvline(needle, color="#2a9d5c", ls="--", lw=1, label="needle")
        ax.set_ylim(-0.12, 1.0)
        ax.set_ylabel(f"layer {i}" + ("" if owner == i else "\n(shared picks)"))
        ax.legend(loc="upper center", fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("position in context (the query token is at the right edge)")
    fig.tight_layout()
    fig.savefig("out/sparse_attention.png", dpi=140)
    print("\nsaved out/sparse_attention.png")
except ImportError:
    print("\n(matplotlib not installed, skipping the plot)")

# --- 4. look at your errors: is the model really matching keys? ---
data.rng.seed(0)
model.sparse = "gather"
by_digit, by_between = {}, {}
with torch.no_grad():
    for _ in range(8):
        x, y, qmask, src = data.batch("val", 32, T)
        correct = model(x)[0].argmax(-1) == y
        for b, t in qmask.nonzero().tolist():
            s = data.decode(x[b])
            first = s[t] == "?"
            hit, n = by_digit.get(first, (0, 0))
            by_digit[first] = (hit + correct[b, t].item(), n + 1)
            if first:  # how many other SETs happened between the needle and this query?
                between = min(s[src[b, t].item() : t].count("="), 3)
                hit, n = by_between.get(between, (0, 0))
                by_between[between] = (hit + correct[b, t].item(), n + 1)
print("\nquery accuracy, sparse attention:")
first_acc, second_acc = (by_digit[key][0] / by_digit[key][1] for key in (True, False))
print(f"  first digit {first_acc:.3f} | second digit {second_acc:.3f}")
print("  first digit, by number of other SETs between the needle and the query:")
for between, (hit, n) in sorted(by_between.items()):
    print(f"    {between}{'+' if between == 3 else ' '}: {hit / n:.3f}  (n={n})")
