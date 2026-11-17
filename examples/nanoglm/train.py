"""
Train nanoGLM on "needles in Shakespeare".

    python train.py --preset tiny          # CPU / laptop
    python train.py --preset single_card   # one accelerator: an Ascend NPU, or a GPU

Two stages, like the DSA recipe of DeepSeek-V3.2 that GLM-5.x inherits:
  1. dense warm-up : normal dense attention. On the side, each lightning
                     indexer learns to predict where the attention goes.
  2. sparse stage  : attention is restricted to the indexer's top-k and the
                     model gets used to seeing only that.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import torch
from data import Needles
from nanoglm import Config, NanoGLM, evaluate, pick_device

PRESETS = {
    "tiny": dict(
        cfg=dict(block_size=128, n_layer=4, n_head=4, n_embd=96, n_experts=4, expert_hidden=96,
                 index_n_head=2, index_head_dim=32, index_topk=16, indexer_layers=(0, 2)),
        batch_size=32, steps=4000, lr=1e-3, gap=(6, 20),
    ),
    "single_card": dict(
        cfg=dict(block_size=512, n_layer=6, n_head=6, n_embd=384, n_experts=8, expert_hidden=384,
                 index_n_head=4, index_head_dim=32, index_topk=48, indexer_layers=(0, 2, 4)),
        batch_size=64, steps=12000, lr=6e-4, gap=(15, 45),
    ),
}  # fmt: skip

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="tiny", choices=PRESETS)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--sparse_frac", type=float, default=0.25, help="fraction of steps spent in the sparse stage")
parser.add_argument("--eval_every", type=int, default=250)
parser.add_argument("--out", default="out")
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

preset = PRESETS[args.preset]
steps = args.steps or preset["steps"]
sparse_from = int(steps * (1 - args.sparse_frac))
device = pick_device()
torch.manual_seed(args.seed)

data = Needles(args.seed, preset["gap"])
cfg = Config(vocab_size=data.vocab_size, **preset["cfg"])
model = NanoGLM(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_indexer = sum(p.numel() for n, p in model.named_parameters() if "indexer" in n)
print(f"{n_params / 1e6:.2f}M parameters, of which lightning indexers: {n_indexer / 1e3:.0f}K | device {device}")

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=preset["lr"], betas=(0.9, 0.95),
)  # fmt: skip


def lr_at(step, warmup=200):
    if step < warmup:
        return preset["lr"] * (step + 1) / warmup
    return preset["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))))


os.makedirs(args.out, exist_ok=True)
log, t0 = [], time.time()
for step in range(steps + 1):
    if step == sparse_from:
        print(f"--- step {step}: switching to sparse attention (top-{cfg.index_topk} of {cfg.block_size}) ---")
        model.sparse = "mask"

    if step % args.eval_every == 0 or step == steps:
        dense_loss, dense_acc = evaluate(model, data, sparse=False)
        sparse_loss, sparse_acc = evaluate(model, data, sparse="mask")
        row = dict(step=step, dense_loss=dense_loss, dense_acc=dense_acc, sparse_loss=sparse_loss,
                   sparse_acc=sparse_acc, time=time.time() - t0)  # fmt: skip
        log.append(row)
        print(
            f"step {step:5d} | val loss dense {dense_loss:.3f} sparse {sparse_loss:.3f} "
            f"| query acc dense {dense_acc:.3f} sparse {sparse_acc:.3f} | {row['time']:.0f}s"
        )
    if step == steps:
        break

    for group in opt.param_groups:
        group["lr"] = lr_at(step)
    x, y, _, _ = data.batch("train", preset["batch_size"], cfg.block_size, device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
        _, losses = model(x, y)
    # three losses: predict the next char, imitate the attention, keep experts busy
    loss = losses["lm"] + losses["index"] + 0.01 * losses["balance"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        print(
            f"  {step:5d} lm {losses['lm'].item():.3f} index {losses['index'].item():.3f} "
            f"balance {losses['balance'].item():.3f}"
        )

torch.save(
    {"model": model.state_dict(), "config": asdict(cfg), "gap": preset["gap"]}, os.path.join(args.out, "nanoglm.pt")
)
with open(os.path.join(args.out, "log.json"), "w") as f:
    json.dump(log, f, indent=1)
print(f"saved {args.out}/nanoglm.pt")
