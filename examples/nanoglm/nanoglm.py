"""
nanoGLM: the smallest model that has the same moving parts as a GLM-5.x
style LLM (HF class `GlmMoeDsaForCausalLM`):

    GPT (decoder-only transformer, RMSNorm, RoPE, SwiGLU)
  + MoE   : a router sends every token to top-k of E expert FFNs (+1 shared expert)
  + DSA   : a cheap "lightning indexer" scores all previous tokens, and the real
            attention only looks at the top-k of them (sparse attention)
  + cache : KV cache (plus an indexer-key cache) for token-by-token decoding

Everything is in this one file. The numbered scripts poke at it.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 80
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    # --- MoE ---
    n_dense_layers: int = 1  # the first layers keep a plain FFN (GLM-5.x: first 3)
    n_experts: int = 8  # GLM-5.x: 256
    moe_top_k: int = 2  # GLM-5.x: 8
    expert_hidden: int = 384
    # --- DSA (sparse attention) ---
    index_n_head: int = 4  # GLM-5.x: 32-64 small heads
    index_head_dim: int = 32  # must be a power of two for the Hadamard trick in part 5
    index_topk: int = 48  # GLM-5.x: 2048 (out of up to ~1M tokens of context)
    indexer_layers: tuple = (0, 2, 4)  # other layers reuse the previous top-k ("index sharing")


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def rope_table(dim, max_len, base=10000.0):
    """cos/sin tables for rotary position embedding, shape [max_len, dim/2]."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    angles = torch.outer(torch.arange(max_len).float(), freqs)
    return angles.cos(), angles.sin()


def rope(x, cos, sin):
    """Rotate pairs of channels of x [..., T, dim] by a position dependent angle.

    After this, (q_t . k_s) depends only on the distance t - s. Positions are
    absolute at rotation time, which is why a rotated key can sit in the KV
    cache forever and never needs to be touched again.
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


class SwiGLU(nn.Module):
    """The FFN of modern LLMs: down( silu(gate(x)) * up(x) ), gate and up fused in one matrix."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


# -----------------------------------------------------------------------------
# MoE
# -----------------------------------------------------------------------------


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_experts, self.top_k = cfg.n_experts, cfg.moe_top_k
        self.router = nn.Linear(cfg.n_embd, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg.n_embd, cfg.expert_hidden) for _ in range(cfg.n_experts))
        self.shared = SwiGLU(cfg.n_embd, cfg.expert_hidden)  # every token goes through this one
        self.last_topi = None  # kept around so scripts can look at the routing
        self.balance_loss = None

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])  # [N, C]: from here on a "batch" is just a bag of tokens
        N = x.shape[0]

        # 1) route: every token picks its top-k experts
        scores = torch.sigmoid(self.router(x).float())  # [N, E]
        topv, topi = scores.topk(self.top_k, dim=-1)  # [N, K]
        gates = topv / topv.sum(-1, keepdim=True)  # mixing weights, sum to 1

        # 2) group: sort the N*K (token, expert) pairs by expert, so that every
        #    expert sees one contiguous slab of tokens. This is what an inference
        #    engine calls "grouped matmul" (GMM).
        token_id = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        order = topi.flatten().argsort(stable=True)
        token_id, gate = token_id[order], gates.flatten()[order]
        counts = torch.bincount(topi.flatten(), minlength=self.n_experts)

        # 3) run each expert on its slab and scatter the results back
        out = self.shared(x)
        start = 0
        for e, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            rows = token_id[start : start + n]
            y = self.experts[e](x[rows]) * gate[start : start + n, None].to(x.dtype)
            out = out.index_add(0, rows, y.to(out.dtype))
            start += n

        # load balancing loss (training only): minimal when all experts get equal load
        if self.training:
            load = counts.float() * self.n_experts / (N * self.top_k)  # 1.0 == fair share
            importance = (scores / scores.sum(-1, keepdim=True)).mean(0) * self.n_experts
            self.balance_loss = (load * importance).mean()
        self.last_topi = topi.view(*shape[:-1], self.top_k)
        return out.view(shape)


# -----------------------------------------------------------------------------
# DSA: lightning indexer + sparse attention
# -----------------------------------------------------------------------------


class LightningIndexer(nn.Module):
    """A tiny attention-like scorer:  I[t, s] = sum_h  w[t, h] * relu( q[t, h] . k[s] )

    Few heads, small dims, ONE key vector per token shared by all heads, no
    softmax, no values. Its only job: say which previous tokens s are worth the
    real attention's time for query token t.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.index_n_head, cfg.index_head_dim
        self.wq = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        # one matrix produces the key (head_dim) AND the per-head weights (n_head),
        # like `wk_weights_proj` in vllm_ascend/attention/sfa_v1.py
        self.wk_weights = nn.Linear(cfg.n_embd, self.head_dim + self.n_head, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.qk_transform = None  # hook for part 5: quantize q and k before the dot product

    def project(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        q = rope(q, cos, sin).transpose(1, 2)  # [B, T, H, d]
        kw = self.wk_weights(x)
        k = rope(self.k_norm(kw[..., : self.head_dim]), cos, sin)  # [B, T, d]   <- goes into the cache
        w = kw[..., self.head_dim :]  # [B, T, H]
        return q, k, w

    def score(self, q, k, w):
        """q [B,T,H,d], k [B,S,d], w [B,T,H]  ->  I [B,T,S]"""
        if self.qk_transform is not None:
            q, k = self.qk_transform(q, k)
        s = torch.einsum("bthd,bsd->bths", q.float(), k.float())
        return torch.einsum("bths,bth->bts", F.relu(s), w.float())


def stable_topk(scores, k):
    """top-k where ties always go to the earlier token.

    relu() makes exact ties (0.0) common. torch.topk breaks ties arbitrarily, so
    prefill and decode could pick different tokens for the same query. A stable
    sort makes the selection a pure function of the scores.
    """
    values, indices = scores.sort(dim=-1, descending=True, stable=True)
    return values[..., :k], indices[..., :k]


class Attention(nn.Module):
    def __init__(self, cfg, has_indexer):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.indexer = LightningIndexer(cfg) if has_indexer else None
        self.select_scores = None  # hook for part 3: replace indexer scores by a baseline
        self.kv_transform = None  # hook for part 5: quantize k and v on their way into the cache
        self.last = {}  # tensors kept for the scripts to inspect

    def forward(self, x, rope_main, rope_idx, cache, topk, sparse, top_k, want_info):
        """
        x      : [B, T, C] the T new tokens
        cache  : None, or a dict holding everything this layer remembers of the S-T tokens before
        topk   : selection made by an earlier layer (index sharing), or None
        sparse : False (dense attention), "mask" (dense compute, sparse result; for training)
                 or "gather" (really only touch top-k keys; for inference)
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)  # each [B, H, T, hd]
        q, k = rope(q, *rope_main), rope(k, *rope_main)
        if self.kv_transform is not None:
            k, v = self.kv_transform(k), self.kv_transform(v)
        if self.indexer is not None:
            # detach(): the indexer learns to imitate the attention, the model never bends to help it
            iq, ik, iw = self.indexer.project(x.detach(), *rope_idx)

        # ---- KV cache: append the new keys/values, then attend over everything so far ----
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), dim=2), torch.cat((cache["v"], v), dim=2)
                if self.indexer is not None:
                    ik = torch.cat((cache["ik"], ik), dim=1)
            cache["k"], cache["v"] = k, v
            if self.indexer is not None:
                cache["ik"] = ik
        S = k.shape[2]  # number of keys = cached + new
        # causal[t, s]: may new token t (absolute position S-T+t) look at key s?
        causal = torch.arange(S, device=x.device)[None, :] <= torch.arange(S - T, S, device=x.device)[:, None]

        # ---- who is worth looking at? ----
        info = None
        scores = None
        if self.indexer is not None and (sparse or want_info):
            scores = self.indexer.score(iq, ik, iw)  # [B, T, S]
            if self.select_scores is not None:
                scores = self.select_scores(scores)
            scores = scores.masked_fill(~causal, float("-inf"))
            if sparse:
                topk = stable_topk(scores, min(top_k, S))  # (values [B,T,k], indices [B,T,k])
        if sparse:
            assert topk is not None, "the first layer must have an indexer"
            idx, valid = topk[1], torch.isfinite(topk[0])  # invalid = future tokens

        scale = 1.0 / math.sqrt(self.head_dim)
        if sparse == "gather":
            # The real thing: pull only the selected k keys/values out of the cache.
            # Cost per query token: O(k), no matter how long the context S is.
            b = torch.arange(B, device=x.device)[:, None, None]
            k_sel = k[b, :, idx].permute(0, 3, 1, 2, 4)  # [B, H, T, k, hd]
            v_sel = v[b, :, idx].permute(0, 3, 1, 2, 4)
            att = torch.einsum("bhtd,bhtkd->bhtk", q, k_sel).float() * scale
            att = att.masked_fill(~valid[:, None], float("-inf")).softmax(-1)
            y = torch.einsum("bhtk,bhtkd->bhtd", att.to(v_sel.dtype), v_sel)
        else:
            allowed = causal[None].expand(B, T, S)
            if sparse == "mask":
                # Same result as "gather", computed the wasteful way. Convenient
                # for training because the full attention matrix exists.
                allowed = torch.zeros(B, T, S, dtype=torch.bool, device=x.device).scatter_(2, idx, valid) & allowed
            att = (q @ k.transpose(-2, -1)).float() * scale  # [B, H, T, S]
            att = att.masked_fill(~allowed[:, None], float("-inf")).softmax(-1)
            y = att.to(v.dtype) @ v
            if want_info:  # what the indexer loss needs: where did this layer really look?
                info = {"att": att.detach().mean(1), "scores": scores, "allowed": allowed}
            self.last = {"att": att.detach(), "scores": None if scores is None else scores.detach()}
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y), topk, info


def index_loss(scores, allowed, attention_maps):
    """Teach an indexer: softmax(I[t, :]) should match where the attention really goes.

    The target is the attention averaged over heads AND over all layers that
    use this indexer's selection (the owner and the layers that share it).
    Cross entropy between two distributions over previous tokens s.
    """
    target = torch.stack(attention_maps).mean(0)  # [B, T, S], rows sum to 1
    logp = scores.masked_fill(~allowed, float("-inf")).log_softmax(-1)
    return -(target * logp.masked_fill(~allowed, 0.0)).sum(-1).mean()


class Block(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.n_embd), RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg, has_indexer=layer in cfg.indexer_layers)
        self.ffn = SwiGLU(cfg.n_embd, 4 * cfg.n_embd) if layer < cfg.n_dense_layers else MoE(cfg)

    def forward(self, x, *args):
        a, topk, info = self.attn(self.norm1(x), *args)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, topk, info


class NanoGLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert 0 in cfg.indexer_layers
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        for name, table in (("main", rope_table(cfg.n_embd // cfg.n_head, cfg.block_size)),
                            ("idx", rope_table(cfg.index_head_dim, cfg.block_size))):  # fmt: skip
            self.register_buffer(f"rope_{name}_cos", table[0], persistent=False)
            self.register_buffer(f"rope_{name}_sin", table[1], persistent=False)
        self.sparse = False  # False | "mask" | "gather"
        self.top_k = cfg.index_topk
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def new_cache(self):
        return {"pos": 0, "layers": [{} for _ in self.blocks]}

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        pos = cache["pos"] if cache is not None else 0  # absolute position of the first new token
        rope_main = (self.rope_main_cos[pos : pos + T], self.rope_main_sin[pos : pos + T])
        rope_idx = (self.rope_idx_cos[pos : pos + T], self.rope_idx_sin[pos : pos + T])
        x = self.wte(idx)
        topk, groups = None, []  # a group = one indexer + the attention maps of the layers it serves
        for i, block in enumerate(self.blocks):
            layer_cache = cache["layers"][i] if cache is not None else None
            x, topk, info = block(
                x, rope_main, rope_idx, layer_cache, topk, self.sparse, self.top_k, targets is not None
            )
            if info is not None:
                if info["scores"] is not None:  # this layer owns an indexer: a new group starts
                    groups.append({"scores": info["scores"], "allowed": info["allowed"], "att": []})
                groups[-1]["att"].append(info["att"])
        if cache is not None:
            cache["pos"] += T
        logits = self.lm_head(self.norm(x)).float()
        if targets is None:
            return logits, None
        losses = {"lm": F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))}
        if groups:
            losses["index"] = torch.stack([index_loss(g["scores"], g["allowed"], g["att"]) for g in groups]).mean()
        balance = [m.balance_loss for m in self.modules() if isinstance(m, MoE) and m.balance_loss is not None]
        if self.training and balance:
            losses["balance"] = torch.stack(balance).mean()
        return logits, losses

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, use_cache=True):
        """Prefill (the whole prompt at once), then decode (one token at a time)."""
        cache = self.new_cache() if use_cache else None
        new = idx
        for _ in range(max_new_tokens):
            logits, _ = self(new if use_cache else idx, cache=cache)
            logits = logits[:, -1]
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            idx = torch.cat((idx, nxt), dim=1)
            new = nxt  # with a cache, only the new token is fed in
        return idx


def pick_device():
    """cuda, npu (Ascend) or cpu.

    torch_npu must be imported before torch.npu exists: importing it is what
    registers the NPU backend, the "npu" device string and its autocast type.
    """
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(path, device="cpu"):
    """Returns (model, dataset) exactly as they were at training time."""
    from data import Needles

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NanoGLM(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), Needles(gap=tuple(ckpt["gap"]))


@torch.no_grad()
def evaluate(model, data, batches=8, batch_size=32, sparse=False, seed=0):
    """Returns (loss, query digit accuracy). The same batches for every call with the same seed."""
    state = data.rng.getstate()
    data.rng.seed(seed)
    device = next(model.parameters()).device
    was_training, was_sparse = model.training, model.sparse
    model.eval()
    model.sparse = sparse
    loss, hit, total = 0.0, 0, 0
    for _ in range(batches):
        x, y, qmask, _ = data.batch("val", batch_size, model.cfg.block_size, device)
        logits, losses = model(x, y)
        loss += losses["lm"].item() / batches
        hit += (logits.argmax(-1) == y)[qmask].sum().item()
        total += qmask.sum().item()
    model.train(was_training)
    model.sparse = was_sparse
    data.rng.setstate(state)
    return loss, hit / max(total, 1)
