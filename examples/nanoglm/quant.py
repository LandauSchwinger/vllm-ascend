"""
W8A8 from scratch: 8-bit Weights, 8-bit Activations.

A float is replaced by a small integer times a shared scale:

    x  ~=  x_q * scale        x_q in {-127, ..., 127}  (int8)

Then a matmul of floats becomes a matmul of int8 with one float multiply at the end:

    y = x @ W.T  ~=  (x_q @ W_q.T) * (scale_x * scale_w)
                      ^^^^^^^^^^^ pure integer arithmetic: this is what the hardware accelerates
"""

import copy

import torch
import torch.nn as nn


def qmax(bits):
    """Largest integer we use: 127 for 8 bits. We give up -128 so the range is symmetric around 0."""
    return 2 ** (bits - 1) - 1


def quantize(x, scale, bits=8):
    return torch.round(x / scale).clamp(-qmax(bits), qmax(bits))


def weight_quant(w, bits=8):
    """Per-channel: one scale per output row of W [out, in]. Done once, offline."""
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax(bits)
    return quantize(w, scale, bits).to(torch.int8), scale.squeeze(1)


def dynamic_quant(x, bits=8):
    """Per-token: one scale per vector (last dim), computed on the fly at inference.

    This is `torch_npu.npu_dynamic_quant` in vllm-ascend.
    """
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax(bits)
    return quantize(x, scale, bits), scale


def fake_quant(x, bits=8):
    """Quantize and immediately dequantize: same values the integer path would see, still a float tensor."""
    q, scale = dynamic_quant(x.float(), bits)
    return (q * scale).to(x.dtype)


class QLinear(nn.Module):
    """Drop-in replacement for nn.Linear with int8 weights.

    act = "float"   : W8A16. Only weights are int8, activations stay float.
    act = "dynamic" : W8A8 dynamic. Each token gets its own activation scale, measured at runtime.
    act = "static"  : W8A8 static. ONE activation scale for the layer, fixed in advance by calibration.
    """

    def __init__(self, linear, act="dynamic", input_scale=None, w_bits=8, a_bits=8):
        super().__init__()
        self.a_bits = a_bits
        w_q, w_scale = weight_quant(linear.weight.detach().float(), w_bits)
        self.register_buffer("weight_q", w_q)  # int8  [out, in]   <- 1 byte per weight
        self.register_buffer("weight_scale", w_scale)  # float [out]
        self.bias = linear.bias
        self.act = act
        if act == "static":
            self.register_buffer("input_scale", torch.as_tensor(float(input_scale)))

    def forward(self, x):
        dtype, x = x.dtype, x.float()
        # We run the integer matmul in float32 because that is what a GPU/CPU is
        # fast at. The values are integers and the sums stay far below 2^24, so
        # it is bit-exact with a real int8 x int8 -> int32 matmul (04 checks this).
        w = self.weight_q.float()
        if self.act == "float":
            y = (x @ w.T) * self.weight_scale
        elif self.act == "dynamic":
            x_q, x_scale = dynamic_quant(x, self.a_bits)  # x_scale [..., 1]
            y = (x_q @ w.T) * x_scale * self.weight_scale
        else:
            x_q = quantize(x, self.input_scale, self.a_bits)
            y = (x_q @ w.T) * (self.input_scale * self.weight_scale)  # "deq_scale" in vllm-ascend
        if self.bias is not None:
            y = y + self.bias
        return y.to(dtype)


@torch.no_grad()
def calibrate(model, batches):
    """Static quantization needs to know the range of every layer's input ahead of time.

    Run some data through the float model and record max |x| at the input of each Linear.
    Returns {layer name: max |x|}.
    """
    peaks, hooks = {}, []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):

            def hook(mod, inp, out, name=name):
                peaks[name] = max(peaks.get(name, 0.0), inp[0].abs().max().item())

            hooks.append(m.register_forward_hook(hook))
    for x in batches:
        model(x)
    for h in hooks:
        h.remove()
    return {name: max(peak, 1e-8) for name, peak in peaks.items()}


def quantize_model(model, act="dynamic", select=lambda name: True, input_peaks=None, w_bits=8, a_bits=8):
    """Return a copy of model where every nn.Linear whose name passes `select` is a QLinear."""
    model = copy.deepcopy(model)
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and select(name):
            parent = model.get_submodule(name.rpartition(".")[0]) if "." in name else model
            scale = None
            if act == "static":
                if name not in input_peaks:
                    continue  # never saw data during calibration (e.g. an expert nobody routed to)
                scale = input_peaks[name] / qmax(a_bits)
            setattr(parent, name.rpartition(".")[2], QLinear(m, act, scale, w_bits, a_bits))
    return model


def hadamard(n):
    """Orthonormal Hadamard matrix (n a power of two): H @ H.T = I, all entries +-1/sqrt(n)."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    assert h.shape[0] == n, "n must be a power of two"
    return h / n**0.5
