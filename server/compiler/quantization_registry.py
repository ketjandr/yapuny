from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from server.compiler.node_registry import NODE_REGISTRY

QUANTIZABLE_NODES = {name for name, d in NODE_REGISTRY.items() if d.quantize_attr}
QUANT_MODES = {"w8", "w4"}


@dataclass
class QuantModeDef:
    quantize_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    linear_cls: type[nn.Module]


try:
    from kernels.quantization.quantize import (
        quantize_weights_int4,
        quantize_weights_int8,
    )
    from kernels.quantization.quantized_linear_w4 import QuantizedLinearW4
    from kernels.quantization.quantized_linear_w8 import QuantizedLinearW8

    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

if QUANTIZATION_AVAILABLE:
    QUANT_REGISTRY: dict[str, QuantModeDef] = {
        "w8": QuantModeDef(quantize_fn=quantize_weights_int8, linear_cls=QuantizedLinearW8),
        "w4": QuantModeDef(quantize_fn=quantize_weights_int4, linear_cls=QuantizedLinearW4),
    }
else:
    QUANT_REGISTRY = {}


def _replace_linear(
    module: nn.Module,
    attr: str,
    mode: str,
) -> None:
    mode_def = QUANT_REGISTRY[mode]
    linear = getattr(module, attr)
    in_f, out_f = linear.in_features, linear.out_features
    has_bias = linear.bias is not None
    weight = linear.weight.data.float()

    w_q, scale = mode_def.quantize_fn(weight)
    q_mod = mode_def.linear_cls(in_f, out_f)

    q_mod.w_q.copy_(w_q)
    q_mod.scale.copy_(scale.squeeze(-1))
    q_mod.bias.copy_(linear.bias.data if has_bias else torch.zeros(out_f))

    setattr(module, attr, q_mod)


def quantize_module(module: nn.Module, node_type: str, mode: str) -> None:
    if not QUANTIZATION_AVAILABLE:
        raise ValueError("quantization requires a CUDA GPU with Triton")
    attr = NODE_REGISTRY[node_type].quantize_attr
    _replace_linear(module, attr, mode)
