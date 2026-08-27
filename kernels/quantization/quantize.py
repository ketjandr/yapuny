import torch


def quantize_weights_int8(
    weight: torch.Tensor,  # (out_features, in_features)
) -> tuple[torch.Tensor, torch.Tensor]:
    # compute per row max
    max_vals = torch.amax(weight.abs(), dim=1, keepdim=True)  # (out_features, 1)
    scale = max_vals / 127.0  # compute scale
    w_int8_tmp = (weight / scale).round()  # quantize
    w_int8 = w_int8_tmp.clamp(-128, 127).to(torch.int8)  # clamp
    return (w_int8, scale)


def quantize_weights_int4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_vals = torch.amax(weight.abs(), dim=1, keepdim=True)  # (out_features, 1)
    scale = max_vals / 7.0  # compute scale (out_features, 1)
    w_int4_tmp = (weight / scale).round()  # quantize
    w_int4 = w_int4_tmp.clamp(-8, 7).to(torch.int8)  # clamp

    out_features, in_features = w_int4.shape
    assert in_features % 2 == 0
    w_int4_reshaped = w_int4.reshape(out_features, in_features // 2, 2)  # pair two INT4s at once
    lo = w_int4_reshaped[..., 0] & 0xF  # low 4 bits
    hi = (w_int4_reshaped[..., 1] & 0xF) << 4  # high 4 bits
    w_packed = (lo | hi).to(torch.uint8)
    return (w_packed, scale)


def dequantize_weights_int8(w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    w_int8_float = w_int8.to(torch.float32)
    w_fp32 = w_int8_float * scale
    return w_fp32


def dequantize_weights_int4(w_packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lo = w_packed & 0xF  # (out_features, in_features // 2)
    hi = (w_packed >> 4) & 0xF  # (out_features, in_features // 2)

    # stack lo and hi together into (out_features, in_features // 2, 2) and flatten
    w_uint4 = torch.stack([lo, hi], dim=-1).flatten(-2)  # (out_features, in_features)
    w_int4 = w_uint4 - (w_uint4 >= 8) * 16
    w_int4_float = w_int4.to(torch.float32)
    w_fp32 = w_int4_float * scale
    return w_fp32
