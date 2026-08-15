import triton

@triton.jit
def fused_layernorm_residual(
    x, y,
    output, weight, bias
):
    