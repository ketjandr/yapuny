// Shared quantization cycle: each click advances a node's weights none -> W8 -> W4 -> none.
export const QUANT_CYCLE: (string | null)[] = [null, "w8", "w4"];

export function nextQuant(current: string | null): string | null {
  return QUANT_CYCLE[(QUANT_CYCLE.indexOf(current ?? null) + 1) % QUANT_CYCLE.length];
}
