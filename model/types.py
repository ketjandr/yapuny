import torch

CacheType = tuple[torch.Tensor, torch.Tensor] | None
CacheListType = list[CacheType] | None
