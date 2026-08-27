from typing import Optional

import torch

CacheType = Optional[tuple[torch.Tensor, torch.Tensor]]
CacheListType = Optional[list[CacheType]]
