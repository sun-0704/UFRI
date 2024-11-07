from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F
from .functional import label_smoothed_nll_loss
import pdb 

__all__ = ["SoftCrossEntropyLoss"]


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.0, ignore_index: Optional[int] = -100, dim=1, weight: Optional[Tensor] = None):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim
        self.weight = weight
        

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        #pdb.set_trace()
        log_prob = F.log_softmax(input, dim=self.dim)

        if self.weight is not None:
            weight_tensor = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()  # 调整维度以匹配 log_prob
            log_prob = log_prob * weight_tensor

        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )
