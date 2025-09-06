# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, output, prev_output):
        if self.drop_prob > 0 and self.training:
            p = torch.rand(output.size(0), dtype=torch.float32, device=output.device)
            p = (p > self.drop_prob)[:, None, None]
            output = torch.where(p, output, prev_output)
        else:
            output = output

        return output

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'