# encoding: utf-8
import torch
import warnings
import numpy as np
import torch.nn.functional as F


def segment_length(segments):
    return (segments[:, 1]-segments[:, 0]).clamp(min=0)


def segment_cw_to_t1t2(x):
    """
    corresponds to box_cxcywh_to_xyxy in detr
    :param x: segments in (center, width) format, shape=(*, 2)
    :return: segments in (t_start, t_end) format, shape=(*, 2)
    """
    if not isinstance(x, np.ndarray):
        x_c, w = x.unbind(-1)
        b = [(x_c - 0.5 * w), (x_c + 0.5 * w)]
        b = torch.stack(b, dim=-1)
        return b
    else:
        x_c, w = x[..., 0], x[..., 1]
        b = [(x_c - 0.5 * w)[..., None], (x_c + 0.5 * w)[..., None]]
        return np.concatenate(b, axis=-1)


def segment_t1t2_to_cw(x):
    """
    corresponds to box_xyxy_to_cxcywh in detr
    :param x: segments in (t_start, t_end) format, shape=(*, 2)
    :return: segments in (center, width) format, shape=(*, 2)
    """
    if not isinstance(x, np.ndarray):
        x1, x2 = x.unbind(-1)
        b = [(x1 + x2) / 2, (x2 - x1)]
        return torch.stack(b, dim=-1)
    else:
        x1, x2 = x[..., 0], x[..., 1]
        b = [((x1 + x2) / 2)[..., None], (x2 - x1)[..., None]]
        return np.concatenate(b, axis=-1)


def segment_iou(segments1, segments2):
    """
    Temporal IoU between the boxes that should be in [x0, y0, x1, y1] format
    :param segments1: segments in (t_start, t_end) format, shape=(N, 2)
    :param segments2: segments in (t_start, t_end) format, shape=(M, 2)
    :return: a [N, M] pairwise matrix
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (segments1[:, 1] >= segments1[:, 0]).all(), f'{segments1.size()}_{(segments1[:, 1] >= segments1[:, 0]).sum()}_{segments1[~(segments1[:, 1] >= segments1[:, 0])]}'

    area1 = segment_length(segments1)
    area2 = segment_length(segments2)

    l = torch.max(segments1[:, None, 0], segments2[:, 0])  # N,M
    r = torch.min(segments1[:, None, 1], segments2[:, 1])  # N,M
    inter = (r - l).clamp(min=0)  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


def segment_diou(src_segments: torch.Tensor, target_segments: torch.Tensor) -> torch.Tensor:
    """
    Calculate the direct DIOU between predicted and target segments.
    :param src_segments: Tensor of shape (N, 2), where N is the number of segments.
    :param target_segments: Tensor of shape (N, 2), where N is the number of segments.
    :return: diou: Tensor of shape (N,)
    """
    # Intersection
    l = torch.max(src_segments[:, 0], target_segments[:, 0])
    r = torch.min(src_segments[:, 1], target_segments[:, 1])
    inter = (r - l).clamp(min=0)

    # Union
    area_pred = src_segments[:, 1] - src_segments[:, 0]
    area_target = target_segments[:, 1] - target_segments[:, 0]
    union = area_pred + area_target - inter

    # IoU
    iou = inter / union

    # Center distance
    center_pred = (src_segments[:, 0] + src_segments[:, 1]) / 2
    center_target = (target_segments[:, 0] + target_segments[:, 1]) / 2
    center_dist = torch.abs(center_pred - center_target)

    # Max possible segment length
    max_length = torch.max(src_segments[:, 1], target_segments[:, 1]) - torch.min(src_segments[:, 0],
                                                                                  target_segments[:, 0])

    # Normalize the center distance by the max_length
    normalized_dist = center_dist / max_length

    # DIOU
    diou = iou - normalized_dist ** 2
    return diou


def pairwise_segment_diou(src_segments: torch.Tensor, target_segments: torch.Tensor) -> torch.Tensor:
    """
    Calculate the pairwise DIOU between predicted and target segments.
    :param src_segments: Tensor of shape (N, 2), where N is the number of predicted segments.
    :param target_segments: Tensor of shape (M, 2), where M is the number of target segments.
    :return: DIOU: Tensor of shape (N, M), where each value is the DIOU between the corresponding predicted and target segment.
    """
    # Intersection
    l = torch.max(src_segments[:, None, 0], target_segments[:, 0])
    r = torch.min(src_segments[:, None, 1], target_segments[:, 1])
    inter = (r - l).clamp(min=0)

    # Union
    area_pred = (src_segments[:, None, 1] - src_segments[:, None, 0])
    area_target = (target_segments[:, 1] - target_segments[:, 0])
    union = area_pred + area_target - inter

    # IoU
    iou = inter / union

    # Center distance
    center_pred = (src_segments[:, None, 0] + src_segments[:, None, 1]) / 2
    center_target = (target_segments[:, 0] + target_segments[:, 1]) / 2
    center_dist = torch.abs(center_pred - center_target)

    # Max possible segment length
    max_length = torch.max(src_segments[:, None, 1], target_segments[:, 1]) - \
                 torch.min(src_segments[:, None, 0], target_segments[:, 0])

    # Normalize the center distance by the max_length
    normalized_dist = center_dist / max_length

    # DIOU
    diou = iou - normalized_dist ** 2
    return diou


def diou_loss(src_segments: torch.Tensor, target_segments: torch.Tensor) -> torch.Tensor:
    diou_values = segment_diou(src_segments, target_segments)
    loss_values = 1.0 - diou_values
    return loss_values


def log_ratio_width_loss(src_segments: torch.Tensor, target_segments: torch.Tensor, beta: float = 1.0, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculate the smooth L1 log ratio width loss between predicted and target segments.
    """
    width_pred = src_segments[..., 1:]
    width_target = target_segments[..., 1:].log()
    loss = F.smooth_l1_loss(width_pred, width_target, reduction='none')

    return loss


def pairwise_log_ratio_width_loss(src_segments: torch.Tensor, target_segments: torch.Tensor, beta: float = 1.0, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculate the pairwise smooth L1 log ratio width loss between predicted and target segments.
    :param src_segments:
    :param target_segments:
    :param beta:
    :param epsilon:
    :return: a tensor of shape [N, M] where N is the number of source segments and M is the number of target segments.
    """
    width_pred = src_segments[:, None, 1]
    width_target = target_segments[None, :, 1].log()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss = F.smooth_l1_loss(width_pred, width_target, reduction='none')

    return loss
