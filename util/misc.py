# encoding: utf-8
import toml
import yaml
import json
import torch
import random
import torchvision
import numpy as np
import torch.distributed as dist

from torch import Tensor
from typing import Optional, List
from collections import namedtuple


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask
        if mask == 'auto':
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError("tensors dim must be 3 or 4 but {}({})".format(self.tensors.dim(), self.tensors.shape))

    def imgsize(self):
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(self, tensor, mask):
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(tensor.dim())
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self):
        """remove the padding and convert to img list

        Returns:
            [type]: [description]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {
            'tensors.shape': self.tensors.shape,
            'mask.shape': self.mask.shape
        }


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def roundup_to_multiple(number, multiple):
    return ((number + multiple - 1) // multiple) * multiple


def nested_tensor_from_tensor_list(tensor_list: List[Tensor], fill_value=0):
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    elif tensor_list[0].ndim == 2 or tensor_list[0].ndim == 4:
        max_size = max([video_ft.shape[1]
                        for video_ft in tensor_list])  # [c,t,h,w] or [c,t]
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        if tensor_list[0].ndim == 2:
            batch_shape = [len(tensor_list), tensor_list[0].shape[0], max_size]
        else:
            batch_shape = [len(tensor_list), tensor_list[0].shape[0],
                           max_size, tensor_list[0].shape[2], tensor_list[0].shape[3]]
        b, c, t = batch_shape[:3]
        t = roundup_to_multiple(t, 256)
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.full((b, c, t), dtype=dtype, device=device, fill_value=fill_value)
        mask = torch.ones((b, t), dtype=torch.bool, device=device)
        for video_ft, pad_video_ft, m in zip(tensor_list, tensor, mask):
            pad_video_ft[:video_ft.shape[0], :video_ft.shape[1]].copy_(video_ft)
            m[:video_ft.shape[1]] = False
    elif tensor_list[0].ndim == 1:
        max_size = max([grid.shape[0]
                        for grid in tensor_list])

        batch_shape = [len(tensor_list), max_size]
        b, t = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, t), dtype=torch.bool, device=device)
        for grid, pad_grid, m in zip(tensor_list, tensor, mask):
            pad_grid[:grid.shape[0]].copy_(grid)
            m[:grid.shape[0]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def make_namedtuple(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = make_namedtuple(value)
        return namedtuple('GenericDict', data.keys())(*data.values())
    else:
        return data


def load_yaml(config_path: str):
    with open(config_path, 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args = make_namedtuple(args)
    return args


def load_toml(config_path: str):
    with open(config_path, 'r') as file:
        toml_data = toml.load(file)
    args = make_namedtuple(toml_data)
    return args


def read_json(path: str, object_hook=None):
    with open(path, 'r') as f:
        return json.load(f, object_hook=object_hook)


def read_video(path: str):
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(3, 0, 1, 2) / 255
    audio = audio.permute(1, 0)
    return video, audio, info


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """
    Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    iou = inter_len / union_len
    return iou


def add_segment_occlusions(features, segment_length, max_occlusion_ratio=0.3, occlusion_value=0):
    """
    Add occlusions within a segment without covering the whole segment.

    :param features: torch.Tensor, the feature tensor to augment.
    :param segment_length: int, the length of the full segment.
    :param max_occlusion_ratio: float, the maximum ratio of the segment that can be occluded.
    :param occlusion_value: int or float, the value to set for occlusion.
    :return: torch.Tensor, the augmented feature tensor with occlusion.
    """
    # occlusions_to_add = random.randint(1, 3)  # Decide on a number of occlusions to add
    max_occlusion_length = int(segment_length * max_occlusion_ratio)
    if max_occlusion_length > 0:
        num_occlusion = random.randint(1, 3)
        for i in range(num_occlusion):
            occlusion_length = random.randint(1, max_occlusion_length)
            start = random.randint(0, segment_length - occlusion_length)
            features[start:start+occlusion_length] = occlusion_value

    return features


def add_background_occlusions(features, segments, max_occlusion_ratio=0.1, occlusion_value=0):
    """
    Add occlusions to the background (non-segment) areas of the features.

    :param features: torch.Tensor, the feature tensor to augment.
    :param segments: torch.Tensor, the segments within the features.
    :param max_occlusion_ratio: float, the maximum ratio of each background area that can be occluded.
    :param occlusion_value: int or float, the value to set for occlusion.
    :return: torch.Tensor, the augmented feature tensor with background occlusions.
    """
    total_length = features.size(1)

    # Create a mask with the same length as the features, initialized to False
    background_mask = torch.ones(total_length, dtype=torch.bool)

    # Mask out the segment areas (these are not background)
    for segment in segments:
        segment_start, segment_end = int(segment[0].item()), int(segment[1].item())
        background_mask[segment_start:segment_end] = False

    # Find indices where the background is True
    background_indices = torch.nonzero(background_mask).squeeze()

    if len(background_indices) > 0:
        # Randomly determine the number of occlusions based on the occlusion ratio
        num_occlusions = int(len(background_indices) * max_occlusion_ratio)

        # Select random indices from the background to occlude
        occlusion_indices = background_indices[random.sample(range(len(background_indices)), num_occlusions)]

        # Apply the occlusion to the selected indices
        features[:, occlusion_indices] = occlusion_value

    return features

def merge_periods(visual_fake_periods, audio_fake_periods):
    if len(visual_fake_periods) > 0:
        v_min, v_max = visual_fake_periods[:, 0], visual_fake_periods[:, 1]
    else:
        v_min, v_max = torch.tensor([]), torch.tensor([])

    if len(audio_fake_periods) > 0:
        a_min, a_max = audio_fake_periods[:, 0], audio_fake_periods[:, 1]
    else:
        a_min, a_max = torch.tensor([]), torch.tensor([])

    merged_tensor = torch.cat((v_min, v_max, a_min, a_max))
    sorted_tensor = torch.unique(merged_tensor)

    v_min, v_max, a_min, a_max = map(lambda x: set(x.tolist()), [v_min, v_max, a_min, a_max])

    labels, segments = [], []
    if len(sorted_tensor) < 2:
        return torch.tensor(labels), torch.tensor(segments)

    pointer = sorted_tensor[0].item()
    if pointer in v_min and pointer in a_min:
        label = 2
    elif pointer in v_min:
        label = 0
    elif pointer in a_min:
        label = 1
    else:
        label = -1

    for element in sorted_tensor[1:]:
        element = element.item()
        if label != -1:
            labels.append(label)
            segments.append([pointer, element])

        if label == 0:
            label = -1 if element in v_max else 2
        elif label == 1:
            label = -1 if element in a_max else 2
        elif label == 2:
            if element in v_max and element in a_max:
                label = -1
            elif element in v_max:
                label = 1
            elif element in a_max:
                label = 0
        else:
            if element in a_min and element in v_min:
                label = 2
            elif element in v_min:
                label = 0
            elif element in a_min:
                label = 1

        pointer = element

    return torch.tensor(labels), torch.tensor(segments)


if __name__ == '__main__':

    visual_fake_periods = torch.tensor([[0.2000, 0.3900], [5.8500, 6.2300]])

    audio_fake_periods = torch.tensor([[0.2000, 0.3900], [5.8500, 6.2300]])

    labels, segments = merge_periods(visual_fake_periods, audio_fake_periods)
    print(labels)
    print(segments)
