"""
@File   : lav_df.py
@Time   : 2025/1/9 9:16
@Author : zxd3099
"""
import os
import torch
import torch.utils.data
import numpy as np

from torch import Tensor
from tqdm.auto import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.nn import functional as F, Identity
from util.misc import read_video, read_json, iou_with_anchors
from typing import Optional, List, Callable


@dataclass
class Metadata:
    file: str
    split: str
    original: str
    n_fakes: int
    fps: int
    duration: float
    fake_periods: List[List[float]]
    modified_either: bool  # "real", "both_modified", "audio_modified", "visual_modified"
    modify_video: bool
    modify_audio: bool
    video_frames: int
    audio_frames: int

    def __init__(self, file: str, original: Optional[str], split: str, n_fakes: int, fake_periods: List[List[float]],
                 modify_video: bool, modify_audio: bool, video_frames: int, audio_frames: int, fps: int=25, *args, **kwargs
                 ):
        self.file = file
        self.original = original
        self.split = split
        self.n_fakes = n_fakes
        self.fps = fps
        self.duration = video_frames / fps
        self.fake_periods = fake_periods
        self.modified_either = modify_video or modify_audio
        self.modify_video = modify_video
        self.modify_audio = modify_audio
        self.video_frames = video_frames
        self.audio_frames = audio_frames


class LAVDF(Dataset):
    def __init__(self, subset: List[str], data_root: str = "Datasets/LAV-DF", temporal_size: int = 100,
                 max_duration: int = 40, fps: int = 25, sampling_rate: int = 16_000, normalized: bool = False,
                 video_transform: Callable[[Tensor], Tensor] = Identity(),
                 audio_transform: Callable[[Tensor], Tensor] = Identity(),
                 with_regs=True, robust_type=None, robust_scale=None
                 ):
        super().__init__()
        self.subset = subset
        self.root = data_root
        self.fps = fps
        self.normalized = normalized
        self.sampling_rate = sampling_rate
        self.temporal_size = temporal_size
        self.audio_temporal_size = int(temporal_size / fps * sampling_rate)
        self.max_duration = max_duration
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.with_regs = with_regs
        self.robust_type = robust_type
        self.robust_scale = robust_scale

        metadata = read_json(os.path.join(self.root, "metadata.min.json"), lambda d: Metadata(**d))
        self.meta_list = []

        for meta in metadata:
            if meta.split in subset:
                self.meta_list.append(meta)

        print(f"Load {len(self.meta_list)} data in {subset}.")

    def __getitem__(self, index: int):
        meta = self.meta_list[index]

        file = meta.file
        video, audio, _ = read_video(os.path.join(self.root, file))  # [C, Tv, H, W]; [Ta, 1]
        if self.with_regs:
            mask = torch.zeros([self.temporal_size + 5], dtype=torch.bool)
        else:
            mask = torch.zeros([self.temporal_size], dtype=torch.bool)

        targets = self.get_label(file, meta)
        targets["labels"] = targets["labels"].long()

        if video.size(1) >= self.temporal_size:
            stride = video.size(1) / self.temporal_size
            video = F.interpolate(video[None], size=(self.temporal_size, 224, 224))[0]  # [C, 200, H, W]
            audio = (F.interpolate(audio.permute(1, 0)[None], size=self.audio_temporal_size, mode="linear")[0]
                      .permute(1, 0))  # [128000, 1]
        else:
            stride = 1
            v_pad_left = (self.temporal_size - video.size(1)) // 2
            v_pad_right = self.temporal_size - video.size(1) - v_pad_left
            a_pad_left = (self.audio_temporal_size - audio.size(0)) // 2
            a_pad_right = self.audio_temporal_size - audio.size(0) - a_pad_left
            video = F.pad(video, (0, 0, 0, 0, v_pad_left, v_pad_right))  # [C, 200, H, W]
            audio = F.pad(audio, (0, 0, a_pad_left, a_pad_right))  # [128000, 1]

            if self.with_regs:
                mask[1:(v_pad_left + 1)] = 1
                mask[(-v_pad_right - 4):-4] = 1
            else:
                mask[:v_pad_left] = 1
                mask[-v_pad_right:] = 1

        feature_duration = video.size(1) * stride / self.fps

        info = {
            "video_name": meta.file,
            "fps": torch.tensor([self.fps]),
            "stride": torch.tensor([stride]),
            "feature_duration": torch.tensor([feature_duration])
        }
        info.update(targets)

        return video, audio, mask, info

    def get_label(self, file: str, meta: Metadata):
        file_name = file.replace("/", "_").split(".")[0] + ".npz"
        path = os.path.join(self.root, "labels", file_name)
        if os.path.exists(path):
            try:
                npz = np.load(path)
            except ValueError:
                pass
            except EOFError:
                pass
            else:
                return {
                    "modified": npz["modified"], "v_modified": npz["v_modified"], "a_modified": npz["a_modified"],
                    "merged_segments": torch.from_numpy(npz["merged_segments"]) if npz["merged_segments"].shape != () else None,
                    "v_segments": torch.from_numpy(npz["v_segments"]) if npz["v_segments"].shape != () else None,
                    "a_segments": torch.from_numpy(npz["a_segments"]) if npz["a_segments"].shape != () else None,
                    "labels": torch.from_numpy(npz["labels"]) if npz["labels"].shape != () else None,
                    "segments": torch.from_numpy(npz["segments"]) if npz["segments"].shape != () else None,
                }

        labels = {
            "modified": int(meta.modified_either),
            "v_modified": int(meta.modify_video),
            "a_modified": int(meta.modify_audio),
        }

        t_bbox = self._get_train_label(meta.video_frames, meta.fake_periods, iou=False)
        visual_bbox = t_bbox if labels["v_modified"] else torch.tensor([])
        audio_bbox = t_bbox if labels["a_modified"] else torch.tensor([])

        if labels["modified"] == 0:
            class_labels, segments = torch.tensor([]), torch.tensor([])
        elif labels["v_modified"] == 0:
            class_labels, segments = torch.ones(audio_bbox.size(0)), audio_bbox
        elif labels["a_modified"] == 0:
            class_labels, segments = torch.zeros(visual_bbox.size(0)), visual_bbox
        else:
            class_labels, segments = torch.full((t_bbox.size(0),), 2), t_bbox

        labels["merged_segments"] = t_bbox
        labels["v_segments"] = visual_bbox
        labels["a_segments"] = audio_bbox
        labels["labels"] = class_labels
        labels["segments"] = segments

        return {
            "modified": labels["modified"], 
            "v_modified": labels["v_modified"], 
            "a_modified": labels["a_modified"],
            "merged_segments": labels["merged_segments"],
            "v_segments": labels["v_segments"],
            "a_segments": labels["a_segments"],
            "labels": labels["labels"],
            "segments": labels["segments"]
        }

    def _get_train_label(self, frames, video_labels, iou=False):
        corrected_second = frames / self.fps
        temporal_scale = self.temporal_size
        temporal_gap = 1 / self.temporal_size

        # change the measurement from second to percentage
        gt_bbox = []
        if self.normalized:
            for j in range(len(video_labels)):
                tmp_start = max(min(1, video_labels[j][0] / corrected_second), 0)
                tmp_end = max(min(1, video_labels[j][1] / corrected_second), 0)
                gt_bbox.append([tmp_start, tmp_end])
        else:
            gt_bbox = video_labels

        # generate R_s and R_e
        gt_bbox = torch.tensor(gt_bbox)
        if iou is False:
            return gt_bbox

        if len(gt_bbox) > 0:
            gt_xmins, gt_xmaxs = gt_bbox[:, 0], gt_bbox[:, 1]
        else:
            gt_xmins, gt_xmaxs = np.array([]), np.array([])

        gt_iou_map = torch.zeros([self.max_duration, temporal_scale])
        if len(gt_bbox) > 0:
            for begin in range(temporal_scale):
                for duration in range(self.max_duration):
                    end = begin + duration
                    if end > temporal_scale:
                        break
                    gt_iou_map[duration, begin] = torch.max(
                        iou_with_anchors(begin * temporal_gap, (end + 1) * temporal_gap, gt_xmins, gt_xmaxs))

        return gt_bbox, gt_iou_map

    def gen_label(self) -> None:
        label_dir = os.path.join(self.root, "labels")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # manually pre-generate label as npy
        for meta in tqdm(self.meta_list):
            self.get_label(meta.file, meta)

    def __len__(self) -> int:
        return len(self.meta_list)


if __name__ == "__main__":
    lavdf = LAVDF(subset="train")
    lavdf.gen_label()
