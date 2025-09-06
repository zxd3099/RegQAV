# encoding: utf-8
import os
import torch
import numpy as np

from torch import Tensor
from tqdm.auto import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch.nn import functional as F, Identity
from util.misc import (
    read_video, read_json, iou_with_anchors,
    merge_periods
)
from typing import Optional, List, Callable


@dataclass
class Metadata:
    file: str
    original: Optional[str]
    split: str
    n_fakes: int
    fps: int
    duration: float
    fake_periods: List[List[float]]
    visual_fake_periods: List[List[float]]
    audio_fake_periods: List[List[float]]
    modified_either: bool  # "real", "both_modified", "audio_modified", "visual_modified"
    modify_video: bool
    modify_audio: bool
    audio_model: str
    video_frames: int
    audio_frames: int

    def __init__(self, file: str, original: Optional[str], split: str, fake_segments: List[List[float]], fps: int,
                 visual_fake_segments: List[List[float]], audio_fake_segments: List[List[float]],
                 audio_model: Optional[str], modify_type: str, video_frames: int, audio_frames: int, *args, **kwargs
                 ):
        self.file = file
        self.original = original
        self.split = split
        self.n_fakes = len(fake_segments)
        self.fps = fps
        self.duration = video_frames / fps
        self.fake_periods = fake_segments
        self.visual_fake_periods = visual_fake_segments
        self.audio_fake_periods = audio_fake_segments
        self.modified_either = modify_type in ("both_modified", "visual_modified", "audio_modified")
        self.modify_video = modify_type in ("both_modified", "visual_modified")
        self.modify_audio = modify_type in ("both_modified", "audio_modified")
        self.audio_model = audio_model
        self.video_frames = video_frames
        self.audio_frames = audio_frames


class AVDeepfake1M(Dataset):
    def __init__(self, subset: str, data_root: str = "Datasets/AV-Deepfake1M/", temporal_size: int = 100,
                 max_duration: int = 30, fps: int = 25, sampling_rate: int = 16_000, normalized: bool = False,
                 video_transform: Callable[[Tensor], Tensor] = Identity(),
                 audio_transform: Callable[[Tensor], Tensor] = Identity(),
                 file_list: Optional[List[str]] = None,
                 with_regs=True
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

        label_dir = os.path.join(self.root, "label")
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        if file_list is None:
            self.file_list = [meta["file"] for meta in read_json(os.path.join(self.root, f"{subset}_metadata.json"))]
        else:
            self.file_list = file_list

        print(f"Load {len(self.file_list)} data in {subset}.")

    def __getitem__(self, index: int):
        file = self.file_list[index]

        video, audio, _ = read_video(os.path.join(self.root, self.subset, file))  # [C, Tv, H, W]; [Ta, 1]
        if self.with_regs:
            mask = torch.zeros([self.temporal_size + 5], dtype=torch.bool)
        else:
            mask = torch.zeros([self.temporal_size], dtype=torch.bool)

        if self.subset != "test":
            meta = read_json(os.path.join(self.root, self.subset + "_metadata", file.replace(".mp4", ".json")))
            meta = Metadata(**meta, fps=self.fps)
            targets = self.get_label(file, meta)
            targets["labels"] = targets["labels"].long()

        if video.size(1) >= self.temporal_size:
            stride = video.size(1) / self.temporal_size
            video = F.interpolate(video[None], size=(self.temporal_size, 224, 224))[0]  # [C, 200, H, W]
            audio = F.interpolate(audio.permute(1, 0)[None], size=self.audio_temporal_size, mode="linear")[0].permute(1,
                                                                                                                      0)  # [128000, 1]
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
        video = self.video_transform(video)
        audio = self.audio_transform(audio)

        info = {
            "video_name": file,
            "fps": torch.tensor([self.fps]),
            "stride": torch.tensor([stride]),
            "feature_duration": torch.tensor([feature_duration])
        }
        if self.subset != "test":
            targets["merged_segments"] = targets["merged_segments"] * info["feature_duration"]
            targets["v_segments"] = targets["v_segments"] * info["feature_duration"]
            targets["a_segments"] = targets["a_segments"] * info["feature_duration"]
            targets["segments"] = targets["segments"] * info["feature_duration"]
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
        visual_bbox = self._get_train_label(meta.video_frames, meta.visual_fake_periods)
        audio_bbox = self._get_train_label(meta.video_frames, meta.audio_fake_periods)

        if labels["modified"] == 0:
            class_labels, segments = torch.tensor([]), torch.tensor([])
        elif labels["v_modified"] == 0:
            class_labels, segments = torch.ones(audio_bbox.size(0)), audio_bbox
        elif labels["a_modified"] == 0:
            class_labels, segments = torch.zeros(visual_bbox.size(0)), visual_bbox
        else:
            class_labels, segments = merge_periods(visual_bbox, audio_bbox)

        labels["merged_segments"] = t_bbox.numpy()
        labels["v_segments"] = visual_bbox.numpy()
        labels["a_segments"] = audio_bbox.numpy()
        labels["labels"] = class_labels.numpy()
        labels["segments"] = segments.numpy()

        np.savez(path, **labels)

        return {
            "modified": labels["modified"], "v_modified": labels["v_modified"], "a_modified": labels["a_modified"],
            "merged_segments": torch.from_numpy(labels["merged_segments"]) if type(labels["merged_segments"]) == np.ndarray else None,
            "v_segments": torch.from_numpy(labels["v_segments"]) if type(labels["v_segments"]) == np.ndarray else None,
            "a_segments": torch.from_numpy(labels["a_segments"]) if type(labels["a_segments"]) == np.ndarray else None,
            "labels": torch.from_numpy(labels["labels"]) if type(labels["labels"]) == np.ndarray else None,
            "segments": torch.from_numpy(labels["segments"]) if type(labels["segments"]) == np.ndarray else None,
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
        # manually pre-generate label as npy
        for file in tqdm(self.file_list):
            meta = read_json(os.path.join(self.root, self.subset + "_metadata", file.replace(".mp4", ".json")))
            meta = Metadata(**meta, fps=self.fps)
            self.get_label(file, meta)

    def __len__(self) -> int:
        return len(self.file_list)


if __name__ == "__main__":
    av1m = AVDeepfake1M(subset="train")
    av1m.gen_label()