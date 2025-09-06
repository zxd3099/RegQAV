# encoding: utf-8
import os
import torch

from util.misc import read_json, NestedTensor
from .av_deepfake1m import AVDeepfake1M
from .lav_df import LAVDF
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from typing import Optional


class AVDeepfake1mDataModule(LightningDataModule):
    def __init__(self, root: str = "Datasets/AV-Deepfake1M", temporal_size: int = 100,
                 max_duration: int = 30, fps: int = 25, batch_size: int = 2, num_workers: int = 0,
                 sampling_rate: int = 16_000, normalized: bool = False, with_regs: bool = True,
                 take_train: int = None, take_val: int = None, take_test: int = None):
        super().__init__()
        self.root = root
        self.temporal_size = temporal_size
        self.max_duration = max_duration
        self.fps = fps
        self.sampling_rate = sampling_rate
        self.normalized = normalized
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.take_train = take_train
        self.take_val = take_val
        self.take_test = take_test
        self.with_regs = with_regs
        self.Dataset = AVDeepfake1M

    def setup(self, stage: Optional[str] = None) -> None:
        train_file_list = [meta["file"] for meta in read_json(os.path.join(self.root, "train_metadata.json"))]
        val_file_list = [meta["file"] for meta in read_json(os.path.join(self.root, "val_metadata.json"))]
        with open(os.path.join(self.root, "test_files.txt"), "r") as f:
            test_file_list = list(filter(lambda x: x != "", f.read().split("\n")))

        # take subset of data if specified
        if self.take_train is not None:
            train_file_list = train_file_list[:self.take_train]
        if self.take_val is not None:
            val_file_list = val_file_list[:self.take_val]
        if self.take_test is not None:
            test_file_list = test_file_list[:self.take_test]

        self.train_dataset = self.Dataset("train", self.root, self.temporal_size, self.max_duration,
                                          self.fps, self.sampling_rate, self.normalized, file_list=train_file_list, with_regs=self.with_regs)
        self.val_dataset = self.Dataset("val", self.root, self.temporal_size, self.max_duration,
                                          self.fps, self.sampling_rate, self.normalized, file_list=val_file_list, with_regs=self.with_regs)
        self.test_dataset = self.Dataset("test", self.root, self.temporal_size, self.max_duration,
                                          self.fps, self.sampling_rate, self.normalized, file_list=test_file_list, with_regs=self.with_regs)

    def collate_fn(self, batch):
        video = torch.stack([item[0] for item in batch], dim=0)
        audio = torch.stack([item[1] for item in batch], dim=0)
        mask = torch.stack([item[2] for item in batch], dim=0)
        info = [item[3] for item in batch]

        return NestedTensor(video, mask), NestedTensor(audio, mask), info

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          drop_last=True, collate_fn=self.collate_fn,
                          sampler=RandomSampler(self.train_dataset, num_samples=self.take_train, replacement=True))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          shuffle=False, drop_last=False, collate_fn=self.collate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          drop_last=False, collate_fn=self.collate_fn)


class LAVDFDataModule(LightningDataModule):
    def __init__(self, root: str = "Datasets/LAV-DF", temporal_size: int = 100,
                 max_duration: int = 40, fps: int = 25, batch_size: int = 2, num_workers: int = 0,
                 sampling_rate: int = 16_000, normalized: bool = False, with_regs: bool = True,
                 robust_type=None, robust_scale=None):
        super().__init__()
        self.root = root
        self.temporal_size = temporal_size
        self.max_duration = max_duration
        self.fps = fps
        self.sampling_rate = sampling_rate
        self.normalized = normalized
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.with_regs = with_regs
        self.Dataset = LAVDF
        self.robust_type = robust_type
        self.robust_scale = robust_scale

    def setup(self, stage: Optional[str] = None) -> None:

        self.train_dataset = self.Dataset(["train"], self.root, self.temporal_size, self.max_duration,
                                          self.fps, self.sampling_rate, self.normalized, with_regs=self.with_regs)
        self.val_dataset = self.Dataset(["test"], self.root, self.temporal_size, self.max_duration,
                                          self.fps, self.sampling_rate, self.normalized, with_regs=self.with_regs)
        self.test_dataset = self.Dataset(["test"], self.root, self.temporal_size, self.max_duration,
                                          self.fps, self.sampling_rate, self.normalized, with_regs=self.with_regs,
                                          robust_type=self.robust_type, robust_scale=self.robust_scale)

    def collate_fn(self, batch):
        video = torch.stack([item[0] for item in batch], dim=0)
        audio = torch.stack([item[1] for item in batch], dim=0)
        mask = torch.stack([item[2] for item in batch], dim=0)
        info = [item[3] for item in batch]

        return NestedTensor(video, mask), NestedTensor(audio, mask), info

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          drop_last=True, collate_fn=self.collate_fn, sampler=RandomSampler(self.train_dataset, replacement=True))

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          shuffle=False, drop_last=False, collate_fn=self.collate_fn)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                          drop_last=False, collate_fn=self.collate_fn)


if __name__ == "__main__":
    lavdf = LAVDFDataModule()
    lavdf.setup()

    dataloader = lavdf.train_dataloader()
    video_samples, audio_samples, info = next(iter(dataloader))
    print(video_samples.shape)   # {'tensors.shape': torch.Size([4, 3, 200, 224, 224]), 'mask.shape': torch.Size([4, 205])}
    print(audio_samples.shape)   # {'tensors.shape': torch.Size([4, 128000, 1]), 'mask.shape': torch.Size([4, 205])}
    print(info)
