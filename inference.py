# encoding: utf-8
"""
@File   : inference.py
@Time   : 2025/1/11 9:51
@Author : zxd3099
"""
import torch
import argparse
import os.path
import json
import pytorch_lightning as pl

from models.tdr.stacking import build
from util.misc import load_yaml
from typing import Any
from pytorch_lightning import LightningModule, Trainer, Callback
from util.segment_ops import segment_cw_to_t1t2
from data.data_interface import AVDeepfake1mDataModule, LAVDFDataModule


class SaveToCsvCallback(Callback):

    def __init__(self, res_dir: str, filename: str):
        super().__init__()
        self.res_dir = res_dir
        self.filename = filename

    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
                            batch: Any, batch_idx: int, dataloader_idx: int = 0):
        b = outputs["pred_logits"].size(0)
        classifier_score = outputs.get("classifier_score", None)
        fake_num = torch.full((b, 1), 40, dtype=torch.int)
        if classifier_score is not None:
            classifier_score = classifier_score.sigmoid()

        pred_logits, out_segments = outputs["pred_logits"].sigmoid(), outputs["pred_segments"]
        out_segments = torch.stack([out_segments[..., 0], out_segments[..., 1].exp()], dim=-1)
        out_segments = segment_cw_to_t1t2(out_segments)
        out_segments = torch.abs(out_segments)

        _, _, info = batch
        video_names = [v["video_name"] for v in info]
        output_path = os.path.join(self.res_dir, self.filename)
        self.postprocess(classifier_score, pred_logits, out_segments, fake_num, video_names, output_path)

    @staticmethod
    def postprocess(classifier_score, prob, segments, fake_num, video_names, output_path):
        confs, _ = prob.max(dim=-1)  # [B, N_q]
        sorted_indices = torch.argsort(confs, dim=-1, descending=True)  # [B, N_q]

        batch_indices = torch.arange(segments.shape[0]).unsqueeze(1)
        top_k_indices = sorted_indices[:, :fake_num.max()]  # [B, max_fake_num]

        top_conf = confs[batch_indices, top_k_indices]  # [B, max_fake_num]
        top_segs = segments[batch_indices, top_k_indices]  # [B, max_fake_num, 2]

        top_conf = torch.round(top_conf * 10000) / 10000
        top_segs = torch.round(top_segs / 0.02) * 0.02
        top_segs = torch.round(top_segs * 1000) / 1000

        if classifier_score is not None:
            classifier_score = classifier_score.squeeze(-1)
            for i in range(top_conf.size(0)):
                if classifier_score[i] >= 0.7:
                    top_conf[i] = torch.where((top_conf[i] < classifier_score[i]) & (top_conf[i] > 0.5),
                                              torch.sqrt(classifier_score[i] * top_conf[i]),
                                              top_conf[i])

        top_conf_list = top_conf.tolist()
        top_segs_list = top_segs.tolist()

        top_conf_list = [[round(conf, 4) for conf in video_conf] for video_conf in top_conf_list]
        top_segs_list = [[[round(seg, 3) for seg in period] for period in video_segs] for video_segs in
                         top_segs_list]

        top_results = [
            [[top_conf_list[i][j], *top_segs_list[i][j]] for j in range(top_conf.size(-1))]
            for i in range(top_conf.size(0))
        ]

        fake_num = fake_num.squeeze(-1).tolist()
        results = {video_names[i]: top_results[i][:fake_num[i]] for i in range(len(fake_num))}

        rank = int(os.environ.get('LOCAL_RANK', 0))
        temp_output_path = f"{output_path}_rank{rank}.json"

        json_data = json.dumps(results)

        # Check if the file exists and is not empty
        if not os.path.exists(temp_output_path) or os.stat(temp_output_path).st_size == 0:
            with open(temp_output_path, 'w') as f:
                # Start a new JSON object with the first set of results
                f.write(f"{{{json_data}}}\n")
        else:
            with open(temp_output_path, 'r+') as f:
                # Read the existing content (the JSON object)
                content = f.read().strip()

                # Remove the trailing '}' to avoid breaking the JSON format
                if content.endswith('}'):
                    content = content[:-1]

                # Add the new results and close the JSON object properly
                content += f",\n{json_data}\n}}"

                # Go back to the beginning of the file and overwrite it
                f.seek(0)
                f.write(content)
                f.truncate()


class Raven_Stacker_Tester(pl.LightningModule):
    def __init__(self, args, v_backbone_args, a_backbone_args):
        super().__init__()
        self.save_hyperparameters()

        model, stacker = build(args, v_backbone_args, a_backbone_args, eval_mode=True)
        self.model = model
        self.stacker = stacker

    def forward(self, video_samples, audio_samples, info):
        out_v, out_a = self.model(video_samples, audio_samples, info)
        out_s = self.stacker(out_v, out_a)
        return out_v, out_a, out_s

    def predict_step(self, batch, batch_idx):
        video_samples, audio_samples, info = batch
        out_v, out_a, out_s = self.forward(video_samples, audio_samples, info)
        return out_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RegQAV Inference")
    parser.add_argument("--model", type=str, default="config/model/stacking.yaml")
    parser.add_argument("--v_backbone", type=str, default="config/model/raven/visual_backbone.yaml")
    parser.add_argument("--a_backbone", type=str, default="config/model/raven/audio_backbone.yaml")
    parser.add_argument("--dataset", type=str, default="config/dataset/lavdf.yaml")
    parser.add_argument("--num_train", type=int, default=None)
    parser.add_argument("--num_val", type=int, default=None)
    parser.add_argument("--num_test", type=int, default=None)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    model_args = load_yaml(args.model)
    dataset_args = load_yaml(args.dataset)
    v_backbone_args = load_yaml(args.v_backbone)
    a_backbone_args = load_yaml(args.a_backbone)

    model = Raven_Stacker_Tester.load_from_checkpoint(
        checkpoint_path=model_args.tdr_ckpt_path,
        args=model_args,
        v_backbone_args=v_backbone_args,
        a_backbone_args=a_backbone_args,
    )
    model.eval()

    # AV-Deepfake1M
    dm = AVDeepfake1mDataModule(
        root=dataset_args.data_root,
        temporal_size=dataset_args.temporal_size,
        max_duration=dataset_args.max_duration,
        fps=dataset_args.fps,
        batch_size=model_args.batch_size,
        num_workers=model_args.num_workers,
        sampling_rate=dataset_args.sampling_rate,
        normalized=dataset_args.normalized,
        take_train=args.num_train,
        take_val=args.num_val,
        take_test=args.num_test,
        with_regs=model_args.with_regs
    )
    dm.setup()

    # LAVDF
    # dm = LAVDFDataModule(
    #     root=dataset_args.data_root,
    #     temporal_size=dataset_args.temporal_size,
    #     max_duration=dataset_args.max_duration,
    #     fps=dataset_args.fps,
    #     batch_size=model_args.batch_size,
    #     num_workers=model_args.num_workers,
    #     sampling_rate=dataset_args.sampling_rate,
    #     normalized=dataset_args.normalized,
    #     with_regs=model_args.with_regs
    # )
    # dm.setup()

    gpus = 4

    trainer = Trainer(
        logger=False,
        accelerator="auto" if gpus > 0 else "cpu",
        devices=[0, 1, 2, 3],
        callbacks=[SaveToCsvCallback(res_dir="", filename=f"result")]
    )

    trainer.predict(model, dm.test_dataloader())
