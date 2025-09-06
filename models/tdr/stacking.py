# encoding: utf-8
import torch

from torch import nn
from models.raven.utils import build_encoder

from .encoder import BinaryClassifier
from .model import TDL
from .utils import MLP, _get_activation_fn
from .transformer import build_deformable_transformer, build_deformable_encoder
from .criterion import SetCriterion
from .matcher import build_matcher


class Stacking(nn.Module):
    def __init__(self, num_reg_head_layers: int = 3, hidden_dim: int = 256, num_cls_head_layers: int = 3,
                 fusion_mode: str = 'concat', count_loss=True, mod_loss=True, o_queries = 40):
        super(Stacking, self).__init__()

        self.fusion_mode = fusion_mode
        self.softplus = nn.Softplus(beta=1.0)

        self.mod_loss = mod_loss
        if self.mod_loss:
            self.av_classifier = BinaryClassifier(hidden_dim, hidden_dim)

        # additional
        # self.proj = nn.Conv1d(o_queries, s_queries, kernel_size=3, padding=1)
        
        self.segments = MLP(hidden_dim, hidden_dim, 2, num_layers=num_reg_head_layers)
        self.classifier = MLP(hidden_dim, hidden_dim, 1, num_layers=num_cls_head_layers)

        self.count_loss = count_loss
        if self.count_loss:
            self.counting = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )

        if self.fusion_mode == 'concat':
            self.f_weight = nn.Linear(2 * hidden_dim, hidden_dim)
            self._init_weights(self.f_weight)
        else:
            self.fusion_act = _get_activation_fn("gelu")
            self.v_weight = nn.Linear(hidden_dim, hidden_dim)
            self.a_weight = nn.Linear(hidden_dim, hidden_dim)
            self._init_weights(self.v_weight)
            self._init_weights(self.a_weight)

        nn.init.zeros_(self.segments.layers[-1].weight)
        nn.init.zeros_(self.segments.layers[-1].bias)
        nn.init.zeros_(self.classifier.layers[-1].weight)
        nn.init.zeros_(self.classifier.layers[-1].bias)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _fusion(self, x1, x2):
        if self.fusion_mode == 'concat':
            x_f = torch.cat([x1, x2], dim=-1)
            # additional
            # temp = x_f
            # temp = self.proj(temp)
            return self.f_weight(x_f)
        elif self.fusion_mode == 'add':
            return x1 + x2
        elif self.fusion_mode == 'star':
            x1 = self.fusion_act(self.v_weight(x1))
            x2 = self.a_weight(x2)
            return x1 * x2

    def _init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, out_v, out_a):
        feature = self._fusion(out_v["feature"], out_a["feature"])

        out_segments = self.segments(feature)
        pos_part = self.softplus(out_segments[..., 0])
        free_part = out_segments[..., 1]
        out_segments = torch.stack([pos_part, free_part], dim=-1)   # [B, num_segs, 2]

        out_class_score = self.classifier(feature)    # [B, num_segs, 1]

        mask = torch.zeros_like(out_class_score.squeeze(-1), device=out_class_score.device, dtype=torch.bool)
        out = {
            "pred_logits": out_class_score, "pred_segments": out_segments, "mask": mask
        }

        if self.mod_loss:
            out["classifier_score"] = self.av_classifier(feature)

        if self.count_loss:
            out['fake_num'] = self.counting(feature.permute(0, -1, 1))  # [B, 1], the model predicts log(λ)

        return out


def build(args, v_backbone_args, a_backbone_args, eval_mode=False):
    v_backbone = build_encoder(v_backbone_args, args.with_regs, args.num_register_tokens)
    a_backbone = build_encoder(a_backbone_args, args.with_regs, args.num_register_tokens)

    if eval_mode is False:
        if args.with_regs:
            ckpt = torch.load(args.backbone_ckpt_path, weights_only=True)
            v_backbone.load_state_dict(ckpt["v_backbone"])
            a_backbone.load_state_dict(ckpt["a_backbone"])
        else:
            ckpt_v = torch.load(args.backbone_v_path, weights_only=True)
            ckpt_a = torch.load(args.backbone_a_path, weights_only=True)

            if 'frontend.frontend3D.0.weight' in ckpt_v:
                del ckpt_v['frontend.frontend3D.0.weight']

            v_backbone.load_state_dict(ckpt_v, strict=False)
            a_backbone.load_state_dict(ckpt_a)

    if eval_mode is False:
        if args.v_backbone_frozen:
            for p in v_backbone.parameters():
                p.requires_grad = False
        if args.a_backbone_frozen:
            for p in a_backbone.parameters():
                p.requires_grad = False

    encoder = build_deformable_encoder(args)
    v_decoder = build_deformable_transformer(args)
    a_decoder = build_deformable_transformer(args)

    model = TDL(
        v_backbone=v_backbone,
        a_backbone=a_backbone,
        encoder=encoder,
        v_decoder=v_decoder,
        a_decoder=a_decoder,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        feature_dim=args.feature_dim,
        num_feature_levels=args.num_feature_levels,
        num_cls_head_layers=args.num_cls_head_layers,
        num_reg_head_layers=args.num_reg_head_layers,
        fusion_mode=args.fusion_mode,
        count_loss=args.tdl_count_loss,
        mod_loss=args.tdl_mod_loss,
        aux_loss=args.aux_loss,
        query_gen=args.query_gen,
        with_regs=args.with_regs,
        share_class_embed=args.share_class_embed,
        share_segment_embed=args.share_segment_embed,
        fix_encoder_proposals=args.fix_encoder_proposals
    )

    stacker = Stacking(
        num_reg_head_layers=args.num_reg_head_layers,
        num_cls_head_layers=args.num_cls_head_layers,
        hidden_dim=args.hidden_dim,
        fusion_mode=args.fusion_mode,
        count_loss=args.stacker_count_loss,
        mod_loss=args.stacker_mod_loss,
    )

    if eval_mode is True:
        return model, stacker

    matcher = build_matcher(args)
    losses = ['labels', 'segments']

    weight_dict = {
        'loss_ce_v': args.cls_loss_coef,
        'loss_segments_v': args.seg_loss_coef,
        'loss_iou_v': args.iou_loss_coef,
        'loss_ce_a': args.cls_loss_coef,
        'loss_segments_a': args.seg_loss_coef,
        'loss_iou_a': args.iou_loss_coef,
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v / 3 for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict.update({
        "loss_mod_v": args.mod_loss_coef,
        "loss_mod_a": args.mod_loss_coef,
        "loss_mod_s": args.mod_loss_coef,
        'loss_segments_s': args.seg_loss_coef,
        'loss_iou_s': args.iou_loss_coef,
        'loss_ce_s': args.cls_loss_coef,
        'loss_pois_v': args.pois_loss_coef,
        'loss_pois_a': args.pois_loss_coef,
        'loss_pois_s': args.pois_loss_coef
    })

    criterion = SetCriterion(matcher, weight_dict, losses,
                             args.focal_alpha, args.diou, args.label_smoothing, args.pois_alpha)
    return model, stacker, criterion
