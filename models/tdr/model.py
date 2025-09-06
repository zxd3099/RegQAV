import copy
import math
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.init import normal_
from .criterion import SetCriterion
from models.raven.utils import build_encoder
from .transformer import build_deformable_transformer, build_deformable_encoder
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from .encoder import BinaryClassifier
from .matcher import build_matcher
from .position_encoding import TimeBasedPositionEmbeddingSine
from .utils import MLP, get_feature_grids, _get_activation_fn, MultiScaleAdapter

# from encoder import BinaryClassifier
# from utils import MLP, get_feature_grids, _get_activation_fn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == 'gn':
        assert num_groups is not None, 'num_groups must be specified'
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class TDL(nn.Module):
    def __init__(self, v_backbone, a_backbone, v_decoder, a_decoder, encoder, num_classes, num_queries, feature_dim,
                 num_feature_levels, num_cls_head_layers=3, num_reg_head_layers=3, fusion_mode='concat', count_loss=True,
                 aux_loss=True, share_class_embed=False, share_segment_embed=False, fix_encoder_proposals=True,
                 query_gen=True, with_regs=True, mod_loss=True):
        super().__init__()
        self.num_queries = num_queries
        self.v_decoder = v_decoder
        self.a_decoder = a_decoder
        self.encoder = encoder
        self.v_backbone = v_backbone
        self.a_backbone = a_backbone
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = v_decoder.d_model
        self.num_feature_levels = num_feature_levels
        self.fix_encoder_proposals = fix_encoder_proposals
        self.num_cls_head_layers = num_cls_head_layers
        self.num_reg_head_layers = num_reg_head_layers
        self.aux_loss = aux_loss
        self.query_gen = query_gen
        self.with_regs = with_regs

        self.fusion_mode = fusion_mode
        self.multi_scale_adapter = MultiScaleAdapter(feature_dim, self.hidden_dim, fusion_mode, num_feature_levels)
        self.position_embedding = TimeBasedPositionEmbeddingSine(self.hidden_dim, v_decoder.temperature)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, self.hidden_dim))
        normal_(self.level_embed)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if num_cls_head_layers > 1:
            self.class_embed_v = MLP(hidden_dim, hidden_dim, num_classes, num_layers=num_cls_head_layers)
            self.class_embed_a = MLP(hidden_dim, hidden_dim, num_classes, num_layers=num_cls_head_layers)
            nn.init.constant_(self.class_embed_v.layers[-1].bias, bias_value)
            nn.init.constant_(self.class_embed_a.layers[-1].bias, bias_value)
        else:
            self.class_embed_v = nn.Linear(hidden_dim, num_classes)
            self.class_embed_a = nn.Linear(hidden_dim, num_classes)
            nn.init.constant_(self.class_embed_v.bias, bias_value)
            nn.init.constant_(self.class_embed_a.bias, bias_value)

        if share_class_embed:
            self.class_embed_v = nn.ModuleList([self.class_embed_v for _ in range(self.v_decoder.num_layers)])
            self.class_embed_a = nn.ModuleList([self.class_embed_a for _ in range(self.a_decoder.num_layers)])
        else:
            self.class_embed_v = _get_clones(self.class_embed_v, self.v_decoder.num_layers)
            self.class_embed_a = _get_clones(self.class_embed_a, self.a_decoder.num_layers)

        self.v_decoder.decoder.class_embed = self.class_embed_v
        self.a_decoder.decoder.class_embed = self.class_embed_a

        if num_reg_head_layers > 1:
            self.segment_embed_v = MLP(hidden_dim, hidden_dim, 2, num_layers=num_reg_head_layers)
            self.segment_embed_a = MLP(hidden_dim, hidden_dim, 2, num_layers=num_reg_head_layers)
            nn.init.zeros_(self.segment_embed_v.layers[-1].bias)
            nn.init.zeros_(self.segment_embed_a.layers[-1].bias)
        else:
            self.segment_embed_v = nn.Linear(hidden_dim, 2)
            self.segment_embed_a = nn.Linear(hidden_dim, 2)
            nn.init.zeros_(self.segment_embed_v.bias)
            nn.init.zeros_(self.segment_embed_a.bias)

        if share_segment_embed:
            self.segment_embed_v = nn.ModuleList([self.segment_embed_v for _ in range(self.v_decoder.num_layers)])
            self.segment_embed_a = nn.ModuleList([self.segment_embed_a for _ in range(self.a_decoder.num_layers)])
            if not self.fix_encoder_proposals:
                enc_embed_v = MLP(hidden_dim, hidden_dim, 2, num_layers=3)
                enc_embed_a = MLP(hidden_dim, hidden_dim, 2, num_layers=3)
                nn.init.zeros_(enc_embed_v.layers[-1].bias)
                nn.init.zeros_(enc_embed_a.layers[-1].bias)
                self.segment_embed_v.append(enc_embed_v)
                self.segment_embed_a.append(enc_embed_a)
        else:
            if not self.fix_encoder_proposals:
                num_layers = self.v_decoder.num_layers + 1
            else:
                num_layers = self.v_decoder.num_layers
            self.segment_embed_v = _get_clones(self.segment_embed_v, num_layers)
            self.segment_embed_a = _get_clones(self.segment_embed_a, num_layers)
        self.v_decoder.decoder.segment_embed = self.segment_embed_v
        self.a_decoder.decoder.segment_embed = self.segment_embed_a

        self.mod_loss = mod_loss
        if self.mod_loss:
            self.v_classifier = BinaryClassifier(hidden_dim, hidden_dim)
            self.a_classifier = BinaryClassifier(hidden_dim, hidden_dim)

        self.count_loss = count_loss
        if self.count_loss:
            self.counting_v = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )

            self.counting_a = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 1)
            )

    def _init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def get_valid_ratio(self, mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio  # shape=(bs)

    def forward(self, video_samples: NestedTensor, audio_samples: NestedTensor, info):
        """
        :param video_samples:  which consists of:
                         - samples.tensors: batched images, of shape [batch_size x 3 x Tv x H x W]
                         - samples.mask: a binary mask of shape [batch_size x Tv + 5], containing 1 on padded pixels
                         or a tuple of tensors and mask
        :param audio_samples:  which consists of:
                         - samples.tensors: batched images, of shape [batch_size x a_points x 1]
                         - samples.mask: a binary mask of shape [batch_size x Tv + 5], containing 1 on padded pixels
                         or a tuple of tensors and mask
        :param info:
        :return: It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(video_samples, NestedTensor):
            if isinstance(video_samples, (list, tuple)):
                video_samples = NestedTensor(*video_samples)
            else:
                video_samples = nested_tensor_from_tensor_list(video_samples)  # (n, c, t)
        if not isinstance(audio_samples, NestedTensor):
            if isinstance(audio_samples, (list, tuple)):
                audio_samples = NestedTensor(*audio_samples)
            else:
                audio_samples = nested_tensor_from_tensor_list(audio_samples)  # (n, c, t)

        video, mask_v = video_samples.tensors, video_samples.mask
        audio, mask_a = audio_samples.tensors, audio_samples.mask

        if self.with_regs:
            # [B, Tv, C], _, [B, Tv+5, Tv+5], [B, 1, C], [B, 4, C]
            v_src, _, _, v_cls, v_regs = self.v_backbone(video, mask_v, return_registers=True)
            # [B, Ta, C], _, [B, Ta+5, Ta+5], [B, 1, C], [B, 4, C]
            a_src, _, _, a_cls, a_regs = self.a_backbone(audio, mask_a, return_registers=True)
            mask_v, mask_a = mask_v[:, 1:-4], mask_a[:, 1:-4]
        else:
            a_src, _, _ = self.a_backbone(audio, mask_a)
            v_src, _, _ = self.v_backbone(video, mask_v)
            self.query_gen = False

        if not self.query_gen:
            v_cls, v_regs, a_cls, a_regs = None, None, None, None

        B, Tv, C = v_src.shape
        _, Ta, _ = a_src.shape
        if Ta != Tv:
            a_src = F.interpolate(a_src.permute(0, 2, 1), size=Tv, mode='linear', align_corners=False).permute(0, 2, 1).contiguous()

        srcs, masks = self.multi_scale_adapter(v_src, a_src, mask_v, mask_a)
        fps = torch.stack([item['fps'] for item in info if 'fps' in item]).squeeze(1)
        stride = torch.stack([item['stride'] for item in info if 'stride' in item]).squeeze(1)
        feature_durations = torch.stack([item['feature_duration'] for item in info if 'feature_duration' in item]).squeeze(1)

        grid = get_feature_grids(mask_v, fps, stride, stride)
        grids = [grid]
        poss = [self.position_embedding(grid)]

        if self.num_feature_levels > 1:
            for l in range(1, self.num_feature_levels):
                mask = masks[l]
                cur_stride = stride * 2 ** l
                grid = get_feature_grids(mask, fps, cur_stride, cur_stride)

                for i, (g, m, m2) in enumerate(zip(grids[-1], mask, masks[l - 1])):
                    g = F.interpolate(g[None, None, ~m2], size=(~m).sum().item(), mode='linear')[0, 0, :]
                    grid[i, :g.size(0)] = g
                pos = self.position_embedding(grid)

                grids.append(grid)
                poss.append(pos)

        # prepare input for decoder
        mask_flatten, grid_flatten, temporal_lens, lvl_pos_embed_flatten = [], [], [], []

        for lvl, (src, mask, pos_embed, grid) in enumerate(zip(srcs, masks, poss, grids)):
            bs, t, c = src.shape
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

            temporal_lens.append(t)
            mask_flatten.append(mask)
            grid_flatten.append(grid)

        src_flatten = torch.cat(srcs, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        grid_flatten = torch.cat(grid_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        temporal_lens = torch.as_tensor(temporal_lens, dtype=torch.long, device=mask_flatten.device)
        level_start_index = torch.cat((temporal_lens.new_zeros((1,)), temporal_lens.cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # (bs, n_levels)

        # deformable encoder
        if self.encoder is not None:
            memory = self.encoder(
                src_flatten, temporal_lens,
                grid_flatten, feature_durations,
                level_start_index, valid_ratios,
                lvl_pos_embed_flatten, mask_flatten
            )   # shape=(bs, t, c)
        else:
            memory = src_flatten

        (
            hs_v, inter_grids_v, enc_memory, enc_outputs_class_v, enc_outputs_segment,
            enc_outputs_mask, output_proposals, sampling_locations_v, query_mask_v
        ) = self.v_decoder(
            memory, mask_flatten, temporal_lens, grid_flatten, v_cls, v_regs,
            level_start_index, valid_ratios, feature_durations, fps,
        )

        (
            hs_a, inter_grids_a, enc_memory, enc_outputs_class_a, enc_outputs_segment,
            enc_outputs_mask, output_proposals, sampling_locations_a, query_mask_a
        ) = self.a_decoder(
            memory, mask_flatten, temporal_lens, grid_flatten, a_cls, a_regs,
            level_start_index, valid_ratios, feature_durations, fps,
        )

        outputs_classes_v, outputs_coords_v, outputs_classes_a, outputs_coords_a = [], [], [], []
        # gather outputs from each decoder layer
        for lvl in range(hs_v.shape[0]):
            outputs_class_v = self.class_embed_v[lvl](hs_v[lvl])
            outputs_class_a = self.class_embed_a[lvl](hs_a[lvl])

            outputs_classes_v.append(outputs_class_v)
            outputs_classes_a.append(outputs_class_a)
            outputs_coords_v.append(inter_grids_v[lvl])
            outputs_coords_a.append(inter_grids_a[lvl])

        outputs_class_v = torch.stack(outputs_classes_v)
        outputs_class_a = torch.stack(outputs_classes_a)
        outputs_coords_v = torch.stack(outputs_coords_v)
        outputs_coords_a = torch.stack(outputs_coords_a)

        out_v = {
            "pred_logits": outputs_class_v[-1], "pred_segments": outputs_coords_v[-1], "feature": hs_v[-1],
            "mask": query_mask_v,
        }
        out_a = {
            "pred_logits": outputs_class_a[-1], "pred_segments": outputs_coords_a[-1], "feature": hs_a[-1],
            "mask": query_mask_a,
        }

        if self.count_loss:
            out_v['fake_num'] = self.counting_v(hs_v[-1].permute(0, -1, 1))
            out_a['fake_num'] = self.counting_a(hs_a[-1].permute(0, -1, 1))

        if self.mod_loss:
            out_v["classifier_score"] = self.v_classifier(hs_v[-1])
            out_a["classifier_score"] = self.a_classifier(hs_a[-1])

        if self.aux_loss:
            out_v['aux_outputs'] = self._set_aux_loss(outputs_class_v, outputs_coords_v, query_mask_v)
            out_a['aux_outputs'] = self._set_aux_loss(outputs_class_a, outputs_coords_a, query_mask_a)

        return out_v, out_a

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, query_mask):
        return [{'pred_logits': a, 'pred_segments': b, 'mask': query_mask}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
