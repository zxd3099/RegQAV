# encoding: utf-8
# ------------------------------------------------------------------------
# Modified from TadTR (https://github.com/xlliu7/TadTR)
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import copy
import math
import torch
import torch.nn as nn

from torch.nn.init import normal_

# from modules import DropPath
# from ops.modules.temporal_deform_attn import DeformAttn
# from utils import MLP, _get_activation_fn, SE_Block

from .modules import DropPath
from .ops.modules.temporal_deform_attn import DeformAttn
from .utils import MLP, _get_activation_fn, SE_Block


class DeformableTransformer(nn.Module):
    def __init__(self, feature_dim=512, d_model=256, n_heads=8, n_deform_heads=8, base_scale=0.02, num_queries=50,
                 max_queries=3000, num_decoder_layers=6, dropout=0.1, droppath=0.1, activation="relu",
                 return_intermediate_dec=False, dim_feedforward=1024, num_feature_levels=3, query_gen=True,
                 num_sampling_levels=3, dec_n_points=4, fix_encoder_proposals=True, temperature=10000):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_deform_heads = n_deform_heads
        self.base_scale = base_scale
        self.num_queries = num_queries
        self.max_queries = max_queries
        self.fix_encoder_proposals = fix_encoder_proposals
        self.num_layers = num_decoder_layers
        self.temperature = temperature

        self.se_block = SE_Block(feature_dim, d_model, out_num=num_queries)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, droppath, activation,
            num_sampling_levels, n_heads, n_deform_heads, dec_n_points
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers,
            d_model=d_model, return_intermediate=return_intermediate_dec,
            temperature=temperature
        )

        if not query_gen:
            self.query_embed = nn.Embedding(num_queries, d_model)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, temporal_lengths, grids, fps):
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, T_ in enumerate(temporal_lengths):
            timeline = grids[:, _cur:(_cur + T_)]

            scale = torch.ones_like(timeline) * (fps[..., None] * self.base_scale) * 2 ** lvl
            proposal = torch.stack((timeline, scale), -1).view(N_, -1, 2)
            proposals.append(proposal)
            _cur += T_

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ~memory_padding_mask
        output_proposals[..., 1] = output_proposals[..., 1].log_()

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask[..., None], float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals, output_proposals_valid

    def forward(self, memory, mask_flatten, temporal_lens, grid_flatten, cls_tokens, reg_tokens,
                level_start_index, valid_ratios, feature_durations, fps, query_embed=None, attn_mask=None):
        bs, t, c = memory.shape

        # generate queries
        if cls_tokens is not None:
            tgt = self.se_block(cls_tokens, reg_tokens)
        else:
            tgt = self.query_embed.weight
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        # generate reference points
        output_memory, output_proposals, output_proposals_valid = self.gen_encoder_output_proposals(
            memory, mask_flatten, temporal_lens, grid_flatten.detach(), fps
        )

        enc_outputs_class = self.decoder.class_embed[-1](output_memory)

        enc_outputs_mask = ~output_proposals_valid
        valid_scores = enc_outputs_class[..., 0].masked_fill(enc_outputs_mask, float('-1e9'))

        if self.fix_encoder_proposals:
            enc_outputs_segment = output_proposals
        else:
            enc_outputs_segment = self.decoder.segment_embed[-1](output_memory)
            enc_outputs_segment = torch.stack([
                output_proposals[..., 0] + enc_outputs_segment[..., 0] * output_proposals[..., 1].exp().detach(),
                output_proposals[..., 1] + enc_outputs_segment[..., 1],
            ], dim=-1)

        num_topk = self.num_queries if t > self.num_queries else t
        topk_valid_scores, topk_proposals = torch.topk(valid_scores, num_topk, dim=1)
        topk_segments = torch.gather(enc_outputs_segment, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, 2))

        if topk_valid_scores.size(-1) > self.max_queries:
            topk_valid_scores, topk_proposals = torch.topk(topk_valid_scores, self.max_queries, dim=1)
            topk_segments = torch.gather(topk_segments, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, 2))

        query_mask = ~(topk_valid_scores > float('-1e9'))
        hs, inter_grids, sampling_locations = self.decoder(
            tgt, topk_segments, feature_durations, memory, temporal_lens, level_start_index,
            valid_ratios, mask_flatten, query_embed, query_mask, attn_mask
        )

        return (
            hs, inter_grids, output_memory, enc_outputs_class, enc_outputs_segment,
            enc_outputs_mask, output_proposals, sampling_locations, query_mask
        )


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, droppath=0.1, activation="relu",
                 n_levels=4, n_deform_heads=2, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = DeformAttn(d_model, n_levels, n_deform_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.droppath = DropPath(droppath)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        input_src = src
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        src = self.droppath(src, input_src)
        return src

    def forward(self, src, pos, reference_points, temporal_lengths, level_start_index, padding_mask=None):
        input_src = src
        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, temporal_lengths, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.droppath(src, input_src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, droppath=0.1):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.droppath = droppath

    @staticmethod
    def get_reference_points(temporal_lengths, valid_ratios, device):
        reference_points_list = []
        for lvl, T_ in enumerate(temporal_lengths):
            ref = torch.linspace(0.5, T_ - 0.5, T_, dtype=torch.float32, device=device)  # (t,)
            ref = ref[None] / (valid_ratios[:, None, lvl] * T_)                          # (bs, t)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]          # (N, t, n_levels)
        return reference_points[..., None]

    def forward(self, src, temporal_lens, grids, feature_durations, level_start_index, valid_ratios, pos=None, padding_mask=None):
        '''
        src: shape=(bs, t, c)
        temporal_lens: shape=(n_levels). content: [t1, t2, t3, ...]
        level_start_index: shape=(n_levels,). [0, t1, t1+t2, ...]
        valid_ratios: shape=(bs, n_levels).
        '''
        output = src
        # (bs, t, levels, 1)
        normalized_grids = grids[:, :, None] / feature_durations[:, None, None]
        reference_points = normalized_grids[..., None] * valid_ratios[:, None, :, None]

        for _, layer in enumerate(self.layers):
            layer_output = layer(output, pos, reference_points, temporal_lens, level_start_index, padding_mask)
            if self.droppath > 0 and self.training:
                p = torch.rand(output.size(0), dtype=torch.float32, device=output.device)
                p = (p > self.droppath)[:, None, None]
                output = torch.where(p, layer_output, output)
            else:
                output = layer_output
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, droppath=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_deform_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = DeformAttn(d_model, n_levels, n_deform_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.droppath = DropPath(droppath)

    def forward_ffn(self, tgt):
        input_tgt = tgt
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        tgt = self.droppath(tgt, input_tgt)
        return tgt

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos, reference_points, src, temporal_lengths,
                level_start_index, src_padding_mask=None, query_mask=None, attn_mask=None):
        """
        :param tgt:
        :param query_pos:
        :param reference_points:
        :param src:
        :param temporal_lengths:
        :param level_start_index:
        :param src_padding_mask:
        :param query_mask:
        :param attn_mask:
        :return:
        """
        # self attention
        input_tgt = tgt
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1),
                              key_padding_mask=query_mask, attn_mask=attn_mask)[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.droppath(tgt, input_tgt)

        # cross attention
        input_tgt = tgt
        tgt2, (sampling_locations, attention_weights) = self.cross_attn(
            self.with_pos_embed(tgt, query_pos), reference_points, src, temporal_lengths, level_start_index, src_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt = self.droppath(tgt, input_tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, sampling_locations


class DeformableTransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, d_model=256, return_intermediate=False, temperature=10000
    ):
        super().__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.segment_embed = None
        self.class_embed = None
        self.d_model = d_model
        self.grid_head = MLP(d_model * 2, d_model, d_model, 2)
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.temperature = temperature

    def get_proposal_pos_embed(self, proposals):
        scale = 2 * math.pi

        dim_t = torch.arange(self.d_model, dtype=torch.float32, device=proposals.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.d_model)
        # N, L, 4
        proposals = proposals * scale
        # N, L, 4, 128
        pos_ct = proposals[:, :, [0]] / dim_t
        pos_w = proposals[:, :, [1]] / dim_t
        pos_ct = torch.stack((pos_ct[:, :, 0::2].sin(), pos_ct[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_ct, pos_w), dim=2)
        # N, L, 4, 64, 2
        return pos

    def forward(self, tgt, enc_output_segments, feature_durations,
                src, temporal_lens, src_level_start_index, src_valid_ratios,
                src_padding_mask=None, query_pos=None, query_mask=None, attn_mask=None):
        """
        :param tgt:                        (N, Length_{query}, C)
        :param enc_output_segments:        (N, Length_{query}, 2)
        :param feature_durations:          (N, )
        :param src:                        (N, \\sum_{l=0}^{L-1} T_l, C)
        :param temporal_lens:
        :param src_level_start_index:      (n_levels, ), [0, T_0, T_1, T_2, ..., T_{L-1}]
        :param src_valid_ratios:           (N, n_levels)
        :param src_padding_mask:           (N, \\sum_{l=0}^{L-1} T_l), True for padding elements, False for non-padding elements
        :param query_pos:
        :param query_mask:
        :param attn_mask:
        :return:
        """
        output = tgt
        intermediate = []
        intermediate_grids = []
        segment_outputs = enc_output_segments.detach()
        reference_points = torch.stack([
            segment_outputs[..., 0], segment_outputs[..., 1].exp()
        ], dim=-1)

        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points / feature_durations[:, None, None]
            reference_points_input = reference_points_input[:, :, None, :] * src_valid_ratios[:, None, :, None]  # (N, Length_{query}, n_levels, 1 or 2)

            grid_sine_embed = self.get_proposal_pos_embed(reference_points)  # [N, Length_{query}, 2C]
            raw_query_pos = self.grid_head(grid_sine_embed)  # [N, Length_{query}, C]
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1  # [N, Length_{query}, C]
            query_pos = pos_scale * raw_query_pos   # [N, Length_{query}, C]

            output, sampling_locations = layer(output, query_pos, reference_points_input, src, temporal_lens,
                                               src_level_start_index, src_padding_mask, query_mask, attn_mask)
            # segment refinement
            if self.segment_embed is not None:
                segment_outputs_detach = segment_outputs.detach()
                segment_outputs = self.segment_embed[lid](output)
                segment_outputs = torch.stack([
                    segment_outputs_detach[..., 0] + segment_outputs[..., 0] * segment_outputs_detach[..., 1].exp(),
                    segment_outputs_detach[..., 1] + segment_outputs[..., 1],
                ], dim=-1)

                new_reference_points = torch.stack([
                    segment_outputs[..., 0], segment_outputs[..., 1].exp()
                ], dim=-1)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_grids.append(segment_outputs)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_grids), sampling_locations

        return output, segment_outputs, sampling_locations


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def test_decoder():
    layer = DeformableTransformerDecoderLayer()
    decoder = DeformableTransformerDecoder(layer, 1)
    tgt = torch.randn([4, 50, 256])
    enc_output_segments = torch.randn([4, 50, 2])
    feature_durations = torch.randn([4, ])
    src_valid_ratios = torch.randn([4, 1])
    decoder(tgt, enc_output_segments, feature_durations,
            None, None, None, src_valid_ratios)


def get_proposal_pos_embed(proposals, d_model=256, temperature=10000):
    scale = 2 * math.pi

    dim_t = torch.arange(d_model, dtype=torch.float32, device=proposals.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / d_model)
    proposals = proposals * scale
    pos_ct = proposals[:, :, :, :, [0]] / dim_t   # [B, Lq, C]
    pos_w = proposals[:, :, :, :, [1]] / dim_t    # [B, Lq, C]
    pos_ct = torch.stack((pos_ct[:, :, :, :, 0::2].sin(), pos_ct[:, :, :, :, 1::2].cos()), dim=5).flatten(4)    # [B, Lq, C]
    pos_w = torch.stack((pos_w[:, :, :, :, 0::2].sin(), pos_w[:, :, :, :, 1::2].cos()), dim=5).flatten(4)   # [B, Lq, C]
    pos = torch.cat((pos_ct, pos_w), dim=-1)     # [B, Lq, 2C]
    return pos


def build_deformable_transformer(args):
    return DeformableTransformer(
        feature_dim=args.feature_dim,
        d_model=args.hidden_dim,
        n_heads=args.n_heads,
        n_deform_heads=args.n_deform_heads,
        base_scale=args.base_scale,
        num_queries=args.num_queries,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        droppath=args.droppath,
        activation=args.transformer_activation,
        return_intermediate_dec=True,
        max_queries=args.max_queries,
        fix_encoder_proposals=args.fix_encoder_proposals,
        num_feature_levels=args.num_feature_levels,
        num_sampling_levels=args.num_sampling_levels,
        dec_n_points=args.dec_n_points,
        temperature=args.temperature,
        query_gen=args.query_gen
    )


def build_deformable_encoder(args):
    if args.enc_layers:
        encoder_layer = DeformableTransformerEncoderLayer(
            args.hidden_dim, args.dim_feedforward, 0.0, 0.0, args.transformer_activation,
            args.num_sampling_levels, args.n_deform_heads, args.enc_n_points
        )
        return DeformableTransformerEncoder(encoder_layer, args.enc_layers, 0.0)
    else:
        return None