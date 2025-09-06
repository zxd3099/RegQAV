# encoding: utf-8
import math
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from einops.layers.torch import Rearrange


class Sequential(nn.Sequential):
    def forward(self, x, mask):
        for module in self:
            x, mask = module(x, mask)
        return x, mask


class LayerNorm(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/libs/modeling/blocks.py#L63
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self, num_channels, eps=1e-5, affine=True, device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x, mask):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out, mask


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GLU, self).__init__()

        self.linear_x = nn.Linear(in_channels, out_channels)
        self.linear_gate = nn.Linear(in_channels, out_channels)
        self._init_weights(self.linear_x)
        self._init_weights(self.linear_gate)

        self.sigmoid = nn.Sigmoid()

    def _init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, cls_token, reg_token):
        cls_token = self.linear_x(cls_token)
        reg_token = self.sigmoid(self.linear_gate(reg_token))
        return cls_token * reg_token


class MaskedConv1D(nn.Module):
    """
    from https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/blocks.py#10
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        # assert T % self.stride == 0
        mask = ~mask.unsqueeze(1)
        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, ~out_mask.squeeze(1)


class SE_Block(nn.Module):
    def __init__(self, in_channels, out_channels, in_num=1, out_num=100):
        super(SE_Block, self).__init__()

        self.glus = nn.ModuleList([GLU(in_channels, in_channels) for _ in range(4)])
        self.cls_linear = nn.Sequential(
            Rearrange('b l c -> b c l'),
            nn.Linear(in_num, out_num),
            Rearrange('b c l -> b l c')
        )
        self.out_linear = nn.Linear(in_channels, out_channels)
        self.out_num = out_num

    def _init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, cls_tokens, reg_tokens):
        """
        :param cls_tokens: [batch_size, C]
        :param reg_tokens: [batch_size, 4, C]
        """
        if cls_tokens.dim() == 2:
            cls_tokens = cls_tokens.unsqueeze(1)   # [B, 1, C]

        glu_outputs = [self.glus[i](cls_tokens, reg_tokens[:, i:i + 1, :]) for i in range(reg_tokens.size(1))]

        cls_token_expanded = self.cls_linear(cls_tokens)   # [B, 100, C]

        result = []
        step = self.out_num // reg_tokens.size(1)
        for i in range(reg_tokens.size(1)):
            result.append(cls_token_expanded[:, i * step:(i + 1) * step, :] * glu_outputs[i])

        result = torch.cat(result, dim=1)
        return self.out_linear(result)


class MultiScaleAdapter(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, fusion_mode="concat", num_feature_levels=4,
                 with_ln=False, scale_factor=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.num_feature_levels = num_feature_levels
        self.activation = nn.ReLU(inplace=True)

        self.v_proj = nn.Linear(feature_dim, hidden_dim)
        self.a_proj = nn.Linear(feature_dim, hidden_dim)

        if self.fusion_mode == 'concat':
            self.f_weight = nn.Linear(2 * hidden_dim, hidden_dim)
            self.__init_weights__(self.f_weight)
        else:
            self.fusion_act = _get_activation_fn("gelu")
            self.v_weight = nn.Linear(hidden_dim, hidden_dim)
            self.a_weight = nn.Linear(hidden_dim, hidden_dim)
            self.__init_weights__(self.v_weight)
            self.__init_weights__(self.a_weight)

        # main branch using convs with pooling
        self.branch_v = nn.ModuleList([
            Sequential(
                MaskedConv1D(hidden_dim, hidden_dim, kernel_size=3, stride=scale_factor, padding=1),
                LayerNorm(hidden_dim),
            )
            for _ in range(num_feature_levels - 1)
        ])
        self.branch_a = nn.ModuleList([
            Sequential(
                MaskedConv1D(hidden_dim, hidden_dim, kernel_size=3, stride=scale_factor, padding=1),
                LayerNorm(hidden_dim),
            )
            for _ in range(num_feature_levels - 1)
        ])

        for layer in self.branch_v:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            if not with_ln:
                nn.init.zeros_(layer[0].conv.bias)

        for layer in self.branch_a:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            if not with_ln:
                nn.init.zeros_(layer[0].conv.bias)

        self.apply(self.__init_weights__)

    def _fusion(self, x1, x2):
        if self.fusion_mode == 'concat':
            x_f = torch.cat([x1, x2], dim=-1)
            return self.f_weight(x_f)
        elif self.fusion_mode == 'add':
            return x1 + x2
        elif self.fusion_mode == 'star':
            x1 = self.activation(self.v_weight(x1))
            x2 = self.a_weight(x2)
            return x1 * x2

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, v_src, a_src, mask_v, mask_a):
        # projection
        v_src = self.v_proj(v_src)
        a_src = self.a_proj(a_src)

        # fusion
        src = self._fusion(v_src, a_src)

        # prep for outputs
        out_feats = (src, )
        out_masks = (mask_v, )

        v_src = v_src.permute(0, -1, 1)
        a_src = a_src.permute(0, -1, 1)
        # main branch with downsampling
        for idx in range(self.num_feature_levels - 1):
            v_src, mask_v = self.branch_v[idx](v_src, mask_v)
            a_src, mask_a = self.branch_a[idx](a_src, mask_a)

            tmp = self._fusion(v_src.permute(0, -1, 1), a_src.permute(0, -1, 1))
            out_feats += (tmp,)
            out_masks += (mask_v,)

        return out_feats, out_masks


class VisualMultiScaleAdapter(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_feature_levels=4, with_ln=False, scale_factor=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.activation = nn.ReLU(inplace=True)

        self.v_proj = nn.Linear(feature_dim, hidden_dim)

        self.branch_v = nn.ModuleList([
            Sequential(
                MaskedConv1D(hidden_dim, hidden_dim, kernel_size=3, stride=scale_factor, padding=1),
                LayerNorm(hidden_dim),
            )
            for _ in range(num_feature_levels - 1)
        ])

        for layer in self.branch_v:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            if not with_ln:
                nn.init.zeros_(layer[0].conv.bias)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, v_src, mask_v):
        v_src = self.v_proj(v_src)
        out_feats = (v_src,)
        out_masks = (mask_v,)

        v_src = v_src.permute(0, -1, 1)
        for idx in range(self.num_feature_levels - 1):
            v_src, mask_v = self.branch_v[idx](v_src, mask_v)
            out_feats += (v_src.permute(0, -1, 1),)
            out_masks += (mask_v,)
        return out_feats, out_masks
    

class AudioMultiScaleAdapter(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_feature_levels=4, with_ln=False, scale_factor=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.activation = nn.ReLU(inplace=True)

        self.a_proj = nn.Linear(feature_dim, hidden_dim)

        self.branch_a = nn.ModuleList([
            Sequential(
                MaskedConv1D(hidden_dim, hidden_dim, kernel_size=3, stride=scale_factor, padding=1),
                LayerNorm(hidden_dim),
            )
            for _ in range(num_feature_levels - 1)
        ])

        for layer in self.branch_a:
            nn.init.xavier_uniform_(layer[0].conv.weight, gain=1)
            if not with_ln:
                nn.init.zeros_(layer[0].conv.bias)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, a_src, mask_a):
        a_src = self.a_proj(a_src)
        out_feats = (a_src,)
        out_masks = (mask_a,)

        a_src = a_src.permute(0, -1, 1)
        for idx in range(self.num_feature_levels - 1):
            a_src, mask_a = self.branch_a[idx](a_src, mask_a)
            out_feats += (a_src.permute(0, -1, 1),)
            out_masks += (mask_a,)
        return out_feats, out_masks


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def get_feature_grids(mask, fps, window_size, stride):
    B, T = mask.size()
    # Create feature indices: [0, 1, 2, ..., total_features - 1]
    feature_indices = torch.arange(0, T, dtype=torch.float32, device=mask.device)
    feature_indices.unsqueeze_(0)

    feature_indices = feature_indices.repeat(B, 1)

    # Calculate the center frame index for each feature
    center_frame_indices = feature_indices * stride[:, None] + window_size[:, None] // 2

    feature_grid = center_frame_indices / fps[:, None]
    return feature_grid


def sigmoid_focal_loss(inputs, targets, mask, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection
    :param inputs: A float tensor of arbitrary shape. The predictions for each example.
    :param targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
    :param mask:
    :param num_boxes:
    :param alpha: (optional) Weighting factor in range (0,1) to balance
                   positive vs negative examples. Default = -1 (no weighting).
    :param gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
    :return: Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if mask is not None:
        valid = ~mask[..., None]
        loss = loss * valid.float()

        num_queries = valid.sum(1).float()
        normalizer = num_queries.clamp(min=1.0)
        return ((loss.sum(1) / normalizer).sum() / num_boxes) * num_queries.mean()
    else:
        num_queries = inputs.size(1)
        return (loss.mean(1).sum() / num_boxes) * num_queries


# def test():
#     model = MultiScaleAdapter()
#     v_src = torch.randn(1, 200, 512)
#     a_src = torch.randn(1, 200, 512)
#     mask = torch.ones([1, 200], dtype=torch.bool)
#     out_feats, out_masks = model(v_src, a_src, mask, mask)
#     print(len(out_feats))
#     print(len(out_masks))
#
#
# test()

