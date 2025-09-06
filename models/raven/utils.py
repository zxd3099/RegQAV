# encoding: utf-8
"""
@File   : utils.py
@Time   : 2025/1/6 14:29
@Author : zxd3099
"""
import torch

from .pytorch_backend.transformer.encoder import Encoder


def build_encoder(args, with_regs, num_register_tokens):
    encoder = Encoder(
            idim=args.idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            frontend=args.transformer_frontend,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
            layerscale=args.layerscale,
            init_values=args.init_values,
            ff_bn_pre=args.ff_bn_pre,
            post_norm=args.post_norm,
            gamma_zero=args.gamma_zero,
            gamma_init=args.gamma_init,
            mask_init_type=args.mask_init_type,
            drop_path=args.drop_path,
            with_regs=with_regs,
            num_register_tokens=num_register_tokens
        )

    return encoder
