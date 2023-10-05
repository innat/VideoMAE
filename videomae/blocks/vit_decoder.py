
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from videomae.layers import TFAttention
from videomae.blocks import TFBlock


class TFPretrainVisionTransformerDecoder(keras.Model):
    def __init__(
        self, 
        patch_size=16, 
        num_classes=768, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.,
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0.,
        norm_layer=layers.LayerNormalization, 
        init_values=None, 
        num_patches=196, 
        tubelet_size=2, 
        **kwargs
    ):
        super().__init__(**kwargs)

        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        dpr = tf.linspace(0., drop_path_rate, depth) 
        
        self.blocks = [
            TFBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer, 
                init_values=init_values,
                attn_func=partial(TFAttention, name=f'TFAttentionDecoder{i+1}'),
                name=f'TFBlockDecoder{i+1}'
            ) for i in range(depth)
        ]

        self.norm = norm_layer(axis=-1, name='norm_decoder')
        self.head = layers.Dense(
            num_classes, name='head_decoder'
        ) if num_classes > 0 else layers.Identity(name='identity_decoder')

    def call(self, x, return_token_num=0):
        for blk in self.blocks:
            x = blk(x)

        condition = tf.greater(return_token_num, 0)
        x = tf.cond(
            condition, 
            lambda: self.head(self.norm(x[:, -return_token_num:])), 
            lambda: self.head(self.norm(x))
        )

        return x