from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal

from blocks import TFBlock
from layers import TFAttention, TFPatchEmbed
from utils import get_sinusoid_encoding_table_tf


class TFPretrainVisionTransformerEncoder(keras.Model):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=layers.LayerNormalization,
        init_values=None,
        tubelet_size=2,
        use_learnable_pos_emb=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.patch_embed = TFPatchEmbed(
            input_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            tubelet_size=tubelet_size,
            name="patch_embed_encoder",
        )
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = self.add_weight(
                "pos_embed_encoder",
                shape=(1, num_patches + 1, embed_dim),
                initializer=TruncatedNormal(stddev=0.02),
                trainable=True,
            )
        else:
            self.pos_embed = get_sinusoid_encoding_table_tf(num_patches, embed_dim)

        dpr = tf.linspace(0.0, drop_path_rate, depth).numpy().tolist()
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
                attn_func=partial(TFAttention, name=f"TFAttentionEncoder{i+1}"),
                name=f"TFBlockEncoder{i+1}",
            )
            for i in range(depth)
        ]

        self.norm = norm_layer(axis=-1, name="norm_encoder")
        self.head = (
            layers.Dense(num_classes, name="head_encoder")
            if num_classes > 0
            else layers.Identity(name="identity_encoder")
        )

    def forward_features(self, x, mask):
        B, T, _, _, _ = tf.unstack(tf.shape(x))
        x = self.patch_embed(x)

        pos_embed_dtype = self.pos_embed.dtype
        x += tf.tile(tf.cast(self.pos_embed, dtype=x.dtype), [B, 1, 1])
        self.pos_embed = tf.cast(self.pos_embed, dtype=pos_embed_dtype)

        B, _, C = tf.unstack(tf.shape(x))
        x_vis = tf.reshape(tf.boolean_mask(x, tf.math.logical_not(mask)), (B, -1, C))

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def call(self, inputs, mask):
        x = self.forward_features(inputs, mask)
        x = self.head(x)
        return x
