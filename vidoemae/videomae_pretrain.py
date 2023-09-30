
from functools import partial
import os
import warnings

warnings.simplefilter(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

from blocks import TFPretrainVisionTransformerEncoder
from blocks import TFPretrainVisionTransformerDecoder
from utils import get_sinusoid_encoding_table_tf


class TFPretrainVisionTransformer(keras.Model):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        encoder_num_classes=0,
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12, 
        decoder_num_classes=1536, 
        decoder_embed_dim=512, 
        decoder_depth=8, 
        decoder_num_heads=8, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=layers.LayerNormalization, 
        init_values=0., 
        use_learnable_pos_emb=False, 
        tubelet_size=2, 
        num_classes=0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.encoder = TFPretrainVisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size, 
            num_classes=encoder_num_classes, embed_dim=encoder_embed_dim, depth=encoder_depth,
            num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size, 
            use_learnable_pos_emb=use_learnable_pos_emb, name='TFPretrainVisionTransformerEncoder',
        )

        self.decoder = TFPretrainVisionTransformerDecoder(
            patch_size=patch_size, num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, depth=decoder_depth,
            num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
            name='TFPretrainVisionTransformerDecoder'
        )

        self.encoder_to_decoder = layers.Dense(
            decoder_embed_dim, use_bias=False, name='encoder_to_decoder'
        )
        self.mask_token = self.add_weight(
            "mask_token_pretrained", 
            shape=(1, 1, decoder_embed_dim), 
            initializer=RandomNormal(stddev=0.02), 
            trainable=True
        )
        self.pos_embed = get_sinusoid_encoding_table_tf(
            self.encoder.patch_embed.num_patches, 
            decoder_embed_dim,
        ) 
        

    def call(self, x, mask):
        B, T, _, _, _ = tf.unstack(tf.shape(x))
        x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)
        B, N, C = tf.unstack(tf.shape(x_vis))
        
        _, num_patches, _ = self.pos_embed.shape
        expand_pos_embed = tf.broadcast_to(self.pos_embed, [B, num_patches, C])
        pos_emd_vis = tf.reshape(
            tf.boolean_mask(expand_pos_embed, tf.math.logical_not(mask)), [B, -1, C]
        )
        pos_emd_mask = tf.reshape(
            tf.boolean_mask(expand_pos_embed, mask), [B, -1, C]
        )
        
        pos_emd_vis = tf.cast(pos_emd_vis, dtype=x_vis.dtype)
        mask_token = tf.cast(self.mask_token, dtype=x_vis.dtype)
        pos_emd_mask = tf.cast(pos_emd_mask, dtype=x_vis.dtype)
        x_full = tf.concat(
            [
                x_vis + pos_emd_vis, 
                mask_token + pos_emd_mask
            ], axis=1
        )
        x = self.decoder(x_full, tf.shape(pos_emd_mask)[1])
        return x
    


def tf_pretrain_videomae_small_patch16_224(**kwargs):
    model = TFPretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        decoder_depth=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
        name='TFVideoMAE_S_16x224_PT',
        **kwargs
    )
    return model


def tf_pretrain_videomae_base_patch16_224(**kwargs):
    model = TFPretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        decoder_depth=4,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
        name='TFVideoMAE_B_16x224_PT',
        **kwargs
    )
    return model


def tf_pretrain_videomae_large_patch16_224(**kwargs):
    model = TFPretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=512,
        decoder_num_heads=8,
        decoder_depth=12,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
        name='TFVideoMAE_L_16x224_PT',
        **kwargs
    )
    return model


def tf_pretrain_videomae_huge_patch16_224(**kwargs):
    model = TFPretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=1280, 
        encoder_depth=32, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536, 
        decoder_embed_dim=640,
        decoder_num_heads=8,
        decoder_depth=12,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6),
        name='TFVideoMAE_H_16x224_PT',
        **kwargs
    )
    return model