
from functools import partial
import os
import warnings

warnings.simplefilter(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

from videomae.blocks import TFPretrainVisionTransformerEncoder
from videomae.blocks import TFPretrainVisionTransformerDecoder
from videomae.utils import get_sinusoid_encoding_table_tf
from .model_configs import MODEL_CONFIGS

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
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), 
        init_values=0., 
        use_learnable_pos_emb=False, 
        tubelet_size=2, 
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
    



def VideoMAE_ViTS16PT(name='TFVideoMAE_S_16x224_PT', **kwargs):
    config = MODEL_CONFIGS[name].copy()
    config.update(kwargs)
    model = TFPretrainVisionTransformer(name=name, **config)
    return model

def VideoMAE_ViTB16PT(name='TFVideoMAE_B_16x224_PT', **kwargs):
    config = MODEL_CONFIGS[name].copy()
    config.update(kwargs)
    model = TFPretrainVisionTransformer(name=name, **config)
    return model

def VideoMAE_ViTL16PT(name='TFVideoMAE_L_16x224_PT', **kwargs):
    config = MODEL_CONFIGS[name].copy()
    config.update(kwargs)
    model = TFPretrainVisionTransformer(name=name, **config)
    return model

def VideoMAE_ViTH16PT(name='TFVideoMAE_H_16x224_PT', **kwargs):
    config = MODEL_CONFIGS[name].copy()
    config.update(kwargs)
    model = TFPretrainVisionTransformer(name=name, **config)
    return model
