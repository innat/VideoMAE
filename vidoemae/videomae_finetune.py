
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal

from layers import TFPatchEmbed
from layers import TFAttention
from blocks import TFBlock
from utils import get_sinusoid_encoding_table_tf


class TFVisionTransformer(keras.Model):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        num_classes=1000, 
        embed_dim=768, 
        depth=12,
        num_heads=12, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        fc_drop_rate=0., 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0., 
        norm_layer=layers.LayerNormalization, 
        init_values=0.,
        use_learnable_pos_emb=False, 
        init_scale=0.,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = TFPatchEmbed(
            input_size=img_size, 
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            num_frames=all_frames, 
            tubelet_size=self.tubelet_size,
            name='patch_embed'
        )
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = self.add_weight(
                "pos_embed", 
                shape=(1, num_patches, embed_dim), 
                initializer=TruncatedNormal(stddev=0.02),
                trainable=True
            )
        else:
            self.pos_embed = get_sinusoid_encoding_table_tf(
                num_patches, embed_dim
            )  

        self.pos_drop = layers.Dropout(drop_rate, name='pos_drop')
        dpr=tf.linspace(0., drop_path_rate, depth).numpy().tolist()
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
                act_layer=layers.Activation(tf.nn.gelu),
                norm_layer=norm_layer,
                attn_func=partial(TFAttention, name=f'TFAttention{i+1}'),
                init_values=init_values,
                name=f'TFBlock{i+1}'
            ) for i in range(depth)
        ]
        
        self.norm = layers.Identity(name='norm_identity') if use_mean_pooling else norm_layer(name='norm')
        self.fc_norm = norm_layer(name='fc_norm') if use_mean_pooling else None
        self.fc_dropout = layers.Dropout(
            fc_drop_rate, name='fc_dropout'
        ) if fc_drop_rate > 0 else layers.Identity(name='fc_dropout_identity')
        self.head = layers.Dense(
            num_classes,
            kernel_initializer=TruncatedNormal(stddev=0.02),
            name='head'
        ) if num_classes > 0 else layers.Identity(name='head_identity')
        

    def forward_features(self, x):
        x = self.patch_embed(x)
        B = tf.shape(x)[0]
        
        if self.pos_embed is not None:
            pos_embed_dtype = self.pos_embed.dtype
            x += tf.tile(tf.cast(self.pos_embed, dtype=x.dtype), [B, 1, 1])
            self.pos_embed = tf.cast(self.pos_embed, dtype=pos_embed_dtype)

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        if self.fc_norm is not None:
            x = tf.reduce_mean(x, axis=1)
            x = self.fc_norm(x)
        else:
            x = x[:, 0]

        return x

    def call(self, inputs):
        x = self.forward_features(inputs)
        x = self.head(self.fc_dropout(x))
        return x
    
    def build(self, input_shape):
        super().build(input_shape)
        self.build_shape = input_shape[1:]

    def build_graph(self):
        x = keras.Input(shape=self.build_shape, name='input_graph')
        return keras.Model(inputs=[x], outputs=self.call(x))
    


def tf_vit_small_patch16_224(**kwargs):
    model = TFVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), drop_path_rate=0.1,
        name='TFVideoMAE_S_16x224_FT', 
        **kwargs
    )
    return model


def tf_vit_base_patch16_224(**kwargs):
    model = TFVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), name='TFVideoMAE_B_16x224_FT', 
        **kwargs
    )
    return model

def tf_vit_base_patch16_384(**kwargs):
    model = TFVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), name='TFVideoMAE_B_16x384_FT', 
        **kwargs
    )
    return model


def tf_vit_large_patch16_224(**kwargs):
    model = TFVisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), name='TFVideoMAE_L_16x224_FT', 
        **kwargs
    )
    return model

def tf_vit_large_patch16_384(**kwargs):
    model = TFVisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), name='TFVideoMAE_L_16x384_FT', 
        **kwargs
    )
    return model

def tf_vit_large_patch16_512(**kwargs):
    model = TFVisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), name='TFVideoMAE_L_16x512_FT', 
        **kwargs
    )
    return model

def tf_vit_huge_patch16_224(**kwargs):
    model = TFVisionTransformer(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-6), name='TFVideoMAE_H_16x224_FT', 
        **kwargs
    )
    return model
