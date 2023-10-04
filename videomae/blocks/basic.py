
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant

from videomae.layers import TFMlp
from videomae.layers import TFAttention
from videomae.layers import TFDropPath

class TFBlock(keras.Model):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=layers.Activation(tf.nn.gelu),
        norm_layer=layers.LayerNormalization,
        attn_func=TFAttention,
        attn_head_dim=None,
        init_values=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.norm1 = norm_layer(axis=-1)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim
        )

        self.drop_path = TFDropPath(drop_path) if drop_path > 0. else layers.Identity()
        self.norm2 = norm_layer(axis=-1)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TFMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        
        if init_values > 0:
            uid = keras.backend.get_uid(prefix="gamma_1")
            self.gamma_1 = self.add_weight(
                shape=(dim,),
                initializer=Constant(value=init_values),
                trainable=True,
                name=f"gamma_1{uid}"
            )
            uid = keras.backend.get_uid(prefix="gamma_2")
            self.gamma_2 = self.add_weight(
                shape=(dim,),
                initializer=Constant(value=init_values),
                trainable=True,
                name=f"gamma_2{uid}"
            )
        else:
            self.gamma_1, self.gamma_2 = None, None


    def call(self, x, training=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)), training=training)
            x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)), training=training)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)), training=training)
        return x