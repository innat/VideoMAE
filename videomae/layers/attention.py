
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Zeros

class TFAttention(keras.Model):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
        attn_head_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
            
        self.num_heads = num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = layers.Dense(all_head_dim * 3, use_bias=False)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
        self.input_size = input_size
        
        if qkv_bias:
            uid = keras.backend.get_uid(prefix="q_bias")
            self.q_bias = self.add_weight(
                name=f"q_bias_id{uid}",
                shape=(all_head_dim,),
                initializer=Zeros(),
                trainable=True
            )
            uid = keras.backend.get_uid(prefix="v_bias")
            self.v_bias = self.add_weight(
                name=f"v_bias_id{uid}",
                shape=(all_head_dim,),
                initializer=Zeros(),
                trainable=True
            )
        else:
            self.q_bias = None
            self.v_bias = None
    

    def call(self, x):
        B, N, C = tf.unstack(tf.shape(x))
        
        qkv_bias = None
        if self.q_bias is not None:
            zeros_v_bias = tf.zeros_like(self.v_bias)
            q_bias = tf.cast(self.q_bias, dtype=zeros_v_bias.dtype)
            v_bias = tf.cast(self.v_bias, dtype=q_bias.dtype)
            qkv_bias = tf.concat([q_bias, zeros_v_bias, v_bias], axis=0)
        qkv = self.qkv(x)
        
        if qkv_bias is not None:
            qkv = qkv + tf.cast(qkv_bias, qkv.dtype)
        
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, [B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x