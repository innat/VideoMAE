import tensorflow as tf
from tensorflow import keras


class TFPatchEmbed(keras.Model):
    def __init__(
        self,
        input_size=224,
        patch_size=16,
        embed_dim=768,
        num_frames=16,
        tubelet_size=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = (patch_size,) * 2
        self.img_size = (input_size,) * 2

        assert self.img_size[1] % self.patch_size[1] == 0
        assert self.img_size[0] % self.patch_size[0] == 0

        self.num_patches = (
            (self.img_size[1] // self.patch_size[1])
            * (self.img_size[0] // self.patch_size[0])
            * (num_frames // tubelet_size)
        )
        kernel_size = [tubelet_size] + list(self.patch_size)

        self.proj = keras.layers.Conv3D(
            embed_dim,
            kernel_size,
            strides=kernel_size,
            padding="valid",
            name="patch_embed_proj",
        )

    def call(self, x):
        B, T, H, W, C = tf.unstack(tf.shape(x))

        tf.debugging.assert_equal(
            H,
            self.img_size[0],
            message=f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        tf.debugging.assert_equal(
            W,
            self.img_size[1],
            message=f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )

        x = self.proj(x)
        new_shape = tf.concat([tf.shape(x)[:1], [-1], tf.shape(x)[-1:]], axis=0)
        x = tf.reshape(x, new_shape)
        return x
