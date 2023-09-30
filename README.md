# Keras Implementation of VideoMAE (NeurIPS 2022 Spotlight).

![videomae](assets\videomae.jpg)


```python
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
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-5),
        **kwargs
    )
    return model

model_tf = tf_vit_small_patch16_224(num_classes=400)
model_tf.load_weights('TFVideoMAE_B_16x224_PT')
```

```python
# tube masking
tube_mask = TubeMaskingGenerator(
    input_size=window_size, 
    mask_ratio=0.75
)
make_bool = tube_mask()
bool_masked_pos_tf = tf.constant(make_bool, dtype=tf.int32)
bool_masked_pos_tf = tf.expand_dims(bool_masked_pos_tf, axis=0)
bool_masked_pos_tf = tf.cast(bool_masked_pos_tf, tf.bool)
bool_masked_pos_tf

# running
pred_tf = model_tf(
    tf.ones(shape=(1, 16, 224, 224, 3)), bool_masked_pos_tf
)
pred_tf.numpy().shape
TensorShape([1, 1176, 1536])
```


# Model Zoo

The pre-trained and fine-tuned models are listed in [MODEL_ZOO.md](MODEL_ZOO.md).



## 🔒 License

The majority of this project is released under the CC-BY-NC 4.0 license as found in the [LICENSE](https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE) file. Portions of the project are available under separate license terms: [SlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) are licensed under the Apache 2.0 license. [BEiT](https://github.com/microsoft/unilm/tree/master/beit) is licensed under the MIT license.

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={arXiv preprint arXiv:2203.12602},
  year={2022}
}
```
