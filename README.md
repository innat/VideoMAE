# VideoMAE

![videomae](./assets/videomae.jpg)


[![keras-2.12.](https://img.shields.io/badge/keras-2.12-darkred)]([?](https://img.shields.io/badge/keras-2.12-darkred)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](?) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/innat/videomae) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Hub-yellow.svg)](?)

This is a unofficial `Keras` reimplementation of [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) model. The official `PyTorch` implementation can be found [here](https://github.com/MCG-NJU/VideoMAE).


## Pre Trained Self-Supervised Model

The pre-trained video-mae model consist of encoder and deconder module. This models are trained in self-supervised manner on the benchmark dataset.

```python
from videomae import VideoMAE_ViTS16PT

# pre-trained self-supervised model
>>> model = VideoMAE_ViTS16PT(num_classes=400)

# tube masking
>>> tube_mask = TubeMaskingGenerator(
    input_size=window_size, 
    mask_ratio=0.75
)
>>> make_bool = tube_mask()
>>> bool_masked_pos_tf = tf.constant(make_bool, dtype=tf.int32)
>>> bool_masked_pos_tf = tf.expand_dims(bool_masked_pos_tf, axis=0)
>>> bool_masked_pos_tf = tf.cast(bool_masked_pos_tf, tf.bool)

# running
>>> pred_tf = model(
    tf.ones(shape=(1, 16, 224, 224, 3)), bool_masked_pos_tf
)
>>> pred_tf.numpy().shape
TensorShape([1, 1176, 1536])
```

## Fine Tuned Model

The fine tuned model is the encoder part of pre-trained model which is used to model for specific target classes.

```python
from videomae import VideoMAE_ViTS16FT

>>> model = VideoMAE_ViTS16FT(num_classes=400)
>>> y = model(np.ones((1, 16, 224, 224, 3)))
>>> y.shape
TensorShape([1, 400])
```


# Model Zoo

The pre-trained and fine-tuned models are listed in [MODEL_ZOO.md](MODEL_ZOO.md). Following are some hightlights.

### Kinetics-400

For Kinetrics-400, VideoMAE is trained around **1600** epoch without **any extra data**. The following checkpoints are available in both tensorflow `SavedModel` and `h5` format.


| Backbone | \#Frame | Top-1 | Top-5 | Params (MB) | FLOPS |
 | :--: | :--: | :---: | :---: | :---: | :---: |
  ViT-S    | 16x5x3  | 79.0 | 93.8   | FT: 51.4 - PT: 89.3 |  ? |
  ViT-B    | 16x5x3  | 81.5  | 95.1  | FT: 196 - PT: 341 |  ? |
  ViT-L    | 16x5x3  | 85.2  | 96.8  | FT: 681 - PT: 1200 |  ? |
  ViT-H    | 16x5x3  | 86.6 | 97.1   | FT: 2360 - PT: ? |  ? |

<sup>?* Official `ViT-H` backbone of VideoMAE has weight issue in pretrained model, details https://github.com/MCG-NJU/VideoMAE/issues/89</sup>

### Something-Something V2

For SSv2, VideoMAE is trained around **2400** epoch without **any extra data**.

| Backbone | \#Frame | Top-1 | Top-5 | Params (MB) | FLOPS |
| :------: | :-----: | :---: | :---: | :---: | :---: |
|  ViT-S    | 16x2x3 | 66.8 | 90.3 | FT: 51.3 - PT: 89.4 |  ? |
|  ViT-B    | 16x2x3 | 70.8  | 92.4  | FT: 196 - PT: 341 |  ? |


### UCF101

For UCF101, VideoMAE is trained around **3200** epoch without **any extra data**.

| Backbone | \#Frame | Top-1 | Top-5 | Params (MB) | FLOPS |
| :---: | :-----: | :---: | :---: | :---: | :---: |
|  ViT-B   |  16x5x3  | 91.3 |  98.5 | FT: 195 - PT: 341 |  ? |


# Visualization 

Masked Autoencoder with `mask_ratio=0.8` from pretrained self-supervised video-mae model.

![](./assets/k400.gif)

![](./assets/ssv2.gif)

![](./assets/ucf101.gif)

# XLA Compatible

All the variants of converted videomae `keras` models are XLA compatible. They are evaluated on **TPU-VM** to reproduce the official reported scores.

# TODO

- [x] Multi-GPU suppport.
- [x] TPU support.
- [ ] Self-supervised training mechanism.
- [ ] Convert to `Keras V3`to support multi-framework backend.
