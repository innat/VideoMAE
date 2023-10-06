# VideoMAE Model Zoo

**Note**

- `#Frame = #input_frame x #clip x #crop.`
- `#input_frame` means how many frames are input for model during the test phase.
- `#crop` means **spatial crops** (e.g., 3 for left/right/center crop).
- `#clip` means **temporal clips** (e.g., 5 means repeted temporal sampling five clips with different start indices).

The official results of torch VideoMAE finetuned with I3D dense sampling on Kinetics400 and TSN uniform sampling on Something-Something V2, respectively.


The name of the weight is followed as `model_{size}_{input_frame}x{input_size}_{mode}`, where `size` can be `S/B/L/H`, and `input_frame` for videomae is `16` for both `Fine Tuned (FT)` and `Pre Trained (PT)` models.

```
TFVideoMAE_B_16x224_FT
TFVideoMAE_B_16x224_PT
```

### Kinetics-400

For Kinetrics-400, VideoMAE is trained around **1600** epoch without **any extra data**. The following checkpoints are available in both tensorflow [`SavedModel`](https://www.tensorflow.org/guide/saved_model) and [`h5`](https://keras.io/api/saving/weights_saving_and_loading/#save_weights-method) format.


| Backbone | \#Frame | Pre-train | Fine-tune | Top-1 | Top-5 |
 | :--: | :--: | :--: | :--: | :---: | :---: |
  ViT-S    | 16x5x3  | [savedmodel]() | [h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_S_16x224_FT.h5) | 79.0 | 93.8   |
  ViT-B    | 16x5x3  | [savedmodel]() | [h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_16x224_FT.h5) | 81.5  | 95.1  |
  ViT-L    | 16x5x3  | checkpoint | [h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_L_16x224_FT.h5) | 85.2  | 96.8  |
  ViT-H    | 16x5x3  | checkpoint | checkpoint | 86.6 | 97.1   |


### Something-Something V2

For SSv2, VideoMAE is trained around **2400** epoch without **any extra data**.

| Backbone | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
|  ViT-S    | 16x2x3  | ? | ? | 66.8 | 90.3 |
|  ViT-B    | 16x2x3  | ? | ? | 70.8  | 92.4  |


### UCF101

For UCF101, VideoMAE is trained around **3200** epoch without **any extra data**.

| Backbone | \#Frame |  Pre-train  |  Fine-tune   | Top-1 | Top-5 |
| :---: | :-----: | :----: | :----: | :---: | :---: |
|  ViT-B   |  16x5x3  | ?  | ? | 91.3 |  98.5 |
