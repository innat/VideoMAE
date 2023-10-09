# VideoMAE Model Zoo

**Note**

- `#Frame = #input_frame x #clip x #crop.`
- `#input_frame` means how many frames are input for model during the test phase.
- `#crop` means **spatial crops** (e.g., 3 for left/right/center crop).
- `#clip` means **temporal clips** (e.g., 5 means repeted temporal sampling five clips with different start indices).

### Kinetics-400

For Kinetrics-400, VideoMAE is trained around **1600** epoch without **any extra data**. The following checkpoints are available in both tensorflow [`SavedModel`](https://www.tensorflow.org/guide/saved_model) and [`h5`](https://keras.io/api/saving/weights_saving_and_loading/#save_weights-method) format.



| Backbone | \#Frame | Pre-train | Fine-tune | Top-1 | Top-5 |
 | :--: | :--: | :--: | :--: | :---: | :---: |
  ViT-S    | 16x5x3  | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_S_K400_16x224_PT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_S_K400_16x224_PT.h5) | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_S_K400_16x224_FT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_S_16x224_FT.h5) | 79.0 | 93.8   |
  ViT-B    | 16x5x3  | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_B_K400_16x224_PT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_K400_16x224_PT.h5) | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_B_K400_16x224_FT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_K400_16x224_FT.h5) | 81.5  | 95.1  |
  ViT-L    | 16x5x3  | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_L_K400_16x224_PT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_L_K400_16x224_PT.h5) | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_L_K400_16x224_FT.h5)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_L_K400_16x224_FT.h5) | 85.2  | 96.8  |
  ViT-H    | 16x5x3  | ? | [SavedModel]()/[h5]() | 86.6 | 97.1   |


### Something-Something V2

For SSv2, VideoMAE is trained around **2400** epoch without **any extra data**.

| Backbone | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
|  ViT-S    | 16x2x3  | [SavedModel]()/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_S_SSv2_16x224_PT.h5) | [SavedModel]()/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_S_SSv2_16x224_FT.h5) | 66.8 | 90.3 |
|  ViT-B    | 16x2x3  | [SavedModel]()/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_SSv2_16x224_PT.h5) | [SavedModel]()/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_SSv2_16x224_FT.h5) | 70.8  | 92.4  |


### UCF101

For UCF101, VideoMAE is trained around **3200** epoch without **any extra data**.

| Backbone | \#Frame |  Pre-train  |  Fine-tune   | Top-1 | Top-5 |
| :---: | :-----: | :----: | :----: | :---: | :---: |
|  ViT-B   |  16x5x3  | [SavedModel]()/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_UCF_16x224_PT.h5)  | [SavedModel]()/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_UCF_16x224_FT.h5) | 91.3 |  98.5 |
