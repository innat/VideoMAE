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
  ViT-H    | 16x5x3  | ? | [SavedModel](https://drive.google.com/drive/folders/1DdwGd-EXD0Rc-05mirZOU3W7b8LlHYJ7?usp=sharing)/[h5](https://drive.google.com/file/d/1ZS7fWw3SbgKpLAdN7QQLm9_pIomwYEtI/view?usp=sharing) | 86.6 | 97.1   |

<sup>?* Official `ViT-H` backbone of VideoMAE has weight issue in pretrained model, details https://github.com/MCG-NJU/VideoMAE/issues/89</sup>

### Something-Something V2

For SSv2, VideoMAE is trained around **2400** epoch without **any extra data**.

| Backbone | \#Frame |                          Pre-train                           |                          Fine-tune                           | Top-1 | Top-5 |
| :------: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: | :---: |
|  ViT-S    | 16x2x3  | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_S_SSv2_16x224_PT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_S_SSv2_16x224_PT.h5) | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_S_SSv2_16x224_FT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_S_SSv2_16x224_FT.h5) | 66.8 | 90.3 |
|  ViT-B    | 16x2x3  | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_B_SSv2_16x224_PT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_SSv2_16x224_PT.h5) | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_B_SSv2_16x224_FT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_SSv2_16x224_FT.h5) | 70.8  | 92.4  |


### UCF101

For UCF101, VideoMAE is trained around **3200** epoch without **any extra data**.

| Backbone | \#Frame |  Pre-train  |  Fine-tune   | Top-1 | Top-5 |
| :---: | :-----: | :----: | :----: | :---: | :---: |
|  ViT-B   |  16x5x3  | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_B_UCF_16x224_PT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_UCF_16x224_PT.h5)  | [SavedModel](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_B_UCF_16x224_FT.zip)/[h5](https://github.com/innat/VideoMAE/releases/download/v1.0/TFVideoMAE_B_UCF_16x224_FT.h5) | 91.3 |  98.5 |



# Weight Comparison

The `torch` video-mae model can be loaded from the official [repo](https://github.com/MCG-NJU/VideoMAE). Following are some quick test of both implementation, showing logit matching. Please note, here only fine-tune models (**UCF-101**) are used to demonstrate. 

```python
inputs_pt = torch.tensor(np.random.rand(4, 3, 16, 224, 224).astype('float32'))
inputs_tf = inputs_pt.detach().numpy().transpose(0,2,3,4,1)

model_pt.eval()
y_pred_pt = model_pt(inputs_pt.float()) # UCF-101 model
y_pred_pt = y_pred_pt.detach().numpy()
y_pred_pt.shape
(4, 101)

y_pred_tf = model_tf(inputs_tf, training=False)
y_pred_tf = y_pred_tf.numpy()
y_pred__tf.shape
(4, 101)

np.testing.assert_allclose(
    y_pred_tf, 
    y_pred_pt, 
    1e-5, 1e-5
) # OK
```

**Saving and Reloading Weight - check if saving and reloading is safe.**

```python
model_tf.save_weights(checkpoint_name + '.h5')
new_model_tf = build_video_mae(...)
new_model_tf.load_weights(checkpoint_name + '.h5')

# Let's check: weight matching
assert len(model_tf.weights) == len(new_model_tf.weights)
for a, b in zip(model_tf.weights, new_model_tf.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy()) # OK
    
# Let's check: inference matching
test_input = tf.random.normal(
    [1, 16, 224, 224, 3], 0, 1, tf.float32
)
tf.nest.map_structure(
    np.testing.assert_allclose,
    model_tf.predict(test_input),
    new_model_tf.predict(test_input),
) # OK
```

**Saving and Reloading TF SavedModel - check if saving and reloading is safe.**

```python
model_tf.save(checkpoint_name)
loaded_model  = keras.models.load_model(
   checkpoint_name
)

assert len(model_tf.weights) == len(loaded_model.weights)
for a, b in zip(model_tf.weights, loaded_model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy()) # OK

# Let's check: inference matching
test_input = tf.random.normal(
    [1, 16, 224, 224, 3], 0, 1, tf.float32
)
tf.nest.map_structure(
    np.testing.assert_allclose,
    model_tf.predict(test_input),
    loaded_model.predict(test_input),
) # OK
```

**Weight matching between TF SavedModel vs `torch` model.**

```python
y_pred_pt = model_pt(inputs_pt.float())
y_pred_tf = loaded_model(inputs_tf, training=False)
print(y_pred_pt.shape, y_pred_tf.shape)
np.testing.assert_allclose(
    y_pred_tf.numpy(), 
    y_pred_pt.detach().numpy(), 
    1e-5, 1e-5
) # OK
```

**XLA compatible - TF SavedModel**

```python
call_fn = tf.function(loaded_model, jit_compile=True)
%timeit _ = call_fn(inputs_tf, training=False)
```
