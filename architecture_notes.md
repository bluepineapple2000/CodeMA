# UNet Architecture Notes

These notes summarize the current training setup for synthetic diffraction spot separation. They are intended as a working reference for debugging the pipeline and later writing the methods chapter.

## Current Task

The current model is no longer a binary spot segmentation model. It is trained to separate one synthetic overlapped diffraction patch into two individual spot-intensity images.

Each training sample is read from:

```text
../data_esrf/augmented_spot_patches.h5
```

Each HDF5 sample group is expected to contain:

```text
image        [H, W]        overlapped input intensity image
spot_images  [2, H, W]     two target intensity images, one per source spot
```

The model input is one grayscale channel:

```text
input:  [B, 1, H, W]
target: [B, 2, H, W]
```

The default augmented patch size is `384 x 384`. With the current default `--scale 1.0`, training receives the full `384 x 384` inputs and targets.

## Augmentation Data

The augmentation notebook is `augment_data/augmentation.ipynb`.

The current augmentation process:

- loads confirmed isolated spots from `confirmed_spots_merged.h5`, falling back to `confirmed_spots_all.h5`
- loads the matching raw detector frame for each spot
- aligns the stored mask and bounding box with the raw frame crop
- estimates a local background from pixels outside the mask
- subtracts that local background from the crop
- sets pixels outside the mask to zero
- clips negative values to zero
- creates random spot variants using flips and arbitrary rotations
- combines two spots from the same dataset/material into one overlapped input image
- stores the two individual source spot images as separate target channels
- writes fixed-size `384 x 384` samples to `augmented_spot_patches.h5`

The saved target is intensity-valued, not binary. This is important: the desired output is not just spot location, but which intensity belongs to each original spot.

## Intensity Normalization

The current dataset class is `H5SpotSeparationDataset` in `train.py`.

For each sample, `normalize_image_and_targets` computes percentile limits from the input image:

```python
lo, hi = np.percentile(image[finite], [1, 99.9])
```

The input image is clipped to `[lo, hi]` and scaled to `[0, 1]`. The two target channels are clipped and scaled with the same `lo` and `hi`.

This preserves relative intensity inside a sample reasonably well, while removing absolute intensity differences between different samples. It also clips very bright peaks. Recent checks showed that changing the upper percentile from `99.8` to `99.9` reduced ratio distortion:

```text
raw spot ratio median:              2.700
normalized spot ratio median:       2.706
normalized/raw ratio-change range:  0.944 to 1.057
samples with clipped target pixels: 282 / 500
```

This suggests the current normalization is probably acceptable as a baseline, but peak clipping remains something to monitor.

## Model Overview

The model is a U-Net-style fully convolutional neural network implemented in `unet_model.py` using blocks from `unet_parts.py`.

The network has:

```text
n_channels = 1
n_classes  = 2
```

The two output channels are unordered. Channel 0 does not always correspond to a specific physical donor spot; the loss handles this by comparing both possible channel assignments.

## Current Architecture

The current network structure is:

```text
Input:       [B, 1, H, W]
inc:         DoubleConv(1 -> 64)
down1:       MaxPool2d(2) + DoubleConv(64 -> 128)
down2:       MaxPool2d(2) + DoubleConv(128 -> 256)
down3:       MaxPool2d(2) + DoubleConv(256 -> 512)
down4:       MaxPool2d(2) + DoubleConv(512 -> 1024)  # or 512 with bilinear=True
up1:         Upsample/ConvTranspose + skip concat + DoubleConv(... -> 512)
up2:         Upsample/ConvTranspose + skip concat + DoubleConv(... -> 256)
up3:         Upsample/ConvTranspose + skip concat + DoubleConv(... -> 128)
up4:         Upsample/ConvTranspose + skip concat + DoubleConv(... -> 64)
outc:        1x1 convolution, 64 -> 2
Output:      [B, 2, H, W] raw logits
```

For a `384 x 384` input without downscaling, the spatial sizes are approximately:

```text
384 -> 192 -> 96 -> 48 -> 24 -> 48 -> 96 -> 192 -> 384
```

If `--scale 0.5` is used for a smaller or faster experiment, the network receives `192 x 192` tensors:

```text
192 -> 96 -> 48 -> 24 -> 12 -> 24 -> 48 -> 96 -> 192


```

The default command-line option uses transposed-convolution upsampling unless `--bilinear` is passed.

## Building Blocks

Each `DoubleConv` block consists of:

```text
3x3 convolution -> BatchNorm2d -> ReLU
3x3 convolution -> BatchNorm2d -> ReLU
```

The downsampling blocks use `MaxPool2d(2)` followed by `DoubleConv`.

The upsampling blocks use either:

- `ConvTranspose2d` followed by concatenation with the encoder skip feature map, or
- bilinear upsampling followed by concatenation, if `--bilinear` is enabled

The final `OutConv` is a `1x1` convolution that maps 64 decoder channels to two output channels.

## Output Activation

The model itself returns raw logits. The training code converts these logits to non-negative intensity predictions with:

```python
prediction = F.softplus(logits)
```

`softplus` is used instead of `sigmoid` because the task is intensity regression, not binary classification. It keeps predictions non-negative while avoiding a hard upper bound inside the model. In practice, the normalized targets are usually in `[0, 1]` because of preprocessing.

## Current Loss

The current loss is a permutation-invariant combination of a weighted Tversky-style mask loss and a foreground intensity loss:

```text
total = mask_tversky + 0.2 * intensity
```

The two output channels are still unordered. For every sample, the loss computes both possible assignments and keeps the lower one:

```text
direct:  pred_0 -> target_0, pred_1 -> target_1
swapped: pred_0 -> target_1, pred_1 -> target_0
loss:    min(direct, swapped)
```

This allows the network to decide which output channel contains which spot, while still forcing the two channels to explain two different target spots.

The mask loss derives binary spot masks from the intensity targets:

```python
target_mask = target > 1e-4
pred_mask = prediction.clamp(0.0, 1.0)
```

For a matched output channel, pixels are treated as follows:

```text
own spot pixel:       positive target
true overlap pixel:   positive target for both spot channels
other-spot-only pixel: false positive with high weight
plain background:     false positive with very low weight
```

The weighted Tversky score is:

```text
TP / (TP + alpha * FP + beta * FN)
```

with the current defaults:

```text
alpha = 0.7
beta = 0.3
background false-positive weight = 0.02
other-spot false-positive weight = 2.0
```

This means the huge empty background is barely counted, but predicting spot 1 in a pixel that belongs only to spot 2 is punished strongly. Pixels where both target spots truly overlap are positive for both channels and are not treated as duplicate errors.

The intensity term is a foreground-only L1 loss. It compares predicted intensity to the matched target intensity only where that target spot exists. True overlap pixels are included with a reduced weight:

```text
overlap intensity weight = 0.5
```

There is currently no reconstruction loss. The earlier reconstruction term encouraged `pred_0 + pred_1` to match the input image, but that can be satisfied by duplicate or averaged outputs and does not directly enforce separation. The new loss focuses on assigning pixels and intensities to the correct spot channel.

Each component is logged to TensorBoard under:

```text
Loss_parts/train_mask_tversky
Loss_parts/train_intensity
```

## Training Setup

The main training script is `train.py`.

Current defaults:

```text
epochs:          5
batch size:      1
learning rate:   1e-4
validation:      10%
image scale:     1.0
optimizer:       AdamW
weight decay:    1e-8
gradient clip:   1.0
scheduler:       ReduceLROnPlateau(mode="min", patience=5)
mixed precision: off unless --amp is passed
```

Validation uses the same composite separation loss. The split is made with `random_split` and a fixed seed of `0`.

The script logs batch loss, epoch loss, validation loss, weights, gradients, learning rate, and image previews to TensorBoard.

## Prediction Previews

After each epoch, `save_prediction_previews` writes validation preview PNGs to:

```text
prediction_previews/<run_name>/epoch_XXX/
```

Each preview shows:

- normalized input
- true spot 1
- predicted spot 1
- spot 1 absolute error
- true target sum
- true spot 2
- predicted spot 2
- summed prediction error

Before plotting, prediction channels are aligned to the target channels with the same direct-vs-swapped Tversky-plus-intensity assignment used by the loss.

## Important Historical Architecture Change

An earlier version used an aggressive strided initial block:

```python
self.inc = DoubleConv(n_channels, 64, stride=4, padding=3)
```

This was removed. For `384 x 384` patches, especially with `--scale 0.5`, applying stride inside both convolutions of `DoubleConv` reduced spatial resolution too quickly and could collapse encoder features to `1 x 1`. With batch size 1, `BatchNorm2d` then failed because it could not compute training statistics from only one value per channel.

The current input block is:

```python
self.inc = DoubleConv(n_channels, 64)
```

Downsampling is now handled only by the four explicit max-pooling stages.

## Current Baseline Interpretation

The current pipeline is best described as:

```text
synthetic overlapping diffraction spots
-> per-sample robust intensity normalization
-> two-channel U-Net intensity regression
-> softplus non-negative prediction
-> permutation-invariant weighted Tversky mask assignment
-> foreground-only matched intensity regression
```

The current baseline directly addresses the suspected failure modes of background domination and weak spot assignment pressure. It intentionally removes reconstruction loss because reconstructing the input can reward duplicated or averaged outputs instead of true separation.

If training still shows little or no improvement, the next issues to check are:

- whether the model can overfit a tiny subset of about 20 samples
- whether batch size 1 plus BatchNorm gives noisy normalization statistics
- whether the Tversky weights need stronger wrong-spot penalties or less false-negative tolerance
- whether `--scale 1.0` causes memory pressure and should be combined with `--amp`
- whether the synthetic target is ambiguous in heavily overlapped cases
