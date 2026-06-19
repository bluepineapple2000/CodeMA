# U-Net Architecture Notes

These notes summarize the current U-Net model used for HDF5 diffraction spot separation. They are intended as a reminder for the architecture and methods chapter of the thesis.

## Task and Input Data

The model is trained to separate two overlapping diffraction spots into two source-specific intensity images and two source-specific spatial masks. Each training sample is read from:

```text
data/augmented_spot_patches_with_masks.h5
```

Each HDF5 sample group contains:

- `image`: the combined single-channel diffraction patch with shape `[H, W]`.
- `spot_images`: two ground-truth intensity frames with shape `[2, H, W]`.
- `spot_masks`: two binary ground-truth masks with shape `[2, H, W]`.

In `train.py`, the input image is returned as a tensor with shape `[1, H, W]`. The intensity target has shape `[2, H, W]`, and the mask target also has shape `[2, H, W]`.

The current patch size is approximately `384 x 384` pixels before optional training downscaling. With the default training scale `--scale 0.5`, the network receives `192 x 192` tensors. Intensity images are resized with bilinear interpolation, while masks are resized with nearest-neighbor interpolation so their binary structure is preserved.

## Normalization

The normalization is applied per sample. The lower and upper intensity limits are computed from the combined input image using the 1st and 99.9th percentiles. The input image and both target spot-intensity frames are clipped to these same limits and divided by the same intensity range.

This shared normalization is important because the loss compares predicted and target intensities. The masks are not intensity-normalized; they are loaded as binary targets and converted to float tensors containing `0` and `1`.

## Model Overview

The model is a U-Net-style fully convolutional neural network. It follows the usual encoder-decoder structure:

- Encoder path: extracts increasingly abstract features while reducing spatial resolution.
- Bottleneck: represents the image at the smallest spatial resolution and largest channel depth.
- Decoder path: upsamples feature maps back to the original resolution.
- Skip connections: concatenate encoder features with decoder features at matching resolutions.
- Output layer: maps the final feature map to four output channels.

The model is implemented in `unet_model.py` using reusable blocks from `unet_parts.py`.

## Current Architecture

The network uses one input channel and four output channels:

```text
Input:       [B, 1, H, W]
inc:         DoubleConv(1 -> 64)
down1:       MaxPool2d(2) + DoubleConv(64 -> 128)
down2:       MaxPool2d(2) + DoubleConv(128 -> 256)
down3:       MaxPool2d(2) + DoubleConv(256 -> 512)
down4:       MaxPool2d(2) + DoubleConv(512 -> 1024)  # or 512 with bilinear=True
up1:         Upsample/ConvTranspose + DoubleConv(... -> 512)
up2:         Upsample/ConvTranspose + DoubleConv(... -> 256)
up3:         Upsample/ConvTranspose + DoubleConv(... -> 128)
up4:         Upsample/ConvTranspose + DoubleConv(... -> 64)
outc:        1x1 convolution, 64 -> 4
Output:      [B, 4, H, W] logits
```

For a `384 x 384` input without additional downscaling, the spatial sizes are approximately:

```text
384 -> 192 -> 96 -> 48 -> 24 -> 48 -> 96 -> 192 -> 384
```

With the default `--scale 0.5`, the spatial sizes are approximately:

```text
192 -> 96 -> 48 -> 24 -> 12 -> 24 -> 48 -> 96 -> 192
```

Both cases keep enough spatial values in the bottleneck for stable training with batch normalization.

## Output Interpretation

The four output channels are interpreted as two mask channels and two intensity-splitting channels:

```python
mask_logits = logits[:, 0:2]
fraction_logits = logits[:, 2:4]

predicted_masks = sigmoid(mask_logits)
fractions = softmax(fraction_logits, dim=1)
predicted_spot_images = fractions * image
```

The two predicted masks are supervised against `spot_masks`. The two predicted intensity images are supervised against `spot_images`.

The softmax over the two fraction channels makes the intensity decomposition conservative:

```text
fraction_1 + fraction_2 = 1
predicted_spot_1 + predicted_spot_2 = image
```

up to normal floating-point precision. This is the main physics-inspired constraint in the current implementation: the model is encouraged by construction to distribute the measured input intensity between the two spots rather than inventing extra total intensity.

## Training Objective and Loss

The current training objective is symmetric: both spots are predicted directly, and both spots are included in the loss. This differs from the earlier one-output model, where spot 1 was predicted directly and spot 2 was treated as a residual.

The total loss is:

```text
L_total =
    L_intensity_dice
  + 0.25 * L_foreground_L1
  + 0.50 * L_mask_BCE
  + 0.50 * L_mask_dice
  + 0.10 * L_reconstruction
  + 0.05 * L_outside_mask
  + 0.01 * L_background
```

where:

```text
predicted_spots = softmax(fraction_logits, channel) * image
predicted_masks = sigmoid(mask_logits)
target_spots    = spot_images
target_masks    = spot_masks
```

### Intensity Dice Term

The intensity Dice term compares the two predicted spot-intensity images with the two target spot-intensity images:

```text
Dice = (2 * sum(prediction * target) + smooth) / (sum(prediction) + sum(target) + smooth)
L_intensity_dice = 1 - mean(Dice)
```

This is a soft intensity Dice, not a hard binary Dice. Brighter pixels contribute more strongly, so the term rewards placing intensity in the correct source-specific spot region.

### Foreground-Weighted L1 Term

The foreground-weighted L1 term penalizes absolute intensity error:

```text
error = abs(predicted_spots - target_spots)
foreground = target_spots > 1e-4
weight = 1 + 8 * foreground
L_foreground_L1 = sum(error * weight) / sum(weight)
```

This term complements the Dice term. Dice rewards overlap and relative support, while L1 encourages the predicted intensity values to match the normalized ground-truth intensities. The foreground weighting is needed because most pixels are background; without it, the model could achieve a low average error by under-predicting signal.

### Mask BCE Term

The mask binary cross-entropy term supervises the two raw mask-logit channels:

```text
L_mask_BCE = BCEWithLogits(mask_logits, target_masks)
```

This term encourages each predicted mask channel to represent the spatial support of its corresponding spot.

### Mask Dice Term

The mask Dice term compares the sigmoid mask probabilities with the binary mask targets:

```text
predicted_masks = sigmoid(mask_logits)
L_mask_dice = 1 - mean(Dice(predicted_masks, target_masks))
```

This is useful because spot masks occupy a relatively small fraction of the image. BCE provides pixelwise supervision, while Dice reduces the effect of background dominance.

### Reconstruction Consistency Term

The reconstruction term compares the sum of the predicted spot intensities with the input image:

```text
reconstruction = predicted_spot_1 + predicted_spot_2
L_reconstruction = mean(abs(reconstruction - image))
```

Because the intensity fractions are produced with a softmax, this term should usually be small. It is retained as an explicit diagnostic and regularizer for the physical conservation idea: the separated spots should reconstruct the measured combined diffraction patch.

### Outside-Mask Penalty

The outside-mask term penalizes predicted source intensity outside the corresponding ground-truth mask:

```text
L_outside_mask = mean(abs(predicted_spots * (1 - target_masks)))
```

This encourages each predicted intensity channel to stay inside the spatial support of its own spot. It links the intensity output and the mask output: intensities should not leak into regions that are not assigned to that spot.

### Background Term

The background term penalizes predicted source intensity where the combined input image is essentially empty:

```text
background = image <= 1e-4
L_background = mean(abs(predicted_spots) over background pixels)
```

This discourages hallucinated signal outside measured diffraction intensity. Its coefficient is small because it is a regularizer rather than the main supervision signal.

## Is This a PIN or PINN?

The current implementation is best described as a physics-informed or physics-constrained U-Net, not a classical PINN.

A classical physics-informed neural network usually includes residuals of governing equations, for example a PDE, ODE, or differentiable physical model evaluated at collocation points. This project does not currently include a PDE residual or an explicit diffraction forward model.

However, the model does include physically motivated constraints:

- nonnegative spot intensities through `softmax(fraction_logits) * image`,
- conservation of measured intensity through `predicted_spot_1 + predicted_spot_2 = image`,
- reconstruction consistency in the loss,
- spatial support supervision through `spot_masks`,
- outside-mask and background penalties to discourage physically implausible signal leakage.

So, for thesis wording, a careful formulation would be:

```text
The model is a physics-informed U-Net for two-source diffraction spot separation.
It is not a classical PINN with a differential-equation residual; instead, physical
prior knowledge is incorporated through intensity-conservation and spatial-support
constraints in the output parameterization and loss function.
```

## Architecture Change From the Earlier Version

An earlier version of the network used this initial layer:

```python
self.inc = DoubleConv(n_channels, 64, stride=4, padding=3)
```

This was useful when input images were much larger, because it performed strong early downsampling. However, `DoubleConv` applies two convolutions, so using `stride=4` twice reduced the spatial size very aggressively before the normal U-Net encoder stages.

For the current `384 x 384` patches, especially with `--scale 0.5`, this caused the encoder feature maps to collapse to `1 x 1` before or around the deepest encoder stage. With batch size 1, a `BatchNorm2d` layer then received a tensor like:

```text
torch.Size([1, 512, 1, 1])
```

Batch normalization cannot compute training statistics from only one value per channel, which caused the error:

```text
ValueError: Expected more than 1 value per channel when training
```

The architecture was therefore adjusted to use the standard U-Net input block:

```python
self.inc = DoubleConv(n_channels, 64)
```

This keeps the original resolution through the first convolutional block and leaves downsampling to the four explicit `MaxPool2d(2)` operations in the encoder.

## Notes for Thesis Writing

The final architecture can be described as a modified U-Net for physics-informed two-source intensity and mask separation from overlapping grayscale diffraction spot patches. The model predicts both spot masks and spot intensities. The intensity channels are parameterized as a softmax split of the input image, which embeds an intensity-conservation prior directly into the prediction.

The loss should be described as a multi-task objective combining intensity reconstruction, mask segmentation, and physically motivated regularization. The most important distinction from the earlier implementation is that the current model is symmetric: both spots are predicted and supervised directly, rather than predicting only one spot and treating the other as a residual.
