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

The four output channels are interpreted as two mask channels and two intensity channels:

```python
mask_logits = logits[:, 0:2]
intensity_logits = logits[:, 2:4]

predicted_masks = sigmoid(mask_logits)
predicted_spot_images = sigmoid(intensity_logits)
```

The two predicted masks are supervised against `spot_masks`. The two predicted intensity images are supervised against `spot_images`.

Important: in the current `train.py`, the intensity channels are independent sigmoid outputs. They are **not** produced by a softmax split of the input image. Therefore the current model does not guarantee:

```text
predicted_spot_1 + predicted_spot_2 = image
```

The conservation idea is only weakly represented through monitoring/preview plots and the background regularizer, not by the output parameterization.

## Training Objective and Loss

The current loss in `train.py` is implemented by `separation_loss_components`:

```text
L_total =
    L_mask_1
  + L_mask_2
  + L_intensity_1
  + L_intensity_2
  + 0.2 * L_background
```

where:

```text
predicted_masks      = sigmoid(mask_logits)
predicted_intensities = sigmoid(intensity_logits)
target_masks         = spot_masks
target_intensities   = spot_images
```

This branch is a four-output multitask model: two channels learn spatial masks, and two channels learn normalized spot intensities.

### Mask Loss

For each matched spot channel, the mask loss combines soft Dice and foreground-weighted binary cross-entropy:

```text
L_mask_i = L_dice_i + L_weighted_BCE_i
```

The Dice part is:

```text
Dice = (2 * sum(predicted_mask * target_mask) + smooth)
       / (sum(predicted_mask) + sum(target_mask) + smooth)
L_dice = 1 - mean(Dice)
```

The BCE part uses the raw `mask_logits` and downweights background pixels:

```python
weights = 1.0 where target_mask > 0.5
weights = 0.05 where target_mask <= 0.5
```

This tries to prevent the many background pixels from dominating the mask objective.

### Intensity Loss

For each matched spot channel, the intensity loss is foreground-only L1:

```text
L_intensity_i = mean(abs(predicted_intensity_i - target_intensity_i))
```

but only over pixels where the corresponding `target_mask_i > 0.5`.

This means intensity values are primarily supervised inside the true spot support. Pixels outside the target mask do not strongly affect the per-spot intensity loss.

### Background Loss

The background loss penalizes predicted intensity outside the corresponding ground-truth mask:

```text
background = target_mask <= 0.5
L_background = mean(abs(predicted_intensity) over background pixels)
```

It is weighted by `0.2` in the total loss. This is the main term that discourages intensity leakage outside the two spot masks.

### Channel Assignment

The current loss is **not permutation invariant**. It assumes:

```text
output mask/intensity channel 0 -> target spot 0
output mask/intensity channel 1 -> target spot 1
```

So if the two target spot channels are arbitrary or can swap meaning between samples, this loss can give contradictory supervision. This is a likely reason the training can plateau or produce confusing outputs. A permutation-invariant variant would compare both channel assignments and use the lower loss.

### Current Limitations

The current loss does not include the newer Tversky-style wrong-spot penalty discussed separately. It also does not use a softmax intensity split, so the model can predict both intensities independently. This gives the network flexibility, but it does not enforce conservation of the input intensity.

## Is This a PIN or PINN?

The current implementation is best described as a physics-informed or physics-constrained U-Net, not a classical PINN.

A classical physics-informed neural network usually includes residuals of governing equations, for example a PDE, ODE, or differentiable physical model evaluated at collocation points. This project does not currently include a PDE residual or an explicit diffraction forward model.

However, the model does include physically motivated structure:

- nonnegative mask probabilities through sigmoid mask outputs,
- normalized nonnegative intensity predictions through sigmoid intensity outputs,
- spatial support supervision through `spot_masks`,
- foreground-only intensity supervision inside the true masks,
- background penalties to discourage physically implausible signal leakage.

Unlike the earlier softmax-split PIN-style experiment, this branch does not enforce exact conservation of measured input intensity in the output parameterization.

So, for thesis wording, a careful formulation would be:

```text
The model is a physics-informed U-Net for two-source diffraction spot separation.
It is not a classical PINN with a differential-equation residual; instead, physical
prior knowledge is incorporated through nonnegative outputs, spatial-support
supervision, and background leakage penalties in the loss function.
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

The final architecture in this branch can be described as a modified U-Net for two-source intensity and mask separation from overlapping grayscale diffraction spot patches. The model predicts both spot masks and spot intensities. The current intensity channels are independent sigmoid outputs, not a softmax split of the input image, so exact intensity conservation is not embedded in this branch.

The loss should be described as a multi-task objective combining mask segmentation, foreground intensity regression, and background leakage regularization. The most important distinction from the earlier implementation is that the current model is symmetric: both spots are predicted and supervised directly, rather than predicting only one spot and treating the other as a residual.
