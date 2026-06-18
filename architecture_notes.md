# UNet Architecture Notes

These notes summarize the U-Net model used for the HDF5 diffraction spot separation training data. They are intended as a reminder for writing the architecture/methods chapter of the thesis.

## Task and Input Data

The model is trained for intensity-based separation of two overlapping diffraction spots. Each training sample is read from the HDF5 file `data/augmented_spot_patches.h5` and contains:

- `image`: the combined single-channel diffraction patch.
- `spot_images`: two target intensity frames with shape `[2, H, W]`, where channel 0 is the reference spot and channel 1 is the second spot.

The current augmented patch size is approximately `384 x 384` pixels. In `train.py`, each input image is loaded as a single grayscale channel and returned as a tensor with shape `[1, H, W]`. The target tensor has shape `[2, H, W]`.

The normalization is applied per sample. The lower and upper intensity limits are computed from the combined input image using the 1st and 99.9th percentiles. The input image and both target spot-intensity frames are clipped to these same limits and divided by the same intensity range. This is important because the loss compares predicted and target intensities, not only binary foreground masks. In the current generated archive, the lower limit is zero for all checked samples, so the normalization preserves the decomposition `spot_1 + spot_2 = image` up to the small effect of high-intensity clipping at the 99.9th percentile.

## Model Overview

The model is a U-Net-style fully convolutional neural network. It follows the common encoder-decoder structure:

- Encoder path: extracts increasingly abstract features while reducing spatial resolution.
- Bottleneck: represents the image at the smallest spatial resolution and largest channel depth.
- Decoder path: upsamples feature maps back to the original resolution.
- Skip connections: concatenate encoder features with decoder features at matching resolutions to preserve spatial detail.
- Output layer: maps the final feature map to one output channel containing raw logits for the first spot.

The model is implemented in `unet_model.py` using reusable blocks from `unet_parts.py`.

## Current Architecture

The current network uses one input channel and one output channel:

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
outc:        1x1 convolution, 64 -> 1
Output:      [B, 1, H, W] logits
```

For a `384 x 384` input without additional training downscaling, the spatial sizes are approximately:

```text
384 -> 192 -> 96 -> 48 -> 24 -> 48 -> 96 -> 192 -> 384
```

If training uses the default `--scale 0.5`, the network receives `192 x 192` inputs and the spatial sizes are approximately:

```text
192 -> 96 -> 48 -> 24 -> 12 -> 24 -> 48 -> 96 -> 192
```

Both cases keep enough spatial values in the bottleneck for stable training with batch normalization.

## Building Blocks

Each `DoubleConv` block consists of two repetitions of:

```text
3x3 convolution -> BatchNorm2d -> ReLU
```

The downsampling blocks use `MaxPool2d(2)` followed by `DoubleConv`. The upsampling blocks use either bilinear upsampling or transposed convolution, followed by concatenation with the corresponding encoder feature map and another `DoubleConv`.

The final `OutConv` is a `1x1` convolution that converts the 64 decoder channels to one logit channel. This output is not a two-channel segmentation mask. It is a single logit image for the first spot. A sigmoid converts the logits to a soft fraction in `[0, 1]`, and this fraction is multiplied by the normalized input image to obtain the predicted first-spot intensity:

```text
predicted_spot_1 = sigmoid(logits) * image
```

For visualization and diagnostics, a second spot can be constructed as the residual:

```text
predicted_spot_2 = max(image - predicted_spot_1, 0)
```

The residual image is shown in TensorBoard and in the saved prediction previews, but it is not a separately predicted network output.

## Training Objective and Loss

The training objective supervises only the first predicted spot image against the first ground-truth spot frame. This matches the architecture: the network has one output channel, and the loss is applied directly to that one predicted intensity image.

The total loss is:

```text
L_total = L_dice(first_spot) + 0.25 * L_foreground_L1(first_spot) + 0.01 * L_background(first_spot)
```

where:

```text
first_spot_prediction = sigmoid(logits) * image
first_spot_target     = spot_images[0]
```

### Soft Dice Term

The soft Dice term compares the predicted first-spot intensity image with the first ground-truth spot image:

```text
Dice = (2 * sum(prediction * target) + smooth) / (sum(prediction) + sum(target) + smooth)
L_dice = 1 - Dice
```

This is a soft intensity Dice rather than a hard binary Dice. It rewards spatial overlap between the predicted and target intensity distributions while still allowing differentiable training. Because the target is an intensity frame, brighter pixels contribute more strongly than weak pixels.

Using Dice is useful for diffraction spots because the foreground region is small compared with the mostly empty background. A pure pixel-average loss could be dominated by background pixels, while Dice emphasizes whether the model places intensity in the correct spot region.

### Foreground-Weighted L1 Term

The foreground-weighted L1 term penalizes absolute intensity differences:

```text
error = abs(prediction - target)
foreground = target > 1e-4
weight = 1 + 8 * foreground
L_foreground_L1 = sum(error * weight) / sum(weight)
```

This term gives extra weight to pixels where the first ground-truth spot is present. It complements the Dice term: Dice mainly captures overlap and relative support, while L1 encourages the predicted intensity values to match the normalized target intensities.

The weighting is important because most pixels are close to zero. Without foreground weighting, a model could obtain a deceptively low average error by predicting too little signal everywhere.

### Background Term

The background term penalizes predicted first-spot intensity in regions where the combined input image is essentially empty:

```text
background = image <= 1e-4
L_background = mean(abs(prediction) over background pixels)
```

This discourages hallucinated spot intensity outside the measured diffraction signal. The term has a small coefficient (`0.01`) because it is a regularizer rather than the main supervision signal.

### Role of the Second Spot

The second ground-truth frame remains important for interpretation, but it is not part of the optimized loss. During logging, the code still forms:

```text
predicted_spot_2 = max(image - predicted_spot_1, 0)
```

and compares this residual prediction with `spot_images[1]` for TensorBoard metrics and preview images. These diagnostics help check whether the first-spot prediction leaves a plausible residual second spot.

This design makes the objective deliberately asymmetric. The model is trained to identify the chosen reference spot, and the second spot is treated as whatever intensity remains in the input image. This is appropriate if the scientific or downstream goal is to recover one selected spot from an overlap, but it should be described explicitly because it differs from a two-output separation model.

### Alternative Loss Design

An alternative would be to include both target frames in the loss:

```text
predicted_spots = [predicted_spot_1, image - predicted_spot_1]
target_spots    = [spot_images[0], spot_images[1]]
```

and compute Dice and L1 over both channels. That would force the residual second spot to match its ground truth directly. The current implementation does not do this. It only trains the network output against the first target frame, while retaining the second frame for qualitative and quantitative monitoring.

For thesis writing, this distinction is crucial: the implemented model is a one-output reference-spot extractor with residual visualization, not a fully symmetric two-channel source-separation network.

## Architecture Change From the Earlier Version

An earlier version of the network used this initial layer:

```python
self.inc = DoubleConv(n_channels, 64, stride=4, padding=3)
```

This was useful when the input images were much larger, because it performed strong early downsampling. However, `DoubleConv` applies two convolutions, so using `stride=4` twice reduced the spatial size very aggressively before the normal U-Net encoder stages.

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

The final architecture can be described as a modified U-Net for reference-spot intensity extraction from overlapping grayscale diffraction spot patches. The main adaptation compared with the earlier large-image setup is the removal of aggressive initial strided convolutions. This change matches the smaller patch-based input data and prevents excessive loss of spatial resolution in the encoder.

The U-Net structure is suitable for this task because the encoder captures contextual information about the overlapping spot pattern, while the skip connections help the decoder recover precise spatial intensity structure. The current loss should be described as an asymmetric first-spot objective: the network predicts one spot directly, and the second spot is retained as a residual diagnostic rather than a directly supervised output.
