# UNet Architecture Notes

These notes summarize the segmentation network used for the HDF5 spot patch training data. They are intended as a reminder for writing the architecture/methods chapter of the thesis.

## Task and Input Data

The model is trained for binary spot segmentation. Each training sample is read from the HDF5 file `../data_esrf/augmented_spot_patches.h5` and contains an `image` dataset and a corresponding binary `mask` dataset.

The current augmented patch size is approximately `384 x 384` pixels. In `train.py`, each image is loaded as a single grayscale channel, normalized to the range `[0, 1]`, and returned as a tensor with shape `[1, H, W]`. The target mask is binarized and returned with the same spatial size.

## Model Overview

The model is a U-Net-style fully convolutional neural network. It follows the common encoder-decoder structure:

- Encoder path: extracts increasingly abstract features while reducing spatial resolution.
- Bottleneck: represents the image at the smallest spatial resolution and largest channel depth.
- Decoder path: upsamples feature maps back to the original resolution.
- Skip connections: concatenate encoder features with decoder features at matching resolutions to preserve spatial detail.
- Output layer: maps the final feature map to one output channel containing the raw segmentation logits.

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

The final `OutConv` is a `1x1` convolution that converts the 64 decoder channels to one logit channel. During training, `BCEWithLogitsLoss` is applied directly to these logits, and Dice loss is computed after applying a sigmoid.

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

The final architecture can be described as a modified U-Net for binary segmentation of grayscale diffraction spot patches. The main adaptation compared with the earlier large-image setup is the removal of aggressive initial strided convolutions. This change matches the smaller patch-based input data and prevents excessive loss of spatial resolution in the encoder.

The U-Net structure is suitable for this task because the encoder captures contextual information about the spot, while the skip connections help the decoder recover precise spatial boundaries in the segmentation mask.
