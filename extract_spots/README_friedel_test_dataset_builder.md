# Friedel Test Dataset Builder

This document describes the workflow implemented in [friedel_test_dataset_builder.ipynb](/var/home/natalieboehm/Documents/Studium/Master/Masterarbeit/Code/extract_spots/friedel_test_dataset_builder.ipynb) for constructing a manual test dataset of overlapping diffraction spots and their spatially separated Friedel-pair ground truths.

The notebook was designed for the following use case:

- An overlapping spot in one frame is chosen as the network input.
- Two manually selected Friedel-pair spots from offset frames are used as ground truth.
- All crops are exported as `128 x 128` HDF5 arrays.
- A preview step allows visual inspection before writing the final sample.

## Goal

The purpose of this workflow is to create evaluation samples for spot-separation models. Each sample contains:

- one overlapping input spot,
- its binary input segmentation mask,
- two isolated Friedel-pair intensity targets,
- two corresponding binary target masks.

This makes it possible to test whether a model can separate an overlapping diffraction signal into its two physically matching components.

## Input Data

The notebook uses two HDF5 sources per scan:

- the raw detector data:
  `../../data_esrf/<SCAN>/<main_file>.h5`
- the segmentation volume:
  `../../data_esrf/<SCAN>/segvol.h5`

The relevant dataset keys are:

- raw detector frames: `instrument/detector_0/data`
- segmentation labels: `segvol`

The raw data provides grayscale diffraction intensities. The segmentation volume provides the spot support that is used for labeling, masking, and crop extraction.

## Overview Of The Procedure

The workflow consists of five main stages:

1. Load the raw frame and segmentation frame for the selected input frame and Friedel frames.
2. Correct the display orientation so that raw data and segmentation labels are spatially consistent.
3. Extract connected components from `segvol` and assign spot numbers within each frame.
4. Build `128 x 128` crops for the input overlap and the two manually selected Friedel targets.
5. Preview and export the final sample as HDF5.

## 1. Manual Sample Definition

Each sample is defined explicitly in the configuration cell of the notebook:

- `SCAN`
- `INPUT_FRAME`
- `INPUT_SPOT_NUMBER`
- `FRIEDEL_TARGETS`

Example:

```python
SCAN = "Al"
INPUT_FRAME = 208
INPUT_SPOT_NUMBER = 2
FRIEDEL_TARGETS = [
    {"frame": 2009, "spot_number": 7, "name": "spot_1"},
    {"frame": 2009, "spot_number": 3, "name": "spot_2"},
]
```

This means:

- the overlapping input is spot `2` in frame `208`,
- the first ground-truth component is spot `7` in frame `2009`,
- the second ground-truth component is spot `3` in frame `2009`.

The notebook does not automatically infer Friedel pairs. Instead, the user decides which exact target spots belong to the overlap. This keeps the dataset creation process transparent and scientifically controllable.

## 2. Orientation Correction

The raw detector images and the segmentation volume are not stored in the same display orientation. Therefore, the notebook first transforms both representations into a common orientation before any labeling or cropping is performed.

Two separate base transforms are used:

- one for raw intensity frames,
- one for segmentation frames.

After that, an additional user-controlled rotation can be applied through:

```python
DISPLAY_ROTATE_K
```

The current notebook keeps the preview in the working orientation and stores the chosen rotation in the exported file attribute:

- `rotation_k_90deg`

This step is essential, because wrong orientation would place spot numbers and masks away from the actual diffraction spots.

## 3. Spot Labeling From The Segmentation Volume

Spot numbering is derived from `segvol`, not from the raw intensity image.

For each frame:

- all nonzero `segvol` pixels are converted into a binary mask,
- connected components are computed with 8-neighbour connectivity,
- each component is treated as one spot candidate,
- very small components are discarded using `MIN_PIXELS`,
- the remaining components are sorted by size,
- spot numbers are assigned in this sorted order.

For each spot, the notebook stores:

- `spot_number`
- pixel count
- bounding box
- centroid
- local binary mask

The segmentation volume is therefore the geometric reference for the whole dataset-building process.

## 4. Full-Frame Preview

The first preview section shows:

- the selected input frame,
- the selected Friedel frame(s),
- the numbered spot boxes.

This step is used to verify that:

- the orientation is correct,
- the chosen spot numbers are correct,
- the user-selected Friedel spots are physically plausible.

The preview is intentionally shown on a white-background segmentation view, because this makes the sparse spots and their labels easier to inspect than raw detector contrast alone.

## 5. Crop Extraction

For each selected spot, the notebook extracts a centered `128 x 128` crop:

- `CROP_SIZE = 128`

The crop is centered on the segmentation centroid of the selected spot. If the crop would extend outside the detector boundary, it is padded with zeros.

For every crop, the notebook creates:

- `raw`: the raw detector crop,
- `mask`: the binary segmentation crop,
- `signal`: the processed intensity crop,
- metadata such as centroid, bounding box, pixel count, frame, and spot number.

## 6. Input Crop Preparation

The input crop corresponds to the overlapping spot. Its purpose is to represent the difficult separation problem that the model will see at inference time.

Two preparation steps are important:

### 6.1 Background Suppression

When

```python
BLACK_OUT_INPUT_BACKGROUND = True
```

the input image is multiplied by its segmentation support so that the background becomes completely black. This ensures that the model focuses on the overlapping signal rather than on unrelated detector background.

### 6.2 Input Segmentation Mask

The input mask is exported separately as:

- `input_mask`

This binary mask preserves the spatial support of the overlapping region and can be used as an auxiliary input or as a reference during evaluation.

## 7. Joint Intensity Construction

The input crop and the two target crops are first background-corrected independently, but they are not normalized independently. Instead, the notebook uses one shared normalization factor for the whole sample so that the relative strength of the overlapping input and the two Friedel targets is preserved.

The processing steps are:

1. Estimate a local background from pixels outside the segmentation mask.
2. Subtract the background baseline.
3. Clip negative values to zero.
4. Set all pixels outside the mask to zero.
5. Collect all masked intensities from the input crop and both target crops.
6. Derive one shared normalization scale from the combined masked intensities.
7. Normalize all three crops with this same scale.
8. Clip values to the `[0, 1]` range.
9. Apply a minimum in-mask intensity floor.

The normalization scope is controlled by:

```python
JOINT_NORMALIZATION_SCOPE = "sample"
```

In the current implementation, `"sample"` means that one scale is computed jointly from:

- the overlapping input crop,
- target 1,
- target 2.

This avoids the main failure mode of per-target normalization: a weak and a strong Friedel component no longer become equally bright just because they were scaled separately.

The minimum intensity floor is controlled by:

```python
MIN_MASK_INTENSITY = 0.1
```

This step was introduced to avoid dark holes inside the target spot that can appear after background subtraction. Such holes are not necessarily physically wrong, but for the intended dataset they reduce visual clarity and may produce unnecessarily harsh targets. The floor preserves the spot structure while keeping the background black.

## 8. Horizontal Flipping Of Ground Truth

The notebook can horizontally flip the two ground-truth crops:

```python
FLIP_GROUND_TRUTH_HORIZONTALLY = True
```

This is applied to:

- target intensity `signal`
- target binary `mask`

The purpose is to match the expected physical or model-oriented orientation of the Friedel-pair targets relative to the input overlap.

## 9. Spatial Alignment Of The Ground-Truth Masks

A central requirement of this workflow is that the target masks should not simply be placed at the crop center. Instead, they should be positioned so that their relative placement resembles the two substructures inside the overlapping input mask.

The notebook therefore performs a simple geometric alignment:

1. The input overlap mask is split into two approximate centers using a two-cluster centroid fit.
2. Each isolated target mask receives an integer translation.
3. The translated target mask is shifted toward one of the two fitted input centers.
4. The same shift is applied to the corresponding target intensity image.

This behaviour is controlled by:

```python
ALIGN_GROUND_TRUTH_TO_INPUT = True
```

The applied shifts are stored in the export metadata as:

- `target_alignment_shift_rc`
- per-target attribute `alignment_shift_rc`

This alignment is important for fair model evaluation. If a model predicts a correct shape but the target mask is artificially centered, the evaluation score would be worse even though the predicted decomposition is physically reasonable.

## 10. Visual Crop Preview

Before exporting, the notebook displays six preview panels:

- input image
- target 1 intensity
- target 2 intensity
- input segmentation mask
- target 1 mask
- target 2 mask

The intensity previews are display-enhanced for inspection only. The input panel keeps the original full aligned `128 x 128` view, while the target intensity panels can use:

- a robust or logarithmic contrast stretch,
- optional tight cropping around the nonzero mask,
- smoother interpolation,
- and an alternate display colormap.

These preview settings are controlled by:

```python
INPUT_PREVIEW_MODE = "full_linear"
TARGET_PREVIEW_MODE = "full_log"
PREVIEW_MARGIN = 6
PREVIEW_INTERPOLATION = "bilinear"
PREVIEW_COLORMAP = "gray"
```

This makes the target diffraction spots look closer to the diagnostic views in `augment_data/intensity_colormap_check.ipynb` while keeping the input panel visually consistent with the original full crop.

Important: these preview settings do not modify the exported HDF5 tensors. The saved arrays remain the aligned `128 x 128` crops built in the preprocessing steps above.

This preview is the main quality-control stage. It allows the user to verify:

- whether the chosen spot numbers are correct,
- whether intensities are visible enough,
- whether the masks are positioned plausibly,
- whether flipping and rotation are correct,
- whether the sample should be exported.

## 11. HDF5 Export

The export is controlled by:

```python
DO_EXPORT = True
```

The output file is written to:

```python
../data/test_friedel_crops/<generated_name>.h5
```

The exported HDF5 structure contains:

### Root datasets

- `image`
  input overlap intensity crop
- `input_mask`
  binary segmentation mask of the overlapping input
- `spot_images`
  stacked target intensity crops with shape `(2, 128, 128)`
- `spot_masks`
  stacked target binary masks with shape `(2, 128, 128)`

### Root attributes

- `scan`
- `rotation_k_90deg`
- `crop_size`
- `raw_path`
- `seg_path`
- `input_frame`
- `input_spot_number`
- `flip_ground_truth_horizontally`
- `black_out_input_background`
- `normalize_intensity_with_mask`
- `joint_normalization_scope`
- `shared_normalization_scale`
- `min_mask_intensity`
- `align_ground_truth_to_input`
- `target_frames`
- `target_spot_numbers`
- `target_names`
- `target_alignment_shift_rc`

### Input group

Group: `input`

Attributes:

- `frame`
- `spot_number`
- `bbox`
- `centroid_rc`
- `pixels`

### Target groups

Groups:

- `target_1`
- `target_2`

Each target group stores:

- dataset `intensity`
- dataset `mask`
- attributes `name`, `frame`, `spot_number`, `bbox`, `centroid_rc`, `pixels`, `alignment_shift_rc`

## 12. Practical Usage

For each new test sample, the recommended procedure is:

1. Open the notebook.
2. Select the scan and define `INPUT_FRAME`, `INPUT_SPOT_NUMBER`, and `FRIEDEL_TARGETS`.
3. Run the configuration and helper cells.
4. Inspect the full-frame preview and confirm that the spot numbering is correct.
5. Inspect the crop preview and confirm that intensities, masks, flipping, and relative placement are correct.
6. Set `DO_EXPORT = True`.
7. Re-run the export cell to write the HDF5 sample.

This manual approach is slower than automatic generation, but it ensures that each exported sample is traceable and physically interpretable.

## 13. Important Assumptions And Limitations

- Spot numbering is frame-local. The number of a spot only has meaning inside its own frame.
- Friedel matching is manual. The notebook does not prove that two spots are a valid pair; it only exports the spots selected by the user.
- Target alignment is translation-only. No scaling, deformation, or nonrigid warping is applied.
- The overlap split inside the input mask is an approximation based on two fitted centers, not on a full physical decomposition.
- The shared normalization preserves relative intensity inside one sample, but not across different exported samples.
- The intensity floor is a visualization and target-stabilization choice. It improves readability but slightly modifies the original masked intensities.

These points should be stated clearly if the dataset is described in a thesis or methods section.

## 14. Methodological Summary For A Thesis

In concise scientific terms, the method can be described as follows:

Manual evaluation samples were generated by combining one overlapping diffraction spot from a selected input frame with two isolated Friedel-pair spots from offset frames. Spot supports were derived from the segmentation volume `segvol` using connected-component labeling. For each selected spot, a `128 x 128` crop was extracted around the segmentation centroid. A local background was subtracted from each crop, after which a single shared normalization factor was computed from the combined masked intensities of the input and both target crops. The overlapping input crop was background-suppressed to retain only the segmented signal. The two isolated Friedel targets were optionally horizontally flipped and translated such that their spatial positions matched the two dominant subregions of the overlapping input mask. The resulting input image, input mask, target intensities, and target masks were previewed visually and exported as structured HDF5 files for downstream model evaluation.

## 15. Main Parameters

The most important notebook parameters are:

- `CROP_SIZE`
- `DISPLAY_ROTATE_K`
- `FLIP_GROUND_TRUTH_HORIZONTALLY`
- `BLACK_OUT_INPUT_BACKGROUND`
- `NORMALIZE_INTENSITY_WITH_MASK`
- `JOINT_NORMALIZATION_SCOPE`
- `MIN_MASK_INTENSITY`
- `ALIGN_GROUND_TRUTH_TO_INPUT`
- `MIN_PIXELS`

Together, these parameters determine the orientation, visual quality, target geometry, and final export format of the dataset sample.
