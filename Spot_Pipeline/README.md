# DCT Spot Pipeline

Friedel-pair extractor that consumes segmentation data and emits high-confidence isolated diffraction-spot patches for downstream synthetic-overlap NN training.

## Pipeline

The pipeline runs in three passes over segmentation data:

1. **Pass 1** — extracts blobs per frame (binarize → connected components → area filter)
2. **Pass 2** — pairs Friedel partners (blobs 180° apart in omega, matched by NCC)
3. **Pass 3** — gates down to high-confidence isolated spots

**Run with a config file (preferred):**

```bash
cd src && python -m seg_pipeline.run --config ../configs/Al_big_grains_logtif.toml
```

**Quick test on a few frames:**

```bash
cd src && python -m seg_pipeline.run --config ../configs/Al_big_grains_logtif.toml \
    --frames 0:5 --output ../output/test.h5
```

CLI flags override any value set in the config file.

## Config Files

Each `.toml` in `configs/` targets one scan. Key fields:

| Field           | Purpose                                                    |
| --------------- | ---------------------------------------------------------- |
| `seg_path`      | Path to LoG TIF directory or seg_vol HDF5                  |
| `seg_type`      | `"log_tif"` or `"seg_vol"`                                 |
| `scan_name`     | Key into `scan_registry.toml` (for raw HDF5 lookup)        |
| `registry`      | Path to `scan_registry.toml`                               |
| `frames`        | Frame range to process, e.g. `"0:1800"`                    |
| `output`        | Where to write the result HDF5                             |
| `log_threshold` | Binarization cutoff (float32 scale for LoG TIFs)           |
| `min_ncc`       | Friedel pair acceptance threshold (0–1, higher = stricter) |

For a full list of parameters with descriptions and defaults, run:

```bash
cd src && python -m seg_pipeline.run --help
```

`scan_registry.toml` is shared across configs — it maps scan names to their raw HDF5 filenames, which are needed for beam-center estimation.

## Notebook (`inspect_spots.ipynb`)

Interactive viewer for pipeline output.

1. Set `OUT_PATH` and `DATASET` at the top of the notebook to point to your output `.h5` file and the scan name inside it.
2. Run all cells.
3. Use the **slider widget** at the bottom to browse Friedel pairs. Each pair shows:
   - Row 1: Spot A — raw / preprocessed / binary mask / overlay with partner
   - Row 2: Spot B — raw / preprocessed / binary mask
   - Row 3: Spatial relationship — positions on full frame, raw A with predicted B location, raw B in predicted region

The cells above the slider also print config metadata and histograms of NCC scores and blob areas.

## Project Structure

```
src/
├── seg_pipeline/
│   ├── seg_types.py   # SegPipelineConfig, SegBlob, SegIsolatedSpot
│   ├── passes.py      # pass1_extract → pass2_pair_friedel → pass3_gate
│   ├── loaders.py     # frame iterators for HDF5, TIF dir, TAR
│   ├── run.py         # CLI entry point
│   └── tests/         # pytest — test_passes.py
configs/
├── scan_registry.toml # shared scan → filename mapping
└── *.toml             # per-scan run configs
inspect_spots.ipynb    # interactive result viewer
```
