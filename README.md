# Code to Masterthesis

## Training path profiles

Edit `training_paths.toml` once on each machine to set the input HDF5 file and output root for each training variant:

```toml
[twooutputs]
input_path = "/path/to/training_data.h5"
output_path = "/path/to/twooutputs"

[oneoutput]
input_path = "/path/to/training_data.h5"
output_path = "/path/to/oneoutput"

[pin]
input_path = "/path/to/training_data.h5"
output_path = "/path/to/pin"
```

Run training with the profile you want:

```sh
python train.py --path-profile twooutputs
python train.py --path-profile oneoutput
python train.py --path-profile pin
```

For each profile, `output_path` contains `checkpoints/`, `runs/`, `prediction_previews/`, and `debug_logs/`. Individual paths can still be overridden with `--h5-file`, `--checkpoint-dir`, `--log-dir`, `--preview-dir`, or `--debug-dir`.

