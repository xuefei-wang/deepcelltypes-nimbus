# Nimbus Baseline

Nimbus UNet marker positivity prediction baseline for the deepcelltypes project.

**Paper:** Rumberger et al., "Nimbus: A deep learning approach for cell segmentation and marker quantification in multiplexed tissue imaging," *Nature Methods* 2025.
DOI: [10.1038/s41592-025-02826-9](https://doi.org/10.1038/s41592-025-02826-9)

**Original code:** https://github.com/angelolab/Nimbus-Inference

This is an **inference-only** baseline that uses a pretrained UNet to predict marker positivity from multiplexed imaging data. It requires a separate **Python 3.11** virtual environment due to Nimbus-Inference dependency constraints (e.g., `numpy<2.0`, older Pillow).

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
python -m nimbus_baseline --model_name nimbus_0 --zarr_dir /path/to/zarr
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `nimbus_0` | Name for output files |
| `--device_num` | `cuda:0` | Device for inference |
| `--zarr_dir` | `$DATA_DIR/tissuenet-caitlin-labels.zarr` | Path to zarr archive |
| `--checkpoint` | `latest` | Nimbus checkpoint (`latest` or path) |
| `--batch_size` | `4` | Batch size for inference |
| `--threshold` | `0.5` | Marker positivity threshold |
| `--max_fovs` | `None` | Limit number of FOVs (for testing) |
| `--enable_wandb` | `False` | Log metrics to Weights & Biases |
| `--skip_datasets` | `[]` | Dataset keys to skip |
| `--keep_datasets` | `[]` | Dataset keys to keep (exclusive with skip) |

## Known Adaptations from Original

- **Per-FOV quantile normalization:** The original Nimbus averages the 99.9th percentile across n=10 FOVs for cross-FOV consistency. This implementation normalizes each FOV independently.
- **TTA flag is non-functional:** The `--test_time_aug` flag is passed to Nimbus but only works with `predict_fovs`, not `predict_segmentation` (which this code uses).
- **No magnification scaling:** The zarr archive already preprocesses images to a standard resolution, so no additional magnification adjustment is needed.

## Bug Fixes Applied

- **Fixed double sigmoid:** The original UNet already applies sigmoid internally; this code does not apply a redundant sigmoid to the output.
- **Fixed binary mask preparation:** Uses `prepare_binary_mask()` from Nimbus-Inference (with boundary erosion) instead of simple thresholding, matching the original preprocessing pipeline.
