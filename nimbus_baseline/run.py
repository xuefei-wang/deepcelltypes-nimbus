"""
Nimbus baseline for marker positivity prediction.

Runs inference with the pretrained Nimbus model to predict marker positivity
for cells in multiplexed imaging data. Compares predictions with ground truth
marker positivity labels.

Reference:
- Paper: Nature Methods 2025, DOI: 10.1038/s41592-025-02826-9
- Code: https://github.com/angelolab/Nimbus-Inference

Installation:
    pip install Nimbus-Inference
"""

import os
import click
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

# Default data directory from environment
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data2"))

from deepcelltypes.config import TissueNetConfig


def check_nimbus_installed():
    """Check if Nimbus-Inference is installed."""
    try:
        import nimbus_inference
        return True
    except ImportError:
        return False


def load_fov_data(zf, dataset_key: str, load_raw: bool = True) -> Tuple[Optional[np.ndarray], np.ndarray, List[str], Dict]:
    """
    Load FOV data from zarr archive.

    Handles three annotation sources (zarr v3 format):
    1. preprocessed/cell_type_info arrays (primary, most datasets)
    2. cell_types/annotations attrs: caitlinb or standardized_source

    Args:
        zf: Zarr file handle
        dataset_key: Dataset key in zarr
        load_raw: Whether to load the raw image array (False to save memory)

    Returns:
        raw: (C, H, W) raw image or None if load_raw=False
        mask: (H, W) segmentation mask
        channel_names: List of channel names
        cell_info: Dict with cell_index, cell_type, centroids

    TODO: This function duplicates logic from ``_get_cell_data_from_ds`` in
    ``deepcelltypes/utils.py``.  In particular, the centroid-matching fallback
    (Path 2) does **not** apply ``preproc.attrs["scale_factor"]`` to the
    annotation centroids before comparing them to the preprocessed centroids,
    which can cause mismatches when the image was rescaled during
    preprocessing.  A future cleanup should replace this with a call to the
    shared utility and fix the scale_factor handling.
    """
    ds = zf[dataset_key]
    preproc = ds["preprocessed"]

    raw = preproc["raw"][:] if load_raw else None  # (C, H, W) or None
    mask = preproc["mask"][:]  # (H, W)
    channel_names = list(preproc.attrs.get("channel_names", []))
    # Zarr v3 serializes dict keys as strings
    centroids_raw = dict(preproc.attrs.get("centroids", {}))

    cell_indices = []
    cell_types = []

    # Path 1: cell_type_info arrays (primary, most datasets)
    if "cell_type_info" in preproc:
        ct_info = preproc["cell_type_info"]
        if "cell_type" in ct_info and "cell_index" in ct_info:
            cell_types = list(ct_info["cell_type"][:])
            cell_indices = [int(idx) for idx in ct_info["cell_index"][:]]

    # Path 2: annotations attrs fallback
    if not cell_indices and "cell_types" in ds and "annotations" in ds["cell_types"]:
        ann_attrs = dict(ds["cell_types/annotations"].attrs)
        source = ann_attrs.get("caitlinb") or ann_attrs.get("standardized_source")
        if source:
            for ct_name, values in source.items():
                if ct_name is None or ct_name == "null":
                    continue
                std_name = ct_name  # cell types already standardized in archive
                for val in values:
                    if isinstance(val, (int, float)) and not isinstance(val, list):
                        cell_indices.append(int(val))
                        cell_types.append(str(std_name))
                    elif isinstance(val, (list, tuple)) and len(val) == 2:
                        # Centroid coordinate — reverse-lookup cell index
                        target = (float(val[0]), float(val[1]))
                        found_idx = None
                        for key, cent in centroids_raw.items():
                            if abs(cent[0] - target[0]) < 0.5 and abs(cent[1] - target[1]) < 0.5:
                                found_idx = int(key)
                                break
                        if found_idx is not None:
                            cell_indices.append(found_idx)
                            cell_types.append(str(std_name))

    if not cell_indices:
        raise ValueError(f"No cell annotations found for {dataset_key}")

    cell_info = {
        "cell_index": np.array(cell_indices),
        "cell_type": np.array(cell_types),
        "centroids": centroids_raw,
    }

    return raw, mask, channel_names, cell_info


def compute_marker_positivity_metrics(
    predictions: pd.DataFrame,
    ground_truth: Dict[str, pd.DataFrame],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute metrics comparing predicted vs ground truth marker positivity.

    Args:
        predictions: DataFrame with predicted marker confidence per cell
        ground_truth: Dict mapping dataset_name to marker positivity DataFrame
        threshold: Threshold for converting predictions to binary

    Returns:
        Dict with accuracy, precision, recall, F1 per marker and overall
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Get marker columns (exclude metadata columns)
    meta_cols = {"cell_index", "cell_type", "dataset_name", "fov_name"}
    marker_cols = [c for c in predictions.columns if c not in meta_cols]

    # Build a ground truth lookup: (dataset_name, cell_type, marker) -> gt_binary
    # by melting each gt_df into long format
    gt_records = []
    for dataset_name, gt_df in ground_truth.items():
        for marker in gt_df.columns:
            for cell_type in gt_df.index:
                val = gt_df.loc[cell_type, marker]
                if pd.isna(val) or val == "?":
                    continue
                gt_records.append((dataset_name, cell_type, marker, float(val) >= 0.5))

    if not gt_records:
        return {"overall": {}, "per_marker": {}}

    gt_long = pd.DataFrame(gt_records, columns=["dataset_name", "cell_type", "marker", "gt_binary"])

    # Melt predictions to long format: one row per (cell, marker)
    pred_long = predictions.melt(
        id_vars=["dataset_name", "cell_type"],
        value_vars=marker_cols,
        var_name="marker",
        value_name="pred_value",
    )
    pred_long["pred_binary"] = pred_long["pred_value"] >= threshold

    # Inner join to keep only (dataset, cell_type, marker) combos with GT
    merged = pred_long.merge(gt_long, on=["dataset_name", "cell_type", "marker"], how="inner")

    if len(merged) == 0:
        return {"overall": {}, "per_marker": {}}

    y_true = merged["gt_binary"].astype(int).values
    y_pred = merged["pred_binary"].astype(int).values

    # Per-marker metrics
    per_marker_metrics = {}
    for marker, group in merged.groupby("marker"):
        yt = group["gt_binary"].astype(int).values
        yp = group["pred_binary"].astype(int).values
        per_marker_metrics[marker] = {
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "n_samples": len(yt),
        }

    # Overall metrics
    overall_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_samples": len(y_true),
    }

    return {
        "overall": overall_metrics,
        "per_marker": per_marker_metrics,
    }


@click.command()
@click.option("--model_name", type=str, default="nimbus_0")
@click.option("--device_num", type=str, default="cuda:0")
@click.option("--enable_wandb", type=bool, default=False)
@click.option(
    "--zarr_dir",
    type=str,
    default=str(DATA_DIR / "tissuenet-caitlin-labels.zarr"),
)
@click.option(
    "--skip_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to skip",
)
@click.option(
    "--keep_datasets",
    type=str,
    multiple=True,
    default=[],
    help="Dataset keys to keep (exclusive with skip_datasets)",
)
@click.option(
    "--checkpoint",
    type=str,
    default="latest",
    help="Nimbus checkpoint to use ('latest' or path to checkpoint file)",
)
@click.option(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size for Nimbus inference",
)
@click.option(
    "--test_time_aug",
    type=bool,
    default=False,
    help="Enable test-time augmentation (slower but more accurate)",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    help="Threshold for marker positivity prediction",
)
@click.option(
    "--max_fovs",
    type=int,
    default=None,
    help="Maximum number of FOVs to process (for testing)",
)
def main(
    model_name: str,
    device_num: str,
    enable_wandb: bool,
    zarr_dir: str,
    skip_datasets: Tuple[str, ...],
    keep_datasets: Tuple[str, ...],
    checkpoint: str,
    batch_size: int,
    test_time_aug: bool,
    threshold: float,
    max_fovs: Optional[int],
):
    """Run Nimbus baseline for marker positivity prediction."""
    import zarr

    # Check if Nimbus is installed
    if not check_nimbus_installed():
        print("Error: Nimbus-Inference is not installed.")
        print("Please install it with: pip install Nimbus-Inference")
        return

    from nimbus_inference.nimbus import Nimbus
    from nimbus_inference.utils import prepare_input_data, prepare_binary_mask
    import torch

    # Set device
    device = torch.device(device_num if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb if enabled
    if enable_wandb:
        import wandb
        wandb.login()
        run = wandb.init(
            project="deepcelltypes-temp-train",
            dir="wandb_tmp",
            job_type="inference",
            name=f"{model_name}_nimbus",
            config={
                "model_type": "nimbus",
                "checkpoint": checkpoint,
                "batch_size": batch_size,
                "test_time_aug": test_time_aug,
                "threshold": threshold,
            },
        )

    # Load config
    dct_config = TissueNetConfig(zarr_dir)
    marker_positivity_gt = dct_config.marker_positivity_labels

    print(f"Loading data from {zarr_dir}")
    print(f"Datasets with marker positivity labels: {list(marker_positivity_gt.keys())}")

    # Open zarr archive
    zf = zarr.open(zarr_dir, mode="r")

    # Get dataset keys
    all_dataset_keys = list(zf.keys())

    # Convert to lists (click returns tuples)
    skip_datasets = list(skip_datasets) if skip_datasets else []
    keep_datasets = list(keep_datasets) if keep_datasets else None

    if keep_datasets:
        dataset_keys = [k for k in all_dataset_keys if k in keep_datasets]
    else:
        dataset_keys = [k for k in all_dataset_keys if k not in skip_datasets]

    # Filter to only datasets with marker positivity ground truth
    dataset_keys = [k for k in dataset_keys if k in marker_positivity_gt]

    print(f"Processing {len(dataset_keys)} datasets with marker positivity labels")

    if len(dataset_keys) == 0:
        print("Error: No datasets with marker positivity labels found.")
        print("Available datasets:", all_dataset_keys[:10], "...")
        return

    # Initialize Nimbus model using the official class (handles tiling for large images)
    print(f"\nLoading Nimbus model (checkpoint: {checkpoint})...")
    nimbus = Nimbus(
        dataset=None,
        output_dir="",
        save_predictions=False,
        batch_size=batch_size,
        test_time_aug=test_time_aug,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint=checkpoint,
        mixed_precision=False,  # mixed_precision=True causes NaN on RTX 6000 Ada
    )
    # Override device to specific GPU
    nimbus.device = device
    nimbus.model = nimbus.model.to(device)
    print("Nimbus model loaded successfully")

    # Process each dataset
    all_predictions = []
    fov_count = 0

    from scipy.ndimage import mean as ndimage_mean

    for dataset_key in tqdm(dataset_keys, desc="Processing datasets"):
        ds = zf[dataset_key]

        # Check if preprocessed data exists
        if "preprocessed" not in ds:
            print(f"Warning: No preprocessed data for {dataset_key}, skipping")
            continue

        try:
            # Load only mask and metadata (not raw — channels loaded lazily below)
            _, mask, channel_names, cell_info = load_fov_data(zf, dataset_key, load_raw=False)
        except Exception as e:
            print(f"Error loading {dataset_key}: {e}")
            continue

        # Prepare binary mask once per FOV (with boundary erosion, matching original Nimbus)
        binary_mask = prepare_binary_mask(mask)

        # Get cell info
        cell_indices = cell_info["cell_index"]
        cell_types = cell_info["cell_type"]
        cell_indices_arr = np.array(cell_indices)

        # Initialize results for this FOV
        fov_results = {
            "cell_index": list(cell_indices),
            "cell_type": list(cell_types),
            "dataset_name": [dataset_key] * len(cell_indices),
        }

        raw_zarr = ds["preprocessed"]["raw"]

        # Process each channel (load lazily from zarr to avoid O(C*H*W) memory)
        for ch_idx, ch_name in enumerate(channel_names):
            # Load single channel from zarr (avoids loading entire C×H×W array)
            channel_img = raw_zarr[ch_idx].astype(np.float32)  # (H, W)

            # Per-channel quantile normalization (Nimbus default)
            foreground = channel_img[channel_img > 0]
            if len(foreground) > 0:
                q999 = np.quantile(foreground, 0.999)
                if q999 > 0:
                    channel_img = np.clip(channel_img / q999, 0, 1)

            # Prepare input: (1, 2, H, W) - [marker, binary_mask]
            input_data = np.stack([channel_img, binary_mask], axis=0)[np.newaxis, ...]

            # Run inference (predict_segmentation handles tiling for large images)
            # Note: UNet forward already applies sigmoid, output is in [0, 1]
            pred = nimbus.predict_segmentation(input_data)
            if isinstance(pred, torch.Tensor):
                pred_np = pred.cpu().numpy()[0, 0]  # (H, W)
            else:
                pred_np = pred[0, 0]  # (H, W)

            # Vectorized mean prediction per cell (single pass over mask)
            channel_preds = ndimage_mean(pred_np, labels=mask, index=cell_indices_arr)
            channel_preds = np.nan_to_num(channel_preds, nan=0.0)

            fov_results[ch_name] = channel_preds.tolist()

        # Convert to DataFrame and append
        fov_df = pd.DataFrame(fov_results)
        all_predictions.append(fov_df)

        fov_count += 1
        if max_fovs and fov_count >= max_fovs:
            print(f"\nReached max_fovs limit ({max_fovs})")
            break

    # Combine all predictions
    if len(all_predictions) == 0:
        print("Error: No predictions generated.")
        return

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    print(f"\nGenerated predictions for {len(predictions_df)} cells")

    # Compute metrics
    print("\nComputing marker positivity metrics...")
    metrics = compute_marker_positivity_metrics(
        predictions_df, marker_positivity_gt, threshold=threshold
    )

    print(f"\nOverall Marker Positivity Metrics:")
    overall = metrics['overall']
    for metric_name in ("accuracy", "precision", "recall", "f1"):
        val = overall.get(metric_name, "N/A")
        print(f"  {metric_name.capitalize()}: {val:.4f}" if isinstance(val, (int, float)) else f"  {metric_name.capitalize()}: {val}")
    print(f"  N Samples: {overall.get('n_samples', 'N/A')}")

    # Print per-marker metrics (top 10 by sample count)
    if metrics["per_marker"]:
        print(f"\nPer-Marker Metrics (top 10 by sample count):")
        sorted_markers = sorted(
            metrics["per_marker"].items(),
            key=lambda x: x[1]["n_samples"],
            reverse=True
        )[:10]
        for marker, m in sorted_markers:
            print(f"  {marker}: Acc={m['accuracy']:.3f}, F1={m['f1']:.3f}, N={m['n_samples']}")

    # Log to wandb
    if enable_wandb:
        wandb.log({
            "marker_positivity/accuracy": metrics["overall"].get("accuracy", 0),
            "marker_positivity/precision": metrics["overall"].get("precision", 0),
            "marker_positivity/recall": metrics["overall"].get("recall", 0),
            "marker_positivity/f1": metrics["overall"].get("f1", 0),
            "marker_positivity/n_samples": metrics["overall"].get("n_samples", 0),
        })

    # Save predictions
    output_path = Path(f"output/{model_name}_nimbus_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    # Save metrics
    metrics_path = Path(f"output/{model_name}_nimbus_metrics.json")
    import json
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {
        "overall": {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                   for k, v in metrics["overall"].items()},
        "per_marker": {
            marker: {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                    for k, v in m.items()}
            for marker, m in metrics["per_marker"].items()
        }
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    if enable_wandb:
        run.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
