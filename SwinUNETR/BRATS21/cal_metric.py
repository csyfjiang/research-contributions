"""
Calculate segmentation metrics between predictions and ground truth.

Metrics: Dice, HD95, mIoU
Evaluated on:
  1. Tumor regions: WT (whole tumor), TC (tumor core), ET (enhancing tumor)
  2. Individual labels: NCR (label 1), ED (label 2), ET (label 3/4)

Usage:
    python cal_metric.py \
        --pred_dir=./outputs/brats23_fold1_inference \
        --gt_dir="I:/data/Brats2025/BraTS 2023 Challenge/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" \
        --output_csv=./outputs/brats23_fold1_metrics.csv
"""

import argparse
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def dice_score(pred, gt):
    intersection = np.sum(pred & gt)
    denom = np.sum(pred) + np.sum(gt)
    if denom == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    return 2.0 * intersection / denom


def iou_score(pred, gt):
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    if union == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    return intersection / union


def hd95_score(pred, gt, voxel_spacing):
    """Compute 95th percentile Hausdorff distance."""
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return np.inf

    # Surface voxels (boundary)
    pred_border = pred ^ _erode(pred)
    gt_border = gt ^ _erode(gt)

    # Distance transforms
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)

    # Distances from pred surface to gt and vice versa
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return np.percentile(all_distances, 95)


def _erode(mask):
    """Simple binary erosion by 1 voxel using 6-connectivity."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1)


def get_regions(seg, et_label=4):
    """
    Convert label map to binary masks for tumor regions.
    BraTS2021: NCR=1, ED=2, ET=4
    BraTS2023: NCR=1, ED=2, ET=3
    """
    ncr = seg == 1
    ed = seg == 2
    et = seg == et_label

    wt = ncr | ed | et   # Whole Tumor: NCR + ED + ET
    tc = ncr | et        # Tumor Core: NCR + ET
    return {"WT": wt, "TC": tc, "ET": et}


def get_labels(seg, et_label=4):
    """Convert label map to individual label binary masks."""
    return {
        "NCR": seg == 1,
        "ED": seg == 2,
        "ET": seg == et_label,
    }


def compute_metrics(pred_mask, gt_mask, voxel_spacing):
    return {
        "Dice": dice_score(pred_mask, gt_mask),
        "HD95": hd95_score(pred_mask, gt_mask, voxel_spacing),
        "mIoU": iou_score(pred_mask, gt_mask),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, help="Directory with prediction nii.gz files")
    parser.add_argument("--gt_dir", required=True, help="Directory with GT patient folders")
    parser.add_argument("--output_csv", default="I:/PhD/BraTS/outputs/brats23_fold1_metrics.csv")
    args = parser.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    print(f"Found {len(pred_files)} prediction files")

    all_results = []

    for i, pred_path in enumerate(pred_files):
        patient_id = os.path.basename(pred_path).replace(".nii.gz", "")
        gt_path = os.path.join(args.gt_dir, patient_id, f"{patient_id}-seg.nii.gz")

        if not os.path.exists(gt_path):
            print(f"[{i+1}/{len(pred_files)}] {patient_id}: GT not found, skipping")
            continue

        pred_nii = nib.load(pred_path)
        gt_nii = nib.load(gt_path)

        pred_data = pred_nii.get_fdata().astype(np.uint8)
        gt_data = gt_nii.get_fdata().astype(np.uint8)
        voxel_spacing = gt_nii.header.get_zooms()[:3]

        # Prediction: BraTS2021 format (ET=4), GT: BraTS2023 format (ET=3)
        pred_regions = get_regions(pred_data, et_label=4)
        gt_regions = get_regions(gt_data, et_label=3)

        pred_labels = get_labels(pred_data, et_label=4)
        gt_labels = get_labels(gt_data, et_label=3)

        result = {"patient_id": patient_id}

        # Region-level metrics (WT, TC, ET)
        for region_name in ["WT", "TC", "ET"]:
            metrics = compute_metrics(pred_regions[region_name], gt_regions[region_name], voxel_spacing)
            for metric_name, value in metrics.items():
                result[f"Region_{region_name}_{metric_name}"] = value

        # Label-level metrics (NCR, ED, ET)
        for label_name in ["NCR", "ED", "ET"]:
            metrics = compute_metrics(pred_labels[label_name], gt_labels[label_name], voxel_spacing)
            for metric_name, value in metrics.items():
                result[f"Label_{label_name}_{metric_name}"] = value

        all_results.append(result)

        print(
            f"[{i+1}/{len(pred_files)}] {patient_id} | "
            f"WT Dice: {result['Region_WT_Dice']:.4f} | "
            f"TC Dice: {result['Region_TC_Dice']:.4f} | "
            f"ET Dice: {result['Region_ET_Dice']:.4f}"
        )

    # Build DataFrame and compute summary
    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # Print summary
    metric_cols = [c for c in df.columns if c != "patient_id"]
    print("\n" + "=" * 80)
    print("REGION-LEVEL METRICS (WT, TC, ET)")
    print("-" * 80)
    region_cols = [c for c in metric_cols if c.startswith("Region_")]
    summary_region = df[region_cols].replace([np.inf], np.nan)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Median':>10}")
    for col in region_cols:
        vals = summary_region[col].dropna()
        print(f"{col:<25} {vals.mean():>10.4f} {vals.std():>10.4f} {vals.median():>10.4f}")

    print("\n" + "=" * 80)
    print("LABEL-LEVEL METRICS (NCR, ED, ET)")
    print("-" * 80)
    label_cols = [c for c in metric_cols if c.startswith("Label_")]
    summary_label = df[label_cols].replace([np.inf], np.nan)
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Median':>10}")
    for col in label_cols:
        vals = summary_label[col].dropna()
        print(f"{col:<25} {vals.mean():>10.4f} {vals.std():>10.4f} {vals.median():>10.4f}")

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
