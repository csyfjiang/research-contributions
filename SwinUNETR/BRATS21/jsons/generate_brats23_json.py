"""
Generate brats23_folds.json from BraTS 2023 GLI Challenge data.

Scans training and validation directories, creates 5-fold cross-validation
splits for training data, and outputs a JSON compatible with the existing
data_utils.py loader.

Usage:
    python jsons/generate_brats23_json.py
"""

import json
import os
from pathlib import Path
from sklearn.model_selection import KFold

TRAIN_DIR = r"I:\data\Brats2025\BraTS 2023 Challenge\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "brats23_folds.json")

NUM_FOLDS = 5
SEED = 42


def get_patient_entry(patient_dir, patient_id, has_label=True):
    """Create a JSON entry for a single patient."""
    entry = {
        "image": [
            os.path.join(patient_dir, f"{patient_id}-t2f.nii.gz"),
            os.path.join(patient_dir, f"{patient_id}-t1c.nii.gz"),
            os.path.join(patient_dir, f"{patient_id}-t1n.nii.gz"),
            os.path.join(patient_dir, f"{patient_id}-t2w.nii.gz"),
        ],
    }
    if has_label:
        entry["label"] = os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")
    return entry


def main():
    # --- Training data: scan and assign folds ---
    train_patients = sorted(os.listdir(TRAIN_DIR))
    print(f"Found {len(train_patients)} training patients")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_map = {}
    for fold_idx, (_, val_indices) in enumerate(kf.split(train_patients)):
        for idx in val_indices:
            fold_map[train_patients[idx]] = fold_idx

    training_entries = []
    for patient_id in train_patients:
        patient_dir = os.path.join(TRAIN_DIR, patient_id)
        if not os.path.isdir(patient_dir):
            continue
        entry = get_patient_entry(patient_id, patient_id, has_label=True)
        entry["fold"] = fold_map[patient_id]
        training_entries.append(entry)

    # --- Write JSON (only training, same as BraTS21 format) ---
    output = {
        "training": training_entries,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved to {OUTPUT_JSON}")
    print(f"  Training entries: {len(training_entries)}")

    # Print fold distribution
    for fold in range(NUM_FOLDS):
        count = sum(1 for e in training_entries if e["fold"] == fold)
        print(f"  Fold {fold}: {count} patients")


if __name__ == "__main__":
    main()
