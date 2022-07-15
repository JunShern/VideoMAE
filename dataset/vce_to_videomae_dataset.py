import argparse
import json
from pathlib import Path

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate adapter dataset for vce_dataset for compatibility with VideoMAE finetuning scripts.')
    parser.add_argument("-d", "--data_dir", default="/data/vce_dataset/", help="Path to VCE dataset.")
    parser.add_argument("-o", "--out_dir", default="./vce_for_videomae/", help="Destination to save adapter dataset.")
    parser.add_argument("--val_samples", type=int, default=5000, help="Number of train samples to hold out as validation set.")
    args = parser.parse_args()

    # Load data
    data_dir = Path(args.data_dir).resolve() # Ensure absolute path
    with open(data_dir / "train_labels.json") as f:
        train_obj = json.load(f)
    with open(data_dir / "test_labels.json") as f:
        test_obj = json.load(f)

    def get_path_and_label(label_obj):
        # Convert from 27-vector of emotion scores to a single classification label corresponding to the max-scoring emotion
        emotions_and_scores = sorted(list(label_obj["emotions"].items())) # Make sure they are sorted alphabetically
        scores = [score for emotion, score in emotions_and_scores]
        label = int(np.argmax(scores))
        path = str(data_dir / label_obj['file'])
        return path, label

    # Generate train.csv, val.csv and test.csv
    out_dir = Path(args.out_dir)

    with open(out_dir / "train.csv", "w") as f:
        for key, val in list(train_obj.items())[:-args.val_samples]:
            path, label = get_path_and_label(val)
            print(path, label, file=f)

    with open(out_dir / "val.csv", "w") as f:
        for key, val in list(train_obj.items())[-args.val_samples:]:
            path, label = get_path_and_label(val)
            print(path, label, file=f)

    with open(out_dir / "test.csv", "w") as f:
        for key, val in list(test_obj.items()):
            path, label = get_path_and_label(val)
            print(path, label, file=f)
