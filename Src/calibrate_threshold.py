"""
Calibrate the anomaly-score threshold used to decide "clean" vs "defect".

Why calibration is needed:
The autoencoder gives a continuous reconstruction-error score. We need a
cutoff: above this score = flag as anomaly. We calibrate this using a
held-out set of KNOWN CLEAN images (not used in training) by computing
their reconstruction error distribution, then setting:

    threshold = mean(clean_scores) + k * std(clean_scores)

k=3 is a common statistical choice (99.7% of clean scores fall below this
under a roughly normal error distribution), but this should be tuned
against a validation set that includes a few known defects if available,
to balance false positives vs false negatives for your specific line.

Usage:
    python calibrate_threshold.py --data_dir data/clean_val --model_path ../models/autoencoder_best.pt
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from autoencoder import ConvAutoencoder, anomaly_score
from train_autoencoder import CleanPCBDataset


def calibrate(data_dir: str, model_path: str, k: float, out_json: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = CleanPCBDataset(data_dir)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            batch_scores = anomaly_score(batch, recon)
            scores.extend(batch_scores.cpu().tolist())

    scores_tensor = torch.tensor(scores)
    mean_score = scores_tensor.mean().item()
    std_score = scores_tensor.std().item()
    threshold = mean_score + k * std_score

    result = {
        "mean_clean_score": mean_score,
        "std_clean_score": std_score,
        "k": k,
        "threshold": threshold,
        "n_calibration_images": len(scores),
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nThreshold saved to {out_json}")
    print("Use this threshold in the Streamlit app / inference script:")
    print(f"  anomaly_score > {threshold:.5f}  =>  flag as DEFECT")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate anomaly detection threshold")
    parser.add_argument("--data_dir", type=str, required=True, help="Held-out clean PCB images")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--k", type=float, default=3.0, help="Std-dev multiplier")
    parser.add_argument("--out_json", type=str, default="../models/threshold.json")
    args = parser.parse_args()

    calibrate(args.data_dir, args.model_path, args.k, args.out_json)
