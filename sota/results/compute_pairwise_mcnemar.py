#!/usr/bin/env python3
"""Pairwise McNemar analysis for SIP-Classifier versus all baselines.

Expected files in --input-dir:
  - kmeans_predictions_llama_nf_msg_con_seq.csv
  - hmm_predictions.csv
  - lstm_predictions.csv
  - cnn_predictions.csv
  - correlation_predictions.csv
  - transformer_predictions.csv

The script aligns every method by test_csv_row, verifies the ground-truth
labels and signaling sequences, builds the complete 2x2 correctness table,
computes the two-sided continuity-corrected McNemar test, and applies Holm's
family-wise error-rate correction across the five comparisons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


ID_COL = "test_csv_row"
TRUE_COL = "true_label"
SIGNAL_COL = "Replaced Signalling Description"

# Filename and prediction column used for each baseline.
BASELINES: dict[str, tuple[str, str]] = {
    "HMM": ("hmm_predictions.csv", "pred_label"),
    "LSTM": ("lstm_predictions.csv", "pred_label_model2_max"),
    "CNN": ("cnn_predictions.csv", "pred_label"),
    "Normalized Correlation": (
        "correlation_predictions.csv",
        "pred_label_normalized",
    ),
    "Transformer Encoder": ("transformer_predictions.csv", "pred_label"),
}


def read_predictions(path: Path, prediction_col: str, method: str) -> pd.DataFrame:
    """Read and validate one per-dialog prediction file."""
    df = pd.read_csv(path)
    required = {ID_COL, TRUE_COL, SIGNAL_COL, prediction_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{method}: missing columns in {path.name}: {sorted(missing)}"
        )

    out = df[[ID_COL, TRUE_COL, SIGNAL_COL, prediction_col]].copy()
    out = out.rename(columns={prediction_col: "prediction"})

    if out[ID_COL].isna().any():
        raise ValueError(f"{method}: {ID_COL} contains missing values")
    if out[ID_COL].duplicated().any():
        duplicates = out.loc[out[ID_COL].duplicated(), ID_COL].tolist()[:10]
        raise ValueError(f"{method}: duplicated {ID_COL} values: {duplicates}")

    for column in (TRUE_COL, "prediction"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(int)
        invalid = sorted(set(out[column].unique()) - {0, 1})
        if invalid:
            raise ValueError(
                f"{method}: column {column} is not binary; found {invalid}"
            )

    return out.sort_values(ID_COL).reset_index(drop=True)


def align_with_reference(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Align a baseline to SIP-Classifier and verify labels/sequences."""
    ref_ids = set(reference[ID_COL])
    candidate_ids = set(candidate[ID_COL])
    if ref_ids != candidate_ids:
        only_ref = sorted(ref_ids - candidate_ids)[:10]
        only_candidate = sorted(candidate_ids - ref_ids)[:10]
        raise ValueError(
            f"{method}: test-row sets differ. "
            f"Only in SIP-Classifier: {only_ref}; only in baseline: {only_candidate}"
        )

    merged = reference.merge(
        candidate,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
        suffixes=("_ours", "_baseline"),
    ).sort_values(ID_COL)

    label_mismatch = merged[f"{TRUE_COL}_ours"] != merged[f"{TRUE_COL}_baseline"]
    if label_mismatch.any():
        bad = merged.loc[label_mismatch, ID_COL].tolist()[:10]
        raise ValueError(f"{method}: ground-truth mismatch at rows {bad}")

    signal_mismatch = merged[f"{SIGNAL_COL}_ours"] != merged[f"{SIGNAL_COL}_baseline"]
    if signal_mismatch.any():
        bad = merged.loc[signal_mismatch, ID_COL].tolist()[:10]
        raise ValueError(f"{method}: signaling-sequence mismatch at rows {bad}")

    return merged


def mcnemar_complete_table(merged: pd.DataFrame) -> dict[str, float | int]:
    """Compute complete paired correctness table and McNemar statistics."""
    y_true = merged[f"{TRUE_COL}_ours"].to_numpy(dtype=int)
    y_ours = merged["prediction_ours"].to_numpy(dtype=int)
    y_baseline = merged["prediction_baseline"].to_numpy(dtype=int)

    ours_correct = y_ours == y_true
    baseline_correct = y_baseline == y_true

    # First index: SIP-Classifier correctness; second: baseline correctness.
    n11 = int(np.sum(ours_correct & baseline_correct))
    n10 = int(np.sum(ours_correct & ~baseline_correct))
    n01 = int(np.sum(~ours_correct & baseline_correct))
    n00 = int(np.sum(~ours_correct & ~baseline_correct))
    total = n11 + n10 + n01 + n00

    discordant = n10 + n01
    if discordant == 0:
        statistic = 0.0
        p_raw = 1.0
    else:
        statistic = ((abs(n10 - n01) - 1.0) ** 2) / discordant
        p_raw = float(chi2.sf(statistic, df=1))

    return {
        "n11_both_correct": n11,
        "n10_ours_correct_baseline_wrong": n10,
        "n01_ours_wrong_baseline_correct": n01,
        "n00_both_wrong": n00,
        "total": total,
        "delta_accuracy_pp": 100.0 * (n10 - n01) / total,
        "chi2_continuity_corrected": statistic,
        "p_raw": p_raw,
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in original order."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, p_value in enumerate(sorted_p):
        candidate = (m - rank) * p_value
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(running_max, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the prediction CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mcnemar_complete_results.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()

    ours = read_predictions(
        input_dir / "kmeans_predictions_llama_nf_msg_con_seq.csv",
        prediction_col="pred_label",
        method="SIP-Classifier",
    ).rename(columns={"prediction": "prediction_ours"})

    rows: list[dict[str, object]] = []
    for method, (filename, prediction_col) in BASELINES.items():
        baseline = read_predictions(
            input_dir / filename,
            prediction_col=prediction_col,
            method=method,
        ).rename(columns={"prediction": "prediction_baseline"})

        merged = align_with_reference(ours, baseline, method)
        result = mcnemar_complete_table(merged)
        rows.append({"baseline": method, **result})

    results = pd.DataFrame(rows)
    results["p_holm"] = holm_adjust(results["p_raw"].to_numpy())

    # Preserve the scientific-paper column order.
    results = results[
        [
            "baseline",
            "n11_both_correct",
            "n10_ours_correct_baseline_wrong",
            "n01_ours_wrong_baseline_correct",
            "n00_both_wrong",
            "total",
            "delta_accuracy_pp",
            "chi2_continuity_corrected",
            "p_raw",
            "p_holm",
        ]
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    display = results.copy()
    display["delta_accuracy_pp"] = display["delta_accuracy_pp"].map(
        lambda value: f"{value:+.1f}"
    )
    display["chi2_continuity_corrected"] = display[
        "chi2_continuity_corrected"
    ].map(lambda value: f"{value:.3f}")
    display["p_raw"] = display["p_raw"].map(lambda value: f"{value:.6e}")
    display["p_holm"] = display["p_holm"].map(lambda value: f"{value:.6e}")

    print(display.to_string(index=False))
    print(f"\nSaved results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
