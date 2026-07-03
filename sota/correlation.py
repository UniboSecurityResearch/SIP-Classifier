import os
import math
import random
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 1. CONFIGURATION
# ============================================================
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"

SIGNAL_COL = "Replaced Signalling Description"
LABEL_COL = "label"

BENIGN_LABEL_VALUE = 0
ANOMALOUS_LABEL_VALUE = 1

PAD_TOKEN = "<PAD>"
PAD_VALUE = 0.0

print("=" * 100)
print("CORRELATION-BASED SIP DIALOG CLASSIFIER IMPLEMENTATION")
print("=" * 100)
print("Implemented exactly as close as possible to the paper:")
print("- padded dialogs")
print("- zero-lag cross-correlation")
print("- vector r of correlations against the benign knowledge base D = DN")
print("- central moments of r")
print("- decision based on M4 (4th central moment)")
print("- threshold theta > max M4 computed on benign dialogs in D")
print()
print("Two variants are computed:")
print("1) non-normalized cross-correlation")
print("2) normalized cross-correlation")
print()
print("Binary interpretation on test.csv:")
print("0 = benign")
print("1 = anomalous")
print("=" * 100)

# ============================================================
# 2. METRICS
# ============================================================
def anomaly_metrics(y_true, y_pred):
    """
    y_true: 0 = benign, 1 = anomalous
    y_pred: 0 = benign, 1 = anomalous
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    return cm, {
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "mcc": mcc,
    }

# ============================================================
# 3. PARSER
# ============================================================
def parse_dialog(dialog_str):
    """
    Extract a SIP token sequence from the signaling description field.

    Current rule:
    - split on ':'
    - for each chunk take the 3rd comma-separated field
    - if token is 'METHOD-CODE', split it into two tokens
    """
    seq = []

    for msg in str(dialog_str).split(':'):
        parts = msg.split(',')
        if len(parts) < 3:
            continue

        tok = parts[2].strip()
        if not tok:
            continue

        if '-' in tok:
            left, right = tok.split('-', 1)
            left = left.strip()
            right = right.strip()

            if left:
                seq.append(left)
            if right:
                seq.append(right)
        else:
            seq.append(tok)

    return seq

# ============================================================
# 4. LOAD DATA
# ============================================================
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

if SIGNAL_COL not in df_train.columns:
    raise ValueError(f"Column '{SIGNAL_COL}' not found in train.csv")
if SIGNAL_COL not in df_test.columns:
    raise ValueError(f"Column '{SIGNAL_COL}' not found in test.csv")
if LABEL_COL not in df_test.columns:
    raise ValueError(f"Column '{LABEL_COL}' not found in test.csv")

train_dialogs_raw = df_train[SIGNAL_COL].astype(str).tolist()
test_dialogs_raw = df_test[SIGNAL_COL].astype(str).tolist()
test_labels_raw = df_test[LABEL_COL].astype(int).to_numpy()

print("\nRAW DATA")
print(f"Train shape: {df_train.shape}")
print(f"Test shape : {df_test.shape}")
print("Test label distribution:")
print(df_test[LABEL_COL].value_counts(dropna=False).sort_index())

# ============================================================
# 5. PARSE AND FILTER
# ============================================================
train_sequences_all = [parse_dialog(d) for d in train_dialogs_raw]
test_sequences_all = [parse_dialog(d) for d in test_dialogs_raw]

train_sequences_all = [s for s in train_sequences_all if len(s) > 0]

filtered_test_pairs = [
    (seq, label)
    for seq, label in zip(test_sequences_all, test_labels_raw)
    if len(seq) > 0
]
test_sequences = [x[0] for x in filtered_test_pairs]
y_test_binary = np.array([x[1] for x in filtered_test_pairs], dtype=int)

if len(train_sequences_all) == 0:
    raise ValueError("No valid training sequences found after parsing.")
if len(test_sequences) == 0:
    raise ValueError("No valid test sequences found after parsing.")

print("\nAFTER PARSING / FILTERING")
print(f"Valid training dialogs: {len(train_sequences_all)}")
print(f"Valid test dialogs    : {len(test_sequences)}")
print(f"Valid benign test dialogs    : {np.sum(y_test_binary == BENIGN_LABEL_VALUE)}")
print(f"Valid anomalous test dialogs : {np.sum(y_test_binary == ANOMALOUS_LABEL_VALUE)}")

# ============================================================
# 6. REMOVE DUPLICATES FROM TRAIN
#    The paper removes repeated dialogs and works on unique dialogs.
# ============================================================
seen = set()
train_unique_sequences = []
for seq in train_sequences_all:
    key = tuple(seq)
    if key not in seen:
        seen.add(key)
        train_unique_sequences.append(seq)

print("\nKNOWLEDGE BASE")
print(f"Unique benign dialogs in D = DN: {len(train_unique_sequences)}")

# ============================================================
# 7. PAD TO FIXED LENGTH LS
#    The paper pads dialogs to the maximum dialog length LS.
# ============================================================
LS = max(
    max(len(s) for s in train_unique_sequences),
    max(len(s) for s in test_sequences),
)

def pad_sequence(seq, LS, pad_token=PAD_TOKEN):
    if len(seq) >= LS:
        return seq[:LS]
    return seq + [pad_token] * (LS - len(seq))

train_padded = [pad_sequence(s, LS) for s in train_unique_sequences]
test_padded = [pad_sequence(s, LS) for s in test_sequences]

print(f"Padded sequence length LS: {LS}")

# ============================================================
# 8. TOKEN -> NUMERIC ENCODING
#    The original paper uses SIP message types encoded as integers.
#    Here we reproduce that with a train-derived codebook.
# ============================================================
vocab_tokens = sorted({tok for seq in train_padded for tok in seq if tok != PAD_TOKEN})

# keep 0 reserved for padding, as in the paper-style zero padding
token_to_id = {tok: i + 1 for i, tok in enumerate(vocab_tokens)}

def encode_numeric(seq):
    out = []
    for tok in seq:
        if tok == PAD_TOKEN:
            out.append(PAD_VALUE)
        else:
            # unseen test token -> 0.0
            out.append(float(token_to_id.get(tok, 0)))
    return np.array(out, dtype=np.float64)

train_numeric = [encode_numeric(s) for s in train_padded]
test_numeric = [encode_numeric(s) for s in test_padded]

# ============================================================
# 9. PAPER CROSS-CORRELATION
#    Eq. (1): use zero-lag correlation by summing products
#    of aligned message positions.
# ============================================================
def calc_cross_correlation_all_lags(x, y):
    """
    Returns cross-correlation values across all lags using numpy.correlate.
    Equivalent paper use focuses on l = 0 for classification.
    """
    return np.correlate(x, y, mode="full")

def zero_lag_value(corr_full, LS):
    """
    For full correlation of two same-length sequences, zero lag is at index LS - 1.
    """
    return float(corr_full[LS - 1])

def normalize_cross_corr_zero(corr_full, LS):
    """
    Eq. (2): ||R_sk,pk[0]||_2 = |R[0]| / sqrt(sum_l |R[l]|^2)
    """
    r0 = abs(float(corr_full[LS - 1]))
    denom = float(np.sqrt(np.sum(np.abs(corr_full) ** 2)))
    if denom == 0.0:
        return 0.0
    return r0 / denom

def build_r_vector(x, knowledge_base, normalized=False):
    r = []
    for pk in knowledge_base:
        corr_full = calc_cross_correlation_all_lags(x, pk)
        if normalized:
            r.append(normalize_cross_corr_zero(corr_full, LS))
        else:
            r.append(zero_lag_value(corr_full, LS))
    return np.array(r, dtype=np.float64)

# ============================================================
# 10. PAPER MOMENTS
#     Eq. (3): central moments of r
# ============================================================
def central_moment(r, n):
    mu_r = float(np.mean(r))
    return float(np.mean((r - mu_r) ** n))

def compute_paper_statistics(x, knowledge_base, normalized=False):
    r = build_r_vector(x, knowledge_base, normalized=normalized)
    m2 = central_moment(r, 2)
    m3 = central_moment(r, 3)
    m4 = central_moment(r, 4)
    return {
        "r": r,
        "mean_r": float(np.mean(r)),
        "M2": m2,
        "M3": m3,
        "M4": m4,
    }

# ============================================================
# 11. LEARN THRESHOLD EXACTLY AS IN THE PAPER
#     theta > max kurtosis (M4) observed on DN
# ============================================================
def learn_theta_from_benign(knowledge_base, normalized=False):
    benign_M4 = []
    benign_M2 = []

    for sk in knowledge_base:
        stats = compute_paper_statistics(sk, knowledge_base, normalized=normalized)
        benign_M2.append(stats["M2"])
        benign_M4.append(stats["M4"])

    benign_M2 = np.array(benign_M2, dtype=np.float64)
    benign_M4 = np.array(benign_M4, dtype=np.float64)

    max_M4 = float(np.max(benign_M4))
    eps = max(1e-12, abs(max_M4) * 1e-12)
    theta = max_M4 + eps

    return theta, benign_M2, benign_M4

# ============================================================
# 12. PAPER DECISION RULE
#     H0: M4 <= theta  -> N
#     H1: M4 >  theta  -> A
# ============================================================
def predict_labels(test_set, knowledge_base, theta, normalized=False):
    preds = []
    test_M2 = []
    test_M4 = []

    for sk in test_set:
        stats = compute_paper_statistics(sk, knowledge_base, normalized=normalized)
        M4 = stats["M4"]
        pred = 0 if M4 <= theta else 1
        preds.append(pred)
        test_M2.append(stats["M2"])
        test_M4.append(M4)

    return (
        np.array(preds, dtype=int),
        np.array(test_M2, dtype=np.float64),
        np.array(test_M4, dtype=np.float64),
    )

# ============================================================
# 13. RUN BOTH PAPER VARIANTS
# ============================================================
print("\n" + "=" * 100)
print("NON-NORMALIZED CLASSIFIER")
print("=" * 100)

theta_non_norm, train_M2_non_norm, train_M4_non_norm = learn_theta_from_benign(
    train_numeric,
    normalized=False
)

print(f"theta_non_normalized = {theta_non_norm:.6e}")
print(f"max benign-train M4  = {np.max(train_M4_non_norm):.6e}")

y_pred_non_norm, test_M2_non_norm, test_M4_non_norm = predict_labels(
    test_numeric,
    train_numeric,
    theta_non_norm,
    normalized=False
)

cm_non_norm, metrics_non_norm = anomaly_metrics(y_test_binary, y_pred_non_norm)

print("Decision rule:")
print("- H0: M4 <= theta  -> benign")
print("- H1: M4 >  theta  -> anomalous")
print()
print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
print(cm_non_norm)
print("\nMetrics:")
for k, v in metrics_non_norm.items():
    print(f"{k}: {v}")

print("\n" + "=" * 100)
print("NORMALIZED CLASSIFIER")
print("=" * 100)

theta_norm, train_M2_norm, train_M4_norm = learn_theta_from_benign(
    train_numeric,
    normalized=True
)

print(f"theta_normalized = {theta_norm:.6e}")
print(f"max benign-train M4 = {np.max(train_M4_norm):.6e}")

y_pred_norm, test_M2_norm, test_M4_norm = predict_labels(
    test_numeric,
    train_numeric,
    theta_norm,
    normalized=True
)

cm_norm, metrics_norm = anomaly_metrics(y_test_binary, y_pred_norm)

print("Decision rule:")
print("- H0: M4 <= theta  -> benign")
print("- H1: M4 >  theta  -> anomalous")
print()
print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
print(cm_norm)
print("\nMetrics:")
for k, v in metrics_norm.items():
    print(f"{k}: {v}")

# ============================================================
# 14. SAVE OUTPUTS
# ============================================================
# Original 0-based row number in test.csv (before empty-sequence filtering):
# stable join key to align predictions across the baseline scripts
test_csv_row = np.array([i for i, s in enumerate(test_sequences_all) if len(s) > 0], dtype=int)

preds_df = pd.DataFrame({
    "idx": np.arange(len(y_test_binary)),
    "test_csv_row": test_csv_row,
    "true_label": y_test_binary,
    "pred_label_non_normalized": y_pred_non_norm,
    "pred_label_normalized": y_pred_norm,

    "M2_non_normalized": test_M2_non_norm,
    "M4_non_normalized": test_M4_non_norm,
    "M2_normalized": test_M2_norm,
    "M4_normalized": test_M4_norm,

    # original signalling description of each test dialog, verbatim from test.csv
    "Replaced Signalling Description": [test_dialogs_raw[i] for i in test_csv_row],
})

summary_df = pd.DataFrame([
    {
        "baseline": "correlation_non_normalized",
        "theta": theta_non_norm,
        **metrics_non_norm,
    },
    {
        "baseline": "correlation_normalized",
        "theta": theta_norm,
        **metrics_norm,
    },
])

preds_df.to_csv("correlation_predictions.csv", index=False)
summary_df.to_csv("correlation_summary.csv", index=False)

print("\nSaved:")
print("- correlation_predictions.csv")
print("- correlation_summary.csv")

# ============================================================
# 15. STATISTICAL TESTS: BOOTSTRAP CI + McNEMAR'S TEST
# ============================================================
from scipy.stats import binomtest, chi2

print("\n" + "=" * 100)
print("STATISTICAL TESTS: BOOTSTRAP CI + McNEMAR'S TEST")
print("=" * 100)

N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 95.0

def bootstrap_metric_cis(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=BOOTSTRAP_CI, seed=SEED):
    """
    Nonparametric bootstrap (percentile method): resample the test set
    with replacement and recompute all metrics on each resample.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    _, point = anomaly_metrics(y_true, y_pred)
    skip = {"TN", "FP", "FN", "TP"}
    metric_names = [k for k in point if k not in skip]
    samples = {name: np.empty(n_boot, dtype=np.float64) for name in metric_names}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, m = anomaly_metrics(y_true[idx], y_pred[idx])
        for name in metric_names:
            samples[name][b] = m[name]

    lo_q = (100.0 - ci) / 2.0
    hi_q = 100.0 - lo_q

    return {
        name: {
            "point": float(point[name]),
            "ci_lo": float(np.percentile(samples[name], lo_q)),
            "ci_hi": float(np.percentile(samples[name], hi_q)),
        }
        for name in metric_names
    }

def print_bootstrap_cis(name, cis, ci=BOOTSTRAP_CI):
    print(f"\nBootstrap {ci:.0f}% CIs ({N_BOOTSTRAP} resamples) - {name}")
    for metric, v in cis.items():
        print(f"{metric:>12}: {v['point']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

def mcnemar_from_counts(n01, n10):
    """
    McNemar's test on discordant counts (n01, n10).
    Exact binomial when the number of discordant pairs is small,
    otherwise chi-square with continuity correction.
    """
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": 0, "n10": 0, "statistic": float("nan"),
                "p_value": 1.0, "method": "no discordant pairs"}
    if n_disc < 25:
        statistic = float(min(n01, n10))
        p_value = float(binomtest(min(n01, n10), n=n_disc, p=0.5).pvalue)
        method = "exact binomial"
    else:
        statistic = (abs(n01 - n10) - 1.0) ** 2 / n_disc
        p_value = float(chi2.sf(statistic, df=1))
        method = "chi-square, continuity corrected"
    return {"n01": int(n01), "n10": int(n10), "statistic": float(statistic),
            "p_value": p_value, "method": method}

def mcnemar_vs_truth(y_true, y_pred):
    """
    McNemar's test of marginal homogeneity between predictions and true
    labels: tests whether FP and FN errors are symmetric.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return mcnemar_from_counts(fp, fn)

def mcnemar_between_models(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test comparing two classifiers on the same test set.
    n01 = A wrong / B right, n10 = A right / B wrong.
    """
    y_true = np.asarray(y_true, dtype=int)
    a_correct = (np.asarray(y_pred_a, dtype=int) == y_true)
    b_correct = (np.asarray(y_pred_b, dtype=int) == y_true)
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    return mcnemar_from_counts(n01, n10)

def print_mcnemar(name, res, labels=("n01", "n10")):
    print(f"\nMcNemar's test - {name}")
    print(f"{labels[0]}: {res['n01']}, {labels[1]}: {res['n10']}")
    print(f"method   : {res['method']}")
    print(f"statistic: {res['statistic']:.4f}")
    print(f"p-value  : {res['p_value']:.6f}")

# Bootstrap CIs for both paper variants
boot_non_norm = bootstrap_metric_cis(y_test_binary, y_pred_non_norm)
print_bootstrap_cis("correlation_non_normalized", boot_non_norm)

boot_norm = bootstrap_metric_cis(y_test_binary, y_pred_norm)
print_bootstrap_cis("correlation_normalized", boot_norm)

# McNemar vs truth (FP vs FN symmetry) for each variant
mcnemar_non_norm = mcnemar_vs_truth(y_test_binary, y_pred_non_norm)
print_mcnemar("correlation_non_normalized (FP vs FN)", mcnemar_non_norm, labels=("FP", "FN"))

mcnemar_norm = mcnemar_vs_truth(y_test_binary, y_pred_norm)
print_mcnemar("correlation_normalized (FP vs FN)", mcnemar_norm, labels=("FP", "FN"))

# McNemar between the two paper variants
mcnemar_variants = mcnemar_between_models(y_test_binary, y_pred_non_norm, y_pred_norm)
print_mcnemar(
    "non_normalized vs normalized",
    mcnemar_variants,
    labels=("non_norm wrong / norm right", "non_norm right / norm wrong"),
)