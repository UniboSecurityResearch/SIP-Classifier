# SIP Correlation-Based Abnormal Dialog Detection
# 3 experiments:
# 1) IV.B-like: PD on benign train/test
# 2) IV.D clean: Gamma from benign train, test on benign test + anomalous
# 3) IV.D full: Gamma from benign train, test on benign train+test + anomalous

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

# ===========================================
# Load benign / anomalous CSVs
# ===========================================

benign_csv    = "benign.csv"
anomalous_csv = "anomalous.csv"

df_benign    = pd.read_csv(benign_csv)
df_anomalous = pd.read_csv(anomalous_csv)

print("Benign dialogs (rows):   ", len(df_benign))
print("Anomalous dialogs (rows):", len(df_anomalous))

# -------------------------------------------
# Parse the "Signaling Flow" column
# Example flow: "GTW,ICSCF,I:ICSCF,..."
# -------------------------------------------

def parse_flow(flow_str: str):
    """Splits a SIP signaling flow string into a list of tokens."""
    return [tok.strip() for tok in str(flow_str).split(",") if tok.strip() != ""]

flows_tokens_benign    = df_benign["Signaling Flow"].apply(parse_flow).tolist()
flows_tokens_anomalous = df_anomalous["Signaling Flow"].apply(parse_flow).tolist()

print("\nExample parsed benign flow:", flows_tokens_benign[0][:15], "...")
print("Benign dialog length:", len(flows_tokens_benign[0]))

# -------------------------------------------
# Build token -> integer dictionary
# (vocabulary from BOTH benign and anomalous)
# -------------------------------------------

all_tokens = set()
for seq in flows_tokens_benign:
    all_tokens.update(seq)
for seq in flows_tokens_anomalous:
    all_tokens.update(seq)

all_tokens = sorted(all_tokens)
token2idx = {tok: i + 1 for i, tok in enumerate(all_tokens)}  # indices start at 1
idx2token = {i: tok for tok, i in token2idx.items()}

print("\nNumber of unique tokens:", len(token2idx))

# Convert dialogs to integer sequences
dialogs_int_benign = [[token2idx[tok] for tok in seq] for seq in flows_tokens_benign]
dialogs_int_anom   = [[token2idx[tok] for tok in seq] for seq in flows_tokens_anomalous]

print("\nExample integer-encoded benign dialog:", dialogs_int_benign[0][:15], "...")
print("Example integer-encoded anomalous dialog:", dialogs_int_anom[0][:15], "...")

dialogs_int_benign = np.array(dialogs_int_benign, dtype=object)
dialogs_int_anom   = np.array(dialogs_int_anom,   dtype=object)

normal_dialogs    = dialogs_int_benign
anomalous_dialogs = dialogs_int_anom

print("\nNormal (benign) dialogs:", len(normal_dialogs))
print("Anomalous dialogs:", len(anomalous_dialogs))


# ===========================================
# Deduplicate dialogs (as done in the paper)
# ===========================================

def unique_sequences(seq_list):
    """Removes duplicate sequences while preserving order."""
    seen = set()
    unique = []
    for s in seq_list:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique

Du_N = unique_sequences(normal_dialogs)
Du_A = unique_sequences(anomalous_dialogs)

print("\nUnique normal dialogs:", len(Du_N))
print("Unique anomalous dialogs:", len(Du_A))

Du_N = np.array(Du_N, dtype=object)
Du_A = np.array(Du_A, dtype=object)


# ===========================================
# Train/test split on UNIQUE normal dialogs (no stratify)
# ===========================================

Du_N_train, Du_N_test = train_test_split(
    Du_N,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

print("\nTrain normal dialogs:", len(Du_N_train))
print("Test  normal dialogs:", len(Du_N_test))

# ===========================================
# Pad dialogs to the same length LS
# ===========================================

max_len_N_train = max(len(s) for s in Du_N_train)
max_len_N_test  = max(len(s) for s in Du_N_test) if len(Du_N_test) > 0 else 0
max_len_A       = max(len(s) for s in Du_A)      if len(Du_A)      > 0 else 0

LS = max(max_len_N_train, max_len_N_test, max_len_A)

print("\nMaximum dialog length (LS):", LS)

def pad_sequence(seq, LS):
    """Pads a sequence with zeros up to length LS."""
    seq = list(seq)
    if len(seq) > LS:
        seq = seq[:LS]
    return np.array(seq + [0] * (LS - len(seq)), dtype=np.int32)

Du_N_train_pad = np.stack([pad_sequence(s, LS) for s in Du_N_train])
Du_N_test_pad  = np.stack([pad_sequence(s, LS) for s in Du_N_test]) if len(Du_N_test) > 0 else np.zeros((0, LS), dtype=np.int32)
Du_A_pad       = np.stack([pad_sequence(s, LS) for s in Du_A])      if len(Du_A)      > 0 else np.zeros((0, LS), dtype=np.int32)

print("Du_N_train_pad shape:", Du_N_train_pad.shape)
print("Du_N_test_pad  shape:", Du_N_test_pad.shape)
print("Du_A_pad       shape:", Du_A_pad.shape)


# ===========================================
# CNN matrix: cross-correlation between TRAIN normal dialogs
# CNN[i, j] = sum_h Du_N_train_pad[i,h] * Du_N_train_pad[j,h]
# ===========================================

CNN = Du_N_train_pad @ Du_N_train_pad.T
print("\nCNN matrix shape:", CNN.shape)

# -------------------------------------------
# 4th central moment (non-normalized kurtosis)
# -------------------------------------------

def central_moment_4(r: np.ndarray) -> float:
    """Computes the 4th central moment of a vector."""
    mu = r.mean()
    return np.mean((r - mu) ** 4)

M4_normals_train = np.apply_along_axis(central_moment_4, axis=1, arr=CNN)

# Gamma slightly above the maximum M4 of TRAIN normal dialogs
Gamma = M4_normals_train.max() * 1.0001

print("Gamma threshold =", Gamma)


# ===========================================
# Classifier: 0 = Normal, 1 = Anomalous
# ===========================================

def classify_dialog_corr(seq_ints, Du_N_pad_train, LS, Gamma):
    """Classifies a dialog based on cross-correlation kurtosis."""
    sk = pad_sequence(seq_ints, LS)
    r  = Du_N_pad_train @ sk  # cross-correlation with all TRAIN normal dialogs
    M4 = central_moment_4(r)
    return 0 if M4 <= Gamma else 1


# ===========================================
# 1) IV.B-like – Detection performance on normal dialogs
#     PD_train: % of TRAIN normals classified as normal
#     PD_test : % of TEST  normals classified as normal
# ===========================================

def detection_pd_normals(Du_list, Du_N_pad_train, LS, Gamma):
    y_true = np.zeros(len(Du_list), dtype=int)   # all should be class 0 (normal)
    y_pred = np.zeros(len(Du_list), dtype=int)
    for i, seq in enumerate(Du_list):
        y_pred[i] = classify_dialog_corr(seq, Du_N_pad_train, LS, Gamma)
    # PD = accuracy over normals (they should all be class 0)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    return acc, cm

pd_train, cm_train = detection_pd_normals(Du_N_train, Du_N_train_pad, LS, Gamma)
pd_test,  cm_test  = detection_pd_normals(Du_N_test,  Du_N_train_pad, LS, Gamma)

print("\n=== IV.B-like – Detection on benign (normal) dialogs ===")
print(f"PD_train (normals) = {pd_train:.4f}")
print("Confusion matrix TRAIN (rows=true [0=normal,1=abnormal], cols=pred):\n", cm_train)
print(f"PD_test  (normals) = {pd_test:.4f}")
print("Confusion matrix TEST (rows=true [0=normal,1=abnormal], cols=pred):\n", cm_test)


# ===========================================
# Helper: paper-style unknown detection metrics
# ===========================================

def report_unknown(y_true, y_pred):
    """
    Paper-style evaluation for unknown SIP dialog detection.

    Here:
        y_true: 0 = known/normal, 1 = unknown/anomalous
        y_pred: 0 = predicted known, 1 = predicted unknown

    We convert to boolean with:
        True  = known
        False = unknown
    to be consistent with the other scripts.

    Returns:
        cm      : 2x2 confusion matrix
        rates   : dict with raw TN, FP, FN, TP and normalized tn, fp, fn, tp
        metrics : dict with specificity, sensitivity, precision, accuracy, f1
    """
    # True = known, False = unknown
    y_true_known = (y_true == 0)
    y_pred_known = (y_pred == 0)

    cm = confusion_matrix(y_true_known, y_pred_known)

    # cm rows: [true unknown, true known], cols: [pred unknown, pred known]
    TN = cm[0, 0]   # true unknown  → predicted unknown
    FP = cm[0, 1]   # true unknown  → predicted known
    FN = cm[1, 0]   # true known    → predicted unknown
    TP = cm[1, 1]   # true known    → predicted known

    total_unknown = TN + FP
    total_known   = TP + FN
    total_all     = TN + FP + FN + TP

    # Normalized rates (same style as other scripts)
    tn_rate = TN / total_unknown if total_unknown > 0 else 0.0
    fp_rate = FP / total_unknown if total_unknown > 0 else 0.0
    tp_rate = TP / total_known   if total_known   > 0 else 0.0
    fn_rate = FN / total_known   if total_known   > 0 else 0.0

    # Metrics
    specificity = tn_rate                  # correct detection of unknown dialogs
    sensitivity = tp_rate                  # correct detection of known dialogs
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy    = (TP + TN) / total_all if total_all > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) > 0 else 0.0)

    rates = dict(
        TN=TN, FP=FP, FN=FN, TP=TP,
        tn=tn_rate, fp=fp_rate, fn=fn_rate, tp=tp_rate
    )

    metrics = dict(
        specificity=specificity,
        sensitivity=sensitivity,
        precision=precision,
        accuracy=accuracy,
        f1=f1,
    )

    return cm, rates, metrics


# ===========================================
# 2) IV.D – Clean Unknown SIP Dialog Detection
#     Eval set = Du_N_test (known) + Du_A (unknown)
# ===========================================

def evaluate_unknown_detection(Du_known, Du_unknown, Du_N_pad_train, LS, Gamma):
    """
    Du_known   : list/array of normal dialogs (ground truth = 0)
    Du_unknown : list/array of anomalous dialogs (ground truth = 1)

    Returns:
        cm, rates, metrics as from report_unknown()
    """
    # Build evaluation set
    all_seqs = list(Du_known) + list(Du_unknown)
    y_true = np.concatenate([
        np.zeros(len(Du_known), dtype=int),  # known = 0
        np.ones(len(Du_unknown), dtype=int)  # unknown = 1
    ])

    y_pred = []
    for seq in all_seqs:
        # classify_dialog_corr returns 0 for normal, 1 for anomalous
        y_pred.append(classify_dialog_corr(seq, Du_N_pad_train, LS, Gamma))
    y_pred = np.array(y_pred)

    cm, rates, metrics = report_unknown(y_true, y_pred)
    return cm, rates, metrics

# Clean: only test normals + all anomalous
cm_clean, rates_clean, metrics_clean = evaluate_unknown_detection(
    Du_N_test, Du_A, Du_N_train_pad, LS, Gamma
)

print("\n=== IV.D – CLEAN (test normals + anomalous) Unknown Detection ===")
print("Confusion Matrix (rows=true [unknown, known], cols=pred [unknown, known]):\n", cm_clean)
print(f"TN={rates_clean['TN']}, FP={rates_clean['FP']}, FN={rates_clean['FN']}, TP={rates_clean['TP']}")
print(
    f"Accuracy={metrics_clean['accuracy']:.4f}, "
    f"Precision(known)={metrics_clean['precision']:.4f}, "
    f"Sensitivity(known)={metrics_clean['sensitivity']:.4f}, "
    f"Specificity(unknown)={metrics_clean['specificity']:.4f}, "
    f"F1(known)={metrics_clean['f1']:.4f}"
)


# ===========================================
# 3) IV.D – Full Unknown SIP Dialog Detection
#     Eval set = Du_N_train + Du_N_test (known) + Du_A (unknown)
# ===========================================

Du_N_all = np.concatenate([Du_N_train, Du_N_test])

cm_full, rates_full, metrics_full = evaluate_unknown_detection(
    Du_N_all, Du_A, Du_N_train_pad, LS, Gamma
)

print("\n=== IV.D – FULL (train+test normals + anomalous) Unknown Detection ===")
print("Confusion Matrix (rows=true [unknown, known], cols=pred [unknown, known]):\n", cm_full)
print(f"TN={rates_full['TN']}, FP={rates_full['FP']}, FN={rates_full['FN']}, TP={rates_full['TP']}")
print(
    f"Accuracy={metrics_full['accuracy']:.4f}, "
    f"Precision(known)={metrics_full['precision']:.4f}, "
    f"Sensitivity(known)={metrics_full['sensitivity']:.4f}, "
    f"Specificity(unknown)={metrics_full['specificity']:.4f}, "
    f"F1(known)={metrics_full['f1']:.4f}"
)
