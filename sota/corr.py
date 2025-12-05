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
    y_true = np.zeros(len(Du_list), dtype=int)
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
print("Confusion matrix TRAIN (rows=true, cols=pred):\n", cm_train)
print(f"PD_test  (normals) = {pd_test:.4f}")
print("Confusion matrix TEST (rows=true, cols=pred):\n", cm_test)


# ===========================================
# 2) IV.D – Clean Unknown SIP Dialog Detection
#     Eval set = Du_N_test (known) + Du_A (unknown)
# ===========================================

def evaluate_unknown_detection(Du_known, Du_unknown, Du_N_pad_train, LS, Gamma, label):
    """
    Du_known: list/array of normal dialogs (ground truth = 0)
    Du_unknown: list/array of anomalous dialogs (ground truth = 1)
    """
    # Build evaluation set
    all_seqs = list(Du_known) + list(Du_unknown)
    y_true = np.concatenate([
        np.zeros(len(Du_known), dtype=int),  # known
        np.ones(len(Du_unknown), dtype=int)  # unknown
    ])

    y_pred = []
    for seq in all_seqs:
        # classify_dialog_corr returns 0 for normal, 1 for anomalous
        y_pred.append(classify_dialog_corr(seq, Du_N_pad_train, LS, Gamma))
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    # Consider "unknown" (1) as the positive class here
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred,  pos_label=1, zero_division=0)
    f1   = f1_score(y_true,  y_pred,     pos_label=1, zero_division=0)

    print(f"\n=== IV.D – {label} Unknown Detection ===")
    print("Confusion Matrix (rows=true [0=known,1=unknown], cols=pred):\n", cm)
    print(f"Accuracy={acc:.4f}, Precision(unknown)={prec:.4f}, Recall(unknown)={rec:.4f}, F1(unknown)={f1:.4f}")

    return cm, acc, prec, rec, f1

# Clean: only test normals + all anomalous
cm_clean, acc_clean, prec_clean, rec_clean, f1_clean = evaluate_unknown_detection(
    Du_N_test, Du_A, Du_N_train_pad, LS, Gamma, label="CLEAN (test normals + anomalous)"
)


# ===========================================
# 3) IV.D – Full Unknown SIP Dialog Detection
#     Eval set = Du_N_train + Du_N_test (known) + Du_A (unknown)
# ===========================================

Du_N_all = np.concatenate([Du_N_train, Du_N_test])

cm_full, acc_full, prec_full, rec_full, f1_full = evaluate_unknown_detection(
    Du_N_all, Du_A, Du_N_train_pad, LS, Gamma, label="FULL (train+test normals + anomalous)"
)
