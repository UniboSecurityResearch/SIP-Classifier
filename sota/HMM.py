import numpy as np
import pandas as pd
from collections import defaultdict
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -------------------------------------------------------------------
# 1. Load benign / anomalous datasets (duplicates already removed)
# -------------------------------------------------------------------
benign_csv    = "benign.csv"
anomalous_csv = "anomalous.csv"

df_benign    = pd.read_csv(benign_csv)
df_anomalous = pd.read_csv(anomalous_csv)

print("Benign dialogs (rows):   ", len(df_benign))
print("Anomalous dialogs (rows):", len(df_anomalous))

benign_dialogs    = df_benign["Signaling Flow"].tolist()
anomalous_dialogs = df_anomalous["Signaling Flow"].tolist()

# -------------------------------------------------------------------
# 2. Build SIP methods and return codes vocabularies from BENIGN only
#    (same parsing logic as in your LSTM/CNN scripts)
# -------------------------------------------------------------------
methods = set()
codes   = set()
seqs_benign = []
max_len = 0

for d in benign_dialogs:
    seq = []
    for msg in str(d).split(":"):
        parts = msg.split(",")
        if len(parts) < 3:
            continue
        tok = parts[2]
        if "-" in tok:
            m, c = tok.split("-", 1)
            m = m.strip()
            c = c.strip()
            seq.append(m)
            seq.append(c)
            methods.add(m)
            codes.add(c)
        else:
            tok = tok.strip()
            seq.append(tok)
            methods.add(tok)
    if len(seq) > max_len:
        max_len = len(seq)
    seqs_benign.append(seq)

symbols = methods | codes
symbols.add("<PAD>")

message2idx = {m: i for i, m in enumerate(sorted(symbols))}

print("Example benign sequence:", seqs_benign[0])
print("Vocabulary size (methods+codes, excl. <PAD>):", len(symbols) - 1)
print("Max benign sequence length:", max_len)
print("message2idx size:", len(message2idx))

# -------------------------------------------------------------------
# 3. Unique benign dialogs (each dialog is a "known" pattern)
# -------------------------------------------------------------------
dialogs_str_benign = [" ".join(s) for s in seqs_benign]

labels_benign, uniques_benign = pd.factorize(dialogs_str_benign)
N_benign = len(uniques_benign)

print("\nNumber of unique benign dialogs:", N_benign)

# -------------------------------------------------------------------
# 4. Map tokens to integers (excluding <PAD>) and build int sequences
# -------------------------------------------------------------------
token2idx = {tok: i for tok, i in message2idx.items() if tok != "<PAD>"}
idx2token = {i: tok for tok, i in token2idx.items()}

# benign integer sequences
int_seqs_benign = []
for s in seqs_benign:
    int_seqs_benign.append([token2idx[tok] for tok in s if tok in token2idx])

# unique benign integer sequences (1:1 with uniques_benign)
unique_int_seqs_benign = []
for dialog_str in uniques_benign:
    idx = dialogs_str_benign.index(dialog_str)
    unique_int_seqs_benign.append(int_seqs_benign[idx])

print("Number of unique_int_seqs_benign:", len(unique_int_seqs_benign))
print("Example integer benign sequence:", unique_int_seqs_benign[0])

# anomalous integer sequences, parsed with the SAME logic/vocab
seqs_anom = []
for d in anomalous_dialogs:
    seq = []
    for msg in str(d).split(":"):
        parts = msg.split(",")
        if len(parts) < 3:
            continue
        tok = parts[2]
        if "-" in tok:
            m, c = tok.split("-", 1)
            m = m.strip()
            c = c.strip()
            seq.append(m)
            seq.append(c)
        else:
            tok = tok.strip()
            seq.append(tok)
    # map to integers; tokens unseen in benign can be skipped or mapped to a dummy
    int_seq = [token2idx[t] for t in seq if t in token2idx]
    if len(int_seq) == 0:
        # if everything is unknown, we can at least put one dummy symbol
        int_seq = [max(token2idx.values()) + 1]
    seqs_anom.append(int_seq)

# deduplicate anomalous integer sequences
unique_int_seqs_anom = []
seen_anom = set()
for s in seqs_anom:
    key = tuple(s)
    if key not in seen_anom:
        seen_anom.add(key)
        unique_int_seqs_anom.append(s)

print("Number of unique_int_seqs_anom:", len(unique_int_seqs_anom))
print("Example integer anomalous sequence:", unique_int_seqs_anom[0])

unique_int_seqs_benign = np.array(unique_int_seqs_benign, dtype=object)
unique_int_seqs_anom   = np.array(unique_int_seqs_anom,   dtype=object)

# -------------------------------------------------------------------
# 5. Train/test split on unique benign dialogs (no stratify)
# -------------------------------------------------------------------
train_seqs_benign, test_seqs_benign = train_test_split(
    unique_int_seqs_benign,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

print("\nTrain benign dialogs:", len(train_seqs_benign))
print("Test  benign dialogs:", len(test_seqs_benign))

# -------------------------------------------------------------------
# 6. Build n-gram counts for each TRAIN benign dialog-class
# -------------------------------------------------------------------
n = 13

# special symbol for padding in the n-gram window
PAD_NGRAM = max(token2idx.values()) + 1

def dialog_ngrams(seq, n=n):
    """
    seq: list of integers (a dialog)
    returns a list of n-grams (tuples of integers) with padding at the beginning and end
    """
    pad = [PAD_NGRAM] * (n - 1)
    padded = pad + seq + pad
    grams = []
    for i in range(len(padded) - n + 1):
        grams.append(tuple(padded[i : i + n]))
    return grams

dialog_ngram_counts = []
dialog_total_counts = []
global_vocabulary_ngrams = set()

for seq in train_seqs_benign:
    grams = dialog_ngrams(seq, n=n)
    counts = defaultdict(int)
    for g in grams:
        counts[g] += 1
        global_vocabulary_ngrams.add(g)
    dialog_ngram_counts.append(counts)
    dialog_total_counts.append(sum(counts.values()))

V = len(global_vocabulary_ngrams)
print("\nNumber of distinct n-grams across TRAIN benign dialogs:", V)

# -------------------------------------------------------------------
# 7. HMM+n-gram log-likelihood and helpers
# -------------------------------------------------------------------
alpha = 1.0  # smoothing

def log_likelihood_for_class(seq, class_id, n=n):
    grams = dialog_ngrams(seq, n=n)
    counts = dialog_ngram_counts[class_id]
    total  = dialog_total_counts[class_id]
    denom  = total + alpha * V

    logp = 0.0
    for g in grams:
        c = counts.get(g, 0)
        prob = (c + alpha) / denom
        logp += math.log(prob)
    return logp

def hmm_best_log_likelihood(seq, n=n):
    """
    seq: list of integers (complete observed dialog)
    returns (best_score, best_class) over TRAIN benign classes
    """
    best_class = None
    best_score = -1e30
    for class_id in range(len(train_seqs_benign)):
        score = log_likelihood_for_class(seq, class_id, n=n)
        if score > best_score:
            best_score = score
            best_class = class_id
    return best_score, best_class

# -------------------------------------------------------------------
# 8. Threshold on best log-likelihood to classify known/unknown
# -------------------------------------------------------------------
# Compute best scores on TRAIN benign dialogs
best_scores_train = []
for seq in train_seqs_benign:
    score, _ = hmm_best_log_likelihood(seq, n=n)
    best_scores_train.append(score)

best_scores_train = np.array(best_scores_train)
Theta = best_scores_train.mean()  # simple threshold = mean of train scores

print("\nMean best log-likelihood on TRAIN benign dialogs:", Theta)
print("Min/Max best train log-likelihood:", best_scores_train.min(), best_scores_train.max())

def classify_known_unknown(seq, Theta, n=n):
    """
    Returns 0 = known (benign-like), 1 = unknown (anomalous-like)
    based on best log-likelihood vs threshold Theta.
    """
    score, _ = hmm_best_log_likelihood(seq, n=n)
    return 0 if score >= Theta else 1

# -------------------------------------------------------------------
# 9. Experiment 1 – IV.B-like Detection on benign (train/test)
#    PD_train / PD_test = fraction of benign classified as known (0)
# -------------------------------------------------------------------
def detection_pd_normals(seq_list, Theta):
    y_true = np.zeros(len(seq_list), dtype=int)  # all known
    y_pred = np.zeros(len(seq_list), dtype=int)
    for i, seq in enumerate(seq_list):
        y_pred[i] = classify_known_unknown(seq, Theta, n=n)
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    return acc, cm

pd_train, cm_train = detection_pd_normals(train_seqs_benign, Theta)
pd_test,  cm_test  = detection_pd_normals(test_seqs_benign,  Theta)

print("\n=== Experiment 1 – IV.B-like Detection on benign ===")
print(f"PD_train (benign) = {pd_train:.4f}")
print("Confusion matrix TRAIN (rows=true [0=benign,1=anom], cols=pred):\n", cm_train)
print(f"PD_test  (benign) = {pd_test:.4f}")
print("Confusion matrix TEST  (rows=true [0=benign,1=anom], cols=pred):\n", cm_test)


# -------------------------------------------------------------------
# 10. Helper: paper-style unknown detection metrics
# -------------------------------------------------------------------
def report_unknown(y_true, y_pred):
    """
    Paper-style evaluation for unknown detection.

    Here:
        y_true: 0 = known/benign, 1 = unknown/anomalous
        y_pred: 0 = predicted known, 1 = predicted unknown

    Internally:
        True  = known
        False = unknown

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

    # Normalized rates
    tn_rate = TN / total_unknown if total_unknown > 0 else 0.0
    fp_rate = FP / total_unknown if total_unknown > 0 else 0.0
    tp_rate = TP / total_known   if total_known   > 0 else 0.0
    fn_rate = FN / total_known   if total_known   > 0 else 0.0

    # Metrics (known as positive class)
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


# -------------------------------------------------------------------
# 11. Helper for Unknown detection experiments (IV.D-style)
# -------------------------------------------------------------------
def evaluate_unknown_detection(benign_seqs, anomalous_seqs, Theta, label):
    """
    benign_seqs   : list/array of benign sequences (ground truth = 0)
    anomalous_seqs: list/array of anomalous sequences (ground truth = 1)
    """
    all_seqs = list(benign_seqs) + list(anomalous_seqs)
    y_true = np.concatenate([
        np.zeros(len(benign_seqs), dtype=int),   # known
        np.ones(len(anomalous_seqs), dtype=int)  # unknown
    ])

    y_pred = []
    for seq in all_seqs:
        y_pred.append(classify_known_unknown(seq, Theta, n=n))
    y_pred = np.array(y_pred)

    cm, rates, metrics = report_unknown(y_true, y_pred)

    print(f"\n=== Experiment 2/3 – {label} ===")
    print("Confusion Matrix (rows=true [unknown, known], cols=pred [unknown, known]):\n", cm)
    print(f"TN={rates['TN']}, FP={rates['FP']}, FN={rates['FN']}, TP={rates['TP']}")
    print(
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Precision(known)={metrics['precision']:.4f}, "
        f"Sensitivity(known)={metrics['sensitivity']:.4f}, "
        f"Specificity(unknown)={metrics['specificity']:.4f}, "
        f"F1(known)={metrics['f1']:.4f}"
    )

    return cm, rates, metrics

# -------------------------------------------------------------------
# 12. Experiment 2 – IV.D CLEAN
#     Eval set = test benign + ALL anomalous
# -------------------------------------------------------------------
cm_clean, rates_clean, metrics_clean = evaluate_unknown_detection(
    test_seqs_benign,
    unique_int_seqs_anom,
    Theta,
    label="IV.D CLEAN – test benign + anomalous"
)

# -------------------------------------------------------------------
# 13. Experiment 3 – IV.D FULL
#     Eval set = train benign + test benign + ALL anomalous
# -------------------------------------------------------------------
benign_all = np.concatenate([train_seqs_benign, test_seqs_benign])

cm_full, rates_full, metrics_full = evaluate_unknown_detection(
    benign_all,
    unique_int_seqs_anom,
    Theta,
    label="IV.D FULL – train+test benign + anomalous"
)
