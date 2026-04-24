import os
import math
import random
import numpy as np
import pandas as pd

from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

VALIDATION_SIZE = 0.2

# HMM / observation model config
NGRAM_ORDER = 3                  # 1=unigram, 2=bigram, 3=trigram
ALPHA_EMISSION = 0.5             # Laplace smoothing for emissions
SELF_TRANSITION = 0.995          # sticky state transitions
THRESHOLD_PERCENTILE = 1.0       # benign validation percentile for rejection
STRICT_EMPTY_PATH_REJECTION = False
USE_FORWARD_SCORE_FOR_REJECTION = True

UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"

# ============================================================
# 2. UTILITIES
# ============================================================
def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

def anomaly_metrics(y_true, y_pred):
    """
    y_true: 0=benign, 1=anomalous
    y_pred: 0=benign, 1=anomalous
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

    metrics = {
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
    }
    return cm, metrics

def print_metrics(name, y_true, y_pred):
    cm, metrics = anomaly_metrics(y_true, y_pred)
    print(f"\n{name}")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)
    print(metrics)
    return cm, metrics

def seq_to_text(seq):
    return " ".join(seq)

def stable_logsumexp(arr):
    arr = np.asarray(arr, dtype=np.float64)
    m = np.max(arr)
    if np.isneginf(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(arr - m))))

def logsubexp(a, b):
    """
    Computes log(exp(a) - exp(b)) for a >= b.
    Returns -inf if result is 0 or invalid.
    """
    if np.isneginf(a):
        return -np.inf
    if b >= a:
        return -np.inf
    return float(a + math.log1p(-math.exp(b - a)))

def argmax_and_second_best(x):
    best_idx = int(np.argmax(x))
    best_val = float(x[best_idx])

    if len(x) == 1:
        return best_idx, best_val, best_idx, -np.inf

    tmp = x.copy()
    tmp[best_idx] = -np.inf
    second_idx = int(np.argmax(tmp))
    second_val = float(tmp[second_idx])
    return best_idx, best_val, second_idx, second_val

# ============================================================
# 3. PARSER
# ============================================================
def parse_dialog(dialog_str):
    """
    Extract SIP token sequence from the signaling description field.

    Current rule:
    - split on ':'
    - for each chunk, take the 3rd comma-separated field
    - if token has form 'METHOD-CODE', split it into two tokens
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
print_header("LOADING DATA")

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
y_test_binary_raw = df_test[LABEL_COL].astype(int).to_numpy()

print(f"Train shape: {df_train.shape}")
print(f"Test shape : {df_test.shape}")
print("Test label distribution:")
print(df_test[LABEL_COL].value_counts(dropna=False).sort_index())

train_sequences_all = [parse_dialog(d) for d in train_dialogs_raw]
test_sequences_all = [parse_dialog(d) for d in test_dialogs_raw]

train_sequences_all = [s for s in train_sequences_all if len(s) > 0]
test_pairs = [(s, y) for s, y in zip(test_sequences_all, y_test_binary_raw) if len(s) > 0]
test_sequences = [x[0] for x in test_pairs]
y_test_binary = np.array([x[1] for x in test_pairs], dtype=int)

print(f"Valid train dialogs: {len(train_sequences_all)}")
print(f"Valid test dialogs : {len(test_sequences)}")
print(f"Valid benign test  : {np.sum(y_test_binary == BENIGN_LABEL_VALUE)}")
print(f"Valid anomalous    : {np.sum(y_test_binary == ANOMALOUS_LABEL_VALUE)}")

if len(train_sequences_all) == 0:
    raise ValueError("No valid training sequences after parsing.")
if len(test_sequences) == 0:
    raise ValueError("No valid test sequences after parsing.")

# benign-only split
train_sequences, val_sequences = train_test_split(
    train_sequences_all,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    shuffle=True
)

print(f"Train benign split: {len(train_sequences)}")
print(f"Val benign split  : {len(val_sequences)}")

# ============================================================
# 5. HIDDEN STATES = KNOWN BENIGN DIALOG CLASSES
# ============================================================
train_texts = [seq_to_text(s) for s in train_sequences]
known_dialogs = list(pd.unique(train_texts))
dialog_to_state = {dlg: i for i, dlg in enumerate(known_dialogs)}
state_to_dialog = {i: dlg for dlg, i in dialog_to_state.items()}
N_STATES = len(known_dialogs)

def map_to_state(seq):
    return dialog_to_state.get(seq_to_text(seq), -1)

y_train_state = np.array([map_to_state(s) for s in train_sequences], dtype=int)
assert np.all(y_train_state >= 0)

print_header("HMM SETUP SUMMARY")
print(f"Known benign states (unique dialogs) in training: {N_STATES}")
print("Main task:")
print("- Train only on benign dialogs")
print("- Evaluate binary anomaly detection on test.csv")
print("- 0 = benign, 1 = anomalous")

# start prior over states
state_counts = np.bincount(y_train_state, minlength=N_STATES).astype(np.float64)
pi = (state_counts + 1.0) / (state_counts.sum() + N_STATES)
log_pi = np.log(pi)

# sticky transitions
if N_STATES == 1:
    LOG_A_SELF = 0.0
    LOG_A_OTHER = -np.inf
else:
    LOG_A_SELF = math.log(SELF_TRANSITION)
    LOG_A_OTHER = math.log((1.0 - SELF_TRANSITION) / (N_STATES - 1))

# ============================================================
# 6. VOCABULARY + NORMALIZATION
# ============================================================
token_vocab = sorted({tok for seq in train_sequences for tok in seq} | {UNK_TOKEN, BOS_TOKEN})
token_set = set(token_vocab)
VOCAB_SIZE = len(token_vocab)

def normalize_seq(seq):
    return [tok if tok in token_set else UNK_TOKEN for tok in seq]

train_sequences_norm = [normalize_seq(s) for s in train_sequences]
val_sequences_norm = [normalize_seq(s) for s in val_sequences]
test_sequences_norm = [normalize_seq(s) for s in test_sequences]

# ============================================================
# 7. CLASS-CONDITIONAL N-GRAM OBSERVATION MODEL
#    emissions[state][order][context][token] = count
# ============================================================
print_header("BUILDING CLASS-CONDITIONAL OBSERVATION MODEL")

emission_counts = [
    [defaultdict(Counter) for _ in range(NGRAM_ORDER)]
    for _ in range(N_STATES)
]
context_totals = [
    [Counter() for _ in range(NGRAM_ORDER)]
    for _ in range(N_STATES)
]

for seq, state in zip(train_sequences_norm, y_train_state):
    padded = [BOS_TOKEN] * (NGRAM_ORDER - 1) + seq

    for t in range(NGRAM_ORDER - 1, len(padded)):
        tok = padded[t]

        for order in range(NGRAM_ORDER):
            if order == 0:
                ctx = ()
            else:
                ctx = tuple(padded[t - order:t])

            emission_counts[state][order][ctx][tok] += 1
            context_totals[state][order][ctx] += 1

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"N-gram order  : {NGRAM_ORDER}")

def emission_logprob_for_state(state, full_context, tok):
    """
    Backoff from highest available order down to unigram.
    full_context is the full (NGRAM_ORDER-1)-length context.
    """
    max_order = min(NGRAM_ORDER - 1, len(full_context))

    for order in range(max_order, -1, -1):
        if order == 0:
            ctx = ()
        else:
            ctx = tuple(full_context[-order:])

        total = context_totals[state][order].get(ctx, 0)
        row = emission_counts[state][order].get(ctx)

        if total > 0:
            count = 0 if row is None else row.get(tok, 0)

            if STRICT_EMPTY_PATH_REJECTION and count == 0:
                continue

            p = (count + ALPHA_EMISSION) / (total + ALPHA_EMISSION * VOCAB_SIZE)
            return math.log(p)

    if STRICT_EMPTY_PATH_REJECTION:
        return -np.inf

    # fallback uniform if literally nothing matches
    return -math.log(VOCAB_SIZE)

def emission_log_matrix(seq):
    """
    Returns matrix E with shape [T, N_STATES],
    where E[t, s] = log p(o_t | state=s, context)
    """
    seq = normalize_seq(seq)
    T = len(seq)
    padded = [BOS_TOKEN] * (NGRAM_ORDER - 1) + seq
    E = np.empty((T, N_STATES), dtype=np.float64)

    for t in range(T):
        tok = padded[t + (NGRAM_ORDER - 1)]
        full_context = tuple(padded[t:t + (NGRAM_ORDER - 1)])

        for s in range(N_STATES):
            E[t, s] = emission_logprob_for_state(s, full_context, tok)

    return E

# ============================================================
# 8. VITERBI DECODING (O(T * N))
#    Optimized for sticky transitions:
#    - self transition has one probability
#    - all switches share the same probability
# ============================================================
def viterbi_decode(emissions):
    """
    emissions: [T, N_STATES] matrix of log-emissions
    Returns:
        path       : most likely hidden-state path
        best_score : log-score of best path
        final_state: final state of the Viterbi path
    """
    T, N = emissions.shape
    psi = np.full((T, N), -1, dtype=np.int32)

    delta = log_pi + emissions[0]
    if np.all(np.isneginf(delta)):
        return [], -np.inf, -1

    for t in range(1, T):
        new_delta = np.full(N, -np.inf, dtype=np.float64)

        best_idx, best_val, second_idx, second_val = argmax_and_second_best(delta)

        for j in range(N):
            stay_score = delta[j] + LOG_A_SELF

            if N == 1:
                switch_score = -np.inf
                switch_prev = -1
            else:
                if best_idx == j:
                    switch_score = second_val + LOG_A_OTHER
                    switch_prev = second_idx
                else:
                    switch_score = best_val + LOG_A_OTHER
                    switch_prev = best_idx

            if stay_score >= switch_score:
                prev_score = stay_score
                prev_state = j
            else:
                prev_score = switch_score
                prev_state = switch_prev

            if np.isneginf(prev_score) or np.isneginf(emissions[t, j]):
                new_delta[j] = -np.inf
                psi[t, j] = -1
            else:
                new_delta[j] = prev_score + emissions[t, j]
                psi[t, j] = prev_state

        delta = new_delta
        if np.all(np.isneginf(delta)):
            return [], -np.inf, -1

    final_state = int(np.argmax(delta))
    best_score = float(delta[final_state])

    path = [final_state]
    for t in range(T - 1, 0, -1):
        prev = psi[t, path[-1]]
        if prev < 0:
            return [], -np.inf, -1
        path.append(int(prev))
    path.reverse()

    return path, best_score, final_state

# ============================================================
# 9. FORWARD LOG-LIKELIHOOD (O(T * N))
# ============================================================
def forward_loglik(emissions):
    """
    Computes log p(observation sequence) under the HMM.
    """
    T, N = emissions.shape
    alpha = log_pi + emissions[0]

    if np.all(np.isneginf(alpha)):
        return -np.inf

    for t in range(1, T):
        total_prev = stable_logsumexp(alpha)
        new_alpha = np.full(N, -np.inf, dtype=np.float64)

        for j in range(N):
            stay_term = alpha[j] + LOG_A_SELF

            if N == 1:
                switch_term = -np.inf
            else:
                others = logsubexp(total_prev, alpha[j])
                switch_term = others + LOG_A_OTHER if not np.isneginf(others) else -np.inf

            trans_term = np.logaddexp(stay_term, switch_term)

            if np.isneginf(trans_term) or np.isneginf(emissions[t, j]):
                new_alpha[j] = -np.inf
            else:
                new_alpha[j] = trans_term + emissions[t, j]

        alpha = new_alpha
        if np.all(np.isneginf(alpha)):
            return -np.inf

    return stable_logsumexp(alpha)

def normalized_score(score, seq_len):
    if np.isneginf(score):
        return -np.inf
    return float(score / max(seq_len, 1))

# ============================================================
# 10. FIT REJECTION THRESHOLDS ON BENIGN VALIDATION
# ============================================================
print_header("FITTING BENIGN REJECTION THRESHOLDS")

val_viterbi_scores = []
val_forward_scores = []

for seq in val_sequences_norm:
    E = emission_log_matrix(seq)
    _, vscore, _ = viterbi_decode(E)
    fscore = forward_loglik(E)

    val_viterbi_scores.append(normalized_score(vscore, len(seq)))
    val_forward_scores.append(normalized_score(fscore, len(seq)))

val_viterbi_scores = np.array(val_viterbi_scores, dtype=np.float64)
val_forward_scores = np.array(val_forward_scores, dtype=np.float64)

lambda_viterbi = float(np.percentile(val_viterbi_scores, THRESHOLD_PERCENTILE))
lambda_forward = float(np.percentile(val_forward_scores, THRESHOLD_PERCENTILE))

print(f"lambda_viterbi (p{THRESHOLD_PERCENTILE:.1f} benign val): {lambda_viterbi:.6f}")
print(f"lambda_forward (p{THRESHOLD_PERCENTILE:.1f} benign val): {lambda_forward:.6f}")

# ============================================================
# 11. TEST-TIME INFERENCE
# ============================================================
print_header("HMM INFERENCE ON TEST SET")

y_pred_hmm = []
prediction_rows = []

for idx, seq in enumerate(test_sequences_norm):
    E = emission_log_matrix(seq)

    path, vscore, final_state = viterbi_decode(E)
    fscore = forward_loglik(E)

    vscore_norm = normalized_score(vscore, len(seq))
    fscore_norm = normalized_score(fscore, len(seq))

    empty_path = (len(path) == 0) or np.isneginf(vscore) or np.isneginf(fscore)

    if USE_FORWARD_SCORE_FOR_REJECTION:
        anomalous = empty_path or (fscore_norm < lambda_forward)
    else:
        anomalous = empty_path or (vscore_norm < lambda_viterbi)

    y_pred = 1 if anomalous else 0
    y_pred_hmm.append(y_pred)

    predicted_dialog = state_to_dialog[final_state] if final_state >= 0 else "<REJECTED>"

    prediction_rows.append({
        "idx": idx,
        "true_label": int(y_test_binary[idx]),
        "pred_label": int(y_pred),
        "seq_len": int(len(seq)),
        "viterbi_score_norm": float(vscore_norm) if not np.isneginf(vscore_norm) else -1e30,
        "forward_score_norm": float(fscore_norm) if not np.isneginf(fscore_norm) else -1e30,
        "predicted_state": int(final_state),
        "predicted_dialog": predicted_dialog,
        "empty_path": int(empty_path),
    })

y_pred_hmm = np.array(y_pred_hmm, dtype=int)

cm_hmm, metrics_hmm = print_metrics("Full HMM baseline", y_test_binary, y_pred_hmm)

# ============================================================
# 12. SAVE RESULTS
# ============================================================
print_header("FINAL SUMMARY")

summary_rows = [{
    "baseline": "Full HMM baseline",
    "ngram_order": NGRAM_ORDER,
    "alpha_emission": ALPHA_EMISSION,
    "self_transition": SELF_TRANSITION,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "use_forward_score_for_rejection": int(USE_FORWARD_SCORE_FOR_REJECTION),
    **metrics_hmm,
}]

summary_df = pd.DataFrame(summary_rows)
pred_df = pd.DataFrame(prediction_rows)

print(summary_df)

summary_df.to_csv("baseline_results_hmm.csv", index=False)
pred_df.to_csv("hmm_predictions.csv", index=False)

print("\nSaved summary to baseline_results_hmm.csv")
print("Saved per-dialog predictions to hmm_predictions.csv")
