import os

# ============================================================
# 0. REPRODUCIBILITY SETUP
# ============================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import skew, kurtosis, binomtest, chi2

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

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

# Paper LSTM parameters
DROPOUT_PROB = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 933
MAX_EPOCHS = 500

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

print("=" * 100)
print("LSTM PAPER-STYLE PIPELINE")
print("=" * 100)
print(f"TensorFlow version: {tf.__version__}")
print(f"Seed: {SEED}")
print("GPU: allowed")
print()
print("This script reproduces the 4 LSTM architectures from the paper:")
print("- Model 1.0 = LSTM(N) -> Dense(N)")
print("- Model 1.1 = LSTM(N) -> Dropout(0.5) -> Dense(N)")
print("- Model 2.0 = LSTM(N, return_sequences=True) -> LSTM(N) -> Dense(N)")
print("- Model 2.1 = LSTM(N, return_sequences=True) -> LSTM(N) -> Dropout(0.5) -> Dense(N)")
print()
print("Unknown / anomalous dialog detection uses ONLY the paper LSTM detector:")
print("- skewness / kurtosis threshold-based classifier")
print("=" * 100)

# ============================================================
# 2. LOAD DATA
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
# 4. PARSE DATA AND REMOVE EMPTY SEQUENCES
# ============================================================
train_sequences_all = [parse_dialog(d) for d in train_dialogs_raw]
test_sequences_all = [parse_dialog(d) for d in test_dialogs_raw]

train_sequences_all = [s for s in train_sequences_all if len(s) > 0]

filtered_test_pairs = [
    (seq, label) for seq, label in zip(test_sequences_all, test_labels_raw) if len(seq) > 0
]
test_sequences = [x[0] for x in filtered_test_pairs]
y_test_binary = np.array([x[1] for x in filtered_test_pairs], dtype=int)

print("\nAFTER PARSING / FILTERING")
print(f"Valid training dialogs: {len(train_sequences_all)}")
print(f"Valid test dialogs    : {len(test_sequences)}")
print(f"Valid benign test dialogs    : {np.sum(y_test_binary == BENIGN_LABEL_VALUE)}")
print(f"Valid anomalous test dialogs : {np.sum(y_test_binary == ANOMALOUS_LABEL_VALUE)}")

if len(train_sequences_all) == 0:
    raise ValueError("No valid training sequences found after parsing.")
if len(test_sequences) == 0:
    raise ValueError("No valid test sequences found after parsing.")

# ============================================================
# 5. TRAIN / VALIDATION SPLIT
# ============================================================
train_sequences, val_sequences = train_test_split(
    train_sequences_all,
    test_size=VALIDATION_SIZE,
    random_state=SEED,
    shuffle=True
)

print("\nBENIGN TRAIN / VALIDATION SPLIT")
print(f"Train benign dialogs: {len(train_sequences)}")
print(f"Val benign dialogs  : {len(val_sequences)}")

# ============================================================
# 6. VOCABULARY FROM TRAIN ONLY
# ============================================================
vocab_symbols = set()
for s in train_sequences:
    vocab_symbols.update(s)

vocab_symbols.update([PAD_TOKEN, UNK_TOKEN])

token_to_index = {tok: i for i, tok in enumerate(sorted(vocab_symbols))}
index_to_token = {i: tok for tok, i in token_to_index.items()}

LM = len(token_to_index)
LN = max(len(s) for s in train_sequences)

print("\nVOCABULARY")
print(f"Vocabulary size (LM): {LM}")
print(f"Max training sequence length (LN): {LN}")

# ============================================================
# 7. KNOWN BENIGN CLASSES
# ============================================================
train_class_strings = [' '.join(s) for s in train_sequences]
known_benign_class_to_id = {
    dlg: i for i, dlg in enumerate(pd.unique(train_class_strings))
}
N = len(known_benign_class_to_id)

print(f"Known benign classes in train (N): {N}")
print("IMPORTANT:")
print("- In the paper, the LSTM layer uses N units, where N is the number of known benign classes.")
print("- Therefore these models are different from your previous 256-unit LSTMs.")

def map_to_known_benign_class(seq):
    s = ' '.join(seq)
    return known_benign_class_to_id.get(s, -1)

y_train_known = np.array([map_to_known_benign_class(s) for s in train_sequences], dtype=int)
y_val_known = np.array([map_to_known_benign_class(s) for s in val_sequences], dtype=int)

assert np.all(y_train_known >= 0), "Training benign classes must all be known."

val_known_mask = (y_val_known >= 0)

# ============================================================
# 8. ONE-HOT ENCODING
# ============================================================
def encode_sequence(seq, LN, LM, token_to_index):
    encoded = np.zeros((LN, LM), dtype=np.float32)

    for i in range(LN):
        if i < len(seq):
            tok = seq[i]
            idx = token_to_index.get(tok, token_to_index[UNK_TOKEN])
        else:
            idx = token_to_index[PAD_TOKEN]

        encoded[i] = to_categorical(idx, num_classes=LM)

    return encoded

def encode_dataset(seqs, LN, LM, token_to_index):
    return np.array([encode_sequence(s[:LN], LN, LM, token_to_index) for s in seqs], dtype=np.float32)

X_train = encode_dataset(train_sequences, LN, LM, token_to_index)
X_val_benign = encode_dataset(val_sequences, LN, LM, token_to_index)
X_test = encode_dataset(test_sequences, LN, LM, token_to_index)

y_train_softmax = to_categorical(y_train_known, num_classes=N)

X_val_known = X_val_benign[val_known_mask]
y_val_known_softmax = (
    to_categorical(y_val_known[val_known_mask], num_classes=N)
    if np.sum(val_known_mask) > 0 else None
)

print("\nENCODED DATA SHAPES")
print(f"X_train    : {X_train.shape}")
print(f"X_val_known: {X_val_known.shape}")
print(f"X_test     : {X_test.shape}")

# ============================================================
# 9. PAPER LSTM MODELS
# ============================================================
def build_model_1_0(LN, LM, N):
    """
    Paper Model 1.0:
    LSTM(N) -> Dense(N)
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential([
        Input(shape=(LN, LM)),
        LSTM(N),
        Dense(N, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_model_1_1(LN, LM, N):
    """
    Paper Model 1.1:
    LSTM(N) -> Dropout(0.5) -> Dense(N)
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential([
        Input(shape=(LN, LM)),
        LSTM(N),
        Dropout(DROPOUT_PROB),
        Dense(N, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_model_2_0(LN, LM, N):
    """
    Paper Model 2.0:
    LSTM(N, return_sequences=True) -> LSTM(N) -> Dense(N)
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential([
        Input(shape=(LN, LM)),
        LSTM(N, return_sequences=True),
        LSTM(N),
        Dense(N, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_model_2_1(LN, LM, N):
    """
    Paper Model 2.1:
    LSTM(N, return_sequences=True) -> LSTM(N) -> Dropout(0.5) -> Dense(N)
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential([
        Input(shape=(LN, LM)),
        LSTM(N, return_sequences=True),
        LSTM(N),
        Dropout(DROPOUT_PROB),
        Dense(N, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)

# ============================================================
# 10. DIAGNOSTIC CLASSIFICATION ACCURACY
# ============================================================
def known_benign_classification_accuracy(model, X, y_true_softmax):
    y_pred = model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true_softmax, axis=1)
    return accuracy_score(y_true_labels, y_pred_labels)

# ============================================================
# 11. PAPER LSTM ANOMALY DETECTOR: SKEW / KURT
# ============================================================
def compute_skew_kurt_thresholds(model, X_known_train):
    """
    Paper semi-supervised threshold-based detector:
    lambda_S = mu_S - var_S
    lambda_K = mu_K - var_K
    computed on benign train outputs only.
    """
    yhat_train = model.predict(X_known_train, batch_size=BATCH_SIZE, verbose=0)

    sk_train = skew(yhat_train, axis=1, bias=False)
    ku_train = kurtosis(yhat_train, axis=1, fisher=False, bias=False)

    mu_S, var_S = float(np.mean(sk_train)), float(np.var(sk_train))
    mu_K, var_K = float(np.mean(ku_train)), float(np.var(ku_train))

    lambda_S = mu_S - var_S
    lambda_K = mu_K - var_K

    return lambda_S, lambda_K

def predict_anomaly_skew_kurt(yhat, lambda_S, lambda_K):
    """
    Binary output:
    0 = benign / already trained
    1 = anomalous / unknown
    """
    sk = skew(yhat, axis=1, bias=False)
    ku = kurtosis(yhat, axis=1, fisher=False, bias=False)
    return np.where((sk < lambda_S) & (ku < lambda_K), 1, 0)

def compute_binary_anomaly_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    precision = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)

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
        "f1": f1
    }

# ============================================================
# 12. TRAIN + EVALUATE
# ============================================================
def train_and_evaluate_lstm(model, model_name):
    print("\n" + "=" * 100)
    print(model_name)
    print("=" * 100)

    print("\nTRAINING PHASE")
    print("- Training uses only benign dialogs from train.csv")
    print("- One softmax class per unique benign training dialog")
    print(f"- LSTM units = N = {N}")
    print(f"- Batch size = {BATCH_SIZE} (paper setting)")
    print(f"- Max epochs = {MAX_EPOCHS} (paper setting)")
    print("- Anomaly detector = skewness/kurtosis threshold-based classifier")

    fit_kwargs = {
        "x": X_train,
        "y": y_train_softmax,
        "epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "callbacks": [early_stop],
        "verbose": 2,
        "shuffle": False
    }

    if len(X_val_known) > 0:
        fit_kwargs["validation_data"] = (X_val_known, y_val_known_softmax)
        print("- Validation uses benign validation dialogs whose class is known from training")
    else:
        print("- No known benign validation samples available")

    history = model.fit(**fit_kwargs)

    print("\nKNOWN-BENIGN CLASSIFICATION (diagnostic only)")
    train_acc = known_benign_classification_accuracy(model, X_train, y_train_softmax)
    print(f"[{model_name}] Train known-benign accuracy: {train_acc:.4f}")

    if len(X_val_known) > 0:
        val_acc = known_benign_classification_accuracy(model, X_val_known, y_val_known_softmax)
        print(f"[{model_name}] Val known-benign accuracy: {val_acc:.4f}")
    else:
        val_acc = None
        print(f"[{model_name}] No known benign validation samples")

    print("\nPAPER ANOMALY DETECTOR THRESHOLDS")
    lambda_S, lambda_K = compute_skew_kurt_thresholds(model, X_train)
    print(f"[{model_name}] lambda_S = {lambda_S:.6f}")
    print(f"[{model_name}] lambda_K = {lambda_K:.6f}")

    print("\nBINARY ANOMALY DETECTION ON TEST.CSV")
    print("- 0 = benign")
    print("- 1 = anomalous")
    print("- Rule: anomalous if skew < lambda_S AND kurtosis < lambda_K")

    yhat_test = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred_binary = predict_anomaly_skew_kurt(yhat_test, lambda_S, lambda_K)

    cm, metrics = compute_binary_anomaly_metrics(y_test_binary, y_pred_binary)

    print("\nCONFUSION MATRIX [rows=true 0/1, cols=pred 0/1]")
    print(cm)

    print("\nMETRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return {
        "history": history.history,
        "lambda_S": lambda_S,
        "lambda_K": lambda_K,
        "known_benign_train_acc": train_acc,
        "known_benign_val_acc": val_acc,
        "binary_metrics": metrics,
        "y_pred_binary": y_pred_binary
    }

# ============================================================
# 13. RUN ALL 4 PAPER LSTM MODELS
# ============================================================
results = {}

model_1_0 = build_model_1_0(LN, LM, N)
results["model_1_0"] = train_and_evaluate_lstm(model_1_0, "MODEL 1.0")

model_1_1 = build_model_1_1(LN, LM, N)
results["model_1_1"] = train_and_evaluate_lstm(model_1_1, "MODEL 1.1")

model_2_0 = build_model_2_0(LN, LM, N)
results["model_2_0"] = train_and_evaluate_lstm(model_2_0, "MODEL 2.0")

model_2_1 = build_model_2_1(LN, LM, N)
results["model_2_1"] = train_and_evaluate_lstm(model_2_1, "MODEL 2.1")

MODEL_NAMES = ["model_1_0", "model_1_1", "model_2_0", "model_2_1"]

# ============================================================
# 14. SAVE PER-SAMPLE PREDICTIONS
#     Needed for a posteriori model comparison (e.g. McNemar)
# ============================================================
# test_csv_row = original 0-based row number in test.csv (before empty-sequence
# filtering): stable join key to align predictions across the baseline scripts
test_csv_row = np.array([i for i, s in enumerate(test_sequences_all) if len(s) > 0], dtype=int)
pred_df = pd.DataFrame({
    "idx": np.arange(len(y_test_binary)),
    "test_csv_row": test_csv_row,
    "true_label": y_test_binary,
    **{f"pred_label_{name}": results[name]["y_pred_binary"] for name in MODEL_NAMES},
    # original signalling description of each test dialog, verbatim from test.csv
    "Replaced Signalling Description": [test_dialogs_raw[i] for i in test_csv_row],
})
pred_df.to_csv("lstm_paper_predictions.csv", index=False)
print("\nSaved per-dialog predictions to lstm_paper_predictions.csv")

# ============================================================
# 15. STATISTICAL TESTS: BOOTSTRAP CI + McNEMAR'S TEST
# ============================================================
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

    _, point = compute_binary_anomaly_metrics(y_true, y_pred)
    skip = {"TN", "FP", "FN", "TP"}
    metric_names = [k for k in point if k not in skip]
    samples = {name: np.empty(n_boot, dtype=np.float64) for name in metric_names}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, m = compute_binary_anomaly_metrics(y_true[idx], y_pred[idx])
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

# Bootstrap CIs and McNemar vs truth for each paper model
for name in MODEL_NAMES:
    y_pred_model = results[name]["y_pred_binary"]

    boot = bootstrap_metric_cis(y_test_binary, y_pred_model)
    print_bootstrap_cis(name, boot)
    results[name]["bootstrap_cis"] = boot

    mcnemar_res = mcnemar_vs_truth(y_test_binary, y_pred_model)
    print_mcnemar(f"{name} (FP vs FN)", mcnemar_res, labels=("FP", "FN"))
    results[name]["mcnemar_vs_truth"] = mcnemar_res

# Pairwise McNemar between the 4 paper models
for i in range(len(MODEL_NAMES)):
    for j in range(i + 1, len(MODEL_NAMES)):
        name_a, name_b = MODEL_NAMES[i], MODEL_NAMES[j]
        mcnemar_pair = mcnemar_between_models(
            y_test_binary,
            results[name_a]["y_pred_binary"],
            results[name_b]["y_pred_binary"],
        )
        print_mcnemar(
            f"{name_a} vs {name_b}",
            mcnemar_pair,
            labels=(f"{name_a} wrong / {name_b} right", f"{name_a} right / {name_b} wrong"),
        )