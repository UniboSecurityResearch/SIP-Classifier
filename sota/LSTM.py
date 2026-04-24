import os
import json
import argparse
import random
from dataclasses import asdict, dataclass

# ============================================================
# ARGPARSE FIRST
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Benign-only LSTM training + anomaly detection for SIP dialogs"
    )

    # Data
    parser.add_argument("--train-csv", type=str, default="train.csv")
    parser.add_argument("--test-csv", type=str, default="test.csv")
    parser.add_argument("--signal-col", type=str, default="Replaced Signalling Description")
    parser.add_argument("--label-col", type=str, default="label")

    # Labels
    parser.add_argument("--benign-label", type=int, default=0)
    parser.add_argument("--anomalous-label", type=int, default=1)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Model selection
    parser.add_argument("--model", type=str, choices=["model1", "model2"], default="model1")

    # Hyperparameters
    parser.add_argument("--units", type=int, default=256)
    parser.add_argument("--dropout-rate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)

    # Sequence length policy
    parser.add_argument("--ln-mode", type=str, choices=["max", "p95", "p90", "fixed"], default="max")
    parser.add_argument("--ln-fixed", type=int, default=None)

    # Threshold policy
    parser.add_argument("--threshold-mode", type=str, choices=["mean", "percentile"], default="mean")
    parser.add_argument("--threshold-percentile", type=float, default=5.0)

    # Output
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)

    return parser


args = build_arg_parser().parse_args()

# ============================================================
# FULL REPRODUCIBILITY SETUP
# Must be set before importing numpy / tensorflow / keras
# ============================================================

os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import skew, kurtosis

# ============================================================
# GLOBAL SEEDS
# ============================================================

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = args.train_csv
TEST_CSV = args.test_csv

SIGNAL_COL = args.signal_col
LABEL_COL = args.label_col

BENIGN_LABEL_VALUE = args.benign_label
ANOMALOUS_LABEL_VALUE = args.anomalous_label

units = args.units
dropout_rate = args.dropout_rate
batch_size = args.batch_size
learning_rate = args.learning_rate
max_epochs = args.max_epochs
validation_size = args.validation_size
patience = args.patience

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# ============================================================
# REPORT HELPERS
# ============================================================

@dataclass
class RunConfig:
    train_csv: str
    test_csv: str
    signal_col: str
    label_col: str
    benign_label: int
    anomalous_label: int
    seed: int
    model: str
    units: int
    dropout_rate: float
    batch_size: int
    learning_rate: float
    max_epochs: int
    validation_size: float
    patience: int
    ln_mode: str
    ln_fixed: int | None
    threshold_mode: str
    threshold_percentile: float
    run_name: str | None

def print_header():
    print("=" * 100)
    print("EXPERIMENT SETUP")
    print("=" * 100)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Seed: {SEED}")
    print("Deterministic ops: enabled")
    print()
    print("RUN CONFIG")
    for k, v in asdict(RunConfig(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        signal_col=SIGNAL_COL,
        label_col=LABEL_COL,
        benign_label=BENIGN_LABEL_VALUE,
        anomalous_label=ANOMALOUS_LABEL_VALUE,
        seed=SEED,
        model=args.model,
        units=units,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        validation_size=validation_size,
        patience=patience,
        ln_mode=args.ln_mode,
        ln_fixed=args.ln_fixed,
        threshold_mode=args.threshold_mode,
        threshold_percentile=args.threshold_percentile,
        run_name=args.run_name
    )).items():
        print(f"{k}: {v}")
    print("=" * 100)

print_header()

# ============================================================
# LOAD DATA
# ============================================================

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

print("\nRAW DATA")
print(f"Train shape: {df_train.shape}")
print(f"Test shape : {df_test.shape}")
print(f"Train columns: {list(df_train.columns)}")
print(f"Test columns : {list(df_test.columns)}")

if SIGNAL_COL not in df_train.columns:
    raise ValueError(f"Column '{SIGNAL_COL}' not found in train.csv")

if SIGNAL_COL not in df_test.columns:
    raise ValueError(f"Column '{SIGNAL_COL}' not found in test.csv")

if LABEL_COL not in df_test.columns:
    raise ValueError(f"Column '{LABEL_COL}' not found in test.csv")

train_dialogs_raw = df_train[SIGNAL_COL].astype(str).tolist()
test_dialogs_raw = df_test[SIGNAL_COL].astype(str).tolist()
test_labels_raw = df_test[LABEL_COL].astype(int).to_numpy()

print(f"Raw training dialogs: {len(train_dialogs_raw)}")
print(f"Raw test dialogs    : {len(test_dialogs_raw)}")
print("Test label distribution:")
print(df_test[LABEL_COL].value_counts(dropna=False).sort_index())

if LABEL_COL in df_train.columns:
    train_label_values = sorted(df_train[LABEL_COL].dropna().astype(int).unique().tolist())
    print(f"Train label values found: {train_label_values}")
    if any(v != BENIGN_LABEL_VALUE for v in train_label_values):
        print("WARNING: train.csv appears to contain labels different from benign label.")

# ============================================================
# PARSER
# ============================================================

def parse_dialog(dialog_str):
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
# PARSE DATA AND REMOVE EMPTY SEQUENCES
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
# TRAIN/VALIDATION SPLIT
# ============================================================

train_sequences, val_sequences = train_test_split(
    train_sequences_all,
    test_size=validation_size,
    random_state=SEED,
    shuffle=True
)

print("\nBENIGN TRAIN/VALIDATION SPLIT")
print(f"Train benign dialogs: {len(train_sequences)}")
print(f"Val benign dialogs  : {len(val_sequences)}")

# ============================================================
# BUILD VOCABULARY
# ============================================================

vocab_symbols = set()
for s in train_sequences:
    vocab_symbols.update(s)

vocab_symbols.update([PAD_TOKEN, UNK_TOKEN])

token_to_index = {tok: i for i, tok in enumerate(sorted(vocab_symbols))}
index_to_token = {i: tok for tok, i in token_to_index.items()}

LM = len(token_to_index)

def compute_LN(seqs, ln_mode, ln_fixed):
    lengths = [len(s) for s in seqs]
    if ln_mode == "max":
        return max(lengths)
    if ln_mode == "p95":
        return max(1, int(np.percentile(lengths, 95)))
    if ln_mode == "p90":
        return max(1, int(np.percentile(lengths, 90)))
    if ln_mode == "fixed":
        if ln_fixed is None or ln_fixed <= 0:
            raise ValueError("--ln-fixed must be a positive integer when --ln-mode fixed")
        return ln_fixed
    raise ValueError(f"Unsupported ln_mode: {ln_mode}")

LN = compute_LN(train_sequences, args.ln_mode, args.ln_fixed)

print("\nVOCABULARY / LENGTH")
print(f"Vocabulary size (LM): {LM}")
print(f"Sequence length (LN): {LN}")
print(f"LN policy: {args.ln_mode}")

# ============================================================
# KNOWN BENIGN CLASSES
# ============================================================

train_class_strings = [' '.join(s) for s in train_sequences]
known_benign_class_to_id = {
    dlg: i for i, dlg in enumerate(pd.unique(train_class_strings))
}
N = len(known_benign_class_to_id)

print(f"Known benign classes in train (N): {N}")

def map_to_known_benign_class(seq):
    s = ' '.join(seq)
    return known_benign_class_to_id.get(s, -1)

y_train_known = np.array([map_to_known_benign_class(s) for s in train_sequences], dtype=int)
y_val_known = np.array([map_to_known_benign_class(s) for s in val_sequences], dtype=int)

assert np.all(y_train_known >= 0), "Training benign classes must all be known."

test_benign_mask = (y_test_binary == BENIGN_LABEL_VALUE)
test_benign_sequences = [s for s, m in zip(test_sequences, test_benign_mask) if m]
y_test_benign_known = np.array([map_to_known_benign_class(s) for s in test_benign_sequences], dtype=int)
test_known_benign_mask = (y_test_benign_known >= 0)
val_known_mask = (y_val_known >= 0)

# ============================================================
# ENCODING
# ============================================================

def encode_sequence(seq, LN, LM, token_to_index):
    encoded = np.zeros((LN, LM), dtype=np.float32)

    trimmed = seq[:LN]

    for i in range(LN):
        if i < len(trimmed):
            tok = trimmed[i]
            idx = token_to_index.get(tok, token_to_index[UNK_TOKEN])
        else:
            idx = token_to_index[PAD_TOKEN]

        encoded[i] = to_categorical(idx, num_classes=LM)

    return encoded

def encode_dataset(seqs, LN, LM, token_to_index):
    return np.array([encode_sequence(s, LN, LM, token_to_index) for s in seqs], dtype=np.float32)

X_train = encode_dataset(train_sequences, LN, LM, token_to_index)
X_val_benign = encode_dataset(val_sequences, LN, LM, token_to_index)
X_test = encode_dataset(test_sequences, LN, LM, token_to_index)

y_train_softmax = to_categorical(y_train_known, num_classes=N)

X_val_known = X_val_benign[val_known_mask]
y_val_known_softmax = to_categorical(y_val_known[val_known_mask], num_classes=N) if np.sum(val_known_mask) > 0 else None

test_benign_sequences_only = [s for s, m in zip(test_sequences, test_benign_mask) if m]
X_test_benign_only = encode_dataset(test_benign_sequences_only, LN, LM, token_to_index)

X_test_known_benign = X_test_benign_only[test_known_benign_mask]
y_test_known_benign_softmax = (
    to_categorical(y_test_benign_known[test_known_benign_mask], num_classes=N)
    if np.sum(test_known_benign_mask) > 0 else None
)

print("\nENCODED DATA SHAPES")
print(f"X_train : {X_train.shape}")
print(f"X_val   : {X_val_benign.shape}")
print(f"X_test  : {X_test.shape}")
print(f"Known benign test samples for multiclass diagnostic: {len(X_test_known_benign)}")
print(f"Unknown benign test samples: {np.sum(test_benign_mask) - len(X_test_known_benign)}")

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_model_1(LN, LM, N):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential([
        Input(shape=(LN, LM)),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(N, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_model_2(LN, LM, N):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential([
        Input(shape=(LN, LM)),
        LSTM(units, return_sequences=True),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(N, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=patience,
    restore_best_weights=True
)

# ============================================================
# METRICS
# ============================================================

def known_benign_classification_accuracy(model, X, y_true_softmax):
    y_pred = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true_softmax, axis=1)
    return accuracy_score(y_true_labels, y_pred_labels)

def compute_anomaly_thresholds(model, X_known_train):
    yhat_train = model.predict(X_known_train, batch_size=batch_size, verbose=0)

    max_scores = np.max(yhat_train, axis=1)

    if args.threshold_mode == "mean":
        lambda_M = float(np.mean(max_scores))
    elif args.threshold_mode == "percentile":
        lambda_M = float(np.percentile(max_scores, args.threshold_percentile))
    else:
        raise ValueError(f"Unsupported threshold_mode: {args.threshold_mode}")

    sk_train = skew(yhat_train, axis=1, bias=False)
    ku_train = kurtosis(yhat_train, axis=1, fisher=False, bias=False)

    mu_S, var_S = float(np.mean(sk_train)), float(np.var(sk_train))
    mu_K, var_K = float(np.mean(ku_train)), float(np.var(ku_train))

    lambda_S = mu_S - var_S
    lambda_K = mu_K - var_K

    return lambda_M, lambda_S, lambda_K

def predict_binary_anomaly_from_max(yhat, lambda_M):
    return np.where(np.max(yhat, axis=1) < lambda_M, 1, 0)

def predict_binary_anomaly_from_moments(yhat, lambda_S, lambda_K):
    sk = skew(yhat, axis=1, bias=False)
    ku = kurtosis(yhat, axis=1, fisher=False, bias=False)
    return np.where((sk < lambda_S) & (ku < lambda_K), 1, 0)

def compute_binary_anomaly_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    TN = int(cm[0, 0])
    FP = int(cm[0, 1])
    FN = int(cm[1, 0])
    TP = int(cm[1, 1])

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
# TRAIN + EVALUATE
# ============================================================

def train_and_evaluate(model, model_name):
    print("\n" + "=" * 100)
    print(model_name)
    print("=" * 100)

    fit_kwargs = {
        "x": X_train,
        "y": y_train_softmax,
        "epochs": max_epochs,
        "batch_size": batch_size,
        "callbacks": [early_stop],
        "verbose": 2,
        "shuffle": False
    }

    if len(X_val_known) > 0:
        fit_kwargs["validation_data"] = (X_val_known, y_val_known_softmax)

    history = model.fit(**fit_kwargs)

    train_acc = known_benign_classification_accuracy(model, X_train, y_train_softmax)

    if len(X_val_known) > 0:
        val_acc = known_benign_classification_accuracy(model, X_val_known, y_val_known_softmax)
    else:
        val_acc = None

    if len(X_test_known_benign) > 0:
        test_known_acc = known_benign_classification_accuracy(
            model, X_test_known_benign, y_test_known_benign_softmax
        )
    else:
        test_known_acc = None

    lambda_M, lambda_S, lambda_K = compute_anomaly_thresholds(model, X_train)

    yhat_test = model.predict(X_test, batch_size=batch_size, verbose=0)

    y_pred_max = predict_binary_anomaly_from_max(yhat_test, lambda_M)
    y_pred_mom = predict_binary_anomaly_from_moments(yhat_test, lambda_S, lambda_K)

    cm_max, metrics_max = compute_binary_anomaly_metrics(y_test_binary, y_pred_max)
    cm_mom, metrics_mom = compute_binary_anomaly_metrics(y_test_binary, y_pred_mom)

    print("\nRESULTS")
    print(f"known_benign_train_acc: {train_acc:.6f}")
    print(f"known_benign_val_acc  : {val_acc}")
    print(f"known_benign_test_acc : {test_known_acc}")
    print(f"lambda_M: {lambda_M:.6f}")
    print(f"lambda_S: {lambda_S:.6f}")
    print(f"lambda_K: {lambda_K:.6f}")

    print("\nMAX SOFTMAX DETECTOR")
    print(cm_max)
    print(metrics_max)

    print("\nMOMENT DETECTOR")
    print(cm_mom)
    print(metrics_mom)

    return {
        "history": history.history,
        "lambda_M": lambda_M,
        "lambda_S": lambda_S,
        "lambda_K": lambda_K,
        "known_benign_train_acc": train_acc,
        "known_benign_val_acc": val_acc,
        "known_benign_test_acc": test_known_acc,
        "binary_metrics_max": metrics_max,
        "binary_metrics_moments": metrics_mom,
        "confusion_matrix_max": cm_max.tolist(),
        "confusion_matrix_moments": cm_mom.tolist(),
        "config": asdict(RunConfig(
            train_csv=TRAIN_CSV,
            test_csv=TEST_CSV,
            signal_col=SIGNAL_COL,
            label_col=LABEL_COL,
            benign_label=BENIGN_LABEL_VALUE,
            anomalous_label=ANOMALOUS_LABEL_VALUE,
            seed=SEED,
            model=args.model,
            units=units,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            validation_size=validation_size,
            patience=patience,
            ln_mode=args.ln_mode,
            ln_fixed=args.ln_fixed,
            threshold_mode=args.threshold_mode,
            threshold_percentile=args.threshold_percentile,
            run_name=args.run_name
        ))
    }

# ============================================================
# MAIN
# ============================================================

if args.model == "model1":
    model = build_model_1(LN, LM, N)
    results = train_and_evaluate(model, "MODEL 1")
else:
    model = build_model_2(LN, LM, N)
    results = train_and_evaluate(model, "MODEL 2")

if args.output_json is not None:
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results JSON to: {args.output_json}")