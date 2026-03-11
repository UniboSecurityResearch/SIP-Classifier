import os

# ============================================================
# 0. FULL REPRODUCIBILITY SETUP
#    These environment variables must be set BEFORE importing
#    numpy / tensorflow / keras.
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
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

from scipy.stats import skew, kurtosis

# Global seeds
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

# Binary test labels assumed from your files:
# 0 = benign
# 1 = anomalous
BENIGN_LABEL_VALUE = 0
ANOMALOUS_LABEL_VALUE = 1

# Model/training hyperparameters
units = 256
dropout_rate = 0.5
batch_size = 64
learning_rate = 1e-3
max_epochs = 200
validation_size = 0.2

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

print("=" * 100)
print("EXPERIMENT SETUP")
print("=" * 100)
print(f"TensorFlow version: {tf.__version__}")
print(f"Seed: {SEED}")
print("Deterministic ops: enabled")
print("Device mode: CPU only")
print()
print("DATASET INTERPRETATION")
print(f"- Training file: {TRAIN_CSV}")
print(f"- Test file    : {TEST_CSV}")
print(f"- Signal column: {SIGNAL_COL}")
print(f"- Label column : {LABEL_COL}")
print()
print("PIPELINE OVERVIEW")
print("- train.csv is used ONLY for benign-only training.")
print("- Each unique benign training dialog is treated as one softmax class.")
print("- Therefore, model training is a benign-only multi-class classification task.")
print()
print("EVALUATION IS SPLIT INTO TWO TASKS")
print("1) Known-benign classification (diagnostic only)")
print("   - Evaluated only on benign test dialogs whose exact benign class was seen in training.")
print("   - This is NOT anomaly detection.")
print()
print("2) Binary anomaly detection (main task)")
print("   - Evaluated on the full test.csv.")
print("   - Binary ground truth is assumed to be:")
print("       0 = benign")
print("       1 = anomalous")
print("   - The model is still trained only on benign dialogs.")
print("   - Anomaly detection is obtained by thresholding the softmax output.")
print("=" * 100)

# ============================================================
# 2. LOAD DATA
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

# Optional sanity check: train should be benign-only
if LABEL_COL in df_train.columns:
    train_label_values = sorted(df_train[LABEL_COL].dropna().astype(int).unique().tolist())
    print(f"Train label values found: {train_label_values}")
    if any(v != BENIGN_LABEL_VALUE for v in train_label_values):
        print("WARNING: train.csv appears to contain labels different from 0.")
        print("The script assumes train.csv is benign-only.")

# ============================================================
# 3. PARSER
#    Adapt this function if your field format changes.
# ============================================================
def parse_dialog(dialog_str):
    """
    Extract a SIP token sequence from the 'Replaced Signalling Description' field.

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

# Keep only non-empty training sequences
train_sequences_all = [s for s in train_sequences_all if len(s) > 0]

# Keep only non-empty test sequences and preserve aligned labels
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
# 5. TRAIN/VALIDATION SPLIT ON BENIGN TRAINING DATA ONLY
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
# 6. BUILD VOCABULARY ONLY FROM TRAINING BENIGN DATA
# ============================================================
vocab_symbols = set()
for s in train_sequences:
    vocab_symbols.update(s)

vocab_symbols.update([PAD_TOKEN, UNK_TOKEN])

token_to_index = {tok: i for i, tok in enumerate(sorted(vocab_symbols))}
index_to_token = {i: tok for tok, i in token_to_index.items()}

LM = len(token_to_index)                     # vocabulary size
LN = max(len(s) for s in train_sequences)   # fixed length from training set

print("\nVOCABULARY")
print(f"Vocabulary size (LM): {LM}")
print(f"Max training sequence length (LN): {LN}")

# ============================================================
# 7. DEFINE KNOWN BENIGN CLASSES FROM TRAINING SET
#    Each unique benign training dialog becomes one softmax class.
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

# Known-benign classification on test:
# only for test samples that are benign and whose exact dialog/class was seen during training
test_benign_mask = (y_test_binary == BENIGN_LABEL_VALUE)
test_benign_sequences = [s for s, m in zip(test_sequences, test_benign_mask) if m]
y_test_benign_known = np.array([map_to_known_benign_class(s) for s in test_benign_sequences], dtype=int)
test_known_benign_mask = (y_test_benign_known >= 0)

# Optional validation-known mask (usually all or almost all known only if duplicates exist)
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

# Validation known-benign subset for monitoring known benign class accuracy
X_val_known = X_val_benign[val_known_mask]
y_val_known_softmax = to_categorical(y_val_known[val_known_mask], num_classes=N) if np.sum(val_known_mask) > 0 else None

# Known-benign classification subset from test
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
# 9. MODEL DEFINITIONS
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
    patience=10,
    restore_best_weights=True
)

# ============================================================
# 10. KNOWN-BENIGN CLASSIFICATION ACCURACY
#     This is NOT anomaly detection.
# ============================================================
def known_benign_classification_accuracy(model, X, y_true_softmax):
    y_pred = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true_softmax, axis=1)
    return accuracy_score(y_true_labels, y_pred_labels)

# ============================================================
# 11. ANOMALY THRESHOLDS
#     Computed ONLY from benign training predictions.
# ============================================================
def compute_anomaly_thresholds(model, X_known_train):
    yhat_train = model.predict(X_known_train, batch_size=batch_size, verbose=0)

    # Detector 1: threshold on max softmax probability
    max_scores = np.max(yhat_train, axis=1)
    lambda_M = float(np.mean(max_scores))

    # Detector 2: thresholds on skewness / kurtosis of the softmax vector
    sk_train = skew(yhat_train, axis=1, bias=False)
    ku_train = kurtosis(yhat_train, axis=1, fisher=False, bias=False)

    mu_S, var_S = float(np.mean(sk_train)), float(np.var(sk_train))
    mu_K, var_K = float(np.mean(ku_train)), float(np.var(ku_train))

    lambda_S = mu_S - var_S
    lambda_K = mu_K - var_K

    return lambda_M, lambda_S, lambda_K

def predict_binary_anomaly_from_max(yhat, lambda_M):
    # 0 = benign, 1 = anomalous
    return np.where(np.max(yhat, axis=1) < lambda_M, 1, 0)

def predict_binary_anomaly_from_moments(yhat, lambda_S, lambda_K):
    sk = skew(yhat, axis=1, bias=False)
    ku = kurtosis(yhat, axis=1, fisher=False, bias=False)
    return np.where((sk < lambda_S) & (ku < lambda_K), 1, 0)

def compute_binary_anomaly_metrics(y_true, y_pred):
    """
    Binary anomaly detection metrics.

    y_true:
        0 = benign
        1 = anomalous

    y_pred:
        0 = benign
        1 = anomalous
    """
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
def train_and_evaluate(model, model_name):
    print("\n" + "=" * 100)
    print(model_name)
    print("=" * 100)

    print("\nTRAINING PHASE")
    print("- Training uses only benign dialogs from train.csv.")
    print("- Each unique benign training dialog is treated as a softmax class.")
    print("- This is benign-only supervised training, not direct benign-vs-anomalous training.")

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
        print("- Validation is performed on validation benign dialogs whose class is known from training.")
    else:
        print("- No known benign validation samples are available.")

    history = model.fit(**fit_kwargs)

    print("\nPHASE A - KNOWN-BENIGN CLASSIFICATION (DIAGNOSTIC)")
    print("- This measures multiclass accuracy only on benign classes seen during training.")
    print("- This is NOT anomaly detection.")

    train_acc = known_benign_classification_accuracy(model, X_train, y_train_softmax)
    print(f"[{model_name}] Known-benign classification accuracy on TRAIN: {train_acc:.4f}")

    if len(X_val_known) > 0:
        val_acc = known_benign_classification_accuracy(model, X_val_known, y_val_known_softmax)
        print(f"[{model_name}] Known-benign classification accuracy on VAL-known: {val_acc:.4f}")
    else:
        val_acc = None
        print(f"[{model_name}] No known benign validation samples available.")

    if len(X_test_known_benign) > 0:
        test_known_acc = known_benign_classification_accuracy(
            model, X_test_known_benign, y_test_known_benign_softmax
        )
        print(f"[{model_name}] Known-benign classification accuracy on TEST-known-benign: {test_known_acc:.4f}")
    else:
        test_known_acc = None
        print(f"[{model_name}] No benign test sample belongs to a known training class.")

    lambda_M, lambda_S, lambda_K = compute_anomaly_thresholds(model, X_train)
    print("\nANOMALY THRESHOLDS (computed only from benign training predictions)")
    print(f"[{model_name}] lambda_M = {lambda_M:.6f}")
    print(f"[{model_name}] lambda_S = {lambda_S:.6f}")
    print(f"[{model_name}] lambda_K = {lambda_K:.6f}")

    print("\nPHASE B - BINARY ANOMALY DETECTION (MAIN TASK)")
    print("- Evaluation is performed on the full test.csv.")
    print("- Ground truth labels are assumed to be:")
    print("    0 = benign")
    print("    1 = anomalous")
    print("- All predictions are converted to binary anomaly decisions by thresholding the softmax output.")

    yhat_test = model.predict(X_test, batch_size=batch_size, verbose=0)

    y_pred_max = predict_binary_anomaly_from_max(yhat_test, lambda_M)
    y_pred_mom = predict_binary_anomaly_from_moments(yhat_test, lambda_S, lambda_K)

    cm_max, metrics_max = compute_binary_anomaly_metrics(y_test_binary, y_pred_max)
    cm_mom, metrics_mom = compute_binary_anomaly_metrics(y_test_binary, y_pred_mom)

    print(f"\n[{model_name}] Binary anomaly detector 1 - MAX softmax threshold")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm_max)
    print(metrics_max)

    print(f"\n[{model_name}] Binary anomaly detector 2 - SKEW/KURT thresholds")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
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
        "binary_metrics_moments": metrics_mom
    }

# ============================================================
# 13. RUN BOTH MODELS
# ============================================================
model1 = build_model_1(LN, LM, N)
results_model1 = train_and_evaluate(model1, "MODEL 1")

model2 = build_model_2(LN, LM, N)
results_model2 = train_and_evaluate(model2, "MODEL 2")