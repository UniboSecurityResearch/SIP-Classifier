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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# Test labels
BENIGN_LABEL_VALUE = 0
ANOMALOUS_LABEL_VALUE = 1

# CNN parameters from the paper
CNN_NUM_FILTERS = 16
POOL_SIZE = (2, 1)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200
VALIDATION_SIZE = 0.2

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

print("=" * 100)
print("CNN-ONLY PIPELINE (PAPER-STYLE)")
print("=" * 100)
print(f"TensorFlow version: {tf.__version__}")
print(f"Seed: {SEED}")
print("GPU: allowed")
print()
print("This script does the following:")
print("- Trains ONLY a CNN model on benign dialogs from train.csv")
print("- Uses one softmax class per unique benign training dialog")
print("- Builds the vocabulary ONLY from benign training data")
print("- Evaluates anomaly detection on test.csv")
print("- Uses ONLY the paper's detector: class-dependent lambdaMax(k)")
print()
print("Binary interpretation in test.csv:")
print("0 = benign")
print("1 = anomalous")
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

# Optional sanity check
if LABEL_COL in df_train.columns:
    train_label_values = sorted(df_train[LABEL_COL].dropna().astype(int).unique().tolist())
    print(f"Train label values found: {train_label_values}")
    if any(v != BENIGN_LABEL_VALUE for v in train_label_values):
        print("WARNING: train.csv contains labels different from 0.")
        print("This script assumes train.csv is benign-only.")

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
# 5. TRAIN / VALIDATION SPLIT ON BENIGN TRAINING DATA ONLY
# ============================================================
train_sequences, val_sequences = train_test_split(
    train_sequences_all,
    test_size=VALIDATION_SIZE,
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

LM = len(token_to_index)                     # encoded message length
LN = max(len(s) for s in train_sequences)   # padded dialog length

print("\nVOCABULARY")
print(f"Vocabulary size (LM): {LM}")
print(f"Max training sequence length (LN): {LN}")

# ============================================================
# 7. DEFINE KNOWN BENIGN CLASSES
#    One class per unique benign training dialog
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

# Only validation samples whose class exists in training can be used for softmax validation
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

# CNN expects channel dimension
X_train_cnn = X_train[..., np.newaxis]
X_val_known_cnn = X_val_known[..., np.newaxis] if len(X_val_known) > 0 else None
X_test_cnn = X_test[..., np.newaxis]

print("\nENCODED DATA SHAPES")
print(f"X_train     : {X_train.shape}")
print(f"X_train_cnn : {X_train_cnn.shape}")
print(f"X_val_known : {X_val_known.shape}")
print(f"X_test      : {X_test.shape}")
print(f"X_test_cnn  : {X_test_cnn.shape}")

# ============================================================
# 9. PAPER CNN MODEL
# ============================================================
def build_cnn_model(LN, LM, N):
    """
    Paper CNN:
    Conv2D -> MaxPooling2D -> Flatten -> Dense(softmax)

    The paper reports:
    - 16 filters
    - filter size = 2 x 56
    - max pooling = 2 x 2
    - final dense softmax
    - Adam lr=0.001
    - categorical crossentropy
    - early stopping on val_loss

    In the original paper, 56 is the padded sequence length LS.
    With Keras (height, width, channels) = (LN, LM, 1),
    the faithful adaptation is kernel_size = (2, LM),
    i.e. span 2 time steps and the full one-hot width.
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = Sequential([
        Input(shape=(LN, LM, 1)),
        Conv2D(
            filters=CNN_NUM_FILTERS,
            kernel_size=(2, LM),
            activation='relu',
            padding='valid'
        ),
        MaxPooling2D(pool_size=POOL_SIZE),
        Flatten(),
        Dense(N, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=False
    )
    return model

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    restore_best_weights=True
)

# ============================================================
# 10. PAPER DETECTOR: lambdaMax(k)
# ============================================================
def compute_lambda_max_per_class(model, X_known_train_cnn, batch_size):
    """
    Paper detector:
    lambdaMax(k) = average of max softmax output for benign training dialogs
    assigned to class k.

    The threshold is class-dependent, not global.
    """
    yhat_train = model.predict(X_known_train_cnn, batch_size=batch_size, verbose=0)

    predicted_class = np.argmax(yhat_train, axis=1)
    max_scores = np.max(yhat_train, axis=1)

    lambda_max_per_class = {}
    for cls in range(yhat_train.shape[1]):
        cls_scores = max_scores[predicted_class == cls]
        if len(cls_scores) > 0:
            lambda_max_per_class[cls] = float(np.mean(cls_scores))

    global_fallback = float(np.mean(max_scores))
    return lambda_max_per_class, global_fallback

def predict_anomaly_lambda_max(model_outputs, lambda_max_per_class, global_fallback):
    """
    Binary output:
    0 = benign / known
    1 = anomalous / unknown
    """
    predicted_class = np.argmax(model_outputs, axis=1)
    max_scores = np.max(model_outputs, axis=1)

    y_pred_binary = np.zeros(len(model_outputs), dtype=int)

    for i, (cls, score) in enumerate(zip(predicted_class, max_scores)):
        threshold = lambda_max_per_class.get(int(cls), global_fallback)
        y_pred_binary[i] = 0 if score >= threshold else 1

    return y_pred_binary

def compute_binary_anomaly_metrics(y_true, y_pred):
    """
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
# 11. TRAIN + EVALUATE
# ============================================================
def train_and_evaluate_cnn():
    model = build_cnn_model(LN, LM, N)

    print("\n" + "=" * 100)
    print("CNN TRAINING")
    print("=" * 100)
    print("- Model architecture: Conv2D -> MaxPooling2D -> Flatten -> Dense(softmax)")
    print("- Detector: ONLY lambdaMax(k), class-dependent, as in the paper")
    print("- No LSTM")
    print("- No skew/kurtosis detector")
    print()

    fit_kwargs = {
        "x": X_train_cnn,
        "y": y_train_softmax,
        "epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "callbacks": [early_stop],
        "verbose": 2,
        "shuffle": False
    }

    if len(X_val_known_cnn) > 0:
        fit_kwargs["validation_data"] = (X_val_known_cnn, y_val_known_softmax)
        print("Validation uses benign validation dialogs whose class is known from training.")
    else:
        print("No known benign validation samples available.")

    history = model.fit(**fit_kwargs)

    print("\n" + "=" * 100)
    print("PAPER DETECTOR THRESHOLDS")
    print("=" * 100)

    lambda_max_per_class, lambda_max_fallback = compute_lambda_max_per_class(
        model, X_train_cnn, batch_size=BATCH_SIZE
    )

    print(f"Number of class-dependent lambdaMax(k): {len(lambda_max_per_class)}")
    print(f"Fallback lambdaMax (global mean max score): {lambda_max_fallback:.6f}")

    print("\n" + "=" * 100)
    print("BINARY ANOMALY DETECTION ON TEST.CSV")
    print("=" * 100)
    print("Detector logic:")
    print("- Predict softmax output y_hat")
    print("- Compute predicted class argmax(y_hat)")
    print("- Compute max(y_hat)")
    print("- Compare max(y_hat) with lambdaMax(k) of the predicted class")
    print("- If max(y_hat) >= lambdaMax(k): benign/known")
    print("- If max(y_hat) <  lambdaMax(k): anomalous/unknown")

    yhat_test = model.predict(X_test_cnn, batch_size=BATCH_SIZE, verbose=0)
    y_pred_binary = predict_anomaly_lambda_max(
        yhat_test,
        lambda_max_per_class=lambda_max_per_class,
        global_fallback=lambda_max_fallback
    )

    cm, metrics = compute_binary_anomaly_metrics(y_test_binary, y_pred_binary)

    print("\nCONFUSION MATRIX [rows=true 0/1, cols=pred 0/1]")
    print(cm)

    print("\nMETRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return {
        "history": history.history,
        "lambda_max_per_class": lambda_max_per_class,
        "lambda_max_fallback": lambda_max_fallback,
        "binary_metrics": metrics
    }

# ============================================================
# 12. RUN
# ============================================================
results = train_and_evaluate_cnn()