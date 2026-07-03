import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

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

# Transformer config
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_TRANSFORMER_BLOCKS = 2
DROPOUT_RATE = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

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
# 5. SHARED REPRESENTATIONS
# ============================================================
def seq_to_text(seq):
    return " ".join(seq)

train_texts = [seq_to_text(s) for s in train_sequences]
val_texts = [seq_to_text(s) for s in val_sequences]
test_texts = [seq_to_text(s) for s in test_sequences]

# known benign classes: one class per unique training dialog
known_benign_class_to_id = {
    dlg: i for i, dlg in enumerate(pd.unique(train_texts))
}
N_CLASSES = len(known_benign_class_to_id)

def map_to_known_benign_class(seq):
    return known_benign_class_to_id.get(seq_to_text(seq), -1)

y_train_known = np.array([map_to_known_benign_class(s) for s in train_sequences], dtype=int)
y_val_known = np.array([map_to_known_benign_class(s) for s in val_sequences], dtype=int)

assert np.all(y_train_known >= 0)

print_header("SETUP SUMMARY")
print(f"Known benign classes in training: {N_CLASSES}")
print("Baseline: Transformer encoder")
print()
print("Main task:")
print("- Train only on benign dialogs")
print("- Evaluate binary anomaly detection on test.csv")
print("- 0 = benign, 1 = anomalous")

# ============================================================
# 6. TRANSFORMER ENCODER
# ============================================================
print_header("TRANSFORMER ENCODER")

# Build vocab only from benign training data
token_vocab = sorted({tok for seq in train_sequences for tok in seq} | {PAD_TOKEN, UNK_TOKEN})
token_to_idx = {tok: i for i, tok in enumerate(token_vocab)}
idx_pad = token_to_idx[PAD_TOKEN]
idx_unk = token_to_idx[UNK_TOKEN]
max_len = max(len(s) for s in train_sequences)

def encode_tokens(seqs):
    out = []
    for seq in seqs:
        ids = [token_to_idx.get(tok, idx_unk) for tok in seq[:max_len]]
        out.append(ids)
    return pad_sequences(out, maxlen=max_len, padding="post", truncating="post", value=idx_pad)

X_train_tok = encode_tokens(train_sequences)
X_val_tok = encode_tokens(val_sequences)
X_test_tok = encode_tokens(test_sequences)

# validation only on benign samples whose class exists in training
val_known_mask = (y_val_known >= 0)
X_val_known_tok = X_val_tok[val_known_mask]
y_val_known = y_val_known[val_known_mask]

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_classifier(maxlen, vocab_size, num_classes):
    inputs = layers.Input(shape=(maxlen,))
    x = TokenAndPositionEmbedding(maxlen, vocab_size, EMBED_DIM)(inputs)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

transformer = build_transformer_classifier(
    maxlen=max_len,
    vocab_size=len(token_to_idx),
    num_classes=N_CLASSES
)

early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    restore_best_weights=True
)

fit_kwargs = {
    "x": X_train_tok,
    "y": y_train_known,
    "epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "callbacks": [early_stop],
    "verbose": 2,
    "shuffle": False,
}
if len(X_val_known_tok) > 0:
    fit_kwargs["validation_data"] = (X_val_known_tok, y_val_known)

history = transformer.fit(**fit_kwargs)

proba_train_tr = transformer.predict(X_train_tok, batch_size=BATCH_SIZE, verbose=0)
lambda_tr = float(np.mean(np.max(proba_train_tr, axis=1)))

proba_test_tr = transformer.predict(X_test_tok, batch_size=BATCH_SIZE, verbose=0)
y_pred_tr = np.where(np.max(proba_test_tr, axis=1) < lambda_tr, 1, 0)

print(f"lambda_transformer: {lambda_tr:.6f}")
cm_tr, metrics_tr = print_metrics("Transformer encoder", y_test_binary, y_pred_tr)

# ============================================================
# 7. SUMMARY TABLE
# ============================================================
print_header("FINAL SUMMARY")

summary_rows = [
    {
        "baseline": "Transformer encoder",
        **metrics_tr,
    },
]

summary_df = pd.DataFrame(summary_rows)
print(summary_df)

summary_df.to_csv("baseline_results.csv", index=False)
print("\nSaved summary to baseline_results.csv")

# Save per-sample predictions for a posteriori model comparison (e.g. McNemar).
# test_csv_row = original 0-based row number in test.csv (before empty-sequence
# filtering): stable join key to align predictions across the baseline scripts
test_csv_row = np.array([i for i, s in enumerate(test_sequences_all) if len(s) > 0], dtype=int)
pred_df = pd.DataFrame({
    "idx": np.arange(len(y_test_binary)),
    "test_csv_row": test_csv_row,
    "true_label": y_test_binary,
    "pred_label": y_pred_tr,
    "max_softmax": np.max(proba_test_tr, axis=1),
    # original signalling description of each test dialog, verbatim from test.csv
    "Replaced Signalling Description": [test_dialogs_raw[i] for i in test_csv_row],
})
pred_df.to_csv("transformer_predictions.csv", index=False)
print("Saved per-dialog predictions to transformer_predictions.csv")

# ============================================================
# 8. STATISTICAL TESTS: BOOTSTRAP CI + McNEMAR'S TEST
# ============================================================
from scipy.stats import binomtest, chi2

print_header("STATISTICAL TESTS: BOOTSTRAP CI + McNEMAR'S TEST")

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

def print_mcnemar(name, res, labels=("n01", "n10")):
    print(f"\nMcNemar's test - {name}")
    print(f"{labels[0]}: {res['n01']}, {labels[1]}: {res['n10']}")
    print(f"method   : {res['method']}")
    print(f"statistic: {res['statistic']:.4f}")
    print(f"p-value  : {res['p_value']:.6f}")

boot_tr = bootstrap_metric_cis(y_test_binary, y_pred_tr)
print_bootstrap_cis("Transformer encoder", boot_tr)

mcnemar_tr = mcnemar_vs_truth(y_test_binary, y_pred_tr)
print_mcnemar("Transformer encoder (FP vs FN)", mcnemar_tr, labels=("FP", "FN"))
