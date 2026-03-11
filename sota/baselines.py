import os
import re
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter, defaultdict
from itertools import islice

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

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

# benign-only split for methods that need validation
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
print("Baselines included:")
print("1) n-gram + Linear SVM")
print("2) Random Forest on SIP engineered features")
print("3) Markov / HMM-style likelihood baseline")
print("4) Transformer encoder")
print()
print("Main task:")
print("- Train only on benign dialogs")
print("- Evaluate binary anomaly detection on test.csv")
print("- 0 = benign, 1 = anomalous")

# ============================================================
# 6. BASELINE 1: N-GRAM + LINEAR SVM
# ============================================================
print_header("BASELINE 1 - N-GRAM + LINEAR SVM")

vectorizer = CountVectorizer(
    analyzer="word",
    token_pattern=r"(?u)\b\w+\b",
    ngram_range=(1, 3),
    min_df=1,
    binary=False,
)

X_train_ngram = vectorizer.fit_transform(train_texts)
X_test_ngram = vectorizer.transform(test_texts)

svm = LinearSVC(random_state=SEED)
svm.fit(X_train_ngram, y_train_known)

# decision score as confidence surrogate
decision = svm.decision_function(X_train_ngram)
if decision.ndim == 1:
    train_max_conf = np.abs(decision)
else:
    train_max_conf = np.max(decision, axis=1)
lambda_svm = float(np.mean(train_max_conf))

decision_test = svm.decision_function(X_test_ngram)
if decision_test.ndim == 1:
    test_max_conf = np.abs(decision_test)
else:
    test_max_conf = np.max(decision_test, axis=1)

# below-threshold => anomalous
y_pred_svm = np.where(test_max_conf < lambda_svm, 1, 0)
print(f"lambda_svm: {lambda_svm:.6f}")
cm_svm, metrics_svm = print_metrics("n-gram + Linear SVM", y_test_binary, y_pred_svm)

# ============================================================
# 7. BASELINE 2: RANDOM FOREST ON SIP FEATURES
# ============================================================
print_header("BASELINE 2 - RANDOM FOREST ON SIP ENGINEERED FEATURES")

REQ_TOKENS = {
    "INVITE", "ACK", "BYE", "CANCEL", "REGISTER", "OPTIONS", "PRACK",
    "SUBSCRIBE", "NOTIFY", "UPDATE", "REFER", "MESSAGE", "INFO", "PUBLISH"
}

def is_response_code(tok):
    return tok.isdigit() and len(tok) == 3

def response_family(tok):
    if is_response_code(tok):
        return tok[0] + "xx"
    return None

def build_engineered_features(seq):
    cnt = Counter(seq)
    codes = [t for t in seq if is_response_code(t)]
    reqs = [t for t in seq if t in REQ_TOKENS]

    transitions = Counter(zip(seq[:-1], seq[1:])) if len(seq) > 1 else Counter()

    feats = {
        "length": len(seq),
        "num_unique_tokens": len(set(seq)),
        "num_requests": len(reqs),
        "num_responses": len(codes),
        "ratio_requests": len(reqs) / len(seq) if len(seq) > 0 else 0.0,
        "ratio_responses": len(codes) / len(seq) if len(seq) > 0 else 0.0,
        "count_1xx": sum(1 for t in codes if t.startswith("1")),
        "count_2xx": sum(1 for t in codes if t.startswith("2")),
        "count_3xx": sum(1 for t in codes if t.startswith("3")),
        "count_4xx": sum(1 for t in codes if t.startswith("4")),
        "count_5xx": sum(1 for t in codes if t.startswith("5")),
        "count_6xx": sum(1 for t in codes if t.startswith("6")),
        "has_INVITE": int("INVITE" in cnt),
        "has_ACK": int("ACK" in cnt),
        "has_BYE": int("BYE" in cnt),
        "has_CANCEL": int("CANCEL" in cnt),
        "has_REGISTER": int("REGISTER" in cnt),
        "has_200": int("200" in cnt),
        "has_401": int("401" in cnt),
        "has_403": int("403" in cnt),
        "has_404": int("404" in cnt),
        "has_407": int("407" in cnt),
        "has_486": int("486" in cnt),
        "has_487": int("487" in cnt),
        "num_repeat_tokens": sum(v - 1 for v in cnt.values() if v > 1),
        "num_transitions": len(transitions),
    }

    tracked_transitions = [
        ("INVITE", "100"),
        ("INVITE", "180"),
        ("INVITE", "183"),
        ("INVITE", "200"),
        ("200", "ACK"),
        ("BYE", "200"),
        ("CANCEL", "487"),
        ("401", "ACK"),
        ("407", "ACK"),
    ]
    for a, b in tracked_transitions:
        feats[f"trans_{a}_{b}"] = transitions.get((a, b), 0)

    return feats

train_feat_df = pd.DataFrame([build_engineered_features(s) for s in train_sequences])
test_feat_df = pd.DataFrame([build_engineered_features(s) for s in test_sequences])

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=SEED,
    n_jobs=-1,
)

rf.fit(train_feat_df, y_train_known)

proba_train_rf = rf.predict_proba(train_feat_df)
lambda_rf = float(np.mean(np.max(proba_train_rf, axis=1)))

proba_test_rf = rf.predict_proba(test_feat_df)
y_pred_rf = np.where(np.max(proba_test_rf, axis=1) < lambda_rf, 1, 0)

print(f"lambda_rf: {lambda_rf:.6f}")
cm_rf, metrics_rf = print_metrics("Random Forest on SIP engineered features", y_test_binary, y_pred_rf)

# ============================================================
# 8. BASELINE 3: MARKOV / HMM-STYLE LIKELIHOOD BASELINE
# ============================================================
print_header("BASELINE 3 - MARKOV / HMM-STYLE LIKELIHOOD BASELINE")

# This is a simple first-order Markov likelihood baseline.
# It is not a full hidden-state HMM implementation, but it is aligned with
# HMM/Markov-style sequence modeling and is easy to defend as a traditional
# probabilistic sequence baseline.

START = "<START>"
END = "<END>"

vocab = sorted({tok for seq in train_sequences for tok in seq} | {UNK_TOKEN, START, END})
vocab_set = set(vocab)

transition_counts = defaultdict(Counter)

for seq in train_sequences:
    seq2 = [START] + [t if t in vocab_set else UNK_TOKEN for t in seq] + [END]
    for a, b in zip(seq2[:-1], seq2[1:]):
        transition_counts[a][b] += 1

V = len(vocab)

def markov_avg_logprob(seq, alpha=1.0):
    seq2 = [START] + [t if t in vocab_set else UNK_TOKEN for t in seq] + [END]
    logp = 0.0
    n = 0
    for a, b in zip(seq2[:-1], seq2[1:]):
        row = transition_counts[a]
        total = sum(row.values())
        count_ab = row.get(b, 0)
        p = (count_ab + alpha) / (total + alpha * V)
        logp += math.log(p)
        n += 1
    return logp / max(n, 1)

train_markov_scores = np.array([markov_avg_logprob(s) for s in train_sequences], dtype=float)
lambda_markov = float(np.mean(train_markov_scores))

test_markov_scores = np.array([markov_avg_logprob(s) for s in test_sequences], dtype=float)

# low likelihood => anomalous
y_pred_markov = np.where(test_markov_scores < lambda_markov, 1, 0)

print(f"lambda_markov (mean avg log-prob): {lambda_markov:.6f}")
cm_markov, metrics_markov = print_metrics("Markov / HMM-style likelihood baseline", y_test_binary, y_pred_markov)

# ============================================================
# 9. BASELINE 4: TRANSFORMER ENCODER
# ============================================================
print_header("BASELINE 4 - TRANSFORMER ENCODER")

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
# 10. SUMMARY TABLE
# ============================================================
print_header("FINAL SUMMARY")

summary_rows = [
    {
        "baseline": "n-gram + Linear SVM",
        **metrics_svm,
    },
    {
        "baseline": "Random Forest (engineered SIP features)",
        **metrics_rf,
    },
    {
        "baseline": "Markov / HMM-style likelihood baseline",
        **metrics_markov,
    },
    {
        "baseline": "Transformer encoder",
        **metrics_tr,
    },
]

summary_df = pd.DataFrame(summary_rows)
print(summary_df)

summary_df.to_csv("baseline_results.csv", index=False)
print("\nSaved summary to baseline_results.csv")