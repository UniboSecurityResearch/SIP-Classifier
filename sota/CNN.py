import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from scipy.stats import skew, kurtosis

# -------------------------------------------------------------------
# 1. Load benign / anomalous datasets (already without duplicates)
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
# 2. Build vocab (methods + codes) and sequences from BENIGN only
#    Parsing logic identical to your LSTM script
# -------------------------------------------------------------------
methods = set()
codes   = set()
seqs_benign = []
max_len_benign = 0

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
    if len(seq) > max_len_benign:
        max_len_benign = len(seq)
    seqs_benign.append(seq)

symbols = methods | codes
symbols.add("<PAD>")

message2idx = {m: i for i, m in enumerate(sorted(symbols))}
idx2message = {i: m for m, i in message2idx.items()}

print("Number of distinct SIP symbols (methods + codes):", len(symbols) - 1)
print("Vocabulary size LM (including <PAD>):", len(symbols))
print("Max benign sequence length LN:", max_len_benign)

# -------------------------------------------------------------------
# 3. Parse anomalous dialogs with same vocab / encoding
# -------------------------------------------------------------------
seqs_anomalous = []
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

    # Map unknown tokens (if any) to <PAD>
    clean_seq = []
    for t in seq:
        if t in message2idx:
            clean_seq.append(t)
        else:
            clean_seq.append("<PAD>")
    seqs_anomalous.append(clean_seq)

# -------------------------------------------------------------------
# 4. Hyperparameters
# -------------------------------------------------------------------
M  = len(message2idx) - 1        # distinct SIP messages (without <PAD>)
LM = len(message2idx)            # one-hot length
LN = max_len_benign              # fixed padded length

dialogs_str_benign = [" ".join(s) for s in seqs_benign]
labels_benign, uniques_benign = pd.factorize(dialogs_str_benign)
N  = len(uniques_benign)         # number of benign dialog classes

units         = 256
dropout_rate  = 0.5
batch_size    = 64
learning_rate = 0.001
max_epochs    = 200

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
early_stop = EarlyStopping(
    monitor="val_loss", mode="min", patience=10, restore_best_weights=True
)

print("M  =", M)
print("LM =", LM)
print("LN =", LN)
print("N  =", N)

# -------------------------------------------------------------------
# 5. Build CNN model
# -------------------------------------------------------------------
def build_cnn_model():
    model = Sequential(
        [
            Conv1D(
                filters=32,
                kernel_size=3,
                activation="relu",
                input_shape=(LN, LM),
            ),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(dropout_rate),
            Dense(N, activation="softmax"),
        ]
    )
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# -------------------------------------------------------------------
# 6. Encode BENIGN dialogs as one-hot tensors X_benign
# -------------------------------------------------------------------
encoded_benign = []
for s in seqs_benign:
    enc = np.zeros((LN, LM), dtype=float)
    for i in range(LN):
        if i < len(s):
            enc[i] = to_categorical(message2idx[s[i]], num_classes=LM)
        else:
            enc[i] = to_categorical(message2idx["<PAD>"], num_classes=LM)
    encoded_benign.append(enc)

X_benign = np.array(encoded_benign)

y_int_benign = labels_benign
y_benign = to_categorical(y_int_benign, num_classes=N)

# 80/20 split ONLY on benign dialogs
X_train, X_test, y_train, y_test = train_test_split(
    X_benign,
    y_benign,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Benign train shape:", X_train.shape, y_train.shape)
print("Benign test shape :", X_test.shape, y_test.shape)

# -------------------------------------------------------------------
# 7. Encode ANOMALOUS dialogs as one-hot tensors X_anomalous
# -------------------------------------------------------------------
encoded_anomalous = []
for s in seqs_anomalous:
    enc = np.zeros((LN, LM), dtype=float)
    for i in range(LN):
        if i < len(s):
            enc[i] = to_categorical(
                message2idx.get(s[i], message2idx["<PAD>"]), num_classes=LM
            )
        else:
            enc[i] = to_categorical(message2idx["<PAD>"], num_classes=LM)
    encoded_anomalous.append(enc)

X_anomalous = np.array(encoded_anomalous)
print("Anomalous shape:", X_anomalous.shape)

# -------------------------------------------------------------------
# 8. Detection performance (IV.B)
# -------------------------------------------------------------------
def detection_perf(model, X, y_true):
    y_pred = model.predict(X, batch_size=batch_size)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    acc = accuracy_score(y_true_labels, y_pred_labels)
    return acc

cnn_model = build_cnn_model()
history_cnn = cnn_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=max_epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=2,
)

pd_train_cnn = detection_perf(cnn_model, X_train, y_train)
pd_test_cnn  = detection_perf(cnn_model, X_test,  y_test)
print(f"\nIV.B – CNN Detection PD_train={pd_train_cnn:.4f}, PD_test={pd_test_cnn:.4f}")

# -------------------------------------------------------------------
# 9. Unknown SIP Dialog Detection (IV.D) – clean test
#    Thresholds from BENIGN TRAIN; eval = BENIGN TEST + ANOMALOUS
# -------------------------------------------------------------------
def compute_thresholds(model, X_known_train):
    """
    Computes λM, λS, λK as in the paper, using known dialogs (train set).
    """
    yhat_train = model.predict(X_known_train, batch_size=batch_size)
    max_train  = np.max(yhat_train, axis=1)
    lambda_M   = max_train.mean()

    sk_train = skew(yhat_train, axis=1)
    ku_train = kurtosis(yhat_train, axis=1)
    mu_S, var_S = sk_train.mean(), sk_train.var()
    mu_K, var_K = ku_train.mean(), ku_train.var()
    lambda_S    = mu_S - var_S
    lambda_K    = mu_K - var_K

    return lambda_M, lambda_S, lambda_K

def classify_max_threshold(yhat, lambda_M):
    # 0 = known, -1 = unknown
    return np.where(np.max(yhat, axis=1) < lambda_M, -1, 0)

def classify_moments(yhat, lambda_S, lambda_K):
    ske = skew(yhat, axis=1)
    kur = kurtosis(yhat, axis=1)
    return np.where((ske < lambda_S) & (kur < lambda_K), -1, 0)

def report_unknown(y_true, y_pred):
    """
    y_true: 0 for known, -1 for unknown
    Evaluates metrics for the "known" class.
    """
    y_true_known = (y_true == 0)
    y_pred_known = (y_pred == 0)

    cm = confusion_matrix(y_true_known, y_pred_known)
    acc = accuracy_score(y_true_known,    y_pred_known)
    prec = precision_score(y_true_known,  y_pred_known, zero_division=0)
    rec  = recall_score(y_true_known,     y_pred_known, zero_division=0)
    f1   = f1_score(y_true_known,         y_pred_known, zero_division=0)
    return cm, acc, prec, rec, f1

# Compute thresholds on benign TRAIN
lambda_M, lambda_S, lambda_K = compute_thresholds(cnn_model, X_train)

# Evaluation set (clean): benign TEST (known) + ALL anomalous (unknown)
X_eval_clean = np.vstack([X_test, X_anomalous])
y_true_clean = np.concatenate([
    np.zeros(len(X_test), dtype=int),    # known
    -1 * np.ones(len(X_anomalous), int)  # unknown
])

yhat_clean = cnn_model.predict(X_eval_clean, batch_size=batch_size)

y_pred_max_clean     = classify_max_threshold(yhat_clean, lambda_M)
y_pred_moments_clean = classify_moments(yhat_clean, lambda_S, lambda_K)

cm_max_c, acc_max_c, prec_max_c, rec_max_c, f1_max_c = report_unknown(y_true_clean, y_pred_max_clean)
cm_mom_c, acc_mom_c, prec_mom_c, rec_mom_c, f1_mom_c = report_unknown(y_true_clean, y_pred_moments_clean)

print("\nIV.D – CNN – CLEAN Max-Threshold Classifier")
print("Confusion Matrix (rows=true known?, cols=pred known?):\n", cm_max_c)
print(f"Accuracy={acc_max_c:.4f}, Precision={prec_max_c:.4f}, Recall={rec_max_c:.4f}, F1={f1_max_c:.4f}")

print("\nIV.D – CNN – CLEAN Skew/Kurtosis Classifier")
print("Confusion Matrix (rows=true known?, cols=pred known?):\n", cm_mom_c)
print(f"Accuracy={acc_mom_c:.4f}, Precision={prec_mom_c:.4f}, Recall={rec_mom_c:.4f}, F1={f1_mom_c:.4f}")

# -------------------------------------------------------------------
# 10. Extended Unknown Detection (IV.D "full")
#     Evaluation = BENIGN TRAIN + BENIGN TEST + ANOMALOUS
# -------------------------------------------------------------------
print("\n" + "="*70)
print("IV.D – CNN – Extended evaluation (benign TRAIN + TEST + anomalous)")
print("="*70)

X_eval_full = np.vstack([X_train, X_test, X_anomalous])
y_true_full = np.concatenate([
    np.zeros(len(X_train) + len(X_test), dtype=int),   # known
    -1 * np.ones(len(X_anomalous), dtype=int)          # unknown
])

yhat_full = cnn_model.predict(X_eval_full, batch_size=batch_size)

y_pred_max_full     = classify_max_threshold(yhat_full, lambda_M)
y_pred_moments_full = classify_moments(yhat_full, lambda_S, lambda_K)

cm_max_f, acc_max_f, prec_max_f, rec_max_f, f1_max_f = report_unknown(y_true_full, y_pred_max_full)
cm_mom_f, acc_mom_f, prec_mom_f, rec_mom_f, f1_mom_f = report_unknown(y_true_full, y_pred_moments_full)

print("\n[ CNN ] Max-Threshold Classifier – FULL (train+test+anomalous)")
print("Confusion Matrix (rows=true known?, cols=pred known?):\n", cm_max_f)
print(f"Accuracy={acc_max_f:.4f}, Precision={prec_max_f:.4f}, Recall={rec_max_f:.4f}, F1={f1_max_f:.4f}")

print("\n[ CNN ] Skew/Kurtosis Classifier – FULL (train+test+anomalous)")
print("Confusion Matrix (rows=true known?, cols=pred known?):\n", cm_mom_f)
print(f"Accuracy={acc_mom_f:.4f}, Precision={prec_mom_f:.4f}, Recall={rec_mom_f:.4f}, F1={f1_mom_f:.4f}")
