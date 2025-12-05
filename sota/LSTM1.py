import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
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
#    (same parsing logic as in your original LSTM script)
# -------------------------------------------------------------------
methods = set()
codes   = set()
seqs_benign = []
max_len_benign = 0

for d in benign_dialogs:
    seq = []
    # same trick as your code: split by ':' then take 3rd field (index 2) after splitting by ','
    for msg in str(d).split(':'):
        parts = msg.split(',')
        if len(parts) < 3:
            continue
        tok = parts[2]
        if '-' in tok:
            m, c = tok.split('-', 1)
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
symbols.add('<PAD>')

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
    for msg in str(d).split(':'):
        parts = msg.split(',')
        if len(parts) < 3:
            continue
        tok = parts[2]
        if '-' in tok:
            m, c = tok.split('-', 1)
            m = m.strip()
            c = c.strip()
            seq.append(m)
            seq.append(c)
        else:
            tok = tok.strip()
            seq.append(tok)
    # WARNING: assume all m,c are in message2idx (same methods/codes)
    # If not, you could map unknowns to <PAD>:
    clean_seq = []
    for t in seq:
        if t in message2idx:
            clean_seq.append(t)
        else:
            # fallback: map unknown tokens to <PAD>
            clean_seq.append('<PAD>')
    seqs_anomalous.append(clean_seq)

# -------------------------------------------------------------------
# 4. Hyperparameters
# -------------------------------------------------------------------
M  = len(message2idx) - 1        # distinct SIP messages (without <PAD>)
LM = len(message2idx)            # one-hot length
LN = max_len_benign              # fixed padded length
# N = number of unique BENIGN dialogs
dialogs_str_benign = [' '.join(s) for s in seqs_benign]
labels_benign, uniques_benign = pd.factorize(dialogs_str_benign)
N  = len(uniques_benign)

units         = 256
dropout_rate  = 0.5
batch_size    = 64
learning_rate = 0.001
max_epochs    = 200

opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

print("M  =", M)
print("LM =", LM)
print("LN =", LN)
print("N  =", N)

# -------------------------------------------------------------------
# 5. Build LSTM model (Model 1)
# -------------------------------------------------------------------
def build_model_1():
    m = Sequential([
        LSTM(units, input_shape=(LN, LM)),
        Dropout(dropout_rate),
        Dense(N, activation='softmax')
    ])
    m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return m

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
            enc[i] = to_categorical(message2idx['<PAD>'], num_classes=LM)
    encoded_benign.append(enc)

X_benign = np.array(encoded_benign)

y_int_benign = labels_benign                    # 0..N-1
y_benign = to_categorical(y_int_benign, num_classes=N)

# Train/test split ONLY on benign dialogs
X_train, X_test, y_train, y_test = train_test_split(
    X_benign, y_benign,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Benign train shape:", X_train.shape, y_train.shape)
print("Benign test shape :", X_test.shape,  y_test.shape)

# -------------------------------------------------------------------
# 7. Encode ANOMALOUS dialogs as one-hot tensors X_anomalous
# -------------------------------------------------------------------
encoded_anomalous = []
for s in seqs_anomalous:
    enc = np.zeros((LN, LM), dtype=float)
    for i in range(LN):
        if i < len(s):
            enc[i] = to_categorical(message2idx.get(s[i], message2idx['<PAD>']), num_classes=LM)
        else:
            enc[i] = to_categorical(message2idx['<PAD>'], num_classes=LM)
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

# Train Model 1
model1 = build_model_1()
history1 = model1.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=max_epochs,
    batch_size=batch_size,
    callbacks=[early_stop],
    verbose=2
)

pd_train_1 = detection_perf(model1, X_train, y_train)
pd_test_1  = detection_perf(model1, X_test,  y_test)
print(f"\nIV.B – Model1 Detection PD_train={pd_train_1:.4f}, PD_test={pd_test_1:.4f}")


# -------------------------------------------------------------------
# 9. Unknown SIP Dialog Detection (IV.D)
#    Thresholds computed on BENIGN TRAIN only.
#    Evaluation set = BENIGN TEST (20% test) + ALL ANOMALOUS (unknown).
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
    # y_true: 0 for known, -1 for unknown
    # convert to boolean: True = known
    y_true_known = (y_true == 0)
    y_pred_known = (y_pred == 0)

    cm = confusion_matrix(y_true_known, y_pred_known)
    acc = accuracy_score(y_true_known,    y_pred_known)
    prec = precision_score(y_true_known,  y_pred_known)
    rec  = recall_score(y_true_known,     y_pred_known)
    f1   = f1_score(y_true_known,         y_pred_known)
    return cm, acc, prec, rec, f1

# ---------------- Model 1 ----------------
lambda_M1, lambda_S1, lambda_K1 = compute_thresholds(model1, X_train)

# Evaluation set: 20% benign test (known) + all anomalous (unknown)
X_eval_1 = np.vstack([X_test, X_anomalous])
y_true_eval_1 = np.concatenate([
    np.zeros(len(X_test), dtype=int),   # known
    -1 * np.ones(len(X_anomalous), int) # unknown
])

yhat_eval_1 = model1.predict(X_eval_1, batch_size=batch_size)

y_pred_max_1     = classify_max_threshold(yhat_eval_1, lambda_M1)
y_pred_moments_1 = classify_moments(yhat_eval_1, lambda_S1, lambda_K1)

cm1_max, acc1_max, prec1_max, rec1_max, f11_max = report_unknown(y_true_eval_1, y_pred_max_1)
cm1_mom, acc1_mom, prec1_mom, rec1_mom, f11_mom = report_unknown(y_true_eval_1, y_pred_moments_1)

print("\nIV.D – Model1 – Max-Threshold Classifier")
print("Confusion Matrix (rows=true known?, cols=pred known?):\n", cm1_max)
print(f"Accuracy={acc1_max:.4f}, Precision={prec1_max:.4f}, Recall={rec1_max:.4f}, F1={f11_max:.4f}")

print("\nIV.D – Model1 – Skew/Kurtosis Classifier")
print("Confusion Matrix (rows=true known?, cols=pred known?):\n", cm1_mom)
print(f"Accuracy={acc1_mom:.4f}, Precision={prec1_mom:.4f}, Recall={rec1_mom:.4f}, F1={f11_mom:.4f}")

# -------------------------------------------------------------------
# 10. Extended Unknown SIP Dialog Detection (IV.D "full test")
#     Evaluation set = BENIGN TRAIN + BENIGN TEST + ALL ANOMALOUS
# -------------------------------------------------------------------

print("\n" + "="*70)
print("IV.D – Extended evaluation including benign TRAIN + TEST + anomalous")
print("="*70)

# ---------------- Model 1 – Extended test ----------------
# Thresholds are still computed ONLY on benign train set (X_train),
# as in the original IV.D.
# We already computed lambda_M1, lambda_S1, lambda_K1 earlier.

# Build full evaluation set: all benign (train + test) + all anomalous
X_eval_full_1 = np.vstack([X_train, X_test, X_anomalous])
y_true_eval_full_1 = np.concatenate([
    np.zeros(len(X_train) + len(X_test), dtype=int),   # known
    -1 * np.ones(len(X_anomalous), dtype=int)          # unknown
])

yhat_eval_full_1 = model1.predict(X_eval_full_1, batch_size=batch_size)

y_pred_max_full_1     = classify_max_threshold(yhat_eval_full_1, lambda_M1)
y_pred_moments_full_1 = classify_moments(yhat_eval_full_1, lambda_S1, lambda_K1)

cm1_max_f, acc1_max_f, prec1_max_f, rec1_max_f, f11_max_f = report_unknown(y_true_eval_full_1, y_pred_max_full_1)
cm1_mom_f, acc1_mom_f, prec1_mom_f, rec1_mom_f, f11_mom_f = report_unknown(y_true_eval_full_1, y_pred_moments_full_1)

print("\n[Model 1] Max-Threshold Classifier – FULL (train+test+anomalous)")
print("Confusion Matrix (rows=true known?, cols=pred known?):\n", cm1_max_f)
print(f"Accuracy={acc1_max_f:.4f}, Precision={prec1_max_f:.4f}, Recall={rec1_max_f:.4f}, F1={f11_max_f:.4f}")

print("\n[Model 1] Skew/Kurtosis Classifier – FULL (train+test+anomalous)")
print("Confusion Matrix (rows=true known?, cols=pred known?):\n", cm1_mom_f)
print(f"Accuracy={acc1_mom_f:.4f}, Precision={prec1_mom_f:.4f}, Recall={rec1_mom_f:.4f}, F1={f11_mom_f:.4f}")
