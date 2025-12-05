import pandas as pd

# Expected output: python3 split.py
# Loaded 28344 rows.
# After deduplication: 11329 unique dialogs.
# After removing unknown dialogs: 4128
# Benign dialogs: 2823
# Anomalous dialogs: 1305

# 1 - Load dataset
input_csv = "../sip_dataset.csv"
df = pd.read_csv(input_csv)

print("Loaded", len(df), "rows.")

# 2 - Remove duplicates based on 'Signaling Flow' column
df_unique = df.drop_duplicates(subset=["Signaling Flow"])

print("After deduplication:", len(df_unique), "unique dialogs.")

# 3 - Filter out UNKNOWN (Class = 2)
df_filtered = df_unique[df_unique["Class"] != 2]

print("After removing unknown dialogs:", len(df_filtered))

# 4 - Split benign / anomalous
#    Class 0 → benign
#    Class 1 → anomalous
benign_df    = df_filtered[df_filtered["Class"] == 0]
anomalous_df = df_filtered[df_filtered["Class"] == 1]

print("Benign dialogs:", len(benign_df))
print("Anomalous dialogs:", len(anomalous_df))


# 5 - Save the outputs
benign_output    = "benign.csv"
anomalous_output = "anomalous.csv"

benign_df.to_csv(benign_output, index=False)
anomalous_df.to_csv(anomalous_output, index=False)

print("\nSaved files:")
print(" -", benign_output)
print(" -", anomalous_output)