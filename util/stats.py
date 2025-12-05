import pandas as pd

# Expected output: python3 stats.py 
# Loaded 28344 rows from ../sip_dataset.csv

# === Statistics for ALL CLASSES (0, 1, 2) ===
# Total signalling flows (with duplicates): 28344
# Number of unique signalling flows       : 11329
# Number of different SIP Request Methods : 15
# Number of different SIP Response Codes  : 26
# Number of different Network Functions   : 16
# Min signalling flow length (tokens)     : 3
# Max signalling flow length (tokens)     : 521

# === Statistics for VALID (Class = 0) ===
# Total signalling flows (with duplicates): 10000
# Number of unique signalling flows       : 2823
# Number of different SIP Request Methods : 10
# Number of different SIP Response Codes  : 17
# Number of different Network Functions   : 9
# Min signalling flow length (tokens)     : 17
# Max signalling flow length (tokens)     : 347

# === Statistics for ANOMALOUS (Class = 1) ===
# Total signalling flows (with duplicates): 8344
# Number of unique signalling flows       : 1305
# Number of different SIP Request Methods : 11
# Number of different SIP Response Codes  : 17
# Number of different Network Functions   : 10
# Min signalling flow length (tokens)     : 3
# Max signalling flow length (tokens)     : 419

# === Statistics for UNKNOWN (Class = 2) ===
# Total signalling flows (with duplicates): 10000
# Number of unique signalling flows       : 7201
# Number of different SIP Request Methods : 14
# Number of different SIP Response Codes  : 22
# Number of different Network Functions   : 10
# Min signalling flow length (tokens)     : 7
# Max signalling flow length (tokens)     : 521

# -----------------------------------------
# Helper: parse a single signalling flow
# -----------------------------------------
def analyze_flow(flow_str, methods_set, codes_set, nfs_set, lengths_list):
    """
    Updates sets of:
      - methods_set  : SIP request methods (compressed codes, e.g., I, P, U, B, ...)
      - codes_set    : SIP response codes (e.g., 200, 404, 487, ...)
      - nfs_set      : network functions (e.g., GTW, SCSCF, PCSCF, MTAS, ASLI, ...)
      - lengths_list : stores the length (in tokens) of the flow
    """
    tokens = [tok.strip() for tok in str(flow_str).split(",") if tok.strip() != ""]
    lengths_list.append(len(tokens))

    for tok in tokens:
        # Case 1: message with explicit NF, like "P-200:SCSCF" or "I:ICSCF"
        if ":" in tok:
            left, right = tok.split(":", 1)
            msg_part = left.strip()   # "P-200" or "I"
            nf_part  = right.strip()  # "SCSCF", "ICSCF", ...

            # NF on the right side of ':'
            if nf_part:
                nfs_set.add(nf_part)

            # Parse msg_part into method and (optional) response code
            if "-" in msg_part:
                method, code = msg_part.split("-", 1)
                method = method.strip()
                code   = code.strip()

                if method:
                    methods_set.add(method)
                if code.isdigit():
                    codes_set.add(code)
            else:
                # Only method (e.g., "I")
                if msg_part:
                    methods_set.add(msg_part)

        # Case 2: message_ID[-code] only (no NF), e.g. "P-200", "I-500"
        elif "-" in tok:
            msg_part = tok.strip()
            method, code = msg_part.split("-", 1)
            method = method.strip()
            code   = code.strip()

            if method:
                methods_set.add(method)
            if code.isdigit():
                codes_set.add(code)

        # Case 3: pure Network Function, e.g. "GTW", "SCSCF", "ASLI", "IBCF"
        else:
            tok_clean = tok.strip()
            # If it's a single alphabetic character, treat it as a method
            if tok_clean.isalpha() and len(tok_clean) == 1:
                methods_set.add(tok_clean)
            else:
                # Otherwise it's a network function (NF)
                nfs_set.add(tok_clean)

# -----------------------------------------
# Helper: analyze a subset of the DataFrame
# -----------------------------------------
def analyze_subset(df_subset, name):
    print(f"\n=== Statistics for {name} ===")

    if df_subset.empty:
        print("No rows in this subset.")
        return

    flows = df_subset["Signaling Flow"].astype(str)

    # TOTAL count (including duplicate flows)
    total_flows = len(flows)

    # UNIQUE signalling flows
    unique_flows = flows.unique()
    num_unique_flows = len(unique_flows)

    methods_set = set()
    codes_set   = set()
    nfs_set     = set()
    lengths_list = []

    for flow in flows:
        analyze_flow(flow, methods_set, codes_set, nfs_set, lengths_list)

    min_len = min(lengths_list)
    max_len = max(lengths_list)

    print(f"Total signalling flows (with duplicates): {total_flows}")
    print(f"Number of unique signalling flows       : {num_unique_flows}")
    print(f"Number of different SIP Request Methods : {len(methods_set)}")
    print(f"Number of different SIP Response Codes  : {len(codes_set)}")
    print(f"Number of different Network Functions   : {len(nfs_set)}")
    print(f"Min signalling flow length (tokens)     : {min_len}")
    print(f"Max signalling flow length (tokens)     : {max_len}")

    # If you want to inspect the actual sets, uncomment:
    # print("\nMethods:", methods_set)
    # print("Response codes:", codes_set)
    # print("Network functions:", nfs_set)


# -----------------------------------------
# Main script
# -----------------------------------------
def main():
    input_csv = "../sip_dataset.csv"
    df = pd.read_csv(input_csv)

    print("Loaded", len(df), "rows from", input_csv)

    # All classes (0, 1, 2)
    analyze_subset(df, "ALL CLASSES (0, 1, 2)")

    # Class 0 = valid
    df_valid = df[df["Class"] == 0]
    analyze_subset(df_valid, "VALID (Class = 0)")

    # Class 1 = anomalous
    df_anomalous = df[df["Class"] == 1]
    analyze_subset(df_anomalous, "ANOMALOUS (Class = 1)")

    # Class 2 = unknown
    df_unknown = df[df["Class"] == 2]
    analyze_subset(df_unknown, "UNKNOWN (Class = 2)")


if __name__ == "__main__":
    main()