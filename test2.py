import pandas as pd
import json
import struct
import hashlib
import numpy as np
import time
import sys

# ================= CONFIG =================
TRAIN_MEMORY_FILE = "train_feature_binary.json"
TEST_FILE = "test.csv"
OUTPUT_FILE = "submission.csv"

SIM_MIN = 0.75
# =========================================

# ================= BINARY UTILS =================
def int_bits(n, b):
    try:
        return format(int(n), f'0{b}b')
    except:
        return "0" * b

def float_bits(f):
    try:
        return format(struct.unpack("!I", struct.pack("!f", float(f)))[0], "032b")
    except:
        return "0" * 32

def text_bits(s, b=32):
    s = "" if pd.isna(s) else str(s)
    h = int(hashlib.md5(s.encode()).hexdigest(), 16)
    return format(h % (2**b), f"0{b}b")

def similarity(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0
    if len(a) != len(b):
        return 0
    return sum(x == y for x, y in zip(a, b)) / len(a)

# ================= LOAD TRAIN MEMORY =================
with open(TRAIN_MEMORY_FILE, "r") as f:
    MEMORY = json.load(f)

FEATURES = [k for k in MEMORY[0] if k not in ["SalePrice", "Id"]]

TRAIN_PRICES = np.array([m["SalePrice"] for m in MEMORY])

# ================= LOAD TEST =================
df = pd.read_csv(TEST_FILE)

df.fillna(0, inplace=True)

# ================= FEATURE → BINARY =================
def to_binary_features(r):
    b = {}

    # numeric
    b["OverallQual"] = int_bits(r.OverallQual, 4)
    b["GrLivArea"] = float_bits(r.GrLivArea / 5000)
    b["TotalBsmtSF"] = float_bits(r.TotalBsmtSF / 3000)
    b["GarageCars"] = int_bits(r.GarageCars, 3)
    b["YearBuilt"] = int_bits(r.YearBuilt - 1800, 9)
    b["LotArea"] = float_bits(r.LotArea / 200000)
    b["1stFlrSF"] = float_bits(r["1stFlrSF"] / 3000)

    # categorical
    cat_cols = [
        "Neighborhood", "HouseStyle", "BldgType",
        "Exterior1st", "KitchenQual", "SaleCondition"
    ]
    for c in cat_cols:
        b[c] = text_bits(r[c], 32)

    return b

# ================= FEATURE WEIGHTS =================
FEATURE_WEIGHTS = {
    "OverallQual": 5,
    "GrLivArea": 4,
    "TotalBsmtSF": 3,
    "GarageCars": 3,
    "YearBuilt": 2,
    "LotArea": 2,
    "1stFlrSF": 2,
    "Neighborhood": 4,
    "KitchenQual": 3,
}

# ================= NATURAL CONVERGENCE =================
def predict_price(test_bin):
    weights = []
    prices = []

    for mem in MEMORY:
        sim_score = 0
        total_weight = 0

        for f, w in FEATURE_WEIGHTS.items():
            if f in test_bin and f in mem:
                s = similarity(test_bin[f], mem[f])
                if s >= SIM_MIN:
                    sim_score += s * w
                    total_weight += w

        if total_weight > 0:
            weights.append(sim_score)
            prices.append(mem["SalePrice"])

    if not weights:
        return int(np.median(TRAIN_PRICES))

    weights = np.array(weights)
    prices = np.array(prices)

    return int(np.sum(weights * prices) / np.sum(weights))

# ================= RUN WITH LIVE PROGRESS =================
results = []
total = len(df)
start = time.time()

for i, row in df.iterrows():
    test_bin = to_binary_features(row)
    price = predict_price(test_bin)
    results.append([int(row.Id), price])

    elapsed = time.time() - start
    progress = (i + 1) / total
    eta = (elapsed / (i + 1)) * (total - i - 1)

    sys.stdout.write(
        f"\rProgress: {progress*100:.2f}% | "
        f"Elapsed: {elapsed:.1f}s | "
        f"ETA: {eta:.1f}s"
    )
    sys.stdout.flush()

# ================= SAVE =================
pd.DataFrame(results, columns=["Id", "SalePrice"]).to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ submission.csv generated | Total time: {time.time() - start:.2f}s")
