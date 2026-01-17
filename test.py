import pandas as pd
import json
import struct
import hashlib
import numpy as np
import time
import sys  # For progress display

# ---------------- CONFIG ----------------
TRAIN_MEMORY_FILE = "train_feature_binary.json"
TEST_FILE = "test.csv"
OUTPUT_FILE = "submission.csv"

SIM_THRESHOLD = 0.92
TOP_FEATURE_LIMIT = 18
MIN_MATCHES = 6
# ---------------------------------------

# ---------------- BINARY UTILS ----------------
def int_bits(n, b):
    try: return format(int(n), f'0{b}b')
    except: return "0"*b

def float_bits(f):
    try: return format(struct.unpack('!I', struct.pack('!f', float(f)))[0], '032b')
    except: return "0"*32

def text_bits(s, b=32):
    s = str(s) if s is not None else ""
    h = int(hashlib.md5(s.encode()).hexdigest(), 16)
    return format(h % (2**b), f'0{b}b')

def similarity(a, b):
    if len(a) != len(b): return 0
    return sum(x==y for x,y in zip(a,b))/len(a)

# ---------------- LOAD TRAIN MEMORY ----------------
with open(TRAIN_MEMORY_FILE, "r") as f:
    MEMORY = json.load(f)

FEATURES = [k for k in MEMORY[0].keys() if k not in ["SalePrice", "Id"]]

# ---------------- LOAD TEST ----------------
df = pd.read_csv(TEST_FILE)

numeric_fill_zero = [
    "LotFrontage","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
    "TotalBsmtSF","GarageYrBlt","GarageArea","WoodDeckSF","OpenPorchSF",
    "EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal"
]
for c in numeric_fill_zero: df[c] = df[c].fillna(0)

categorical_fill = [
    "Alley","MasVnrType","BsmtQual","BsmtCond","BsmtExposure",
    "BsmtFinType1","BsmtFinType2","FireplaceQu","GarageType",
    "GarageFinish","GarageQual","GarageCond","PoolQC","Fence",
    "MiscFeature"
]
for c in categorical_fill: df[c] = df[c].fillna("")

# ---------------- FEATURE → BINARY ----------------
def to_binary_features(r):
    b = {}
    # numeric
    b["MSSubClass"] = int_bits(r.MSSubClass, 8)
    b["LotFrontage"] = float_bits(r.LotFrontage / 300)
    b["LotArea"] = float_bits(r.LotArea / 200000)
    b["OverallQual"] = int_bits(r.OverallQual, 4)
    b["OverallCond"] = int_bits(r.OverallCond, 4)
    b["YearBuilt"] = int_bits(r.YearBuilt - 1800, 9)
    b["YearRemodAdd"] = int_bits(r.YearRemodAdd - 1800, 9)
    b["MasVnrArea"] = float_bits(r.MasVnrArea / 1000)
    b["TotalBsmtSF"] = float_bits(r.TotalBsmtSF / 3000)
    b["1stFlrSF"] = float_bits(r["1stFlrSF"] / 3000)
    b["2ndFlrSF"] = float_bits(r["2ndFlrSF"] / 3000)
    b["GrLivArea"] = float_bits(r.GrLivArea / 5000)
    b["GarageCars"] = int_bits(r.GarageCars, 3)
    b["GarageArea"] = float_bits(r.GarageArea / 1500)
    b["FullBath"] = int_bits(r.FullBath, 2)
    b["HalfBath"] = int_bits(r.HalfBath, 2)
    b["BedroomAbvGr"] = int_bits(r.BedroomAbvGr, 3)
    b["TotRmsAbvGrd"] = int_bits(r.TotRmsAbvGrd, 4)
    b["Fireplaces"] = int_bits(r.Fireplaces, 2)
    b["MoSold"] = int_bits(r.MoSold, 4)
    b["YrSold"] = int_bits(r.YrSold - 2000, 5)

    # categorical
    cat_cols = [
        "MSZoning","Street","Alley","LotShape","LandContour","Utilities",
        "LotConfig","LandSlope","Neighborhood","Condition1","Condition2",
        "BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st",
        "Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation",
        "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
        "Heating","HeatingQC","CentralAir","Electrical","KitchenQual",
        "Functional","FireplaceQu","GarageType","GarageFinish",
        "GarageQual","GarageCond","PavedDrive","PoolQC","Fence",
        "MiscFeature","SaleType","SaleCondition"
    ]
    for c in cat_cols: b[c] = text_bits(r[c], 32)
    return b

# ---------------- WEIGHTED PRICE PREDICT ----------------
def weighted_price_predict(test_features):
    feature_weights = {f:1 for f in FEATURES}
    # increase weights for important features
    feature_weights["OverallQual"] = 5
    feature_weights["GrLivArea"] = 4
    feature_weights["TotalBsmtSF"] = 3
    feature_weights["GarageCars"] = 3

    votes = []
    for mem in MEMORY:
        score = 0
        for f, w in feature_weights.items():
            if f in test_features and f in mem:
                sim = similarity(test_features[f], mem[f])
                if sim >= 0.92:
                    score += w
        if score > 0:
            votes.append((score, mem["SalePrice"]))
    if not votes:
        return int(np.median([m["SalePrice"] for m in MEMORY]))
    votes = sorted(votes, key=lambda x:x[0], reverse=True)
    weights = np.array([v[0] for v in votes])
    prices = np.array([v[1] for v in votes])
    return int(np.average(prices, weights=weights))

# ---------------- RUN WITH PROGRESS ----------------
results = []
total = len(df)
start_time = time.time()

for idx, row in df.iterrows():
    features = to_binary_features(row)
    price = weighted_price_predict(features)
    results.append([int(row.Id), price])

    # live progress
    elapsed = time.time() - start_time
    progress = (idx+1) / total * 100
    est_total = elapsed / (idx+1) * total
    est_remaining = est_total - elapsed
    sys.stdout.write(
        f"\rProgress: {progress:.2f}% | "
        f"Elapsed: {elapsed:.2f}s | "
        f"Est. remaining: {est_remaining:.2f}s"
    )
    sys.stdout.flush()

# ---------------- SAVE SUBMISSION ----------------
pd.DataFrame(results, columns=["Id","SalePrice"]).to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ submission.csv generated | Total time: {time.time()-start_time:.2f}s")
