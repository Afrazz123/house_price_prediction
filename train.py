import pandas as pd
import json
import struct
import hashlib

# ------------------ helpers ------------------

def int_bits(n, bits):
    n = 0 if pd.isna(n) else int(n)
    return format(n % (2**bits), f"0{bits}b")

def float_bits(x):
    x = 0.0 if pd.isna(x) else float(x)
    return format(struct.unpack(">I", struct.pack(">f", x))[0], "032b")

def text_bits(s, bits=32):
    s = "" if pd.isna(s) else str(s)
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
    return format(h % (2**bits), f"0{bits}b")

# ------------------ load data ------------------

df = pd.read_csv("train.csv")

# ------------------ fill missing ------------------

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("")
    else:
        df[col] = df[col].fillna(df[col].median())

# ------------------ binary extraction ------------------

memory = []

for _, r in df.iterrows():
    memory.append({
        "Id": int(r.Id),

        # -------- numeric / ordinal --------
        "MSSubClass": int_bits(r.MSSubClass, 8),
        "LotFrontage": float_bits(r.LotFrontage / 100),
        "LotArea": float_bits(r.LotArea / 100000),
        "OverallQual": int_bits(r.OverallQual, 4),
        "OverallCond": int_bits(r.OverallCond, 4),
        "YearBuilt": int_bits(r.YearBuilt, 12),
        "YearRemodAdd": int_bits(r.YearRemodAdd, 12),
        "MasVnrArea": float_bits(r.MasVnrArea / 1000),
        "BsmtFinSF1": float_bits(r.BsmtFinSF1 / 2000),
        "BsmtFinSF2": float_bits(r.BsmtFinSF2 / 2000),
        "BsmtUnfSF": float_bits(r.BsmtUnfSF / 2000),
        "TotalBsmtSF": float_bits(r.TotalBsmtSF / 3000),
        "1stFlrSF": float_bits(r["1stFlrSF"] / 3000),
        "2ndFlrSF": float_bits(r["2ndFlrSF"] / 3000),
        "LowQualFinSF": float_bits(r.LowQualFinSF / 1000),
        "GrLivArea": float_bits(r.GrLivArea / 4000),
        "GarageYrBlt": int_bits(r.GarageYrBlt, 12),
        "GarageCars": int_bits(r.GarageCars, 3),
        "GarageArea": float_bits(r.GarageArea / 1500),
        "WoodDeckSF": float_bits(r.WoodDeckSF / 1000),
        "OpenPorchSF": float_bits(r.OpenPorchSF / 500),
        "EnclosedPorch": float_bits(r.EnclosedPorch / 500),
        "3SsnPorch": float_bits(r["3SsnPorch"] / 500),
        "ScreenPorch": float_bits(r.ScreenPorch / 500),
        "PoolArea": float_bits(r.PoolArea / 1000),
        "MiscVal": float_bits(r.MiscVal / 10000),
        "MoSold": int_bits(r.MoSold, 4),
        "YrSold": int_bits(r.YrSold, 12),

        # -------- categorical (hashed) --------
        "MSZoning": text_bits(r.MSZoning),
        "Street": text_bits(r.Street),
        "Alley": text_bits(r.Alley),
        "LotShape": text_bits(r.LotShape),
        "LandContour": text_bits(r.LandContour),
        "Utilities": text_bits(r.Utilities),
        "LotConfig": text_bits(r.LotConfig),
        "LandSlope": text_bits(r.LandSlope),
        "Neighborhood": text_bits(r.Neighborhood),
        "Condition1": text_bits(r.Condition1),
        "Condition2": text_bits(r.Condition2),
        "BldgType": text_bits(r.BldgType),
        "HouseStyle": text_bits(r.HouseStyle),
        "RoofStyle": text_bits(r.RoofStyle),
        "RoofMatl": text_bits(r.RoofMatl),
        "Exterior1st": text_bits(r.Exterior1st),
        "Exterior2nd": text_bits(r.Exterior2nd),
        "MasVnrType": text_bits(r.MasVnrType),
        "ExterQual": text_bits(r.ExterQual),
        "ExterCond": text_bits(r.ExterCond),
        "Foundation": text_bits(r.Foundation),
        "BsmtQual": text_bits(r.BsmtQual),
        "BsmtCond": text_bits(r.BsmtCond),
        "BsmtExposure": text_bits(r.BsmtExposure),
        "BsmtFinType1": text_bits(r.BsmtFinType1),
        "BsmtFinType2": text_bits(r.BsmtFinType2),
        "Heating": text_bits(r.Heating),
        "HeatingQC": text_bits(r.HeatingQC),
        "CentralAir": text_bits(r.CentralAir),
        "Electrical": text_bits(r.Electrical),
        "KitchenQual": text_bits(r.KitchenQual),
        "Functional": text_bits(r.Functional),
        "FireplaceQu": text_bits(r.FireplaceQu),
        "GarageType": text_bits(r.GarageType),
        "GarageFinish": text_bits(r.GarageFinish),
        "GarageQual": text_bits(r.GarageQual),
        "GarageCond": text_bits(r.GarageCond),
        "PavedDrive": text_bits(r.PavedDrive),
        "PoolQC": text_bits(r.PoolQC),
        "Fence": text_bits(r.Fence),
        "MiscFeature": text_bits(r.MiscFeature),
        "SaleType": text_bits(r.SaleType),
        "SaleCondition": text_bits(r.SaleCondition),

        # -------- label --------
        "SalePrice": int(r.SalePrice)
    })

# ------------------ save ------------------

with open("train_feature_binary.json", "w") as f:
    json.dump(memory, f, indent=2)

print(f"✅ train_feature_binary.json created ({len(memory)} rows)")
