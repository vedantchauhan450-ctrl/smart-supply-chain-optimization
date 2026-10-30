"""Feature engineering + encoding for demand forecasting."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["sku", "category", "region"]
NUM_COLS = ["dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "trend",
            "promo", "holiday", "log_price", "lead_time_days",
            "lag_1", "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28", "roll_std_7"]
FEATURE_COLS = CAT_COLS + NUM_COLS


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + lag/rolling features. Lags are log1p-scaled (target is log1p too)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    dow, month = df["date"].dt.dayofweek, df["date"].dt.month
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(int)
    df["trend"] = (df["date"] - pd.Timestamp("2020-01-01")).dt.days / 365.25
    df["log_price"] = np.log1p(df["unit_price"])
    g = df.groupby("sku")["demand"]
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = np.log1p(g.shift(lag))
    for w in (7, 28):
        df[f"roll_mean_{w}"] = np.log1p(g.transform(lambda s: s.shift(1).rolling(w).mean()))
    df["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7).std())
    df["log_demand"] = np.log1p(df["demand"])
    return df.dropna(subset=NUM_COLS).reset_index(drop=True)


def time_split(df: pd.DataFrame, val_days: int = 60, test_days: int = 60):
    """Chronological split - never shuffle time series."""
    last = df["date"].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    key = ["date", "sku"]
    train = df[df["date"] < val_start].sort_values(key)
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)].sort_values(key)
    test = df[df["date"] >= test_start].sort_values(key)
    return train, val, test


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)])
