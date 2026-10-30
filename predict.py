"""Inference: python -m src.predict --model_path models/ensemble.joblib --input_csv data/new.csv
Input CSV needs >=28 days of history per SKU (lag features)."""
import argparse
import joblib
import pandas as pd
from .preprocessing import make_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", default="outputs/predictions.csv")
    a = ap.parse_args()
    ens = joblib.load(a.model_path)
    feats = make_features(pd.read_csv(a.input_csv, parse_dates=["date"]))
    feats["predicted_demand"] = ens.predict_demand(feats).round(1)
    feats[["date", "sku", "predicted_demand"]].to_csv(a.output_csv, index=False)
    print(f"Wrote {len(feats):,} predictions -> {a.output_csv}")


if __name__ == "__main__":
    main()
