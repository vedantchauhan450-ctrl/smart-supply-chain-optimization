"""Inventory policy: safety stock, reorder point, EOQ (service-level driven)."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def inventory_policy(df, pred_col="pred", service_level=0.95, holding_rate=0.25,
                     order_cost=75.0, unit_cost_ratio=0.6) -> pd.DataFrame:
    """df needs: sku, demand (actual), pred_col, lead_time_days, unit_price, category, region."""
    z = norm.ppf(service_level)
    d = df.assign(error=df["demand"] - df[pred_col])
    p = d.groupby("sku").agg(category=("category", "first"), region=("region", "first"),
                             unit_price=("unit_price", "first"), lead_time_days=("lead_time_days", "first"),
                             avg_daily_demand=(pred_col, "mean"), forecast_std=("error", "std")).reset_index()
    L = p["lead_time_days"]
    p["safety_stock"] = z * p["forecast_std"] * np.sqrt(L)
    p["reorder_point"] = p["avg_daily_demand"] * L + p["safety_stock"]
    H = holding_rate * p["unit_price"] * unit_cost_ratio          # holding cost / unit / year
    annual = p["avg_daily_demand"] * 365
    p["eoq"] = np.sqrt(2 * annual * order_cost / H)
    p["order_cycle_days"] = p["eoq"] / p["avg_daily_demand"]
    p["annual_cost"] = (p["eoq"] / 2 + p["safety_stock"]) * H + (annual / p["eoq"]) * order_cost
    num = p.select_dtypes("number").columns.drop(["unit_price", "lead_time_days"])
    p[num] = p[num].round(1)
    return p
