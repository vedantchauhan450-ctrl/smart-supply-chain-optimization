"""Synthetic multi-SKU daily demand generator (trend + seasonality + promo + holiday)."""
import numpy as np
import pandas as pd

CATEGORIES = ["Electronics", "Grocery", "Apparel", "Home"]
REGIONS = ["North", "South", "East", "West"]


def generate_demand_data(n_days: int = 730, n_skus: int = 12, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    weekend = np.asarray(dates.dayofweek >= 5)
    holiday = np.asarray((dates.month == 12) & (dates.day >= 15)).astype(int)
    frames = []
    for s in range(n_skus):
        base, trend = rng.uniform(20, 120), rng.uniform(-0.0003, 0.0008)
        wk_amp, yr_amp = rng.uniform(0.1, 0.4), rng.uniform(0.1, 0.35)
        promo = (rng.random(n_days) < 0.08).astype(int)
        mu = (base * (1 + trend * t) * (1 + yr_amp * np.sin(2 * np.pi * t / 365.25 + s))
              * (1 + wk_amp * weekend) * (1 + 0.35 * promo) * (1 + 0.5 * holiday))
        demand = rng.poisson(mu * rng.lognormal(0, 0.15, n_days))
        frames.append(pd.DataFrame({
            "date": dates, "sku": f"SKU{s + 1:02d}",
            "category": CATEGORIES[s % 4], "region": REGIONS[(s // 3) % 4],
            "unit_price": round(rng.uniform(5, 200), 2),
            "lead_time_days": int(rng.integers(3, 15)),
            "promo": promo, "holiday": holiday, "demand": demand}))
    return pd.concat(frames, ignore_index=True)
