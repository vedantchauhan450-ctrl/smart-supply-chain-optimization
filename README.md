# 🚚 Smart Supply Chain Optimization
### Demand Forecasting → Inventory Policy → Logistics Optimisation

> An end-to-end pipeline: regularised regression forecasts SKU demand, forecast errors drive safety stock / reorder points / EOQ, and a linear program routes stock from warehouses to regions at minimum cost.

## Key techniques
| Technique | Purpose |
|---|---|
| log1p transform of demand | Stabilise variance on right-skewed target |
| Lag & rolling features (1/7/14/28d) | Capture demand momentum (leak-free: shifted) |
| Cyclical encoding (sin/cos) | Day-of-week & month seasonality |
| One-Hot Encoding | SKU, category, region |
| Ridge / Lasso / ElasticNet | Regularised models, 60+ features |
| TimeSeriesSplit + GridSearch | Tuning without look-ahead leakage |
| NNLS stacked ensemble | Blend models using validation set |
| Safety stock / ROP / EOQ | Service-level driven inventory policy |
| Linear programming (HiGHS) | Min-cost warehouse → region shipping |

## Structure
```
src/  data_generator.py · preprocessing.py · models.py · inventory.py
      logistics.py · visualise.py · train.py · predict.py
data/ models/ outputs/ (auto-created)   requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt
python -m src.train --tune --save_models          # synthetic data
python -m src.train --data data/my_sales.csv      # your own data
python -m src.predict --model_path models/ensemble.joblib --input_csv data/new.csv
```
CSV columns: `date, sku, category, region, unit_price, lead_time_days, promo, holiday, demand`

## Methodology
1. **Forecast** – chronological train/val/test split (no shuffling). Target = `log1p(demand)`.
2. **Inventory** – `SS = z·σ_error·√L`, `ROP = μ·L + SS`, `EOQ = √(2DS/H)` (service level via `--service_level`).
3. **Logistics** – minimise Σ cost·x subject to warehouse capacity and regional demand (`scipy.optimize.linprog`).

## Sample results (synthetic data, 60-day hold-out)
| Model | RMSLE | WAPE % | R² (log) |
|---|---|---|---|
| Naive (lag-7) | 0.313 | 23.3 | 0.720 |
| Ridge | 0.208 | 16.1 | 0.876 |
| Lasso | 0.205 | 15.6 | 0.880 |
| ElasticNet | 0.205 | 15.6 | 0.880 |
| Stacked Ensemble | 0.206 | 15.7 | 0.879 |

Logistics: LP plan cut transport cost **~35%** vs proportional-split baseline.

## Notes
- Data is synthetic; swap in real sales data (e.g. Kaggle Store Item Demand / M5) via `--data`.
- Logistics uses hold-out forecasts as the planning horizon; recursive multi-step forecasting is a natural extension.
- Extensions: LightGBM, Prophet, vehicle routing (OR-Tools), multi-echelon inventory.

MIT License
