"""End-to-end pipeline: forecast -> inventory policy -> logistics optimisation.
Run from project root:  python -m src.train --tune --save_models
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_generator import generate_demand_data
from .inventory import inventory_policy
from .logistics import build_network, optimize_transport
from .models import DemandModel, StackedEnsemble, evaluate_demand
from .preprocessing import make_features, time_split
from . import visualise as viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV: date,sku,category,region,unit_price,lead_time_days,promo,holiday,demand")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--service_level", type=float, default=0.95)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(a.data, parse_dates=["date"]) if a.data else generate_demand_data(a.days)
    train, val, test = time_split(make_features(raw))
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

    models = [DemandModel.ridge(), DemandModel.lasso(), DemandModel.elasticnet()]
    rows = {"Naive (lag-7)": evaluate_demand(test["demand"], np.expm1(test["lag_7"]))}
    for m in models:
        if a.tune:
            print(f"{m.name} best params:", m.tune(train))
        m.fit(train)
        rows[m.name] = evaluate_demand(test["demand"], m.predict_demand(test))
    ens = StackedEnsemble(models).fit_weights(val)
    test = test.assign(pred=ens.predict_demand(test))
    rows["Stacked Ensemble"] = evaluate_demand(test["demand"], test["pred"])
    table = pd.DataFrame(rows).T.round(4)
    print("\n== Hold-out forecast accuracy ==\n", table)
    print("Ensemble weights:", dict(zip([m.name for m in models], ens.weights.round(3))))
    table.to_csv(out / "model_comparison.csv")

    policy = inventory_policy(test, service_level=a.service_level)
    policy.to_csv(out / "inventory_policy.csv", index=False)
    print("\n== Inventory policy (first 5 SKUs) ==\n",
          policy.head()[["sku", "avg_daily_demand", "safety_stock", "reorder_point", "eoq"]])

    horizon = test[test["date"] > test["date"].max() - pd.Timedelta(days=30)]
    demand = horizon.groupby("region")["pred"].sum().round()
    supply, cost = build_network(demand)
    plan, opt_cost, base_cost = optimize_transport(supply, demand, cost)
    plan.to_csv(out / "transport_plan.csv")
    print(f"\n== Logistics (30-day horizon) ==\nOptimised cost ${opt_cost:,.0f} vs baseline ${base_cost:,.0f} "
          f"-> saving {100 * (1 - opt_cost / base_cost):.1f}%\n", plan)

    if not a.no_plots:
        viz.plot_model_comparison(table, out / "figures/model_comparison.png")
        viz.plot_forecast(test, test["sku"].iloc[0], out / "figures/forecast_example.png")
        viz.plot_inventory(policy, out / "figures/inventory_policy.png")
        viz.plot_transport(plan, out / "figures/transport_plan.png")
    if a.save_models:
        Path(a.models_dir).mkdir(exist_ok=True)
        joblib.dump(ens, Path(a.models_dir) / "ensemble.joblib")
        print("Saved models/ensemble.joblib")


if __name__ == "__main__":
    main()
