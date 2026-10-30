"""Figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_model_comparison(table, path):
    ax = table["WAPE_%"].plot.bar(color="#3776ab", figsize=(7, 4), rot=20)
    ax.set_ylabel("WAPE % (lower is better)"); ax.set_title("Demand forecast error by model")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_forecast(test_df, sku, path, pred_col="pred"):
    d = test_df[test_df["sku"] == sku]
    plt.figure(figsize=(9, 4))
    plt.plot(d["date"], d["demand"], label="Actual"); plt.plot(d["date"], d[pred_col], label="Forecast")
    plt.title(f"{sku}: actual vs forecast (hold-out)"); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_inventory(policy, path):
    ax = policy.set_index("sku")[["safety_stock", "reorder_point"]].plot.bar(figsize=(9, 4), rot=45)
    ax.set_title("Safety stock & reorder point by SKU"); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_transport(plan, path):
    fig, ax = plt.subplots(figsize=(6, 3.5)); im = ax.imshow(plan.values, cmap="Blues")
    ax.set_xticks(range(plan.shape[1]), plan.columns); ax.set_yticks(range(plan.shape[0]), plan.index)
    for i in range(plan.shape[0]):
        for j in range(plan.shape[1]):
            ax.text(j, i, int(plan.iloc[i, j]), ha="center", va="center")
    ax.set_title("Optimal shipment plan (units)"); fig.colorbar(im); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()
