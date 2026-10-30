"""Transportation optimisation (warehouse -> region) via linear programming."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def build_network(regional_demand: pd.Series, seed: int = 7):
    rng = np.random.default_rng(seed)
    wh = ["WH-A", "WH-B", "WH-C"]
    supply = regional_demand.sum() * np.array([0.45, 0.40, 0.35])     # 20% spare capacity
    cost = pd.DataFrame(np.round(rng.uniform(1, 6, (len(wh), len(regional_demand))), 2),
                        index=wh, columns=regional_demand.index)         # $/unit
    return pd.Series(supply, index=wh), cost


def optimize_transport(supply: pd.Series, demand: pd.Series, cost: pd.DataFrame):
    m, n = cost.shape
    A_ub, A_eq = np.zeros((m, m * n)), np.zeros((n, m * n))
    for i in range(m):
        A_ub[i, i * n:(i + 1) * n] = 1                 # sum_j x_ij <= supply_i
    for j in range(n):
        A_eq[j, j::n] = 1                              # sum_i x_ij == demand_j
    res = linprog(cost.values.flatten(), A_ub=A_ub, b_ub=supply.values,
                  A_eq=A_eq, b_eq=demand.values, bounds=(0, None), method="highs")
    if not res.success:
        raise ValueError(f"Infeasible network: {res.message}")
    plan = pd.DataFrame(res.x.reshape(m, n), index=cost.index, columns=cost.columns).round(0)
    naive = np.outer(supply / supply.sum(), demand.values)           # proportional split baseline
    base_cost = float((naive * cost.values).sum())
    return plan, float(res.fun), base_cost
