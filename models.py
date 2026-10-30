"""Ridge / Lasso / ElasticNet demand models, time-series tuning, stacked ensemble."""
import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .preprocessing import FEATURE_COLS, build_preprocessor


def evaluate_demand(y_true, y_pred) -> dict:
    """RMSLE-style error on log scale + business metrics in units."""
    y_true, y_pred = np.asarray(y_true, float), np.clip(np.asarray(y_pred, float), 0, None)
    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    return {"RMSLE": float(np.sqrt(np.mean((lt - lp) ** 2))),
            "MAE_units": float(mean_absolute_error(y_true, y_pred)),
            "WAPE_%": float(100 * np.abs(y_true - y_pred).sum() / y_true.sum()),
            "R2_log": float(r2_score(lt, lp))}


class DemandModel:
    def __init__(self, name, estimator):
        self.name = name
        self.pipeline = Pipeline([("prep", build_preprocessor()), ("est", estimator)])

    @classmethod
    def ridge(cls, alpha=10.0):
        return cls("Ridge", Ridge(alpha=alpha))

    @classmethod
    def lasso(cls, alpha=0.001):
        return cls("Lasso", Lasso(alpha=alpha, max_iter=20000))

    @classmethod
    def elasticnet(cls, alpha=0.001, l1_ratio=0.5):
        return cls("ElasticNet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000))

    def fit(self, df):
        self.pipeline.fit(df[FEATURE_COLS], df["log_demand"])
        return self

    def predict_log(self, df):
        return self.pipeline.predict(df[FEATURE_COLS])

    def predict_demand(self, df):
        return np.clip(np.expm1(self.predict_log(df)), 0, None)

    def tune(self, df, n_splits=4):
        """Grid search with TimeSeriesSplit (df must be sorted by date)."""
        grids = {"Ridge": {"est__alpha": [0.1, 1, 10, 50, 100]},
                 "Lasso": {"est__alpha": [1e-4, 5e-4, 1e-3, 5e-3]},
                 "ElasticNet": {"est__alpha": [1e-4, 5e-4, 1e-3, 5e-3],
                                "est__l1_ratio": [0.2, 0.5, 0.8]}}
        gs = GridSearchCV(self.pipeline, grids[self.name], cv=TimeSeriesSplit(n_splits),
                          scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(df[FEATURE_COLS], df["log_demand"])
        self.pipeline = gs.best_estimator_
        return gs.best_params_


class StackedEnsemble:
    """Non-negative least-squares blend of base models, weights fit on validation."""
    def __init__(self, models):
        self.models, self.weights = models, np.ones(len(models)) / len(models)

    def fit_weights(self, val_df):
        P = np.column_stack([m.predict_log(val_df) for m in self.models])
        w, _ = nnls(P, val_df["log_demand"].values)
        self.weights = w / w.sum() if w.sum() > 0 else self.weights
        return self

    def predict_demand(self, df):
        P = np.column_stack([m.predict_log(df) for m in self.models])
        return np.clip(np.expm1(P @ self.weights), 0, None)
