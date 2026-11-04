"""Short-term oil-rate forecasting with chronological validation.

Design (kept deliberately simple and leak-free)
------------------------------------------------
* Target: daily oil rate (Sm3/d) on *producing days* only (oil > 0 and on-stream hours > 0).
  Shut-in days are excluded: they are operational events, not decline behaviour.
* The model predicts the *relative* rate ``y_t / mean7_t`` (mean7 = mean of the previous 7
  producing days) and the oil rate is recovered as ``mean7_t * predicted_ratio``. This makes
  the target roughly stationary, so tree models are not asked to extrapolate absolute levels
  on a declining well (a known failure mode of Random Forest / Gradient Boosting).
* Features are scale-free ratios of the past only (lags / mean7, mean7 / mean30, mean14 / mean30,
  std7 / mean7). All rolling statistics are shifted by one step, so the value being predicted
  is never part of its own features.
* Split: first ``1 - test_frac`` of the series for training, last ``test_frac`` for testing.
  Scaling/fitting only ever sees training data.
* Baselines: persistence (last observed value) and 7-day trailing mean.
* Models: Linear Regression, Random Forest, Gradient Boosting.
* Two evaluation views:
    1. one-step-ahead on the test period (features come from real past values);
    2. recursive multi-step backtest (horizon H, several rolling origins in the test period),
       which is what a real forecast looks like, because predictions feed the next step.
* Forecast bands are empirical: per-horizon RMSE from the recursive backtest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MIN_SERIES_LEN = 365
LAGS = (1, 2, 3, 7)
FEATURES = ["r_lag1", "r_lag2", "r_lag3", "r_lag7", "r_m7_m30", "r_m14_m30", "r_std7"]
BASELINES = ("Baseline: persistence", "Baseline: 7-day mean")


def _model_factories() -> Dict[str, Callable[[], object]]:
    return {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=1),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42),
    }


def get_series(prod: pd.DataFrame, well: str) -> pd.DataFrame:
    """Producing-day oil series of a well (date, oil_sm3)."""
    g = prod[(prod["well"] == well) & prod["is_producing"] & (prod["oil_sm3"] > 0)]
    return g[["date", "oil_sm3"]].sort_values("date").reset_index(drop=True)


def feature_row(history: np.ndarray) -> np.ndarray:
    """Scale-free feature vector for the step following ``history`` (needs >= 30 values)."""
    h = history
    m7, m14, m30 = h[-7:].mean(), h[-14:].mean(), h[-30:].mean()
    m7 = max(m7, 1e-6)
    m30 = max(m30, 1e-6)
    return np.array([h[-1] / m7, h[-2] / m7, h[-3] / m7, h[-7] / m7,
                     m7 / m30, m14 / m30, h[-7:].std(ddof=0) / m7], dtype=float)


def build_supervised(y: np.ndarray) -> pd.DataFrame:
    """Vectorised, leak-free design matrix: row i describes y[i] using y[:i] only.

    Columns: pos, FEATURES..., mean7 (scale for back-transform), y (actual rate), ratio (target).
    """
    s = pd.Series(y)
    m7 = s.shift(1).rolling(7).mean()
    m14 = s.shift(1).rolling(14).mean()
    m30 = s.shift(1).rolling(30).mean()
    sd7 = s.shift(1).rolling(7).std(ddof=0)
    X = pd.DataFrame({
        "r_lag1": s.shift(1) / m7, "r_lag2": s.shift(2) / m7,
        "r_lag3": s.shift(3) / m7, "r_lag7": s.shift(7) / m7,
        "r_m7_m30": m7 / m30, "r_m14_m30": m14 / m30, "r_std7": sd7 / m7,
        "mean7": m7, "y": s,
    })
    X["ratio"] = X["y"] / X["mean7"]
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.dropna().reset_index().rename(columns={"index": "pos"})


def _metrics(y_true, y_pred) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else np.nan,
    }


def _recursive(model, history: np.ndarray, steps: int) -> np.ndarray:
    """Multi-step forecast: each prediction is appended to the history and reused."""
    hist = list(history)
    out = []
    for _ in range(steps):
        arr = np.array(hist)
        x = pd.DataFrame(feature_row(arr).reshape(1, -1), columns=FEATURES)
        ratio = float(np.clip(model.predict(x)[0], 0.0, 3.0))
        yhat = max(arr[-7:].mean() * ratio, 0.0)
        out.append(yhat)
        hist.append(yhat)
    return np.array(out)


@dataclass
class ForecastEvaluation:
    well: str
    status: str                       # "ok" or "insufficient"
    message: str = ""
    n_points: int = 0
    split_date: Optional[pd.Timestamp] = None
    metrics_onestep: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_recursive: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    horizon_rmse: Dict[str, np.ndarray] = field(default_factory=dict)
    best_model: Optional[str] = None
    reliability: str = "n/a"
    reliability_note: str = ""
    series: pd.DataFrame = field(default_factory=pd.DataFrame)


def evaluate_well(prod: pd.DataFrame, well: str, test_frac: float = 0.2, horizon: int = 30,
                  n_origins: int = 8) -> ForecastEvaluation:
    """Train/evaluate all models for one well. Never raises on thin data; reports status instead."""
    ser = get_series(prod, well)
    ev = ForecastEvaluation(well=well, status="ok", n_points=len(ser), series=ser)
    if len(ser) < MIN_SERIES_LEN:
        ev.status = "insufficient"
        ev.message = (f"Only {len(ser)} producing days available (minimum {MIN_SERIES_LEN}). "
                      "A forecast would be misleading, so none is generated.")
        return ev

    y = ser["oil_sm3"].to_numpy(float)
    sup = build_supervised(y)
    split_pos = int(len(y) * (1 - test_frac))
    train = sup[sup["pos"] < split_pos]
    test = sup[sup["pos"] >= split_pos]
    if len(train) < 100 or len(test) < horizon + 5:
        ev.status = "insufficient"
        ev.message = "Not enough data for a chronological train/test split with the chosen horizon."
        return ev
    ev.split_date = ser["date"].iloc[split_pos]

    # ---- baselines & one-step-ahead -------------------------------------------------
    preds: Dict[str, np.ndarray] = {
        BASELINES[0]: (test["r_lag1"] * test["mean7"]).to_numpy(),
        BASELINES[1]: test["mean7"].to_numpy(),
    }
    fitted = {}
    for name, fac in _model_factories().items():
        m = fac().fit(train[FEATURES], train["ratio"])
        fitted[name] = m
        preds[name] = np.maximum(np.clip(m.predict(test[FEATURES]), 0, 3) * test["mean7"], 0)
    rows = [{"Model": n, **_metrics(test["y"], p)} for n, p in preds.items()]
    ev.metrics_onestep = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    ev.test_predictions = pd.DataFrame({
        "date": ser["date"].iloc[test["pos"].to_numpy()].to_numpy(),
        "actual": test["y"].to_numpy(),
        **{n: p for n, p in preds.items()},
    })

    # ---- recursive multi-step backtest ---------------------------------------------
    origins = np.unique(np.linspace(split_pos, len(y) - horizon, n_origins).astype(int))
    errs: Dict[str, List[np.ndarray]] = {n: [] for n in list(fitted) + list(BASELINES)}
    pool_true, pool_pred = [], {n: [] for n in errs}
    for o in origins:
        hist, actual = y[:o], y[o:o + horizon]
        pool_true.append(actual)
        for n, m in fitted.items():
            p = _recursive(m, hist, horizon)
            errs[n].append(p - actual); pool_pred[n].append(p)
        flat = np.full(horizon, hist[-1])
        mean7 = np.full(horizon, hist[-7:].mean())
        for n, p in ((BASELINES[0], flat), (BASELINES[1], mean7)):
            errs[n].append(p - actual); pool_pred[n].append(p)
    yt = np.concatenate(pool_true)
    ev.metrics_recursive = pd.DataFrame([
        {"Model": n, **_metrics(yt, np.concatenate(pool_pred[n]))} for n in errs
    ]).sort_values("RMSE").reset_index(drop=True)
    ev.horizon_rmse = {n: np.sqrt(np.mean(np.vstack(e) ** 2, axis=0)) for n, e in errs.items()}

    # ---- model choice & reliability ---------------------------------------------------
    ml_rows = ev.metrics_recursive[~ev.metrics_recursive["Model"].isin(BASELINES)]
    ev.best_model = str(ml_rows.iloc[0]["Model"])
    best_rmse = float(ml_rows.iloc[0]["RMSE"])
    base_rmse = float(ev.metrics_recursive[ev.metrics_recursive["Model"].isin(BASELINES)]["RMSE"].min())
    best_r2 = float(ml_rows.iloc[0]["R2"])
    improve = 1 - best_rmse / base_rmse
    rel = f"{abs(improve):.0%} {'lower' if improve >= 0 else 'higher'}"
    if improve >= 0.05 and best_r2 > 0.3:
        ev.reliability = "Moderate"
        ev.reliability_note = (f"Best model ({ev.best_model}) has {rel} recursive RMSE than the best "
                               f"baseline, R2 = {best_r2:.2f} over a {horizon}-day backtest.")
    else:
        ev.reliability = "Low"
        ev.reliability_note = (
            f"Best ML model ({ev.best_model}) has {rel} recursive RMSE than the best baseline "
            f"(R2 = {best_r2:.2f}). It does not clearly beat a naive method, so treat the "
            "forecast as indicative only.")
    return ev


def forecast_future(prod: pd.DataFrame, well: str, model_name: str, horizon: int,
                    ev: Optional[ForecastEvaluation] = None) -> pd.DataFrame:
    """Refit the chosen model on the full history and forecast ``horizon`` producing days.

    Assumes uninterrupted production after the last observation (no shut-ins), with calendar
    dates continuing daily. Returns date, forecast, lower, upper (empirical ~80 % band).
    """
    ser = get_series(prod, well)
    if len(ser) < MIN_SERIES_LEN:
        return pd.DataFrame(columns=["date", "forecast", "lower", "upper"])
    y = ser["oil_sm3"].to_numpy(float)
    sup = build_supervised(y)
    model = _model_factories()[model_name]().fit(sup[FEATURES], sup["ratio"])
    pred = _recursive(model, y, horizon)
    dates = pd.date_range(ser["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    out = pd.DataFrame({"date": dates, "forecast": pred})
    if ev is not None and model_name in ev.horizon_rmse:
        hr = ev.horizon_rmse[model_name]
        r = np.interp(np.arange(horizon), np.arange(len(hr)), hr) if horizon > len(hr) else hr[:horizon]
        if horizon > len(hr):
            r = r * np.sqrt(np.arange(1, horizon + 1) / len(hr)).clip(min=1)
        out["lower"] = np.maximum(out["forecast"] - 1.28 * r, 0)
        out["upper"] = out["forecast"] + 1.28 * r
    else:
        out["lower"] = out["upper"] = np.nan
    return out
