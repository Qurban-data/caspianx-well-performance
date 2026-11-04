"""Statistical anomaly detection on producing-day data.

Two complementary detectors, both flagged as *statistical* anomalies (not confirmed
engineering problems):

1. Trailing robust z-score (median / MAD over the previous ``window`` producing days,
   current day excluded) on oil rate, water cut and GOR (the gas indicator, because raw gas
   volume mostly follows oil rate). A minimum relative change is also required so that tiny
   absolute wiggles on near-zero baselines are not reported.
2. Isolation Forest on [oil rate, water cut, GOR, 1-day oil change, 1-day water-cut change],
   fitted per well and standardised, to catch unusual *combinations*.

Consecutive flagged days (gap <= ``merge_gap`` days) for the same well/variable/direction are
merged into one event; the day with the largest deviation is reported.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

COLUMNS = ["well", "date", "variable", "direction", "observed", "expected_low", "expected_high",
           "deviation_score", "severity", "method", "days_in_event", "interpretation"]

_VARS = {
    "oil_sm3": ("Oil rate", "Sm3/d", 0.25),
    "water_cut": ("Water cut", "fraction", 0.15),
    "gor": ("GOR", "Sm3/Sm3", 0.20),
}

_INTERP = {
    ("Oil rate", "drop"): "Sudden fall versus recent behaviour. Possible causes include choke/rate "
                          "change, partial shut-in or restart transient, lift/flow-assurance issue, "
                          "or a metering/allocation artefact. Needs engineering review.",
    ("Oil rate", "spike"): "Rate above recent behaviour. Possible choke opening, post-shut-in "
                           "clean-up flush, or allocation effect.",
    ("Water cut", "spike"): "Water cut jumped above recent level. Possible water breakthrough, "
                            "change in well allocation, or a rate/choke change.",
    ("Water cut", "drop"): "Water cut fell versus recent level. Possible rate change, restart "
                           "effect or metering artefact.",
    ("GOR", "spike"): "Gas-oil ratio rose above recent level. Possible gas coning/breakthrough "
                      "or a low-oil-rate effect on the ratio.",
    ("GOR", "drop"): "Gas-oil ratio fell below recent level. Possible rate change or allocation effect.",
    ("Multivariate", "mixed"): "Unusual combination of rate, water cut and GOR compared with this "
                               "well's own history. Review alongside operational logs.",
}


def _severity(score: float, thr: float) -> str:
    a = abs(score)
    if a >= 2.0 * thr:
        return "High"
    if a >= 1.4 * thr:
        return "Medium"
    return "Low"


def _merge_events(df: pd.DataFrame, merge_gap: int) -> pd.DataFrame:
    if df.empty:
        return df
    out: List[pd.Series] = []
    for _, g in df.sort_values("date").groupby(["well", "variable", "direction"]):
        g = g.reset_index(drop=True)
        grp = (g["date"].diff().dt.days.fillna(0) > merge_gap).cumsum()
        for _, ev in g.groupby(grp):
            peak = ev.loc[ev["deviation_score"].abs().idxmax()].copy()
            peak["days_in_event"] = len(ev)
            out.append(peak)
    return pd.DataFrame(out)


def detect_zscore(prod: pd.DataFrame, well: str, z_threshold: float = 4.0, window: int = 60,
                  merge_gap: int = 3) -> pd.DataFrame:
    g = prod[(prod["well"] == well) & prod["is_producing"]].sort_values("date").reset_index(drop=True)
    rows = []
    if len(g) < window // 2 + 10:
        return pd.DataFrame(columns=COLUMNS)
    for col, (label, _unit, min_rel) in _VARS.items():
        s = g[col].astype(float)
        past = s.shift(1)
        med = past.rolling(window, min_periods=20).median()
        mad = past.rolling(window, min_periods=20).apply(
            lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True)
        floor = np.maximum(0.05 * med.abs(), 1e-3 if col == "water_cut" else 1e-2)
        sigma = np.maximum(1.4826 * mad, floor)
        z = (s - med) / sigma
        rel = (s - med).abs() / med.abs().replace(0, np.nan)
        if col == "water_cut":                      # absolute change matters near zero
            ok = (s - med).abs() >= min_rel
        else:
            ok = rel >= min_rel
        flag = (z.abs() >= z_threshold) & ok & z.notna()
        for i in np.flatnonzero(flag.to_numpy()):
            direction = "drop" if z.iloc[i] < 0 else "spike"
            rows.append({
                "well": well, "date": g["date"].iloc[i], "variable": label, "direction": direction,
                "observed": float(s.iloc[i]),
                "expected_low": float(med.iloc[i] - z_threshold * sigma.iloc[i]),
                "expected_high": float(med.iloc[i] + z_threshold * sigma.iloc[i]),
                "deviation_score": float(z.iloc[i]),
                "severity": _severity(z.iloc[i], z_threshold),
                "method": "Robust z-score", "days_in_event": 1,
                "interpretation": _INTERP.get((label, direction), ""),
            })
    ev = pd.DataFrame(rows, columns=COLUMNS)
    return _merge_events(ev, merge_gap)


def detect_isolation_forest(prod: pd.DataFrame, well: str, contamination: float = 0.02,
                            merge_gap: int = 3) -> pd.DataFrame:
    g = prod[(prod["well"] == well) & prod["is_producing"]].sort_values("date").reset_index(drop=True)
    feat = pd.DataFrame({
        "oil": g["oil_sm3"], "wc": g["water_cut"], "gor": g["gor"],
        "d_oil": g["oil_sm3"].pct_change().clip(-1, 3), "d_wc": g["water_cut"].diff(),
    })
    valid = feat.dropna()
    if len(valid) < 100:
        return pd.DataFrame(columns=COLUMNS)
    X = StandardScaler().fit_transform(valid)
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42).fit(X)
    pred = iso.predict(X)
    score = -iso.score_samples(X)              # higher = more abnormal
    thr = -iso.offset_
    rows = []
    for k in np.flatnonzero(pred == -1):
        idx = valid.index[k]
        rows.append({
            "well": well, "date": g["date"].iloc[idx], "variable": "Multivariate",
            "direction": "mixed", "observed": float(g["oil_sm3"].iloc[idx]),
            "expected_low": float(valid["oil"].quantile(0.05)),
            "expected_high": float(valid["oil"].quantile(0.95)),
            "deviation_score": float(score[k]),
            "severity": "High" if score[k] >= thr + 0.10 else ("Medium" if score[k] >= thr + 0.05 else "Low"),
            "method": "Isolation Forest", "days_in_event": 1,
            "interpretation": _INTERP[("Multivariate", "mixed")],
        })
    ev = pd.DataFrame(rows, columns=COLUMNS)
    return _merge_events(ev, merge_gap)


def detect_all(prod: pd.DataFrame, wells: List[str], z_threshold: float = 4.0,
               contamination: float = 0.02, window: int = 60) -> pd.DataFrame:
    """Run both detectors on all wells; return one tidy event table (empty frame if none)."""
    frames = []
    for w in wells:
        frames.append(detect_zscore(prod, w, z_threshold, window))
        frames.append(detect_isolation_forest(prod, w, contamination))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    res = pd.concat(frames, ignore_index=True)
    order = {"High": 0, "Medium": 1, "Low": 2}
    res["_o"] = res["severity"].map(order)
    return res.sort_values(["_o", "date"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
