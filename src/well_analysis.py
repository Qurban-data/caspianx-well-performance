"""Per-well KPIs, decline estimation, data-sufficiency classes and transparent ranking."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

MIN_DAYS_RELIABLE = 365     # producing days required for "Sufficient"
MIN_DAYS_LIMITED = 90       # below this -> "Insufficient"
RECENT_N = 90               # number of last producing days used as "recent" window
DECLINE_WINDOW_DAYS = 365   # calendar window (before last producing day) for decline fit
DECLINE_MIN_POINTS = 60
DECLINE_MIN_R2 = 0.20       # below this the exponential fit is considered unreliable
DECLINE_FLAG = 0.30         # illustrative cut-off for the "Declining" label

DEFAULT_WEIGHTS: Dict[str, float] = {
    "recent_rate": 0.35,   # recent average oil rate (Sm3/d), higher is better
    "uptime": 0.20,        # share of calendar days on stream, higher is better
    "decline": 0.25,       # lower annual decline is better
    "water_cut": 0.20,     # lower recent water cut is better
}


def data_sufficiency(producing_days: int) -> str:
    if producing_days >= MIN_DAYS_RELIABLE:
        return "Sufficient"
    if producing_days >= MIN_DAYS_LIMITED:
        return "Limited"
    return "Insufficient"


def estimate_decline(df: pd.DataFrame) -> Dict[str, float]:
    """Exponential (Arps b=0) decline fitted to the last ``DECLINE_WINDOW_DAYS`` of producing days.

    ln(q) = a - D*t  with t in years. Returns nominal D (1/yr) and effective annual decline
    (1 - exp(-D)); positive values mean production is falling, negative mean it is rising.
    Returns NaN when fewer than ``DECLINE_MIN_POINTS`` points exist.
    """
    d = df[df["is_producing"] & (df["oil_sm3"] > 0)]
    if d.empty:
        return {"nominal_decline": np.nan, "effective_decline": np.nan, "n_points": 0, "r2": np.nan}
    end = d["date"].max()
    w = d[d["date"] > end - pd.Timedelta(days=DECLINE_WINDOW_DAYS)]
    if len(w) < DECLINE_MIN_POINTS:
        return {"nominal_decline": np.nan, "effective_decline": np.nan, "n_points": len(w), "r2": np.nan}
    t = (w["date"] - w["date"].min()).dt.days.to_numpy() / 365.25
    y = np.log(w["oil_sm3"].to_numpy())
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    D = -slope
    return {"nominal_decline": D, "effective_decline": 1 - np.exp(-D), "n_points": len(w), "r2": r2}


def well_kpis(prod: pd.DataFrame, well: str) -> Dict[str, object]:
    """Compute KPIs for one well from the producing-frame. Missing inputs yield NaN, never guesses."""
    g = prod[prod["well"] == well].sort_values("date")
    pdays = g[g["is_producing"]]
    out: Dict[str, object] = {"well": well}
    if pdays.empty:
        out.update({"data_class": "Insufficient", "producing_days": 0})
        return out
    first, last = pdays["date"].min(), pdays["date"].max()
    cal_days = (last - first).days + 1
    oil, gas, wat = g["oil_sm3"].sum(), g["gas_sm3"].sum(), g["water_sm3"].sum(skipna=True)
    recent = pdays.tail(RECENT_N)
    last_year = g[g["date"] > last - pd.Timedelta(days=365)]
    first_n = pdays.head(RECENT_N)
    hrs = g["on_stream_hrs"].fillna(0)
    out.update({
        "first_production": first,
        "last_production": last,
        "calendar_days": cal_days,
        "producing_days": int(len(pdays)),
        "total_oil_sm3": float(oil),
        "total_gas_sm3": float(gas),
        "total_water_sm3": float(wat),
        "avg_oil_rate_producing": float(pdays["oil_sm3"].mean()),
        "avg_oil_rate_calendar": float(oil / cal_days),
        "peak_oil_rate": float(pdays["oil_sm3"].max()),
        "first_rate": float(first_n["oil_sm3"].mean()),
        "recent_rate": float(recent["oil_sm3"].mean()),
        "recent_rate_change_pct": float(recent["oil_sm3"].mean() / first_n["oil_sm3"].mean() - 1) * 100,
        "cum_water_cut": float(wat / (oil + wat)) if (oil + wat) > 0 else np.nan,
        "recent_water_cut": float(recent["water_cut"].mean()),
        "cum_gor": float(gas / oil) if oil > 0 else np.nan,
        "recent_gor": float(recent["gas_sm3"].sum() / recent["oil_sm3"].sum()),
        "uptime": float(min(hrs[g["date"].between(first, last)].sum() / (24.0 * cal_days), 1.0)),
        "recent_uptime_365d": float(min(
            last_year["on_stream_hrs"].fillna(0).sum() / (24.0 * min(365, cal_days)), 1.0)),
        "data_class": data_sufficiency(len(pdays)),
    })
    dec = estimate_decline(g)
    reliable = pd.notna(dec["r2"]) and dec["r2"] >= DECLINE_MIN_R2
    out.update({
        "annual_decline_raw": dec["effective_decline"],
        "annual_decline": dec["effective_decline"] if reliable else np.nan,
        "decline_r2": dec["r2"],
        "decline_points": dec["n_points"],
        "decline_reliable": bool(reliable),
    })
    return out


def all_kpis(prod: pd.DataFrame) -> pd.DataFrame:
    wells = sorted(prod["well"].unique())
    return pd.DataFrame([well_kpis(prod, w) for w in wells]).set_index("well")


def _minmax(s: pd.Series, higher_is_better: bool) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or hi == lo:
        return pd.Series(0.5, index=s.index).where(s.notna())
    n = (s - lo) / (hi - lo)
    return n if higher_is_better else 1 - n


def rank_wells(kpis: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Transparent weighted score (0-100) among wells with 'Sufficient' data only.

    Each criterion is min-max normalised across eligible wells. If a criterion is NaN
    for a well (e.g. decline not estimable), its weight is redistributed over the
    remaining criteria for that well. Wells with 'Limited' or 'Insufficient' history are not
    ranked (their KPIs are shown, but a ranking would be misleading). The exponential decline
    is only used when its fit R2 >= DECLINE_MIN_R2; otherwise it is treated as missing.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    elig = kpis[kpis["data_class"] == "Sufficient"].copy()
    if elig.empty:
        return kpis.assign(score=np.nan, rank=np.nan)
    comp = pd.DataFrame({
        "recent_rate": _minmax(elig["recent_rate"], True),
        "uptime": _minmax(elig["uptime"], True),
        "decline": _minmax(elig["annual_decline"], False),
        "water_cut": _minmax(elig["recent_water_cut"], False),
    })
    wt = pd.Series(w)[comp.columns]
    mask = comp.notna()
    num = (comp.fillna(0) * wt).sum(axis=1)
    den = (mask * wt).sum(axis=1)
    elig["score"] = (num / den * 100).where(den > 0)
    elig["rank"] = elig["score"].rank(ascending=False, method="min")
    res = kpis.join(elig[["score", "rank"]])
    return res


def classify_wells(kpis: pd.DataFrame) -> pd.Series:
    """Rule-based labels using only computed KPIs (no hidden assumptions).

    * Insufficient data : fewer than MIN_DAYS_LIMITED producing days
    * Limited data      : MIN_DAYS_LIMITED <= producing days < MIN_DAYS_RELIABLE (indicative only)
    * Declining         : reliable effective annual decline >= DECLINE_FLAG, or recent-vs-initial
                          rate change <= -30 %
    * High performer    : recent rate >= median of 'Sufficient' wells (and not declining)
    * Low performer     : recent rate below that median
    """
    labels = {}
    suff = kpis[kpis["data_class"] == "Sufficient"]
    med = suff["recent_rate"].median() if not suff.empty else np.nan
    for w, r in kpis.iterrows():
        if r["data_class"] == "Insufficient":
            labels[w] = "Insufficient data"
        elif r["data_class"] == "Limited":
            labels[w] = "Limited data"
        else:
            dec = r.get("annual_decline", np.nan)
            falling = (pd.notna(dec) and dec >= DECLINE_FLAG) or r["recent_rate_change_pct"] <= -30
            if falling:
                labels[w] = "Declining"
            elif r["recent_rate"] >= med:
                labels[w] = "High performer"
            else:
                labels[w] = "Low performer"
    return pd.Series(labels, name="classification")
