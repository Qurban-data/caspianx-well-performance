"""Configurable engineering-limit monitoring.

IMPORTANT: the default thresholds below are *illustrative placeholders chosen for demonstration*.
They are NOT official reservoir, facility or regulatory limits, and nothing here is derived
from Volve design documents. Operators must set thresholds that match their own constraints.

Evaluation basis: recent window = the last ``RECENT_N`` producing days of each well
(relative to the well's own last production date, because Volve wells stopped in 2016).
Status logic: a parameter breaching its *warning* threshold is Warning; breaching its
*critical* threshold is Critical; otherwise Normal; missing input -> "No data".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class Thresholds:
    min_oil_warn: float = 200.0       # Sm3/d, recent average oil rate below this -> Warning
    min_oil_crit: float = 100.0       # Sm3/d -> Critical
    max_wc_warn: float = 0.60         # fraction
    max_wc_crit: float = 0.85
    max_gor_warn: float = 200.0       # Sm3/Sm3
    max_gor_crit: float = 300.0
    decline_warn: float = 0.30        # effective annual decline (fraction)
    decline_crit: float = 0.60
    min_uptime_warn: float = 0.85     # fraction of last 365 d on stream
    min_uptime_crit: float = 0.60

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _status_high_bad(v, warn, crit):
    if pd.isna(v):
        return "No data"
    return "Critical" if v >= crit else ("Warning" if v >= warn else "Normal")


def _status_low_bad(v, warn, crit):
    if pd.isna(v):
        return "No data"
    return "Critical" if v <= crit else ("Warning" if v <= warn else "Normal")


def evaluate_limits(kpis: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Long-format table: one row per well and parameter."""
    rows = []
    for well, r in kpis.iterrows():
        if r.get("data_class") == "Insufficient":
            continue
        dec_val = r.get("annual_decline_raw", np.nan)
        checks = [
            ("Recent oil rate (last 90 producing days)", "Sm3/d", r.get("recent_rate"),
             f"< {th.min_oil_warn:g} warn / < {th.min_oil_crit:g} crit",
             _status_low_bad(r.get("recent_rate"), th.min_oil_warn, th.min_oil_crit),
             "Average oil rate over the last 90 producing days compared with the configured minimum."),
            ("Recent water cut", "fraction", r.get("recent_water_cut"),
             f"> {th.max_wc_warn:g} warn / > {th.max_wc_crit:g} crit",
             _status_high_bad(r.get("recent_water_cut"), th.max_wc_warn, th.max_wc_crit),
             "Water / (oil + water), last 90 producing days."),
            ("Recent GOR", "Sm3/Sm3", r.get("recent_gor"),
             f"> {th.max_gor_warn:g} warn / > {th.max_gor_crit:g} crit",
             _status_high_bad(r.get("recent_gor"), th.max_gor_warn, th.max_gor_crit),
             "Total gas / total oil over the last 90 producing days."),
            ("Annual decline (exponential fit)", "fraction/yr", dec_val,
             f"> {th.decline_warn:g} warn / > {th.decline_crit:g} crit",
             _status_high_bad(dec_val, th.decline_warn, th.decline_crit),
             "Effective annual decline from an exponential fit on the last 365 days"
             + ("" if r.get("decline_reliable", False) else " (low fit quality - interpret with care)") + "."),
            ("Uptime (last 365 days)", "fraction", r.get("recent_uptime_365d"),
             f"< {th.min_uptime_warn:g} warn / < {th.min_uptime_crit:g} crit",
             _status_low_bad(r.get("recent_uptime_365d"), th.min_uptime_warn, th.min_uptime_crit),
             "On-stream hours / (24 x days) in the last 365 days before the last production date."),
        ]
        for name, unit, val, thr, status, expl in checks:
            rows.append({"well": well, "parameter": name, "unit": unit,
                         "actual": None if pd.isna(val) else float(val),
                         "threshold": thr, "status": status, "explanation": expl})
    return pd.DataFrame(rows, columns=["well", "parameter", "unit", "actual", "threshold",
                                        "status", "explanation"])


def well_status(checks: pd.DataFrame) -> pd.DataFrame:
    """Worst status per well plus counts."""
    if checks.empty:
        return pd.DataFrame(columns=["well", "overall", "n_warning", "n_critical"])
    rank = {"Normal": 0, "No data": 0, "Warning": 1, "Critical": 2}
    inv = {0: "Normal", 1: "Warning", 2: "Critical"}
    out = []
    for w, g in checks.groupby("well"):
        worst = max(rank[s] for s in g["status"])
        out.append({"well": w, "overall": inv[worst],
                    "n_warning": int((g["status"] == "Warning").sum()),
                    "n_critical": int((g["status"] == "Critical").sum())})
    return pd.DataFrame(out)
