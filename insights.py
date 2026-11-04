"""Rule-based, evidence-linked insights. Every sentence is built from computed values;
nothing is inferred beyond what the data show."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def _fmt(v, nd=0):
    return "n/a" if pd.isna(v) else f"{v:,.{nd}f}"


def build_insights(kpis: pd.DataFrame, anomalies: pd.DataFrame, checks: pd.DataFrame) -> List[dict]:
    """Return a list of dicts: {title, text, evidence(DataFrame or None)}."""
    out: List[dict] = []
    suff = kpis[kpis["data_class"] == "Sufficient"]
    if suff.empty:
        return [{"title": "No wells with sufficient history",
                 "text": "No well has enough producing days for reliable comparison.", "evidence": None}]

    # 1 - strongest decline (use recent vs initial rate change: robust to poor exp fits)
    d = suff.sort_values("recent_rate_change_pct").head(3)
    txt = "; ".join(f"{w}: {_fmt(r['recent_rate_change_pct'],0)} % "
                    f"({_fmt(r['first_rate'])} -> {_fmt(r['recent_rate'])} Sm3/d)" for w, r in d.iterrows())
    out.append({"title": "Strongest production decline",
                "text": "Comparing the average oil rate over the first 90 and last 90 producing days: " + txt + ".",
                "evidence": d[["first_rate", "recent_rate", "recent_rate_change_pct", "annual_decline_raw", "decline_r2"]]})

    # 2 - highest water cut
    wc = kpis.dropna(subset=["recent_water_cut"]).sort_values("recent_water_cut", ascending=False).head(3)
    txt = "; ".join(f"{w}: {r['recent_water_cut']:.0%} recent vs {r['cum_water_cut']:.0%} cumulative"
                    for w, r in wc.iterrows())
    out.append({"title": "Highest water cut",
                "text": "Recent (last 90 producing days) water cut: " + txt + ". "
                        "Water cut rising above the cumulative value signals growing water production.",
                "evidence": wc[["recent_water_cut", "cum_water_cut"]]})

    # 3 - relative stability: lowest rate coefficient of variation is not stored; use uptime & |change|
    stab = suff.assign(abs_change=suff["recent_rate_change_pct"].abs()).sort_values("abs_change").head(2)
    txt = "; ".join(f"{w}: rate change {_fmt(r['recent_rate_change_pct'],0)} %, uptime {r['uptime']:.0%}"
                    for w, r in stab.iterrows())
    out.append({"title": "Relatively stable production",
                "text": "Smallest change between initial and recent rate among wells with sufficient data: "
                        + txt + ". 'Stable' is relative - all wells in this dataset are in late-life decline.",
                "evidence": stab[["first_rate", "recent_rate", "recent_rate_change_pct", "uptime"]]})

    # 4 - availability
    low_up = suff.sort_values("recent_uptime_365d").head(2)
    out.append({"title": "Lowest availability",
                "text": "Uptime over the final 365 days before last production: " + "; ".join(
                    f"{w}: {r['recent_uptime_365d']:.0%}" for w, r in low_up.iterrows()) +
                        ". Low uptime reduces cumulative output independent of reservoir decline.",
                "evidence": low_up[["recent_uptime_365d", "uptime"]]})

    # 5 - candidates for investigation: combine checks and anomalies
    cand = []
    if not checks.empty:
        bad = checks[checks["status"].isin(["Warning", "Critical"])]
        cnt = bad.groupby("well").size()
        crit = bad[bad["status"] == "Critical"].groupby("well").size()
        for w in cnt.index:
            cand.append((w, int(cnt[w]), int(crit.get(w, 0))))
    if cand:
        cand.sort(key=lambda x: (-x[2], -x[1]))
        out.append({"title": "Wells to prioritise for engineering review",
                    "text": "Based on the currently configured (illustrative) thresholds: " + "; ".join(
                        f"{w} ({c} Critical, {n} flagged checks)" for w, n, c in cand[:4]) +
                            ". These are screening results, not confirmed problems.",
                    "evidence": None})
    if not anomalies.empty:
        hi = anomalies[anomalies["severity"] == "High"].groupby("well").size().sort_values(ascending=False)
        if not hi.empty:
            out.append({"title": "Anomaly concentration",
                        "text": "High-severity statistical anomaly events per well: " + "; ".join(
                            f"{w}: {n}" for w, n in hi.head(4).items()) +
                                ". Statistical anomalies need correlation with operational logs before any "
                                "engineering conclusion.", "evidence": None})

    # 6 - gas
    g = kpis["recent_gor"].dropna()
    if not g.empty:
        spread = g.max() / g.min() if g.min() > 0 else np.inf
        note = ("The spread across wells is narrow, so no well stands out with a gas-breakthrough "
                "signature in this data." if spread < 1.3 else
                "The spread across wells is wide; wells at the upper end deserve a closer GOR review.")
        out.append({"title": "Gas-oil ratio behaviour",
                    "text": f"Recent GOR ranges from {_fmt(g.min(),0)} to {_fmt(g.max(),0)} Sm3/Sm3 across wells "
                            f"(cumulative {_fmt(kpis['cum_gor'].min(),0)}-{_fmt(kpis['cum_gor'].max(),0)}). " + note,
                    "evidence": None})
    return out
