"""Cleaning and feature derivation for the Volve daily production data.

Every correction is recorded in a ``flag_*`` column and summarised by
:func:`quality_report`, so nothing is silently altered.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

RENAME = {
    "DATEPRD": "date",
    "NPD_WELL_BORE_NAME": "well",
    "ON_STREAM_HRS": "on_stream_hrs",
    "AVG_DOWNHOLE_PRESSURE": "bhp_bar",
    "AVG_DOWNHOLE_TEMPERATURE": "bht_c",
    "AVG_DP_TUBING": "dp_tubing_bar",
    "AVG_ANNULUS_PRESS": "annulus_bar",
    "AVG_CHOKE_SIZE_P": "choke_pct",
    "AVG_WHP_P": "whp_bar",
    "AVG_WHT_P": "wht_c",
    "BORE_OIL_VOL": "oil_sm3",
    "BORE_GAS_VOL": "gas_sm3",
    "BORE_WAT_VOL": "water_sm3",
    "BORE_WI_VOL": "water_inj_sm3",
    "FLOW_KIND": "flow_kind",
    "WELL_TYPE": "well_type",
}

KEEP = list(RENAME.values())


def clean_daily(daily_raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, sort, and apply documented, flagged corrections.

    Rules
    -----
    * Negative water volumes are physically impossible -> set to NaN (flag_neg_water).
    * On-stream hours > 24 are clipped to 24 (flag_hrs_clipped).
    * Downhole pressure/temperature equal to exactly 0 while the well is flowing
      indicate gauge outage, not a real reading -> set to NaN (flag_gauge_zero).
    """
    df = daily_raw.rename(columns=RENAME)[KEEP].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["well", "date"]).reset_index(drop=True)

    df["flag_neg_water"] = df["water_sm3"] < 0
    df.loc[df["flag_neg_water"], "water_sm3"] = np.nan

    df["flag_hrs_clipped"] = df["on_stream_hrs"] > 24
    df["on_stream_hrs"] = df["on_stream_hrs"].clip(upper=24)

    flowing = df["oil_sm3"] > 0
    zero_gauge = flowing & (df["bhp_bar"] == 0)
    df["flag_gauge_zero"] = zero_gauge
    df.loc[zero_gauge, ["bhp_bar", "bht_c", "dp_tubing_bar"]] = np.nan
    return df


def producing_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Daily rows of production-mode records with derived water cut and GOR.

    * ``water_cut`` = water / (oil + water)  (fraction, volumes at standard conditions)
    * ``gor``       = gas / oil              (Sm3/Sm3)
    * ``is_producing`` = oil > 0 and on-stream hours > 0
    Water cut and GOR are only computed on producing days; otherwise NaN.
    """
    p = clean[(clean["well_type"] == "OP") & clean["oil_sm3"].notna()].copy()
    p["is_producing"] = (p["oil_sm3"] > 0) & (p["on_stream_hrs"].fillna(0) > 0)
    liquid = p["oil_sm3"] + p["water_sm3"]
    p["water_cut"] = np.where(p["is_producing"] & (liquid > 0), p["water_sm3"] / liquid, np.nan)
    p["gor"] = np.where(p["is_producing"], p["gas_sm3"] / p["oil_sm3"].replace(0, np.nan), np.nan)
    return p.reset_index(drop=True)


def injection_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Daily rows recorded as injection (water injection wells)."""
    return clean[clean["flow_kind"] == "injection"].copy().reset_index(drop=True)


def quality_report(daily_raw: pd.DataFrame, clean: pd.DataFrame) -> Dict[str, object]:
    """Summarise data-quality findings computed directly from the data."""
    key_dups = int(daily_raw.duplicated(["NPD_WELL_BORE_CODE", "DATEPRD"]).sum())
    prod = clean[(clean["well_type"] == "OP") & clean["oil_sm3"].notna()]
    gaps = {}
    for w, g in clean.groupby("well"):
        span = (g["date"].max() - g["date"].min()).days + 1
        gaps[w] = int(span - g["date"].nunique())
    return {
        "rows": int(len(daily_raw)),
        "duplicate_well_date_keys": key_dups,
        "negative_water_rows": int(clean["flag_neg_water"].sum()),
        "on_stream_hours_over_24_rows": int(clean["flag_hrs_clipped"].sum()),
        "flowing_rows_with_zero_gauge": int(clean["flag_gauge_zero"].sum()),
        "missing_oil_rows_all": int(clean["oil_sm3"].isna().sum()),
        "missing_oil_rows_producers": int(prod["oil_sm3"].isna().sum()),
        "date_gaps_days_by_well": gaps,
        "date_min": clean["date"].min(),
        "date_max": clean["date"].max(),
    }
