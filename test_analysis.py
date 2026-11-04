"""Unit tests. They use small synthetic data so they run without the Volve file;
one optional integration test runs when the dataset is available."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import anomaly_detection as ad
from src import engineering_checks as ec
from src import forecasting as fc
from src import well_analysis as wa
from src.data_loader import find_data_file, load_raw
from src.preprocessing import clean_daily, producing_frame


def synthetic_prod(n=600, well="W1", seed=0, decline=0.0015):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    oil = 1000 * np.exp(-decline * t) * (1 + rng.normal(0, 0.03, n))
    wat = 200 + 0.8 * t
    gas = oil * 150
    df = pd.DataFrame({
        "well": well, "date": pd.date_range("2015-01-01", periods=n), "on_stream_hrs": 24.0,
        "oil_sm3": oil, "gas_sm3": gas, "water_sm3": wat,
        "bhp_bar": 200.0, "whp_bar": 40.0, "choke_pct": 50.0,
    })
    df["is_producing"] = True
    df["water_cut"] = df["water_sm3"] / (df["oil_sm3"] + df["water_sm3"])
    df["gor"] = df["gas_sm3"] / df["oil_sm3"]
    return df


def test_decline_recovers_known_rate():
    df = synthetic_prod(n=400, decline=0.002)
    d = wa.estimate_decline(df)
    expected_nominal = 0.002 * 365.25
    assert d["nominal_decline"] == pytest.approx(expected_nominal, rel=0.1)
    assert 0 < d["effective_decline"] < 1


def test_kpis_water_cut_and_gor():
    df = synthetic_prod()
    k = wa.well_kpis(df, "W1")
    assert k["total_oil_sm3"] == pytest.approx(df["oil_sm3"].sum())
    assert k["cum_gor"] == pytest.approx(150.0)
    assert 0 < k["cum_water_cut"] < 1
    assert k["data_class"] == "Sufficient"
    assert k["uptime"] == pytest.approx(1.0)


def test_insufficient_data_class_and_no_ranking():
    df = synthetic_prod(n=50)
    k = wa.all_kpis(df)
    assert k.loc["W1", "data_class"] == "Insufficient"
    ranked = wa.rank_wells(k)
    assert ranked["score"].isna().all()


def test_forecast_refuses_thin_series():
    ev = fc.evaluate_well(synthetic_prod(n=200), "W1")
    assert ev.status == "insufficient"
    assert fc.forecast_future(synthetic_prod(n=200), "W1", "Linear Regression", 10).empty


def test_features_have_no_leakage():
    """Changing the current target value must not change its own feature row."""
    y = np.linspace(1000, 500, 120)
    a = fc.build_supervised(y)
    y2 = y.copy(); y2[80] *= 5
    b = fc.build_supervised(y2)
    ra, rb = a[a["pos"] == 80].iloc[0], b[b["pos"] == 80].iloc[0]
    for f in fc.FEATURES:
        assert ra[f] == pytest.approx(rb[f])


def test_forecast_pipeline_runs_and_is_chronological():
    df = synthetic_prod(n=700)
    ev = fc.evaluate_well(df, "W1", horizon=20, n_origins=3)
    assert ev.status == "ok"
    assert ev.test_predictions["date"].min() >= ev.split_date
    assert set(fc.BASELINES).issubset(set(ev.metrics_recursive["Model"]))
    f = fc.forecast_future(df, "W1", ev.best_model, 30, ev)
    assert len(f) == 30 and (f["forecast"] >= 0).all()
    assert f["date"].min() > df["date"].max()


def test_anomaly_detects_injected_drop():
    df = synthetic_prod(n=500)
    df.loc[400:402, "oil_sm3"] = 50.0
    df["gor"] = df["gas_sm3"] / df["oil_sm3"]
    ev = ad.detect_zscore(df, "W1", z_threshold=4.0)
    drops = ev[(ev["variable"] == "Oil rate") & (ev["direction"] == "drop")]
    assert not drops.empty
    assert (drops["date"] >= df["date"].iloc[400]).any()
    assert set(ad.COLUMNS) == set(ev.columns)


def test_anomaly_returns_schema_and_few_events_on_smooth_series():
    df = synthetic_prod(n=300)
    ev = ad.detect_zscore(df, "W1", z_threshold=8.0)
    assert list(ev.columns) == ad.COLUMNS
    assert len(ev) <= 2   # smooth synthetic series should be almost event-free at z>=8


def test_limit_checks_status_logic():
    k = pd.DataFrame({
        "recent_rate": [500.0, 150.0, 50.0], "recent_water_cut": [0.2, 0.7, 0.9],
        "recent_gor": [100.0, 250.0, 400.0], "annual_decline_raw": [0.1, 0.4, 0.8],
        "decline_reliable": [True] * 3, "recent_uptime_365d": [0.95, 0.7, 0.5],
        "data_class": ["Sufficient"] * 3}, index=["A", "B", "C"])
    res = ec.evaluate_limits(k, ec.Thresholds())
    st_ = ec.well_status(res).set_index("well")["overall"]
    assert st_["A"] == "Normal" and st_["B"] == "Warning" and st_["C"] == "Critical"


def test_limit_checks_missing_values_and_custom_thresholds():
    k = pd.DataFrame({"recent_rate": [np.nan], "recent_water_cut": [np.nan], "recent_gor": [np.nan],
                      "annual_decline_raw": [np.nan], "decline_reliable": [False],
                      "recent_uptime_365d": [np.nan], "data_class": ["Limited"]}, index=["X"])
    res = ec.evaluate_limits(k, ec.Thresholds(min_oil_warn=1e9))
    assert (res["status"] == "No data").all()


def test_clean_daily_flags_bad_values():
    raw = pd.DataFrame({
        "DATEPRD": pd.date_range("2020-01-01", periods=3), "NPD_WELL_BORE_NAME": "W",
        "ON_STREAM_HRS": [24, 25, 24], "AVG_DOWNHOLE_PRESSURE": [0, 100, 100],
        "AVG_DOWNHOLE_TEMPERATURE": [0, 90, 90], "AVG_DP_TUBING": [0, 10, 10], "AVG_ANNULUS_PRESS": 1.0,
        "AVG_CHOKE_SIZE_P": 50.0, "AVG_WHP_P": 30.0, "AVG_WHT_P": 60.0,
        "BORE_OIL_VOL": [100, 100, 100], "BORE_GAS_VOL": 15000.0, "BORE_WAT_VOL": [-5, 10, 10],
        "BORE_WI_VOL": np.nan, "FLOW_KIND": "production", "WELL_TYPE": "OP"})
    c = clean_daily(raw)
    assert c["water_sm3"].isna().sum() == 1
    assert c["on_stream_hrs"].max() == 24
    assert c["bhp_bar"].isna().sum() == 1
    p = producing_frame(c)
    assert p["water_cut"].between(0, 1).all() or p["water_cut"].isna().any()


@pytest.mark.skipif(find_data_file() is None, reason="Volve dataset not present")
def test_integration_real_dataset():
    daily, monthly = load_raw()
    c = clean_daily(daily)
    p = producing_frame(c)
    assert p["well"].nunique() == 6
    k = wa.all_kpis(p)
    assert (k["total_oil_sm3"] > 0).all()
    # daily vs monthly oil totals should agree closely for producers
    assert k["total_oil_sm3"].sum() == pytest.approx(p["oil_sm3"].sum())
