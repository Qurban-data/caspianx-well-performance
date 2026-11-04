"""CASPIANX | WELL INTELLIGENCE - Streamlit prototype.

Reference dataset: Equinor Volve production data (not SOCAR operational data).
Run with: streamlit run app.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import anomaly_detection as ad
from src import engineering_checks as ec
from src import forecasting as fc
from src import insights as ins
from src import well_analysis as wa
from src.data_loader import DATA_FILENAME, DataNotFoundError, find_data_file, load_raw, parse_monthly
from src.preprocessing import clean_daily, injection_frame, producing_frame, quality_report

warnings.filterwarnings("ignore", category=FutureWarning)

NAVY, BLUE, LIGHT_BLUE, GREY = "#0B1F3A", "#1F5FBF", "#7FA8E0", "#6B7A90"
STATUS_COLORS = {"Normal": "#2E8B6F", "Warning": "#D9A400", "Critical": "#C0392B", "No data": "#9AA5B4"}
SEV_COLORS = {"High": "#C0392B", "Medium": "#D9A400", "Low": "#7FA8E0"}
WELL_PALETTE = ["#0B1F3A", "#1F5FBF", "#5B8DD6", "#9DBBE8", "#6B7A90", "#2E8B6F"]

st.set_page_config(page_title="CASPIANX | Well Intelligence", page_icon="🛢️", layout="wide")

CSS = f"""
<style>
.block-container {{ padding-top: 1.4rem; max-width: 1400px; }}
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #E8EEF8 !important; }}
.brand {{ font-weight: 800; letter-spacing: .12em; font-size: 1.05rem; color: #fff; }}
.brand small {{ display:block; letter-spacing:.2em; color:{LIGHT_BLUE}; font-weight:600; font-size:.7rem; }}
.kpi {{ background:#F2F5FA; border-left:4px solid {BLUE}; border-radius:6px; padding:.8rem 1rem; height:100%; }}
.kpi .l {{ font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; color:{GREY}; font-weight:600; }}
.kpi .v {{ font-size:1.55rem; font-weight:700; color:{NAVY}; line-height:1.25; }}
.kpi .s {{ font-size:.75rem; color:{GREY}; }}
.pagehead {{ border-bottom:2px solid {NAVY}; padding-bottom:.4rem; margin-bottom:1rem; }}
.pagehead h2 {{ margin:0; color:{NAVY}; font-size:1.5rem; }}
.pagehead p {{ margin:.15rem 0 0 0; color:{GREY}; font-size:.9rem; }}
.disclaimer {{ background:#FFF8E5; border:1px solid #F0DC9B; border-radius:6px; padding:.6rem .9rem;
               font-size:.82rem; color:#5B4A00; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<div class="pagehead"><h2>{title}</h2><p>{subtitle}</p></div>', unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=NAVY), legend=dict(orientation="h", y=-0.18),
                      colorway=WELL_PALETTE)
    return fig


def fmt_num(v, nd=0):
    return "n/a" if v is None or pd.isna(v) else f"{v:,.{nd}f}"


def short(v, unit):
    """Human-friendly large volume formatting (Sm3 -> k / M / G)."""
    if pd.isna(v):
        return "n/a"
    a = abs(v)
    if a >= 1e9:
        return f"{v/1e9:,.2f} G{unit}"
    if a >= 1e6:
        return f"{v/1e6:,.2f} M{unit}"
    if a >= 1e3:
        return f"{v/1e3:,.1f} k{unit}"
    return f"{v:,.0f} {unit}"


def style_status(df, col="status"):
    def color(v):
        c = STATUS_COLORS.get(v)
        return f"background-color:{c}; color:white; font-weight:600" if c else ""
    return df.style.map(color, subset=[col])


# --------------------------------------------------------------------------------------
# Data layer (cached)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning Volve production data...")
def load_pipeline(file_bytes: bytes | None, path: str | None):
    import io
    src = io.BytesIO(file_bytes) if file_bytes is not None else path
    daily_raw, monthly_raw = load_raw(src)
    clean = clean_daily(daily_raw)
    prod = producing_frame(clean)
    inj = injection_frame(clean)
    return {
        "daily_raw_rows": len(daily_raw),
        "clean": clean, "prod": prod, "inj": inj,
        "monthly": parse_monthly(monthly_raw),
        "quality": quality_report(daily_raw, clean),
        "kpis": wa.all_kpis(prod),
        "dataset_label": "Equinor Volve production data",
    }


@st.cache_data(show_spinner="Running anomaly detection...")
def run_anomalies(_prod, wells, z, contamination, window):
    return ad.detect_all(_prod, list(wells), z_threshold=z, contamination=contamination, window=window)


@st.cache_data(show_spinner="Training and back-testing models (chronological split)...")
def run_forecast_eval(_prod, well, horizon):
    return fc.evaluate_well(_prod, well, test_frac=0.2, horizon=horizon)


def get_data():
    if "uploaded_bytes" in st.session_state:
        return load_pipeline(st.session_state["uploaded_bytes"], None)
    path = find_data_file()
    if path is not None:
        return load_pipeline(None, str(path))
    return None


def get_thresholds() -> ec.Thresholds:
    if "thresholds" not in st.session_state:
        st.session_state["thresholds"] = ec.Thresholds().as_dict()
    return ec.Thresholds(**st.session_state["thresholds"])


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
def page_dashboard(D):
    page_header("Executive Dashboard", "Field-level production overview and wells requiring attention")
    prod, kp = D["prod"], D["kpis"]
    th = get_thresholds()
    checks = ec.evaluate_limits(kp, th)
    status = ec.well_status(checks)
    attention = int((status["overall"].isin(["Warning", "Critical"])).sum()) if not status.empty else 0

    field = prod.groupby("date")[["oil_sm3", "gas_sm3", "water_sm3"]].sum(min_count=1)
    field_days = field[field["oil_sm3"] > 0]
    n_suff = int((kp["data_class"] == "Sufficient").sum())

    c = st.columns(5)
    kpi(c[0], "Producer wells analysed", f"{len(kp)}", f"{n_suff} with sufficient history")
    kpi(c[1], "Cumulative oil", short(kp["total_oil_sm3"].sum(), "Sm³"), "all producers, whole record")
    kpi(c[2], "Cumulative gas", short(kp["total_gas_sm3"].sum(), "Sm³"), "reported bore gas volume")
    kpi(c[3], "Avg field oil rate", f"{fmt_num(field_days['oil_sm3'].mean())} Sm³/d",
        "mean over days with field production")
    kpi(c[4], "Wells requiring attention", f"{attention}",
        "Warning/Critical under current thresholds")

    st.markdown("&nbsp;")
    m = prod.assign(month=prod["date"].dt.to_period("M").dt.to_timestamp())
    monthly = m.groupby(["month", "well"])["oil_sm3"].sum().reset_index()
    fig = go.Figure()
    for i, (w, g) in enumerate(monthly.groupby("well")):
        fig.add_bar(x=g["month"], y=g["oil_sm3"], name=w, marker_color=WELL_PALETTE[i % 6])
    fig.update_layout(barmode="stack", title="Monthly oil production by well (Sm³/month)")
    st.plotly_chart(style_fig(fig, 400), width="stretch")

    left, right = st.columns(2)
    with left:
        cum = kp["total_oil_sm3"].sort_values()
        fig = go.Figure(go.Bar(x=cum.values, y=cum.index, orientation="h", marker_color=BLUE))
        fig.update_layout(title="Cumulative oil by well (Sm³)")
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with right:
        fm = m.groupby("month")[["oil_sm3", "water_sm3"]].sum()
        wc = fm["water_sm3"] / (fm["oil_sm3"] + fm["water_sm3"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=fm.index, y=fm["oil_sm3"], name="Oil (Sm³/month)", line=dict(color=NAVY), secondary_y=False)
        fig.add_scatter(x=wc.index, y=wc.values, name="Field water cut", line=dict(color=LIGHT_BLUE), secondary_y=True)
        fig.update_yaxes(title_text="Sm³/month", secondary_y=False)
        fig.update_yaxes(title_text="Water cut", tickformat=".0%", range=[0, 1], secondary_y=True)
        fig.update_layout(title="Field oil and water cut")
        st.plotly_chart(style_fig(fig, 340), width="stretch")

    if not status.empty:
        st.subheader("Well status summary")
        disp = status.rename(columns={"overall": "status"})
        st.dataframe(style_status(disp), width="stretch", hide_index=True)
    st.caption("Status uses the illustrative thresholds configured on the Engineering Monitoring page. "
               f"Data source: {D.get('dataset_label', 'production dataset')}; not SOCAR operational data.")


def page_explorer(D):
    page_header("Well Explorer", "Production history, trends and KPIs for an individual producer")
    prod, kp = D["prod"], D["kpis"]
    wells = list(kp.index)
    c1, c2 = st.columns([1, 2])
    well = c1.selectbox("Well", wells, index=min(2, len(wells) - 1))
    g_all = prod[prod["well"] == well].sort_values("date")
    dmin, dmax = g_all["date"].min().date(), g_all["date"].max().date()
    rng = c2.slider("Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax))
    g = g_all[(g_all["date"].dt.date >= rng[0]) & (g_all["date"].dt.date <= rng[1])]
    if g.empty:
        st.warning("No records in the selected date range.")
        return
    pdays = g[g["is_producing"]]
    oil, gas, wat = g["oil_sm3"].sum(), g["gas_sm3"].sum(), g["water_sm3"].sum()
    cal = (rng[1] - rng[0]).days + 1
    c = st.columns(6)
    kpi(c[0], "Oil produced", short(oil, "Sm³"), f"{len(pdays)} producing days")
    kpi(c[1], "Gas produced", short(gas, "Sm³"))
    kpi(c[2], "Water produced", short(wat, "Sm³"))
    kpi(c[3], "Avg oil rate", f"{fmt_num(pdays['oil_sm3'].mean())} Sm³/d", "producing days only")
    kpi(c[4], "Water cut / GOR", f"{wat/(oil+wat):.0%} / {fmt_num(gas/oil) if oil > 0 else 'n/a'}"
        if (oil + wat) > 0 else "n/a", "cumulative in range")
    kpi(c[5], "Uptime", f"{min(g['on_stream_hrs'].fillna(0).sum()/(24*cal), 1):.0%}", "on-stream hours / calendar")
    dq = wa.data_sufficiency(int(g_all["is_producing"].sum()))
    if dq != "Sufficient":
        st.warning(f"History class for this well: **{dq}** ({int(g_all['is_producing'].sum())} producing days). "
                   "KPIs are indicative only.")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Oil rate (Sm³/d)", "Gas rate (Sm³/d)", "Water rate (Sm³/d)"))
    for r, (col, colr) in enumerate([("oil_sm3", NAVY), ("gas_sm3", BLUE), ("water_sm3", LIGHT_BLUE)], 1):
        fig.add_scatter(x=g["date"], y=g[col], mode="lines", line=dict(color=colr, width=1.4), showlegend=False, row=r, col=1)
    st.plotly_chart(style_fig(fig, 620), width="stretch")

    a, b = st.columns(2)
    with a:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=g["date"], y=g["water_cut"], name="Water cut", line=dict(color=BLUE), secondary_y=False)
        fig.add_scatter(x=g["date"], y=g["gor"], name="GOR (Sm³/Sm³)", line=dict(color=GREY), secondary_y=True)
        fig.update_yaxes(tickformat=".0%", title_text="Water cut", secondary_y=False)
        fig.update_yaxes(title_text="GOR", secondary_y=True)
        fig.update_layout(title="Water cut and GOR (producing days)")
        st.plotly_chart(style_fig(fig, 340), width="stretch")
    with b:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=g["date"], y=g["bhp_bar"], name="Downhole pressure (bar)", line=dict(color=NAVY), secondary_y=False)
        fig.add_scatter(x=g["date"], y=g["whp_bar"], name="Wellhead pressure (bar)", line=dict(color=LIGHT_BLUE), secondary_y=False)
        fig.add_scatter(x=g["date"], y=g["choke_pct"], name="Choke (%)", line=dict(color=GREY, dash="dot"), secondary_y=True)
        fig.update_layout(title="Pressure and choke")
        st.plotly_chart(style_fig(fig, 340), width="stretch")
        if g["bhp_bar"].notna().sum() == 0:
            st.caption("No valid downhole pressure data for this well/range (gauge zeros were excluded).")
        elif D["clean"].query("well == @well")["flag_gauge_zero"].any():
            st.caption("Downhole gauge values equal to 0 while flowing were treated as gauge outage and excluded.")

    with st.expander("Data table"):
        cols = ["date", "on_stream_hrs", "oil_sm3", "gas_sm3", "water_sm3", "water_cut", "gor", "bhp_bar", "whp_bar", "choke_pct"]
        st.dataframe(g[cols], width="stretch", hide_index=True)
        st.download_button("Download CSV", g[cols].to_csv(index=False).encode(), f"{well.replace('/', '_')}_daily.csv", "text/csv")


def page_forecast(D):
    page_header("Production Forecast", "Oil-rate forecasting with chronological validation and honest uncertainty")
    prod, kp = D["prod"], D["kpis"]
    c1, c2 = st.columns(2)
    well = c1.selectbox("Well", list(kp.index), index=min(2, len(kp) - 1), key="fc_well")
    horizon = c2.slider("Forecast horizon (days)", 14, 120, 60, step=7)
    ev = run_forecast_eval(prod, well, 30)
    ser = ev.series
    if ev.status != "ok":
        st.error(ev.message)
        if not ser.empty:
            fig = go.Figure(go.Scatter(x=ser["date"], y=ser["oil_sm3"], line=dict(color=NAVY)))
            fig.update_layout(title="Historical oil rate (no forecast generated)")
            st.plotly_chart(style_fig(fig), width="stretch")
        return

    badge = {"Moderate": "#D9A400", "Low": "#C0392B"}.get(ev.reliability, GREY)
    st.markdown(f"**Forecast confidence:** <span style='background:{badge};color:white;padding:2px 10px;border-radius:10px'>"
                f"{ev.reliability}</span> &nbsp; {ev.reliability_note}", unsafe_allow_html=True)
    st.caption(f"Series: {ev.n_points} producing days. Chronological split at {ev.split_date.date()} "
               "(80 % train / 20 % test). Backtest: recursive 30-day forecasts from 8 origins in the test period.")

    model_choice = st.selectbox("Model used for the forecast", ["Best (auto)"] + list(fc._model_factories().keys()))
    model = ev.best_model if model_choice == "Best (auto)" else model_choice
    f = fc.forecast_future(prod, well, model, horizon, ev)

    fig = go.Figure()
    fig.add_scatter(x=ser["date"], y=ser["oil_sm3"], name="Historical (producing days)", line=dict(color=NAVY, width=1.3))
    fig.add_scatter(x=f["date"], y=f["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=f["date"], y=f["lower"], fill="tonexty", fillcolor="rgba(31,95,191,0.18)",
                    line=dict(width=0), name="~80 % empirical band")
    fig.add_scatter(x=f["date"], y=f["forecast"], name=f"Forecast ({model})", line=dict(color=BLUE, width=2.5))
    fig.add_vline(x=ser["date"].iloc[-1], line_dash="dot", line_color=GREY)
    fig.update_layout(title=f"{well} - oil rate forecast (Sm³/d)", yaxis_title="Sm³/d")
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    st.caption("Forecast assumes uninterrupted production after the last observation (no shut-ins, workovers "
               "or choke changes). The band is derived from backtest errors, not from a probabilistic model.")

    st.subheader("Actual vs predicted (test period, one-step-ahead)")
    tp = ev.test_predictions
    sel = st.multiselect("Models to display", [c for c in tp.columns if c not in ("date", "actual")],
                         default=[ev.best_model])
    fig = go.Figure()
    fig.add_scatter(x=tp["date"], y=tp["actual"], name="Actual", line=dict(color=NAVY, width=1.6))
    for i, s in enumerate(sel):
        fig.add_scatter(x=tp["date"], y=tp[s], name=s, line=dict(width=1.3, dash="dot", color=WELL_PALETTE[(i + 1) % 6]))
    st.plotly_chart(style_fig(fig, 360), width="stretch")

    m1, m2 = st.columns(2)
    m1.markdown("**One-step-ahead (test period)**")
    m1.dataframe(ev.metrics_onestep.round(3), width="stretch", hide_index=True)
    m2.markdown("**Recursive 30-day backtest (realistic forecast skill)**")
    m2.dataframe(ev.metrics_recursive.round(3), width="stretch", hide_index=True)
    st.caption("MAE/RMSE in Sm³/d. R² can be negative when a model is worse than predicting the mean.")

    with st.expander("Forecasting limitations", expanded=True):
        st.markdown(
            "- Univariate: only past oil rates are used. Pressures, chokes, shut-ins and interventions are not modelled.\n"
            "- One-step metrics look optimistic because the previous day's real value is always known; "
            "the recursive backtest is the fair measure.\n"
            "- Tree models cannot extrapolate beyond training ratios; if decline accelerates they may lag.\n"
            "- Small test windows (few hundred days) make metrics noisy; compare with the baselines before trusting a model.\n"
            "- Volve wells are in late life; behaviour may not transfer to other fields.")


def page_anomaly(D):
    page_header("Anomaly Detection", "Statistical anomalies (robust z-score + Isolation Forest) - not confirmed engineering problems")
    prod, kp = D["prod"], D["kpis"]
    c = st.columns(3)
    z = c[0].slider("Robust z threshold", 3.0, 8.0, 5.0, 0.5)
    cont = c[1].slider("Isolation Forest contamination", 0.005, 0.05, 0.02, 0.005, format="%.3f")
    win = c[2].select_slider("Trailing window (producing days)", [30, 45, 60, 90], value=60)
    res = run_anomalies(prod, tuple(kp.index), z, cont, win)
    if res.empty:
        st.info("No anomalies detected with the current settings.")
        return
    f1, f2, f3 = st.columns(3)
    wsel = f1.multiselect("Wells", list(kp.index), default=list(kp.index))
    ssel = f2.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium"])
    vsel = f3.multiselect("Variable", sorted(res["variable"].unique()), default=sorted(res["variable"].unique()))
    view = res[res["well"].isin(wsel) & res["severity"].isin(ssel) & res["variable"].isin(vsel)]
    k = st.columns(4)
    kpi(k[0], "Events shown", f"{len(view)}", f"of {len(res)} detected")
    kpi(k[1], "High severity", f"{int((view['severity'] == 'High').sum())}")
    kpi(k[2], "Wells affected", f"{view['well'].nunique()}")
    kpi(k[3], "Most affected variable", view["variable"].mode().iat[0] if not view.empty else "n/a")
    st.markdown('<div class="disclaimer">Anomalies are statistical deviations from each well\'s own recent behaviour. '
                'Many coincide with routine events (restarts, choke changes, shut-ins). Confirm against operational '
                'records before drawing engineering conclusions. For Isolation Forest rows, "deviation score" is the '
                'model anomaly score, not a z-score.</div>', unsafe_allow_html=True)
    st.markdown("&nbsp;")
    if view.empty:
        st.warning("No events match the selected filters.")
        return
    cnt = view.groupby(["well", "severity"]).size().reset_index(name="n")
    fig = go.Figure()
    for sev in ["High", "Medium", "Low"]:
        s = cnt[cnt["severity"] == sev]
        fig.add_bar(x=s["well"], y=s["n"], name=sev, marker_color=SEV_COLORS[sev])
    fig.update_layout(barmode="stack", title="Anomaly events per well")
    st.plotly_chart(style_fig(fig, 300), width="stretch")

    well = st.selectbox("Visualise abnormal periods for well", sorted(view["well"].unique()))
    g = prod[(prod["well"] == well) & prod["is_producing"]].sort_values("date")
    ev = view[view["well"] == well]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("Oil rate (Sm³/d)", "Water cut"))
    fig.add_scatter(x=g["date"], y=g["oil_sm3"], line=dict(color=NAVY, width=1.2), name="Oil rate", row=1, col=1)
    fig.add_scatter(x=g["date"], y=g["water_cut"], line=dict(color=BLUE, width=1.2), name="Water cut", row=2, col=1)
    for sev, color in SEV_COLORS.items():
        e = ev[ev["severity"] == sev]
        if e.empty:
            continue
        oil_at = g.set_index("date")["oil_sm3"].reindex(e["date"]).to_numpy()
        fig.add_scatter(x=e["date"], y=oil_at, mode="markers", marker=dict(color=color, size=9, line=dict(color="white", width=1)),
                        name=f"{sev} event", text=e["variable"] + " (" + e["method"] + ")", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(style_fig(fig, 520), width="stretch")

    show = view.copy()
    show["date"] = show["date"].dt.date
    show = show.rename(columns={"deviation_score": "deviation score"})
    for col in ["observed", "expected_low", "expected_high", "deviation score"]:
        show[col] = show[col].round(3)
    st.dataframe(show, width="stretch", hide_index=True, height=360)
    st.download_button("Download anomalies (CSV)", show.to_csv(index=False).encode(), "caspianx_anomalies.csv", "text/csv")


def page_engineering(D):
    page_header("Engineering Monitoring", "Configurable limit checks on recent well performance")
    kp = D["kpis"]
    st.markdown('<div class="disclaimer"><b>Illustrative defaults.</b> The thresholds below are placeholders for '
                'demonstration. They are not official reservoir, facility or regulatory limits and are not derived '
                'from Volve design documents. Set values that match the actual operating envelope.</div>',
                unsafe_allow_html=True)
    st.markdown("&nbsp;")
    th = get_thresholds()
    with st.expander("Configure thresholds", expanded=True):
        with st.form("th_form"):
            a, b, c, d, e = st.columns(5)
            v = {}
            a.markdown("**Min oil rate (Sm³/d)**")
            v["min_oil_warn"] = a.number_input("Warning below", value=float(th.min_oil_warn), min_value=0.0, step=10.0)
            v["min_oil_crit"] = a.number_input("Critical below", value=float(th.min_oil_crit), min_value=0.0, step=10.0)
            b.markdown("**Max water cut (fraction)**")
            v["max_wc_warn"] = b.number_input("Warning above", value=float(th.max_wc_warn), min_value=0.0, max_value=1.0, step=0.05)
            v["max_wc_crit"] = b.number_input("Critical above", value=float(th.max_wc_crit), min_value=0.0, max_value=1.0, step=0.05)
            c.markdown("**Max GOR (Sm³/Sm³)**")
            v["max_gor_warn"] = c.number_input("Warning above ", value=float(th.max_gor_warn), min_value=0.0, step=10.0)
            v["max_gor_crit"] = c.number_input("Critical above ", value=float(th.max_gor_crit), min_value=0.0, step=10.0)
            d.markdown("**Annual decline (fraction)**")
            v["decline_warn"] = d.number_input("Warning above  ", value=float(th.decline_warn), min_value=0.0, max_value=1.0, step=0.05)
            v["decline_crit"] = d.number_input("Critical above  ", value=float(th.decline_crit), min_value=0.0, max_value=1.0, step=0.05)
            e.markdown("**Min uptime, last 365 d**")
            v["min_uptime_warn"] = e.number_input("Warning below  ", value=float(th.min_uptime_warn), min_value=0.0, max_value=1.0, step=0.05)
            v["min_uptime_crit"] = e.number_input("Critical below  ", value=float(th.min_uptime_crit), min_value=0.0, max_value=1.0, step=0.05)
            s1, s2 = st.columns([1, 5])
            apply = s1.form_submit_button("Apply")
            reset = s2.form_submit_button("Reset to defaults")
        if apply:
            st.session_state["thresholds"] = v
            st.rerun()
        if reset:
            st.session_state["thresholds"] = ec.Thresholds().as_dict()
            st.rerun()

    th = get_thresholds()
    bad = [k for k in ("min_oil", "max_wc", "max_gor", "decline", "min_uptime")
           if (getattr(th, f"{k}_warn") < getattr(th, f"{k}_crit") if k.startswith("min")
               else getattr(th, f"{k}_warn") > getattr(th, f"{k}_crit"))]
    if bad:
        st.warning("For some parameters the warning threshold is more severe than the critical one "
                   f"({', '.join(bad)}). Results may be inconsistent - check the values.")
    checks = ec.evaluate_limits(kp, th)
    if checks.empty:
        st.info("No wells with enough data to evaluate.")
        return
    status = ec.well_status(checks)
    cols = st.columns(max(len(status), 1))
    for col, (_, r) in zip(cols, status.iterrows()):
        color = STATUS_COLORS[r["overall"]]
        col.markdown(f'<div class="kpi" style="border-left-color:{color}"><div class="l">{r["well"]}</div>'
                     f'<div class="v" style="color:{color}">{r["overall"]}</div>'
                     f'<div class="s">{r["n_critical"]} critical / {r["n_warning"]} warning</div></div>',
                     unsafe_allow_html=True)
    st.markdown("&nbsp;")
    wsel = st.multiselect("Wells", list(status["well"]), default=list(status["well"]))
    view = checks[checks["well"].isin(wsel)].copy()
    view["actual"] = view["actual"].round(3)
    st.dataframe(style_status(view), width="stretch", hide_index=True, height=420)
    st.download_button("Download limit checks (CSV)", view.to_csv(index=False).encode(), "caspianx_limit_checks.csv", "text/csv")
    st.caption("Wells with 'Limited' history are evaluated but their KPIs are indicative. Wells with 'Insufficient' data are skipped.")


def page_insights(D):
    page_header("Technical Insights", "Evidence-based findings generated from computed KPIs")
    prod, kp = D["prod"], D["kpis"]
    th = get_thresholds()
    checks = ec.evaluate_limits(kp, th)
    anomalies = run_anomalies(prod, tuple(kp.index), 5.0, 0.02, 60)
    tab1, tab2, tab3 = st.tabs(["Insights", "Well ranking", "Dataset & data quality"])

    with tab1:
        for item in ins.build_insights(kp, anomalies, checks):
            st.markdown(f"#### {item['title']}")
            st.write(item["text"])
            if item["evidence"] is not None:
                st.dataframe(item["evidence"].round(3), width="stretch")
        st.caption("Insights are rule-based summaries of the computed numbers; they describe patterns in the data, "
                   "not confirmed root causes.")

    with tab2:
        st.markdown("Transparent weighted score (0-100). Each criterion is min-max normalised across wells with "
                    "**Sufficient** history; weights are adjustable.")
        w = st.columns(4)
        weights = {
            "recent_rate": w[0].slider("Recent oil rate", 0, 100, 35),
            "uptime": w[1].slider("Uptime", 0, 100, 20),
            "decline": w[2].slider("Low decline", 0, 100, 25),
            "water_cut": w[3].slider("Low water cut", 0, 100, 20),
        }
        if sum(weights.values()) == 0:
            st.warning("At least one weight must be > 0.")
        else:
            norm = {k: v / sum(weights.values()) for k, v in weights.items()}
            ranked = wa.rank_wells(kp, norm)
            ranked["classification"] = wa.classify_wells(kp)
            cols = ["rank", "score", "classification", "data_class", "recent_rate", "recent_rate_change_pct",
                    "annual_decline", "recent_water_cut", "recent_gor", "uptime", "producing_days"]
            tbl = ranked[cols].sort_values("rank").round(3)
            st.dataframe(tbl, width="stretch")
            st.caption("Decline is used only when the exponential fit has R² ≥ 0.20; otherwise it is excluded and the "
                       "remaining weights are renormalised for that well. 'Limited' wells are not ranked. "
                       "'Declining' = reliable annual decline ≥ 30 % or recent-vs-initial rate change ≤ -30 %.")
            st.download_button("Download ranking (CSV)", tbl.to_csv().encode(), "caspianx_ranking.csv", "text/csv")

    with tab3:
        q = D["quality"]
        st.markdown(f"**Source:** {D.get('dataset_label', DATA_FILENAME)}. Daily rows: {q['rows']:,} "
                    f"({q['date_min'].date()} to {q['date_max'].date()}); monthly sheet: {len(D['monthly'])} well-months.")
        st.markdown("**Findings computed from the file**")
        st.markdown(
            f"- Duplicate well/date keys: **{q['duplicate_well_date_keys']}**\n"
            f"- Negative water volumes (set to missing): **{q['negative_water_rows']}**\n"
            f"- On-stream hours above 24 (clipped): **{q['on_stream_hours_over_24_rows']}**\n"
            f"- Flowing days with downhole gauge = 0 (treated as gauge outage): **{q['flowing_rows_with_zero_gauge']}**\n"
            f"- Missing oil volume on production rows: **{q['missing_oil_rows_producers']}** "
            f"(the {q['missing_oil_rows_all']:,} blanks are injection-mode rows)\n"
            f"- Missing calendar days within each well's record: "
            + ", ".join(f"{w}: {n}" for w, n in q["date_gaps_days_by_well"].items()))
        inv = D["clean"].groupby(["well", "well_type"]).agg(
            rows=("date", "size"), start=("date", "min"), end=("date", "max"),
            oil_sm3=("oil_sm3", "sum"), water_inj_sm3=("water_inj_sm3", "sum")).reset_index()
        inv["start"], inv["end"] = inv["start"].dt.date, inv["end"].dt.date
        st.dataframe(inv.round(0), width="stretch", hide_index=True)
        st.caption("Injection wells (15/9-F-4, and 15/9-F-5 before April 2016) carry water-injection volumes and are "
                   "excluded from producer analytics. The dataset has no geology, PVT, well-test or reservoir pressure "
                   "files, and daily data has no gas-lift or ESP parameters.")


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------
PAGES = {
    "Executive Dashboard": page_dashboard,
    "Well Explorer": page_explorer,
    "Production Forecast": page_forecast,
    "Anomaly Detection": page_anomaly,
    "Engineering Monitoring": page_engineering,
    "Technical Insights": page_insights,
}

with st.sidebar:
    st.markdown('<div class="brand">CASPIANX<small>WELL INTELLIGENCE</small></div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("Reference dataset: Equinor Volve field (North Sea). Not SOCAR operational data.")

try:
    data = get_data()
except DataNotFoundError as exc:
    data = None
    st.error(str(exc))
except Exception as exc:  # corrupted / wrong file
    data = None
    st.error(f"Could not read the dataset: {exc}")

if data is None:
    page_header("CASPIANX | WELL INTELLIGENCE", "Upload the Volve production dataset to begin")
    st.info(f"The raw dataset is not bundled with this repository (see data/README.md). Place "
            f"`{DATA_FILENAME}` in the `data/` folder, or upload it (or the original ZIP) below.")
    up = st.file_uploader("Volve production data (.xlsx or .zip)", type=["xlsx", "zip"])
    if up is not None:
        st.session_state["uploaded_bytes"] = up.getvalue()
        st.rerun()
else:
    try:
        PAGES[page](data)
    except Exception as exc:
        st.error(f"Unexpected error on this page: {exc}")
        st.exception(exc)
