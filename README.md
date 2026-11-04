# CASPIANX | Well Intelligence

Streamlit prototype for exploring well production, forecasting oil rate, detecting statistical anomalies, and reviewing illustrative engineering checks. It uses the Equinor Volve production workbook. The workbook is not included in this repository; upload it in the app or provide it through `CASPIANX_DATA`.

**Live app:** Add the Streamlit Community Cloud URL after the first deployment.

## What it includes

- Executive dashboard for field and well KPIs
- Well explorer with production history and decline indicators
- Oil-rate forecasting with a chronological 80/20 split and recursive 30-day backtests from eight origins
- Robust z-score and Isolation Forest anomaly detection
- Configurable engineering limit checks with clearly labeled illustrative defaults
- Technical insights, well ranking, and data-quality review

## Screenshots

Screenshots will be added after the app is deployed and captured from the running interface.

## Data and attribution

The app does not generate substitute data. Upload the Volve workbook (`.xlsx`) or its ZIP archive in the app, place it at `data/Volve production data.xlsx`, or set `CASPIANX_DATA` to its path. To have a public deployment load the workbook automatically, commit it at that exact `data/` path. Until the workbook is available, the deployed app shows the upload screen. Equinor allows sharing the Volve licensed material and adapted material with attribution, a link to the applicable terms, and any applicable copyright notice; the material may not be sold or presented in a misleading or distorted way. Review the current [Volve data-sharing page](https://www.equinor.com/energy/volve-data-sharing) and [licence terms](https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf) before redistributing data. See [data/README.md](data/README.md).

## Limitations

- This is a prototype, not an operational decision system.
- Volve is a public North Sea reference dataset, not SOCAR operational data.
- Forecasts use historical oil rates only. They do not model interventions, shut-ins, choke changes, or other operating inputs.
- Forecast evaluation is backtesting, not a guarantee of future performance.
- Anomaly flags are statistical signals, not confirmed equipment or reservoir problems.
- Engineering thresholds are illustrative placeholders and must be configured for the relevant operating envelope.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To run the app, upload the Volve workbook, place it at `data/Volve production data.xlsx`, or set `CASPIANX_DATA` to its path. Without the workbook, the app displays its upload screen.

## Deploy with Git

1. In [Streamlit Community Cloud](https://share.streamlit.io), create an app from this repository.
2. Select branch `main` and entrypoint `app.py`.
3. After the first deployment, each `git push origin main` updates the app automatically.

Set the resulting app URL in this README and in the GitHub repository's **About** website field. Suggested GitHub topics: `streamlit`, `well-performance`, `petroleum-engineering`, `production-forecasting`, `anomaly-detection`, `volve`.

## Project layout

```text
app.py                  Streamlit UI and page routing
src/                    Data loading, preprocessing, analytics, forecasting
.streamlit/config.toml  Streamlit theme and server settings
data/README.md          Data instructions and licence attribution
```
