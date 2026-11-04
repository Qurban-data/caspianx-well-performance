"""Data loading for the Equinor Volve production dataset.

The loader never hardcodes machine-specific paths. It looks for the workbook in
(1) the ``CASPIANX_DATA`` environment variable, (2) the project ``data/`` folder,
(3) the current working directory. A file-like object (e.g. a Streamlit upload)
or a ``.zip`` archive containing the workbook is also accepted.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, Union

import pandas as pd

DATA_FILENAME = "Volve production data.xlsx"
DAILY_SHEET = "Daily Production Data"
MONTHLY_SHEET = "Monthly Production Data"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Source = Union[str, os.PathLike, BinaryIO]


class DataNotFoundError(FileNotFoundError):
    """Raised when the Volve workbook cannot be located."""


def find_data_file() -> Optional[Path]:
    """Return the first existing candidate path of the Volve workbook (or a zip)."""
    candidates = []
    env = os.environ.get("CASPIANX_DATA")
    if env:
        candidates.append(Path(env))
    candidates += [
        PROJECT_ROOT / "data" / DATA_FILENAME,
        PROJECT_ROOT / "data" / "archive.zip",
        Path.cwd() / DATA_FILENAME,
        Path.cwd() / "archive.zip",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _is_wrapper_zip(raw: bytes) -> bool:
    """An .xlsx is itself a ZIP; distinguish a wrapper archive from a workbook."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            return "xl/workbook.xml" not in zf.namelist()
    except zipfile.BadZipFile:
        return False


def _open_workbook_bytes(source: Source) -> io.BytesIO:
    """Resolve any supported source to an in-memory XLSX buffer."""
    if hasattr(source, "read"):
        raw = source.read()
    else:
        raw = Path(source).read_bytes()
    if _is_wrapper_zip(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            xlsx = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
            if not xlsx:
                raise DataNotFoundError("The ZIP archive does not contain an .xlsx file.")
            preferred = [n for n in xlsx if "volve" in n.lower()] or xlsx
            return io.BytesIO(zf.read(preferred[0]))
    return io.BytesIO(raw)


def load_raw(source: Optional[Source] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw daily and monthly sheets exactly as stored in the workbook."""
    if source is None:
        source = find_data_file()
        if source is None:
            raise DataNotFoundError(
                f"'{DATA_FILENAME}' not found. Place it in the 'data/' folder, set the "
                "CASPIANX_DATA environment variable, or upload it in the app."
            )
    buf = _open_workbook_bytes(source)
    sheets = pd.read_excel(buf, sheet_name=[DAILY_SHEET, MONTHLY_SHEET])
    return sheets[DAILY_SHEET], sheets[MONTHLY_SHEET]


def parse_monthly(monthly_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the monthly sheet (first data row holds units, values stored as text)."""
    m = monthly_raw.iloc[1:].dropna(subset=["Wellbore name"]).copy()
    for c in ["On Stream", "Oil", "Gas", "Water", "GI", "WI"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["Year"] = m["Year"].astype(int)
    m["Month"] = m["Month"].astype(int)
    m = m.rename(columns={
        "Wellbore name": "well", "Year": "year", "Month": "month", "On Stream": "on_stream_hrs",
        "Oil": "oil_sm3", "Gas": "gas_sm3", "Water": "water_sm3", "GI": "gas_inj_sm3",
        "WI": "water_inj_sm3",
    })
    return m.drop(columns=["NPDCode"], errors="ignore").reset_index(drop=True)
