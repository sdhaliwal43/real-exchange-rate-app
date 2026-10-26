"""
Real & Nominal Exchange Rate Explorer (IMF SDMX 3.0)
====================================================

Downloads monthly CPI and exchange-rate data from the IMF SDMX 3.0 API and computes

    E_i(t)   = units of country i currency per 1 USD (IMF period average)
    S(t)     = E_2(t) / E_1(t)
             = units of Country 2 currency per 1 unit of Country 1 currency
    RER(t)   = S(t) * P1(t) / P2(t)
    RER*     = mean of RER(t) over valid months in the selected window
    Index(t) = 100 * RER(t) / RER*

Direction convention: an INCREASE in the Real Index means Country 1's currency has
APPRECIATED in real terms against Country 2's currency.

Euro area prices: the euro area publishes an HICP rather than a national CPI. If the IMF
has no euro-area CPI, the app falls back to an HICP key at the IMF, then to Eurostat's
euro-area HICP.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================
# Configuration
# =============================================================================
IMF_BASE_URL = "https://api.imf.org/external/sdmx/3.0"
AGENCY = "IMF.STA"            # IMF Statistics Department
CPI_DATAFLOW = "CPI"
ER_DATAFLOW = "ER"

# Dataflow versions verified live after the IMF's May-2026 SDMX 3.0 refactor.
# If the IMF publishes new versions, update these (later URL candidates use "+" = latest).
DATAFLOW_VERSIONS = {CPI_DATAFLOW: "5.0.0", ER_DATAFLOW: "4.0.1"}

CPI_KEY_TEMPLATE = "{iso3}.CPI._T.IX.M"   # headline CPI, all items, index, monthly
# Exchange-rate keys tried in order. The first is the specified series
# (domestic currency per USD, period average). The second is the code seen in
# post-refactor live tests; if it is used, check the orientation (see sidebar toggle).
ER_KEY_TEMPLATES = ("{iso3}.XDC_USD.PA_RT.M", "{iso3}.USD_XDC.PA_RT.M")

# Extra price-index keys to try when the main CPI key returns nothing.
# The euro area publishes an HICP (Harmonised Index of Consumer Prices), not a national CPI.
CPI_FALLBACK_KEYS = {
    "G163": ["G163.HICP._T.IX.M"],
}
# Last resort for the euro area: Eurostat HICP, all items (CP00), euro area (EA), monthly index.
EUROSTAT_HICP_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/"
    "prc_hicp_midx/M..CP00.EA?format=SDMX-CSV&startPeriod=1996-01"
)

START_PERIOD = "1976-01"
START_TS = pd.Timestamp("1976-01-01")

REQUEST_TIMEOUT = (10, 45)    # (connect, read) seconds
CACHE_TTL = 24 * 60 * 60      # 24 hours
USER_AGENT = (
    "RER-Explorer/1.2 (Streamlit Community Cloud; educational real-exchange-rate app; "
    "python-requests)"
)

ACCEPT_CSV = "application/vnd.sdmx.data+csv;version=2.0.0"
ACCEPT_JSON = "application/vnd.sdmx.data+json;version=2.0.0"
ACCEPT_STRUCT_JSON = "application/vnd.sdmx.structure+json;version=2.0.0"
ACCEPT_STRUCT_XML = "application/vnd.sdmx.structure+xml;version=3.0.0"

REAL_COLOR = "#2563EB"     # blue: Real Index
NOMINAL_COLOR = "#F97316"  # orange: nominal S(t)

FALLBACK_COUNTRIES: dict[str, tuple[str, str]] = {
    "USA": ("United States", "USD"),
    "G163": ("Euro area", "EUR"),
    "GBR": ("United Kingdom", "GBP"),
    "JPN": ("Japan", "JPY"),
    "CHN": ("China", "CNY"),
    "CAN": ("Canada", "CAD"),
    "CHE": ("Switzerland", "CHF"),
    "AUS": ("Australia", "AUD"),
    "NZL": ("New Zealand", "NZD"),
    "IND": ("India", "INR"),
    "BRA": ("Brazil", "BRL"),
    "MEX": ("Mexico", "MXN"),
    "KOR": ("Korea, Republic of", "KRW"),
    "ZAF": ("South Africa", "ZAR"),
    "SWE": ("Sweden", "SEK"),
    "NOR": ("Norway", "NOK"),
    "DNK": ("Denmark", "DKK"),
    "POL": ("Poland", "PLN"),
    "CZE": ("Czech Republic", "CZK"),
    "HUN": ("Hungary", "HUF"),
    "TUR": ("Türkiye", "TRY"),
    "RUS": ("Russian Federation", "RUB"),
    "IDN": ("Indonesia", "IDR"),
    "MYS": ("Malaysia", "MYR"),
    "THA": ("Thailand", "THB"),
    "PHL": ("Philippines", "PHP"),
    "SGP": ("Singapore", "SGD"),
    "HKG": ("Hong Kong SAR", "HKD"),
    "ISR": ("Israel", "ILS"),
    "SAU": ("Saudi Arabia", "SAR"),
    "ARE": ("United Arab Emirates", "AED"),
    "EGY": ("Egypt", "EGP"),
    "NGA": ("Nigeria", "NGN"),
    "KEN": ("Kenya", "KES"),
    "ARG": ("Argentina", "ARS"),
    "CHL": ("Chile", "CLP"),
    "COL": ("Colombia", "COP"),
    "PER": ("Peru", "PEN"),
    "PAK": ("Pakistan", "PKR"),
    "VNM": ("Vietnam", "VND"),
}
CURRENCY_CODES = {iso: ccy for iso, (_, ccy) in FALLBACK_COUNTRIES.items()}


def _empty_series() -> pd.Series:
    return pd.Series(dtype="float64")


# =============================================================================
# Styling
# =============================================================================
def apply_styling() -> None:
    """Custom look: Inter font, banner header, card-style metrics, chart and expander panels."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, p, li, label, h1, h2, h3, h4, .stMarkdown { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }

        .hero {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #0EA5E9 100%);
            color: #fff; padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.18);
        }
        .hero h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0.2rem 0 0.3rem 0; padding: 0; }
        .hero p { color: rgba(255, 255, 255, 0.88); margin: 0; font-size: 1rem; }
        .hero-kicker {
            text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem;
            font-weight: 600; color: rgba(255, 255, 255, 0.75);
        }

        [data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
            padding: 0.9rem 1rem; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        [data-testid="stMetricLabel"] p { font-size: 0.78rem; color: #64748B; font-weight: 500; }
        [data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 700; color: #0F172A; }

        [data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }
        [data-testid="stPlotlyChart"] {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 0.5rem;
        }
        [data-testid="stExpander"] details {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
        }
        .stDownloadButton button { border-radius: 10px; font-weight: 600; }
        #MainMenu, footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    """Banner header at the top of the page."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">IMF · SDMX 3.0 · Monthly data since 1976</div>
            <h1>Real &amp; Nominal Exchange Rates</h1>
            <p>Compare any two currencies in nominal terms and after adjusting for inflation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# URL construction — EDIT HERE IF THE IMF CHANGES ITS API
# =============================================================================
def build_data_urls(dataflow: str, key: str, start_period: str = START_PERIOD) -> list[str]:
    """Return candidate IMF SDMX 3.0 data-query URLs, most preferred first.

    *** If the IMF changes its API structure, this is the one function to edit. ***

    1. Compact form with pinned version (verified working after the May-2026 refactor):
         {base}/data/IMF.STA,CPI,5.0.0/USA.CPI._T.IX.M?startPeriod=1976-01
    2. Compact form without a version (server picks the latest).
    3. Standard SDMX 3.0 REST form with "+" (= latest version):
         {base}/data/dataflow/IMF.STA/CPI/+/USA.CPI._T.IX.M?startPeriod=1976-01
    4. Standard form with the SDMX 3.0 time filter c[TIME_PERIOD]=ge:...
    5. Pinned compact form with no time filter (app trims to 1976-01 itself).
    """
    base = IMF_BASE_URL.rstrip("/")
    version = DATAFLOW_VERSIONS.get(dataflow, "+")
    sp = f"startPeriod={start_period}"
    return [
        f"{base}/data/{AGENCY},{dataflow},{version}/{key}?{sp}",
        f"{base}/data/{AGENCY},{dataflow}/{key}?{sp}",
        f"{base}/data/dataflow/{AGENCY}/{dataflow}/+/{key}?{sp}",
        f"{base}/data/dataflow/{AGENCY}/{dataflow}/+/{key}?c[TIME_PERIOD]=ge:{start_period}",
        f"{base}/data/{AGENCY},{dataflow},{version}/{key}",
    ]


def build_codelist_urls() -> list[str]:
    """Candidate URLs for the IMF country codelist (CL_COUNTRY)."""
    base = IMF_BASE_URL.rstrip("/")
    return [
        f"{base}/structure/codelist/IMF/CL_COUNTRY/+",
        f"{base}/structure/codelist/IMF.STA/CL_COUNTRY/+",
        f"{base}/structure/codelist/IMF/CL_COUNTRY/latest",
    ]


def _subscription_key() -> str | None:
    """OPTIONAL. The app works without a key. Only if the IMF starts returning 401/403,
    add IMF_SUBSCRIPTION_KEY under Streamlit Cloud → App settings → Secrets."""
    try:
        return st.secrets.get("IMF_SUBSCRIPTION_KEY")
    except Exception:  # noqa: BLE001 - no secrets file is the normal case
        return None


def _headers(accept: str) -> dict[str, str]:
    h = {"User-Agent": USER_AGENT, "Accept": accept, "Accept-Encoding": "gzip, deflate"}
    key = _subscription_key()
    if key:
        h["Ocp-Apim-Subscription-Key"] = key
    return h


# =============================================================================
# Parsing helpers (CSV, JSON and XML SDMX messages)
# =============================================================================
_PERIOD_RE = re.compile(r"^\s*(\d{4})-?M?(\d{1,2})")


def parse_period(value) -> pd.Timestamp:
    """Convert '1976-01', '1976-M01' or '1976M01' to a month-start Timestamp."""
    m = _PERIOD_RE.match(str(value))
    if not m:
        return pd.NaT
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return pd.NaT
    return pd.Timestamp(year=year, month=month, day=1)


def _series_from_pairs(periods: list, values: list) -> pd.Series:
    if not periods:
        return _empty_series()
    idx = pd.DatetimeIndex([parse_period(p) for p in periods])
    s = pd.Series(pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(), index=idx)
    s = s[s.index.notna()]
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s[s.index >= START_TS]
    return s.dropna().astype("float64")


def _parse_sdmx_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), dtype=str)
    if df.empty:
        return _empty_series()

    def find(name: str):
        return next((c for c in df.columns if c.split(":")[0].strip().upper() == name), None)

    time_col, val_col = find("TIME_PERIOD"), find("OBS_VALUE")
    if time_col is None or val_col is None:
        return _empty_series()
    return _series_from_pairs(df[time_col].tolist(), df[val_col].tolist())


def _parse_sdmx_json(payload: dict) -> pd.Series:
    """Parse SDMX-JSON (2.0 'structures' list or 1.0 'structure' object)."""
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    structures = data.get("structures") or ([data["structure"]] if "structure" in data else [])
    datasets = data.get("dataSets") or []
    if not structures or not datasets:
        return _empty_series()

    obs_dims = structures[0].get("dimensions", {}).get("observation", [])
    if not obs_dims:
        return _empty_series()
    t_pos = next((i for i, d in enumerate(obs_dims) if d.get("id") == "TIME_PERIOD"), 0)
    time_values = [v.get("id") or v.get("value") for v in obs_dims[t_pos].get("values", [])]

    ds = datasets[0]
    periods, values = [], []

    def collect(observations: dict, pos: int):
        for okey, val in observations.items():
            try:
                t_idx = int(str(okey).split(":")[pos])
                periods.append(time_values[t_idx])
                values.append(val[0] if isinstance(val, list) and val else val)
            except (ValueError, IndexError, TypeError):
                continue

    if ds.get("series"):
        best = max(ds["series"].values(), key=lambda s: len(s.get("observations", {})))
        collect(best.get("observations", {}), 0)
    elif ds.get("observations"):
        collect(ds["observations"], t_pos)
    return _series_from_pairs(periods, values)


def _parse_sdmx_xml(text: str) -> pd.Series:
    root = ET.fromstring(text.encode("utf-8"))
    periods, values = [], []
    for el in root.iter():
        if el.tag.split("}")[-1] != "Obs":
            continue
        p, v = el.attrib.get("TIME_PERIOD"), el.attrib.get("OBS_VALUE")
        if p is None:
            for ch in el:
                tag = ch.tag.split("}")[-1]
                if tag in ("ObsDimension", "ObsKey"):
                    p = ch.attrib.get("value", p)
                elif tag == "ObsValue":
                    v = ch.attrib.get("value", v)
        if p is not None:
            periods.append(p)
            values.append(v)
    return _series_from_pairs(periods, values)


def _parse_response(resp: requests.Response) -> pd.Series:
    text = resp.text.lstrip("\ufeff").lstrip()
    if text.startswith("{"):
        return _parse_sdmx_json(resp.json())
    if text.startswith("<"):
        return _parse_sdmx_xml(text)
    return _parse_sdmx_csv(text)


# =============================================================================
# Data access
# =============================================================================
class IMFFetchError(Exception):
    """Exceptions are NOT cached by st.cache_data, so outages are retried next run."""

    def __init__(self, message: str, tried: list[str]):
        super().__init__(message)
        self.tried = tried


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _fetch_series_cached(dataflow: str, key: str) -> tuple[pd.Series, str]:
    tried: list[str] = []
    errors: list[str] = []
    session = requests.Session()
    for url in build_data_urls(dataflow, key):
        tried.append(url)
        for accept in (ACCEPT_CSV, ACCEPT_JSON):
            try:
                resp = session.get(url, headers=_headers(accept), timeout=REQUEST_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as exc:
                raise IMFFetchError(f"Network error ({type(exc).__name__}): {exc}", tried)
            except requests.RequestException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            if resp.status_code in (401, 403):
                errors.append(f"HTTP {resp.status_code} (access denied — the IMF may require a subscription key)")
                break
            if resp.status_code == 404:
                errors.append("HTTP 404 (series not found / no results)")
                break
            if resp.status_code >= 400:
                errors.append(f"HTTP {resp.status_code}")
                continue
            try:
                series = _parse_response(resp)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Could not parse response ({exc})")
                continue
            if not series.empty:
                return series, url
            errors.append("Response contained no observations")
    raise IMFFetchError("; ".join(dict.fromkeys(errors)) or "No data returned", tried)


def fetch_series(dataflow: str, key: str) -> dict:
    """Fetch an IMF series safely. Returns {series, url, error, tried}."""
    try:
        series, url = _fetch_series_cached(dataflow, key)
        return {"series": series, "url": url, "error": None, "tried": [url]}
    except IMFFetchError as exc:
        return {"series": _empty_series(), "url": exc.tried[-1] if exc.tried else "",
                "error": str(exc), "tried": exc.tried}
    except Exception as exc:  # noqa: BLE001
        return {"series": _empty_series(), "url": "", "error": f"Unexpected error: {exc}", "tried": []}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _fetch_country_codelist() -> dict[str, str]:
    last_error = "no candidate URL succeeded"
    for url in build_codelist_urls():
        for accept in (ACCEPT_STRUCT_JSON, ACCEPT_STRUCT_XML):
            try:
                resp = requests.get(url, headers=_headers(accept), timeout=REQUEST_TIMEOUT)
            except requests.RequestException as exc:
                raise RuntimeError(f"Network error: {exc}")
            if resp.status_code >= 400:
                last_error = f"HTTP {resp.status_code} at {url}"
                continue
            text = resp.text.lstrip("\ufeff").lstrip()
            codes: dict[str, str] = {}
            try:
                if text.startswith("{"):
                    data = resp.json().get("data", {})
                    for cl in data.get("codelists", []) or []:
                        for code in cl.get("codes", []) or []:
                            cid = str(code.get("id", "")).strip()
                            name = code.get("name") or code.get("names")
                            if isinstance(name, dict):
                                name = name.get("en") or next(iter(name.values()), "")
                            if cid:
                                codes[cid] = str(name or cid)
                elif text.startswith("<"):
                    root = ET.fromstring(text.encode("utf-8"))
                    for el in root.iter():
                        if el.tag.split("}")[-1] != "Code":
                            continue
                        cid, name = el.attrib.get("id"), None
                        for ch in el:
                            if ch.tag.split("}")[-1] == "Name":
                                lang = ch.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "en")
                                if lang == "en" or name is None:
                                    name = (ch.text or "").strip()
                        if cid:
                            codes[cid] = name or cid
            except Exception as exc:  # noqa: BLE001
                last_error = f"Parse error at {url}: {exc}"
                continue
            if codes:
                return codes
            last_error = f"Empty codelist at {url}"
    raise RuntimeError(last_error)


def get_country_list() -> tuple[dict[str, str], str]:
    """Return ({ISO3: name}, source). IMF codelist when reachable, merged with the fallback list."""
    countries = {iso: name for iso, (name, _) in FALLBACK_COUNTRIES.items()}
    try:
        codes = _fetch_country_codelist()
        for cid, name in codes.items():
            if re.fullmatch(r"[A-Z]{3}", cid) and cid not in countries:
                countries[cid] = name
        countries["G163"] = "Euro area"
        return countries, f"IMF codelist CL_COUNTRY ({len(countries)} entries)"
    except Exception as exc:  # noqa: BLE001
        return countries, f"built-in list of {len(countries)} major economies (IMF codelist unavailable: {exc})"


def us_dollar_rate_series() -> pd.Series:
    """USD per USD is identically 1, so it is never requested from the API."""
    end = pd.Timestamp.today().to_period("M").to_timestamp()
    idx = pd.date_range(START_TS, end, freq="MS")
    return pd.Series(1.0, index=idx, dtype="float64")


def fetch_exchange_rate(iso: str) -> dict:
    """Try each exchange-rate key in ER_KEY_TEMPLATES; return the first that has data."""
    all_tried: list[str] = []
    errors: list[str] = []
    for tmpl in ER_KEY_TEMPLATES:
        key = tmpl.format(iso3=iso)
        res = fetch_series(ER_DATAFLOW, key)
        if not res["error"]:
            res["key"] = key
            return res
        all_tried.extend(res["tried"])
        errors.append(f"{key}: {res['error']}")
    return {"series": _empty_series(), "url": all_tried[-1] if all_tried else "",
            "error": " | ".join(errors), "tried": all_tried, "key": ER_KEY_TEMPLATES[0].format(iso3=iso)}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _fetch_eurostat_hicp_cached() -> pd.Series:
    """Download the euro-area HICP from Eurostat. Raises on failure (failures are not cached)."""
    resp = requests.get(EUROSTAT_HICP_URL, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), dtype=str)
    # Eurostat returns several index bases (e.g. 2025=100, 2015=100). Keep one, newest base first.
    unit_col = next((c for c in df.columns if c.split(":")[0].strip().lower() == "unit"), None)
    if unit_col is not None:
        units = df[unit_col].str.split(":").str[0].str.strip()
        for pref in ("I25", "I15", "I05", "I96"):
            if (units == pref).any():
                df = df[units == pref]
                break
    s = _parse_sdmx_csv(df.to_csv(index=False))
    if s.empty:
        raise RuntimeError("Eurostat returned no HICP observations")
    return s


def fetch_cpi(iso: str) -> dict:
    """Fetch a monthly consumer price index: IMF CPI first, then fallback keys,
    then (euro area only) Eurostat HICP. Returns {series, url, error, tried, key, source}."""
    keys = [CPI_KEY_TEMPLATE.format(iso3=iso)] + CPI_FALLBACK_KEYS.get(iso, [])
    tried: list[str] = []
    errors: list[str] = []
    for key in keys:
        res = fetch_series(CPI_DATAFLOW, key)
        if not res["error"]:
            res["key"], res["source"] = key, "IMF"
            return res
        tried.extend(res["tried"])
        errors.append(f"{key}: {res['error']}")
    if iso == "G163":
        try:
            s = _fetch_eurostat_hicp_cached()
            return {"series": s, "url": EUROSTAT_HICP_URL, "error": None, "tried": tried + [EUROSTAT_HICP_URL],
                    "key": "Eurostat HICP prc_hicp_midx (all items, EA)", "source": "Eurostat"}
        except Exception as exc:  # noqa: BLE001
            tried.append(EUROSTAT_HICP_URL)
            errors.append(f"Eurostat HICP: {exc}")
    return {"series": _empty_series(), "url": tried[-1] if tried else "", "error": " | ".join(errors),
            "tried": tried, "key": keys[0], "source": "IMF"}


def build_dataframe(iso1: str, iso2: str, names: dict[str, str],
                    invert_er: bool = False) -> tuple[pd.DataFrame, list, list]:
    """Download E1, E2, P1, P2 and merge them on month (outer join, gaps kept as NaN)."""
    messages: list[tuple[str, str]] = []
    sources: list[tuple[str, str]] = []
    cols: dict[str, pd.Series] = {}

    for slot, iso in (("1", iso1), ("2", iso2)):
        name = names.get(iso, iso)

        # --- Exchange rate: E_i(t) = local currency units per 1 USD ---------------
        e_col = f"E{slot}_LCU_per_USD"
        if iso == "USA":
            cols[e_col] = us_dollar_rate_series()
            sources.append((f"E{slot}: {name} exchange rate", "Not requested: USD per USD = 1 by definition"))
        else:
            res = fetch_exchange_rate(iso)
            series = res["series"]
            if invert_er and not series.empty:
                series = 1.0 / series.where(series > 0)
            cols[e_col] = series
            if res["error"]:
                tried = "\n".join(f"- {u}" for u in res["tried"]) or "- (none)"
                messages.append(("error", f"**No exchange-rate data for {name} ({iso})** — dataflow "
                                          f"`{ER_DATAFLOW}`.\n\nReason: {res['error']}\n\nURLs tried:\n{tried}"))
            elif res["key"] != ER_KEY_TEMPLATES[0].format(iso3=iso):
                messages.append(("info", f"{name}: the IMF returned data under the alternative key "
                                         f"`{res['key']}`. Check that S(t) looks right. If it looks like "
                                         "the reciprocal, tick **Invert IMF exchange rates** in the sidebar."))
            sources.append((f"E{slot}: {name} exchange rate ({res['key']})"
                            + (" — inverted" if invert_er else ""), res["url"]))

        # --- CPI: P_i(t) ---------------------------------------------------------------
        p_col = f"P{slot}"
        res = fetch_cpi(iso)
        cols[p_col] = res["series"]
        if res["error"]:
            tried = "\n".join(f"- {u}" for u in res["tried"]) or "- (none)"
            messages.append(("error", f"**No CPI data for {name} ({iso})**.\n\nReason: {res['error']}"
                                      f"\n\nURLs tried:\n{tried}"))
        elif res["source"] == "Eurostat":
            messages.append(("info", f"{name}: the IMF has no series under `{CPI_KEY_TEMPLATE.format(iso3=iso)}`, "
                                     "so P is Eurostat's euro-area HICP (all items, monthly index). The index base "
                                     "doesn't matter because the Real Index is normalised to its own mean."))
        sources.append((f"P{slot}: {name} price index ({res['key']})", res["url"]))

    order = ["E1_LCU_per_USD", "E2_LCU_per_USD", "P1", "P2"]
    df = pd.concat([cols[c].rename(c) for c in order], axis=1)
    df = df.reindex(columns=order)
    if not df.empty:
        full_idx = pd.date_range(max(df.index.min(), START_TS), df.index.max(), freq="MS")
        df = df.reindex(full_idx)
    df.index.name = "date"
    return df, messages, sources


# =============================================================================
# Economics
# =============================================================================
def compute_rer(df: pd.DataFrame, start: pd.Timestamp | None = None,
                end: pd.Timestamp | None = None) -> tuple[pd.DataFrame, float]:
    """Compute S, RER, RER* and the Real Index over [start, end].

    Cross rate via the dollar:
        S = E_2 / E_1 = (LCU2/USD) / (LCU1/USD) = LCU2 per 1 LCU1
    Real exchange rate:
        RER = S * P1 / P2  (a rise = real appreciation of Country 1's currency)
    Normalisation (CPI bases are arbitrary):
        Real Index = 100 * RER / RER*, RER* = mean RER over valid months.
    Valid month: S, P1, P2 all present and positive.
    """
    w = df.copy()
    if start is not None:
        w = w[w.index >= start]
    if end is not None:
        w = w[w.index <= end]

    e1 = w["E1_LCU_per_USD"].where(w["E1_LCU_per_USD"] > 0)
    e2 = w["E2_LCU_per_USD"].where(w["E2_LCU_per_USD"] > 0)
    w["S"] = e2 / e1                                   # Country 2 currency per 1 Country 1 currency
    w["RER"] = w["S"] * w["P1"] / w["P2"]

    valid = w["S"].gt(0) & w["P1"].gt(0) & w["P2"].gt(0) & w["RER"].notna()
    w["valid_flag"] = valid
    rer_star = float(w.loc[valid, "RER"].mean()) if valid.any() else float("nan")
    w["RER_star"] = rer_star
    w["Real_Index"] = (100.0 * w["RER"] / rer_star).where(valid)
    return w, rer_star


# =============================================================================
# Charts
# =============================================================================
def make_chart(w: pd.DataFrame, mode: str, ccy1: str, ccy2: str) -> go.Figure:
    """Plotly chart. mode in {"Real", "Nominal", "Both"}. S(t) is never normalised."""
    x = w.index
    nominal_axis = f"{ccy2} per 1 {ccy1}"
    real_trace = go.Scatter(
        x=x, y=w["Real_Index"], mode="lines", name="Real Index (mean = 100)",
        line=dict(color=REAL_COLOR, width=2.5), connectgaps=False,
        hovertemplate="Real Index: %{y:.2f}<extra></extra>",
    )
    nominal_trace = go.Scatter(
        x=x, y=w["S"], mode="lines", name=f"Nominal S(t): {nominal_axis}",
        line=dict(color=NOMINAL_COLOR, width=2.5), connectgaps=False,
        hovertemplate="S(t): %{y:,.4f}<extra></extra>",
    )

    if mode == "Both":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(real_trace, secondary_y=False)
        fig.add_trace(nominal_trace, secondary_y=True)
        fig.update_yaxes(title_text="Real Index (mean over window = 100)", color=REAL_COLOR, secondary_y=False)
        fig.update_yaxes(title_text=nominal_axis, color=NOMINAL_COLOR, secondary_y=True, showgrid=False)
    elif mode == "Real":
        fig = go.Figure(real_trace)
        fig.update_yaxes(title_text="Real Index (mean over window = 100)")
    else:
        fig = go.Figure(nominal_trace)
        fig.update_yaxes(title_text=nominal_axis)

    if mode in ("Real", "Both"):
        # Reference line at 100 on the (left) Real Index axis.
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=100, y1=100,
                      line=dict(color="#94A3B8", dash="dash", width=1))

    fig.update_xaxes(
        title_text="Month",
        rangeslider_visible=True,
        rangeselector=dict(buttons=[
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(count=10, label="10Y", step="year", stepmode="backward"),
            dict(count=20, label="20Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
    )
    fig.update_layout(
        template="plotly_white",
        height=600,
        hovermode="x unified",
        showlegend=True,
        font=dict(family="Inter, sans-serif", size=13, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", bordercolor="#E2E8F0", font=dict(family="Inter, sans-serif", size=12)),
        margin=dict(l=60, r=60, t=50, b=30),
    )
    fig.update_xaxes(
        gridcolor="#EEF2F7", linecolor="#CBD5E1",
        showspikes=True, spikecolor="#94A3B8", spikethickness=1, spikedash="dot",
        rangeslider=dict(thickness=0.08, bgcolor="#F1F5F9"),
        rangeselector=dict(bgcolor="#F1F5F9", activecolor="#DBEAFE", font=dict(size=12)),
    )
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False)
    return fig


# =============================================================================
# UI helpers
# =============================================================================
def currency_label(iso: str, names: dict[str, str]) -> str:
    return CURRENCY_CODES.get(iso, f"{names.get(iso, iso)} currency")


def fmt_num(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"


def series_coverage_table(df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for col, label in labels.items():
        s = df[col].dropna() if col in df else _empty_series()
        rows.append({
            "Series": label,
            "First month": s.index.min().strftime("%Y-%m") if not s.empty else "—",
            "Last month": s.index.max().strftime("%Y-%m") if not s.empty else "—",
            "Observations": int(s.size),
        })
    return pd.DataFrame(rows)


# =============================================================================
# App
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="Real & Nominal Exchange Rates", page_icon="💱", layout="wide")
    apply_styling()
    hero()

    with st.spinner("Loading country list…"):
        countries, list_source = get_country_list()
    codes = sorted(countries, key=lambda c: countries[c].lower())

    # ---- Sidebar: selections ------------------------------------------------------
    st.sidebar.header("Settings")
    fmt = lambda c: f"{countries.get(c, c)} ({c})"  # noqa: E731
    iso1 = st.sidebar.selectbox("Country 1", codes, index=codes.index("USA") if "USA" in codes else 0, format_func=fmt)
    iso2 = st.sidebar.selectbox("Country 2", codes, index=codes.index("G163") if "G163" in codes else 1, format_func=fmt)
    invert_er = st.sidebar.checkbox(
        "Invert IMF exchange rates",
        value=False,
        help="Leave unticked normally. Tick only if the IMF series turns out to be USD per local currency "
             "(e.g. USA vs GBR shows ~1.3 instead of ~0.75 GBP per USD).",
    )
    st.sidebar.caption(f"Country list source: {list_source}")

    if iso1 == iso2:
        st.error("Country 1 and Country 2 are the same. Please choose two different countries.")
        st.stop()

    name1, name2 = countries.get(iso1, iso1), countries.get(iso2, iso2)
    ccy1, ccy2 = currency_label(iso1, countries), currency_label(iso2, countries)

    with st.spinner(f"Downloading data for {name1} and {name2}…"):
        df, messages, sources = build_dataframe(iso1, iso2, countries, invert_er)
    for level, text in messages:
        getattr(st, level)(text)

    coverage_labels = {
        "E1_LCU_per_USD": f"E1: {name1} per USD", "E2_LCU_per_USD": f"E2: {name2} per USD",
        "P1": f"P1: {name1} prices", "P2": f"P2: {name2} prices",
    }

    if df.empty:
        st.error("No data could be retrieved for either country. See the messages above for the URLs tried.")
        st.stop()

    full, _ = compute_rer(df)
    valid_months = full.index[full["valid_flag"]]
    if len(valid_months) == 0:
        st.error(f"There are no months where the exchange rates and price indices of {name1} and {name2} all overlap.")
        st.dataframe(series_coverage_table(df, coverage_labels), hide_index=True, use_container_width=True)
        st.stop()

    first_all, last_all = valid_months.min(), valid_months.max()
    month_labels = [m.strftime("%Y-%m") for m in pd.date_range(first_all, last_all, freq="MS")]

    # ---- Sample window (resets automatically when countries change) ---------------
    if len(month_labels) >= 2:
        start_lbl, end_lbl = st.sidebar.select_slider(
            "Sample period (start – end month)", options=month_labels,
            value=(month_labels[0], month_labels[-1]), key=f"window_{iso1}_{iso2}",
        )
    else:
        start_lbl = end_lbl = month_labels[0]
    if start_lbl > end_lbl:
        start_lbl, end_lbl = end_lbl, start_lbl
    start_ts, end_ts = pd.Timestamp(f"{start_lbl}-01"), pd.Timestamp(f"{end_lbl}-01")

    # ---- Economics over the selected window (recomputed on every change) ----------
    w, rer_star = compute_rer(df, start_ts, end_ts)
    w = w.reindex(pd.date_range(start_ts, end_ts, freq="MS"))
    w["valid_flag"] = w["valid_flag"].fillna(False).astype(bool)
    w["RER_star"] = rer_star
    n_valid = int(w["valid_flag"].sum())
    if n_valid == 0 or not np.isfinite(rer_star):
        st.warning("No valid months in the selected window. Widen the sample period.")
        st.stop()

    v = w[w["valid_flag"]]
    first_v, last_v = v.index.min(), v.index.max()
    s_nonnull = w["S"].dropna()
    latest_s = float(s_nonnull.iloc[-1]) if not s_nonnull.empty else float("nan")
    latest_s_month = s_nonnull.index[-1].strftime("%Y-%m") if not s_nonnull.empty else "n/a"
    latest_idx = float(v["Real_Index"].iloc[-1])

    # ---- Coverage warnings --------------------------------------------------------
    if first_all > START_TS:
        late = [label for col, label in coverage_labels.items()
                if df[col].first_valid_index() is not None and df[col].first_valid_index() >= first_all]
        st.warning(f"Limited coverage: months with all four series begin in **{first_all:%Y-%m}**, not 1976-01. "
                   f"Latest-starting series: {', '.join(late) if late else 'n/a'}.")
    if "G163" in (iso1, iso2):
        st.info("Euro-area note: the euro (and its IMF exchange rate) begins in 1999, so euro-area comparisons "
                "cannot start earlier.")
    n_window = len(w)
    if n_valid < n_window:
        st.warning(f"{n_window - n_valid} of {n_window} months in the window lack S, P1 or P2. They are shown as gaps "
                   "and excluded from RER* (no interpolation or forward-filling).")
    if n_valid < 36:
        st.warning(f"Only {n_valid} valid months: RER* (and therefore the Real Index level) is sensitive to the window.")

    # ---- Metrics ------------------------------------------------------------------
    st.markdown(f"#### {name1} ({ccy1}) vs {name2} ({ccy2})")
    c = st.columns(6)
    c[0].metric("RER*", fmt_num(rer_star), help="Mean of RER(t) = S·P1/P2 over valid months in the window")
    c[1].metric("Valid months", f"{n_valid}")
    c[2].metric("First valid month", first_v.strftime("%Y-%m"))
    c[3].metric("Last valid month", last_v.strftime("%Y-%m"))
    c[4].metric(f"Latest S ({latest_s_month})", fmt_num(latest_s), help=f"{ccy2} per 1 {ccy1}")
    c[5].metric(f"Latest Real Index ({last_v:%Y-%m})", f"{latest_idx:.2f}")
    st.caption(f"Check: mean Real Index over valid months = **{v['Real_Index'].mean():.4f}** (should be 100).")

    st.info(f"**Interpretation:** an increase in the Real Index means **{name1}'s currency ({ccy1}) has appreciated "
            f"in real terms** against {name2}'s currency ({ccy2}). S(t) is shown in actual units: {ccy2} per 1 {ccy1}.")

    # ---- Chart --------------------------------------------------------------------
    mode = st.radio("Chart", ["Real", "Nominal", "Both"], index=2, horizontal=True)
    try:
        st.plotly_chart(make_chart(w, mode, ccy1, ccy2), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not draw the chart: {exc}")

    # ---- CSV download -------------------------------------------------------------
    out = w.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m"))
    out = out[["date", "E1_LCU_per_USD", "E2_LCU_per_USD", "S", "P1", "P2", "RER",
               "valid_flag", "RER_star", "Real_Index"]]
    st.download_button(
        "⬇️ Download data (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"RER_{iso1}_{iso2}_{start_lbl}_{end_lbl}.csv",
        mime="text/csv",
    )

    # ---- Documentation ------------------------------------------------------------
    with st.expander("Data coverage by series"):
        st.dataframe(series_coverage_table(df, coverage_labels), hide_index=True, use_container_width=True)

    with st.expander("Formulas, conventions and data sources"):
        st.markdown(f"""
**Definitions**

- $E_i(t)$ = IMF period-average exchange rate: units of country *i* currency per 1 USD
  (dataflow `{AGENCY},{ER_DATAFLOW},{DATAFLOW_VERSIONS[ER_DATAFLOW]}`, key `{{ISO3}}.XDC_USD.PA_RT.M`,
  fallback `{{ISO3}}.USD_XDC.PA_RT.M`). For the United States $E=1$ every month.
- $S(t) = E_2(t)/E_1(t)$ = units of **Country 2** currency per 1 unit of **Country 1** currency.
- $P_1(t), P_2(t)$ = monthly headline CPI (dataflow `{AGENCY},{CPI_DATAFLOW},{DATAFLOW_VERSIONS[CPI_DATAFLOW]}`,
  key `{{ISO3}}.CPI._T.IX.M`). For the euro area, which publishes an HICP rather than a CPI, the app falls back
  to an IMF HICP key and then to Eurostat's euro-area HICP (all items, monthly index).
- $RER(t) = S(t)\\,P_1(t)/P_2(t)$
- $RER^* = $ mean of $RER(t)$ over valid months in the selected window.
- $\\text{{Real Index}}(t) = 100 \\times RER(t)/RER^*$

**Direction:** a higher Real Index = real appreciation of Country 1's currency against Country 2's.

**Notes:** price-index bases differ, so only movements in the index are meaningful. Rebased series can show
breaks. Gaps stay gaps. Euro-area exchange rates start in 1999. Data are cached for 24 hours.

**URLs used in this run**
""")
        for label, url in sources:
            st.markdown(f"- {label}: `{url}`")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}. Try reloading the page or choosing different countries.")
