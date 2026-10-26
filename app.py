"""
RealFX — Real & Nominal Exchange Rate Analytics (IMF SDMX 3.0)
==============================================================

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

import html
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
APP_NAME = "RealFX"

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
    "RealFX/2.0 (Streamlit Community Cloud; educational real-exchange-rate app; "
    "python-requests)"
)

ACCEPT_CSV = "application/vnd.sdmx.data+csv;version=2.0.0"
ACCEPT_JSON = "application/vnd.sdmx.data+json;version=2.0.0"
ACCEPT_STRUCT_JSON = "application/vnd.sdmx.structure+json;version=2.0.0"
ACCEPT_STRUCT_XML = "application/vnd.sdmx.structure+xml;version=3.0.0"

# Design tokens
REAL_COLOR = "#4F46E5"            # indigo: Real Index
REAL_FILL = "rgba(79, 70, 229, 0.10)"
NOMINAL_COLOR = "#F97316"         # orange: nominal S(t)
INK = "#0F172A"
MUTED = "#64748B"
GRID = "#EDF1F7"

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

# One-click pairs shown in the sidebar (Country 1, Country 2, label).
QUICK_PAIRS = [
    ("USA", "G163", "USD · EUR"),
    ("USA", "GBR", "USD · GBP"),
    ("USA", "JPN", "USD · JPY"),
    ("USA", "CHN", "USD · CNY"),
    ("USA", "CAN", "USD · CAD"),
    ("USA", "MEX", "USD · MXN"),
]


def _empty_series() -> pd.Series:
    return pd.Series(dtype="float64")


# =============================================================================
# Styling & HTML helpers
# =============================================================================
def _html(s: str) -> str:
    """Strip indentation and blank lines so Streamlit's Markdown renders raw HTML
    instead of treating indented lines as a code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())


def render(s: str) -> None:
    st.markdown(_html(s), unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def md_to_html(text: str) -> str:
    """Tiny Markdown subset (bold, inline code, line breaks) for styled notes."""
    t = esc(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t.replace("\n", "<br>")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, p, li, label, h1, h2, h3, h4, h5, .stMarkdown, button, input { font-family: 'Inter', sans-serif; }
.stApp {
background:
radial-gradient(1100px 500px at 0% -10%, rgba(99, 102, 241, 0.10), transparent 60%),
radial-gradient(900px 400px at 100% 0%, rgba(14, 165, 233, 0.08), transparent 60%),
#F6F8FC;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }
/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1220 0%, #111A2E 100%); border-right: 1px solid #1E293B; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small, section[data-testid="stSidebar"] summary { color: #CBD5E1; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #0F172A; border-color: #334155; border-radius: 10px; }
section[data-testid="stSidebar"] [data-baseweb="select"] div, section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #F1F5F9 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #94A3B8; }
section[data-testid="stSidebar"] .stButton button {
background: rgba(255, 255, 255, 0.04); border: 1px solid #334155; color: #E2E8F0;
border-radius: 10px; font-weight: 600; font-size: 0.82rem; transition: all 0.15s ease;
}
section[data-testid="stSidebar"] .stButton button:hover { border-color: #6366F1; background: rgba(99, 102, 241, 0.15); color: #fff; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details { background: rgba(255, 255, 255, 0.03); border: 1px solid #1E293B; border-radius: 12px; }
.sb-brand { display: flex; align-items: center; gap: 10px; margin: 4px 0 18px 0; }
.sb-logo { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; font-weight: 800; color: #fff;
background: linear-gradient(135deg, #6366F1, #0EA5E9); box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45); }
.sb-name { font-weight: 800; font-size: 1.15rem; color: #F8FAFC; letter-spacing: -0.01em; line-height: 1.1; }
.sb-tag { font-size: 0.72rem; color: #94A3B8; }
.sb-section { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #64748B; margin: 18px 0 6px 0; }
/* ---------- Hero ---------- */
.hero {
position: relative; overflow: hidden; border-radius: 22px; padding: 28px 32px; margin-bottom: 22px; color: #fff;
background:
radial-gradient(800px 280px at 90% -30%, rgba(56, 189, 248, 0.35), transparent 60%),
radial-gradient(700px 300px at -10% 130%, rgba(99, 102, 241, 0.55), transparent 60%),
linear-gradient(135deg, #0B1220 0%, #1E1B4B 100%);
box-shadow: 0 24px 48px -18px rgba(15, 23, 42, 0.45);
display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap;
}
.hero::after {
content: ""; position: absolute; inset: 0; pointer-events: none;
background-image: linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
background-size: 34px 34px;
-webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,0.7), transparent 85%);
mask-image: linear-gradient(to bottom, rgba(0,0,0,0.7), transparent 85%);
}
.hero > * { position: relative; z-index: 1; }
.badges { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
.badge { display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 600; padding: 4px 10px;
border-radius: 999px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.14); color: #E0E7FF; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: #22C55E; box-shadow: 0 0 0 0 rgba(34,197,94,0.6); animation: pulse 2s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.55); } 70% { box-shadow: 0 0 0 8px rgba(34,197,94,0); } 100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); } }
.hero-title { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05; margin: 0; color: #fff; }
.hero-title .arrow { color: #818CF8; font-weight: 700; padding: 0 6px; }
.hero-sub { color: rgba(226, 232, 240, 0.82); font-size: 0.98rem; margin-top: 8px; }
.hero-stats { display: flex; gap: 12px; flex-wrap: wrap; }
.hero-stat { min-width: 150px; padding: 12px 16px; border-radius: 14px; background: rgba(255,255,255,0.07);
border: 1px solid rgba(255,255,255,0.14); -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px); }
.hero-stat .k { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: #A5B4FC; font-weight: 700; }
.hero-stat .v { font-size: 1.45rem; font-weight: 800; color: #fff; font-variant-numeric: tabular-nums; margin-top: 2px; }
.hero-stat .s { font-size: 0.74rem; color: #CBD5E1; }
/* ---------- KPI cards ---------- */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 14px; margin: 6px 0 18px 0; }
.kpi { background: #fff; border: 1px solid #E6EAF2; border-radius: 16px; padding: 16px 18px;
box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px -16px rgba(15,23,42,0.18); transition: transform 0.15s ease, box-shadow 0.15s ease; }
.kpi:hover { transform: translateY(-2px); box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 14px 30px -16px rgba(79,70,229,0.35); }
.kpi-top { display: flex; justify-content: space-between; align-items: center; }
.kpi-label { font-size: 0.74rem; font-weight: 600; color: #64748B; letter-spacing: 0.02em; }
.kpi-icon { width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center; font-size: 0.9rem; background: #EEF2FF; }
.kpi-value { font-size: 1.6rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em; margin-top: 8px; font-variant-numeric: tabular-nums; }
.kpi-sub { font-size: 0.76rem; margin-top: 4px; color: #64748B; }
.pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-weight: 700; font-size: 0.72rem; margin-right: 4px; }
.pill.pos { background: #DCFCE7; color: #15803D; }
.pill.neg { background: #FEE2E2; color: #B91C1C; }
.pill.neu { background: #F1F5F9; color: #475569; }
/* ---------- Cards ---------- */
.card { background: #fff; border: 1px solid #E6EAF2; border-radius: 18px; padding: 20px 22px; height: 100%;
box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px -16px rgba(15,23,42,0.18); }
.eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #6366F1; margin-bottom: 8px; }
.insight-head { font-size: 1.22rem; font-weight: 700; color: #0F172A; letter-spacing: -0.01em; line-height: 1.35; }
.insight-head .pos { color: #15803D; } .insight-head .neg { color: #B91C1C; }
.insight-body { color: #475569; font-size: 0.92rem; margin-top: 8px; line-height: 1.55; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.chip { font-size: 0.76rem; font-weight: 600; color: #334155; background: #F1F5F9; border: 1px solid #E2E8F0; padding: 5px 10px; border-radius: 999px; }
.dec-row { display: grid; grid-template-columns: 1fr 70px; gap: 6px 12px; align-items: center; margin-top: 12px; }
.dec-label { font-size: 0.82rem; font-weight: 600; color: #334155; }
.dec-val { font-size: 0.86rem; font-weight: 800; text-align: right; font-variant-numeric: tabular-nums; }
.dec-val.pos { color: #15803D; } .dec-val.neg { color: #B91C1C; }
.dec-track { grid-column: 1 / span 2; height: 8px; border-radius: 999px; background: #F1F5F9; overflow: hidden; }
.dec-bar { height: 100%; border-radius: 999px; }
.dec-bar.pos { background: linear-gradient(90deg, #4ADE80, #16A34A); }
.dec-bar.neg { background: linear-gradient(90deg, #F87171, #DC2626); }
.dec-note { font-size: 0.74rem; color: #94A3B8; margin-top: 14px; }
/* ---------- Notes ---------- */
.notes { display: flex; flex-direction: column; gap: 8px; margin: 2px 0 16px 0; }
.note { font-size: 0.84rem; padding: 10px 14px; border-radius: 12px; border: 1px solid; line-height: 1.5; }
.note.warn { background: #FFFBEB; border-color: #FDE68A; color: #92400E; }
.note.info { background: #EEF2FF; border-color: #C7D2FE; color: #3730A3; }
.note code { background: rgba(15,23,42,0.06); padding: 1px 5px; border-radius: 5px; font-size: 0.78rem; }
/* ---------- Section header ---------- */
.section-head { display: flex; justify-content: space-between; align-items: baseline; margin: 22px 0 8px 0; }
.section-title { font-size: 1.05rem; font-weight: 700; color: #0F172A; }
.section-meta { font-size: 0.8rem; color: #64748B; }
/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #E9EDF5; padding: 4px; border-radius: 12px; width: fit-content; }
.stTabs [data-baseweb="tab"] { border-radius: 9px; padding: 6px 18px; height: auto; font-weight: 600; color: #475569; }
.stTabs [aria-selected="true"] { background: #fff; color: #0F172A; box-shadow: 0 1px 3px rgba(15,23,42,0.12); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
/* ---------- Segmented radio (chart mode) ---------- */
[data-testid="stMain"] div[role="radiogroup"], section.main div[role="radiogroup"] {
display: inline-flex; gap: 0; background: #E9EDF5; padding: 4px; border-radius: 10px; }
[data-testid="stMain"] div[role="radiogroup"] label, section.main div[role="radiogroup"] label {
padding: 5px 16px; border-radius: 8px; margin: 0 !important; cursor: pointer; }
[data-testid="stMain"] div[role="radiogroup"] label:has(input:checked), section.main div[role="radiogroup"] label:has(input:checked) {
background: #fff; box-shadow: 0 1px 3px rgba(15,23,42,0.12); }
[data-testid="stMain"] div[role="radiogroup"] label > div:first-child, section.main div[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stMain"] div[role="radiogroup"] p, section.main div[role="radiogroup"] p { font-weight: 600; font-size: 0.85rem; }
/* ---------- Charts, tables, expanders, buttons ---------- */
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid #E6EAF2; border-radius: 18px; padding: 10px 8px 4px 8px;
box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px -16px rgba(15,23,42,0.18); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid #E6EAF2; }
[data-testid="stMain"] [data-testid="stExpander"] details, section.main [data-testid="stExpander"] details {
background: #fff; border: 1px solid #E6EAF2; border-radius: 14px; }
.stDownloadButton button { border-radius: 12px; font-weight: 700; padding: 0.6rem 1.1rem; }
/* ---------- Footer ---------- */
.app-footer { margin-top: 36px; padding-top: 16px; border-top: 1px solid #E2E8F0; display: flex; justify-content: space-between;
flex-wrap: wrap; gap: 8px; font-size: 0.76rem; color: #94A3B8; }
@media (max-width: 640px) { .hero { padding: 22px; } .hero-title { font-size: 1.7rem; } }
</style>
"""


def apply_styling() -> None:
    render(CSS)


def hero_html(ccy1: str, ccy2: str, name1: str, name2: str, stats: list[tuple[str, str, str]] | None = None) -> str:
    """Dark banner with the currency pair and (once computed) headline stats."""
    stat_html = "".join(
        f'<div class="hero-stat"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
        for k, v, s in (stats or [])
    )
    return f"""
    <div class="hero">
    <div>
    <div class="badges">
    <span class="badge"><span class="dot"></span>Live IMF data</span>
    <span class="badge">Monthly · since 1976</span>
    <span class="badge">Cached 24h</span>
    </div>
    <h1 class="hero-title">{esc(ccy1)}<span class="arrow">→</span>{esc(ccy2)}</h1>
    <div class="hero-sub">{esc(name1)} vs {esc(name2)} · real &amp; nominal exchange rates</div>
    </div>
    <div class="hero-stats">{stat_html}</div>
    </div>
    """


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
    text = resp.text.lstrip("﻿").lstrip()
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
            text = resp.text.lstrip("﻿").lstrip()
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
        return countries, f"IMF codelist · {len(countries)} economies"
    except Exception:  # noqa: BLE001
        return countries, f"Built-in list · {len(countries)} major economies"


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
                                     "so prices come from Eurostat's euro-area HICP (all items, monthly index). "
                                     "The index base doesn't matter because the Real Index is normalised to its own mean."))
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


def pct_change(new: float, old: float) -> float:
    """Percentage change from old to new; NaN if either is missing or old is zero."""
    try:
        if np.isfinite(new) and np.isfinite(old) and old != 0:
            return (new / old - 1.0) * 100.0
    except TypeError:
        pass
    return float("nan")


def value_at(w: pd.DataFrame, col: str, ts: pd.Timestamp) -> float:
    if ts in w.index:
        v = w.at[ts, col]
        return float(v) if pd.notna(v) else float("nan")
    return float("nan")


# =============================================================================
# Charts
# =============================================================================
def make_chart(w: pd.DataFrame, mode: str, ccy1: str, ccy2: str) -> go.Figure:
    """Plotly chart. mode in {"Real", "Nominal", "Both"}. S(t) is never normalised."""
    x = w.index
    nominal_axis = f"{ccy2} per 1 {ccy1}"
    both = mode == "Both"
    fig = make_subplots(specs=[[{"secondary_y": True}]]) if both else go.Figure()

    def add(trace, secondary: bool = False):
        if both:
            fig.add_trace(trace, secondary_y=secondary)
        else:
            fig.add_trace(trace)

    if mode in ("Real", "Both"):
        ri = w["Real_Index"]
        # Invisible baseline at 100 so the Real Index can be shaded against it.
        add(go.Scatter(x=x, y=np.full(len(x), 100.0), mode="lines", line=dict(width=0),
                       hoverinfo="skip", showlegend=False))
        add(go.Scatter(
            x=x, y=ri, mode="lines", name="Real Index (window mean = 100)",
            line=dict(color=REAL_COLOR, width=2.6, shape="spline", smoothing=0.3),
            fill="tonexty", fillcolor=REAL_FILL, connectgaps=False,
            hovertemplate="Real Index: <b>%{y:.2f}</b><extra></extra>",
        ))
        valid_ri = ri.dropna()
        if len(valid_ri) > 2:
            pk, tr, last = valid_ri.idxmax(), valid_ri.idxmin(), valid_ri.index[-1]
            add(go.Scatter(
                x=[pk, tr], y=[valid_ri[pk], valid_ri[tr]], mode="markers+text",
                text=[f"Peak {valid_ri[pk]:.1f}", f"Trough {valid_ri[tr]:.1f}"],
                textposition=["top center", "bottom center"],
                textfont=dict(size=11, color=MUTED),
                marker=dict(size=8, color="#fff", line=dict(color=REAL_COLOR, width=2)),
                hoverinfo="skip", showlegend=False,
            ))
            add(go.Scatter(
                x=[last], y=[valid_ri[last]], mode="markers",
                marker=dict(size=11, color=REAL_COLOR, line=dict(color="#fff", width=3)),
                hoverinfo="skip", showlegend=False,
            ))
        # Dashed reference line at 100 on the (left) Real Index axis.
        fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=100, y1=100,
                      line=dict(color="#94A3B8", dash="dot", width=1.2))

    if mode in ("Nominal", "Both"):
        add(go.Scatter(
            x=x, y=w["S"], mode="lines", name=f"Nominal S(t): {nominal_axis}",
            line=dict(color=NOMINAL_COLOR, width=2.2 if both else 2.6), connectgaps=False,
            hovertemplate="S(t): <b>%{y:,.4f}</b><extra></extra>",
        ), secondary=True)
        s_valid = w["S"].dropna()
        if not s_valid.empty:
            add(go.Scatter(
                x=[s_valid.index[-1]], y=[s_valid.iloc[-1]], mode="markers",
                marker=dict(size=10, color=NOMINAL_COLOR, line=dict(color="#fff", width=3)),
                hoverinfo="skip", showlegend=False,
            ), secondary=True)

    axis_font = dict(size=12, color=MUTED)
    if both:
        fig.update_yaxes(title_text="Real Index", title_font=dict(color=REAL_COLOR), tickfont=axis_font,
                         gridcolor=GRID, zeroline=False, secondary_y=False)
        fig.update_yaxes(title_text=nominal_axis, title_font=dict(color=NOMINAL_COLOR), tickfont=axis_font,
                         showgrid=False, zeroline=False, secondary_y=True)
    else:
        fig.update_yaxes(title_text="Real Index (window mean = 100)" if mode == "Real" else nominal_axis,
                         tickfont=axis_font, gridcolor=GRID, zeroline=False)

    fig.update_xaxes(
        showgrid=False, linecolor="#CBD5E1", tickfont=axis_font,
        showspikes=True, spikecolor="#94A3B8", spikethickness=1, spikedash="dot", spikemode="across",
        rangeslider=dict(visible=True, thickness=0.07, bgcolor="#F1F5F9", bordercolor="#E2E8F0"),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(count=20, label="20Y", step="year", stepmode="backward"),
                dict(step="all", label="Max"),
            ],
            bgcolor="#F1F5F9", activecolor="#E0E7FF", bordercolor="#E2E8F0", borderwidth=1,
            font=dict(size=11, color="#334155"), x=0, y=1.08,
        ),
    )
    fig.update_layout(
        template="plotly_white",
        height=560,
        hovermode="x unified",
        showlegend=True,
        font=dict(family="Inter, sans-serif", size=13, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#0F172A",
                        font=dict(family="Inter, sans-serif", size=12, color="#F8FAFC")),
        margin=dict(l=56, r=56 if both else 24, t=56, b=16),
    )
    return fig


# =============================================================================
# UI building blocks
# =============================================================================
def currency_label(iso: str, names: dict[str, str]) -> str:
    return CURRENCY_CODES.get(iso, f"{names.get(iso, iso)} currency")


def fmt_num(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"


def fmt_pct(v: float) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:+.1f}%"


def tone(v: float) -> str:
    if not np.isfinite(v) or abs(v) < 0.05:
        return "neu"
    return "pos" if v > 0 else "neg"


def delta_pill(v: float, suffix: str) -> str:
    if not np.isfinite(v):
        return f'<span class="pill neu">—</span>{esc(suffix)}'
    arrow = "▲" if v > 0 else ("▼" if v < 0 else "•")
    return f'<span class="pill {tone(v)}">{arrow} {abs(v):.1f}%</span>{esc(suffix)}'


def kpi_card(label: str, value: str, sub_html: str, icon: str) -> str:
    return (f'<div class="kpi"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span>'
            f'<span class="kpi-icon">{icon}</span></div><div class="kpi-value">{esc(value)}</div>'
            f'<div class="kpi-sub">{sub_html}</div></div>')


def notes_html(notes: list[tuple[str, str]]) -> str:
    if not notes:
        return ""
    items = "".join(f'<div class="note {kind}">{md_to_html(text)}</div>' for kind, text in notes)
    return f'<div class="notes">{items}</div>'


def decomposition_html(real_chg: float, nom_chg: float, rel_chg: float, name1: str, name2: str) -> str:
    rows = [
        ("Real exchange rate", real_chg),
        ("Nominal rate · S", nom_chg),
        (f"Relative prices · {name1} ÷ {name2}", rel_chg),
    ]
    finite = [abs(v) for _, v in rows if np.isfinite(v)]
    scale = max(finite) if finite and max(finite) > 0 else 1.0
    body = ""
    for label, v in rows:
        width = 0 if not np.isfinite(v) else max(3.0, abs(v) / scale * 100)
        body += (f'<div class="dec-row"><div class="dec-label">{esc(label)}</div>'
                 f'<div class="dec-val {tone(v)}">{fmt_pct(v)}</div>'
                 f'<div class="dec-track"><div class="dec-bar {tone(v)}" style="width:{width:.1f}%"></div></div></div>')
    return (f'<div class="card"><div class="eyebrow">What drove it</div>{body}'
            f'<div class="dec-note">Effects multiply: (1 + real) = (1 + nominal) × (1 + relative prices). '
            f'Measured from the first to the last valid month in the window.</div></div>')


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


def _set_pair(a: str, b: str) -> None:
    st.session_state["iso1"], st.session_state["iso2"] = a, b


def _swap_pair() -> None:
    st.session_state["iso1"], st.session_state["iso2"] = st.session_state["iso2"], st.session_state["iso1"]


# =============================================================================
# App
# =============================================================================
def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} · Real & Nominal Exchange Rates", page_icon="💱",
                       layout="wide", initial_sidebar_state="expanded")
    apply_styling()

    countries, list_source = get_country_list()
    codes = sorted(countries, key=lambda c: countries[c].lower())
    if st.session_state.get("iso1") not in codes:
        st.session_state["iso1"] = "USA" if "USA" in codes else codes[0]
    if st.session_state.get("iso2") not in codes:
        st.session_state["iso2"] = "G163" if "G163" in codes else codes[1]

    # ---- Sidebar --------------------------------------------------------------------
    with st.sidebar:
        render(f"""
        <div class="sb-brand"><div class="sb-logo">R</div>
        <div><div class="sb-name">{APP_NAME}</div><div class="sb-tag">Exchange-rate analytics</div></div></div>
        <div class="sb-section">Currency pair</div>
        """)
        fmt = lambda c: f"{countries.get(c, c)} ({c})"  # noqa: E731
        st.selectbox("Country 1 (base)", codes, key="iso1", format_func=fmt)
        st.button("⇅  Swap countries", on_click=_swap_pair, use_container_width=True)
        st.selectbox("Country 2 (quote)", codes, key="iso2", format_func=fmt)

        render('<div class="sb-section">Quick pairs</div>')
        qcols = st.columns(2)
        for i, (a, b, label) in enumerate(p for p in QUICK_PAIRS if p[0] in codes and p[1] in codes):
            qcols[i % 2].button(label, key=f"qp_{a}_{b}", on_click=_set_pair, args=(a, b),
                                use_container_width=True)

        render('<div class="sb-section">Sample period</div>')
        period_slot = st.container()

        render('<div class="sb-section">Settings</div>')
        with st.expander("Advanced", expanded=False):
            invert_er = st.checkbox(
                "Invert IMF exchange rates",
                value=False,
                help="Leave unticked normally. Tick only if the IMF series turns out to be USD per local "
                     "currency (e.g. USA vs GBR shows ~1.3 instead of ~0.75 GBP per USD).",
            )
        st.caption(f"Countries: {list_source}")

    iso1, iso2 = st.session_state["iso1"], st.session_state["iso2"]
    name1, name2 = countries.get(iso1, iso1), countries.get(iso2, iso2)
    ccy1, ccy2 = currency_label(iso1, countries), currency_label(iso2, countries)

    hero_slot = st.empty()
    hero_slot.markdown(_html(hero_html(ccy1, ccy2, name1, name2)), unsafe_allow_html=True)

    if iso1 == iso2:
        st.error("Country 1 and Country 2 are the same. Choose two different countries, or pick a quick pair.")
        st.stop()

    with st.spinner(f"Fetching IMF data for {name1} and {name2}…"):
        df, messages, sources = build_dataframe(iso1, iso2, countries, invert_er)

    notes: list[tuple[str, str]] = []
    for level, text in messages:
        if level == "error":
            st.error(text)
        else:
            notes.append(("warn" if level == "warning" else "info", text))

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
    with period_slot:
        if len(month_labels) >= 2:
            start_lbl, end_lbl = st.select_slider(
                "Start – end month", options=month_labels,
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
    latest_s_ts = s_nonnull.index[-1] if not s_nonnull.empty else last_v
    latest_s_month = latest_s_ts.strftime("%Y-%m") if not s_nonnull.empty else "n/a"
    latest_idx = float(v["Real_Index"].iloc[-1])

    # 12-month changes
    idx_12m = pct_change(latest_idx, value_at(w, "Real_Index", last_v - pd.DateOffset(months=12)))
    s_12m = pct_change(latest_s, value_at(w, "S", latest_s_ts - pd.DateOffset(months=12)))

    # Decomposition from first to last valid month: RER = S × (P1/P2)
    real_chg = pct_change(value_at(w, "RER", last_v), value_at(w, "RER", first_v))
    nom_chg = pct_change(value_at(w, "S", last_v), value_at(w, "S", first_v))
    rel_chg = pct_change(value_at(w, "P1", last_v) / value_at(w, "P2", last_v),
                         value_at(w, "P1", first_v) / value_at(w, "P2", first_v))

    dev = latest_idx - 100.0
    pctile = float((v["Real_Index"] <= latest_idx).mean() * 100)
    pk_ts, tr_ts = v["Real_Index"].idxmax(), v["Real_Index"].idxmin()

    # ---- Hero with headline stats ----------------------------------------------------
    hero_slot.markdown(_html(hero_html(ccy1, ccy2, name1, name2, stats=[
        (f"1 {ccy1} =", f"{fmt_num(latest_s)} {ccy2}" if len(ccy2) <= 4 else fmt_num(latest_s), f"as of {latest_s_month}"),
        ("Real Index", f"{latest_idx:.1f}", f"{fmt_pct(dev)} vs window mean"),
    ])), unsafe_allow_html=True)

    # ---- Coverage notes ----------------------------------------------------------------
    if first_all > START_TS:
        late = [label for col, label in coverage_labels.items()
                if df[col].first_valid_index() is not None and df[col].first_valid_index() >= first_all]
        notes.append(("warn", f"**Limited coverage:** months with all four series begin in **{first_all:%Y-%m}**, "
                              f"not 1976-01. Latest-starting series: {', '.join(late) if late else 'n/a'}."))
    if "G163" in (iso1, iso2):
        notes.append(("info", "**Euro area:** the euro (and its IMF exchange rate) begins in 1999, so euro-area "
                              "comparisons cannot start earlier."))
    n_window = len(w)
    if n_valid < n_window:
        notes.append(("warn", f"**Gaps:** {n_window - n_valid} of {n_window} months in the window lack S, P1 or P2. "
                              "They are shown as gaps and excluded from RER* (no interpolation or forward-filling)."))
    if n_valid < 36:
        notes.append(("warn", f"**Short sample:** only {n_valid} valid months, so RER* (and the Real Index level) "
                              "is sensitive to the window."))

    tab_overview, tab_data, tab_method = st.tabs(["Overview", "Data", "Methodology"])

    # =========================== Overview ===========================
    with tab_overview:
        render(notes_html(notes))

        cards = [
            kpi_card("Real Index", f"{latest_idx:.2f}", delta_pill(idx_12m, " vs 12m ago"), "📈"),
            kpi_card(f"S · {ccy2} per {ccy1}", fmt_num(latest_s), delta_pill(s_12m, " vs 12m ago"), "💱"),
            kpi_card("RER*", fmt_num(rer_star), "Window mean of S·P1/P2", "⚖️"),
            kpi_card("Valid months", f"{n_valid:,}", f"of {n_window:,} in window", "🗓️"),
            kpi_card("First valid month", first_v.strftime("%b %Y"), first_v.strftime("%Y-%m"), "⏮️"),
            kpi_card("Last valid month", last_v.strftime("%b %Y"), last_v.strftime("%Y-%m"), "⏭️"),
        ]
        render(f'<div class="kpi-grid">{"".join(cards)}</div>')

        direction = "appreciated" if real_chg > 0 else "depreciated"
        chg_cls = "pos" if real_chg > 0 else "neg"
        head = (f'{esc(ccy1)} has <span class="{chg_cls}">{direction} {abs(real_chg):.1f}%</span> '
                f'in real terms against {esc(ccy2)}') if np.isfinite(real_chg) else "Real exchange rate"
        body = (f"Between {first_v:%b %Y} and {last_v:%b %Y}, the nominal rate moved {fmt_pct(nom_chg)} and "
                f"relative prices ({esc(name1)} vs {esc(name2)}) moved {fmt_pct(rel_chg)}. "
                f"An increase in the Real Index means {esc(name1)}'s currency has appreciated in real terms "
                f"against {esc(name2)}'s currency.")
        chips = [
            f"{abs(dev):.1f}% {'above' if dev >= 0 else 'below'} window average",
            f"Higher than {pctile:.0f}% of months",
            f"Peak {pk_ts:%b %Y} · {v.at[pk_ts, 'Real_Index']:.1f}",
            f"Trough {tr_ts:%b %Y} · {v.at[tr_ts, 'Real_Index']:.1f}",
        ]
        insight = (f'<div class="card"><div class="eyebrow">Insight</div><div class="insight-head">{head}</div>'
                   f'<div class="insight-body">{body}</div>'
                   f'<div class="chips">{"".join(f"<span class=chip>{esc(c)}</span>" for c in chips)}</div></div>')

        c_left, c_right = st.columns([1.55, 1], gap="medium")
        with c_left:
            render(insight)
        with c_right:
            render(decomposition_html(real_chg, nom_chg, rel_chg, ccy1 if ccy1 in CURRENCY_CODES.values() else name1,
                                      ccy2 if ccy2 in CURRENCY_CODES.values() else name2))

        h_left, h_right = st.columns([1, 1], vertical_alignment="bottom")
        with h_left:
            render(f'<div class="section-head"><div><div class="section-title">Exchange-rate history</div>'
                   f'<div class="section-meta">{start_lbl} → {end_lbl} · monthly · {esc(ccy2)} per 1 {esc(ccy1)}</div>'
                   f'</div></div>')
        with h_right:
            mode = st.radio("Chart", ["Real", "Nominal", "Both"], index=2, horizontal=True,
                            label_visibility="collapsed")
        try:
            st.plotly_chart(
                make_chart(w, mode, ccy1, ccy2), use_container_width=True,
                config={
                    "displaylogo": False,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
                    "toImageButtonOptions": {"filename": f"RealFX_{iso1}_{iso2}_{start_lbl}_{end_lbl}", "scale": 2},
                },
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not draw the chart: {exc}")

    # =========================== Data ===========================
    out = w.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m"))
    out = out[["date", "E1_LCU_per_USD", "E2_LCU_per_USD", "S", "P1", "P2", "RER",
               "valid_flag", "RER_star", "Real_Index"]]
    with tab_data:
        d_left, d_right = st.columns([3, 1], vertical_alignment="bottom")
        with d_left:
            render(f'<div class="section-head"><div><div class="section-title">Monthly observations</div>'
                   f'<div class="section-meta">{len(out):,} months · {start_lbl} → {end_lbl} · newest first</div>'
                   f'</div></div>')
        with d_right:
            st.download_button(
                "⬇  Download CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name=f"RER_{iso1}_{iso2}_{start_lbl}_{end_lbl}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
        st.dataframe(
            out.iloc[::-1],
            hide_index=True,
            use_container_width=True,
            height=480,
            column_config={
                "date": st.column_config.TextColumn("Month"),
                "E1_LCU_per_USD": st.column_config.NumberColumn(f"E1 · {ccy1}/USD", format="%.4f"),
                "E2_LCU_per_USD": st.column_config.NumberColumn(f"E2 · {ccy2}/USD", format="%.4f"),
                "S": st.column_config.NumberColumn(f"S · {ccy2}/{ccy1}", format="%.4f"),
                "P1": st.column_config.NumberColumn(f"P1 · {ccy1} prices", format="%.2f"),
                "P2": st.column_config.NumberColumn(f"P2 · {ccy2} prices", format="%.2f"),
                "RER": st.column_config.NumberColumn("RER", format="%.4f"),
                "valid_flag": st.column_config.CheckboxColumn("Valid"),
                "RER_star": st.column_config.NumberColumn("RER*", format="%.4f"),
                "Real_Index": st.column_config.NumberColumn("Real Index", format="%.2f"),
            },
        )
        render('<div class="section-head"><div><div class="section-title">Coverage by series</div>'
               '<div class="section-meta">Full download, before the sample window is applied</div></div></div>')
        st.dataframe(series_coverage_table(df, coverage_labels), hide_index=True, use_container_width=True)

    # =========================== Methodology ===========================
    with tab_method:
        m_left, m_right = st.columns([1.3, 1], gap="large")
        with m_left:
            st.markdown(f"""
#### Formulas

- $E_i(t)$ = IMF period-average exchange rate: units of country *i* currency per 1 USD.
  For the United States, $E = 1$ every month.
- $S(t) = E_2(t) / E_1(t)$ = units of **Country 2** currency per 1 unit of **Country 1** currency
  (the dollar cancels: LCU2/USD ÷ LCU1/USD = LCU2/LCU1).
- $P_1(t), P_2(t)$ = monthly headline consumer price indices.
- $RER(t) = S(t)\\,P_1(t)/P_2(t)$
- $RER^*$ = mean of $RER(t)$ over valid months in the selected window
  (S, P1 and P2 all present and positive).
- $\\text{{Real Index}}(t) = 100 \\times RER(t)/RER^*$

**Direction:** an increase in the Real Index means Country 1's currency has appreciated in real terms
against Country 2's currency.

**Check:** mean Real Index over valid months in this window = **{v['Real_Index'].mean():.4f}** (should be 100).
""")
        with m_right:
            st.markdown(f"""
#### Data sources

- **IMF SDMX 3.0 API** · `{IMF_BASE_URL}`
- Exchange rates · dataflow `{AGENCY},{ER_DATAFLOW},{DATAFLOW_VERSIONS[ER_DATAFLOW]}`,
  key `{{ISO3}}.XDC_USD.PA_RT.M` (fallback `{{ISO3}}.USD_XDC.PA_RT.M`)
- Prices · dataflow `{AGENCY},{CPI_DATAFLOW},{DATAFLOW_VERSIONS[CPI_DATAFLOW]}`, key `{{ISO3}}.CPI._T.IX.M`
- Euro area (`G163`) prices · IMF HICP key if available, otherwise **Eurostat HICP**
  (all items, monthly index)

#### Limitations

Price-index bases differ, so only movements in the index are meaningful. Rebased series can show breaks.
Gaps stay gaps. Euro-area exchange rates start in 1999. Data are cached for 24 hours.
""")
        with st.expander("URLs used in this run"):
            for label, url in sources:
                st.markdown(f"- {label}: `{url}`")

    render(f"""
    <div class="app-footer">
    <div>{APP_NAME} · Data: IMF SDMX 3.0 (CPI, ER) · Eurostat HICP fallback for the euro area</div>
    <div>For research and education · not investment advice</div>
    </div>
    """)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}. Try reloading the page or choosing different countries.")
