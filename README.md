
# Real & Nominal Exchange Rate Explorer

A Streamlit app that downloads monthly CPI and exchange-rate data from the IMF SDMX 3.0 API and plots
the **nominal** and **real** exchange rate between any two countries (or the Euro area).

**Live app:** _paste your Streamlit URL here_

## What it calculates

| Symbol | Definition |
|---|---|
| E_i(t) | IMF period-average rate: units of country *i* currency per 1 USD (USD = 1 for the United States) |
| S(t) | E_2(t) / E_1(t) = units of **Country 2** currency per 1 unit of **Country 1** currency |
| RER(t) | S(t) × P1(t) / P2(t), with P1, P2 = monthly CPI |
| RER* | Mean of RER(t) over *valid* months in the selected window (S, P1, P2 all present and positive) |
| Real Index(t) | 100 × RER(t) / RER* (averages exactly 100 over the window) |

**Direction convention:** an increase in the Real Index means Country 1's currency has **appreciated in real
terms** against Country 2's currency. S(t) is plotted in actual currency units and is never normalised.

## Data

- Source: IMF SDMX 3.0 API, `https://api.imf.org/external/sdmx/3.0`, agency `IMF.STA`
- CPI: dataflow `CPI`, key `{ISO3}.CPI._T.IX.M`
- Exchange rates: dataflow `ER`, key `{ISO3}.XDC_USD.PA_RT.M`
- Euro area code: `G163`
- Monthly data from January 1976 where available; cached for 24 hours.
- All URL construction lives in `build_data_urls()` in `app.py`. If the IMF changes its API, edit that one function.

**Limitations:** Euro-area exchange rates begin in 1999. Some countries have CPI gaps, late starts, or rebased
series. Gaps are left missing (no interpolation). Because CPI bases are arbitrary, only movements in the Real
Index are meaningful, not its level.

## Deploy using only a web browser

1. **Create a GitHub repository**
   - Sign in at github.com → **New repository** → name it (e.g. `rer-explorer`) → **Public** → **Create repository**.
   - Click **Add file → Create new file**, name it `app.py`, paste the code, click **Commit changes**.
   - Repeat for `requirements.txt` and `README.md`.
2. **Deploy on Streamlit Community Cloud**
   - Go to share.streamlit.io and sign in with GitHub.
   - Click **Create app** → **Deploy a public app from GitHub**.
   - Repository: your repo; Branch: `main`; Main file path: `app.py`.
   - Open **Advanced settings** and choose **Python 3.12**.
   - Click **Deploy**. The first build takes a few minutes.
3. Copy the app URL into the "Live app" line above.

## Troubleshooting

- **IMF API error (HTTP 4xx/5xx or network error):** the app lists every URL it tried. Paste one into your browser.
  If the IMF has changed dataflow IDs, versions, or URL patterns, update `build_data_urls()` (and the key templates
  at the top of `app.py`). If the IMF begins requiring a subscription key, the requests will return 401/403.
- **A country has no data:** not every country publishes monthly CPI or exchange rates to the IMF. Choose another
  country or check the "Data coverage by series" panel.
- **Empty or gappy charts:** the selected window may contain no months where both exchange rates and both CPIs
  overlap. Widen the sample period, or pick countries with longer coverage. Use **Manage app → Reboot** to clear
  the cache after an IMF outage.
