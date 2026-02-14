# ============================================================
# NSE INSTITUTIONAL DASHBOARD (3 MONTH VERSION - CLOUD SAFE)
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import ta

from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import breaks_cusumolsresid

st.set_page_config(page_title="NSE Institutional Dashboard", layout="wide")

# ============================================================
# CONFIG
# ============================================================

NIFTY_50_TICKERS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS",
    "TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","WIPRO.NS","HCLTECH.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","POWERGRID.NS","NTPC.NS","ONGC.NS",
    "JSWSTEEL.NS","TATASTEEL.NS","TECHM.NS","INDUSINDBK.NS","ADANIENT.NS",
    "ADANIPORTS.NS","GRASIM.NS","BRITANNIA.NS","CIPLA.NS","EICHERMOT.NS",
    "COALINDIA.NS","HEROMOTOCO.NS","DRREDDY.NS","APOLLOHOSP.NS","DIVISLAB.NS",
    "HDFCLIFE.NS","SBILIFE.NS","UPL.NS","BAJAJ-AUTO.NS","TATACONSUM.NS",
    "SHREECEM.NS","M&M.NS","HINDALCO.NS","BPCL.NS","LTIM.NS"
]

INDEX_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX"
}

# ============================================================
# SAFE DOWNLOAD
# ============================================================

@st.cache_data(ttl=600)
def safe_download(tickers, **kwargs):
    try:
        data = yf.download(
            tickers,
            progress=False,
            auto_adjust=True,
            threads=False,
            **kwargs
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    except:
        return pd.DataFrame()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configuration")
run_garch = st.sidebar.button("Run GARCH Screening")

tabs = st.tabs([
    "Market Overview",
    "Stock Analytics (3 Months)",
    "GARCH Screening"
])

# ============================================================
# MARKET OVERVIEW
# ============================================================

with tabs[0]:
    st.header("Market Overview")

    for name, ticker in INDEX_SYMBOLS.items():
        data = safe_download(ticker, period="5d", interval="1d")

        if not data.empty and len(data) >= 2:
            last = float(data["Close"].iloc[-1])
            prev = float(data["Close"].iloc[-2])
            change = ((last - prev) / prev) * 100
            st.metric(name, f"{last:.2f}", f"{change:.2f}%")

# ============================================================
# STOCK ANALYTICS (3 MONTH DATA)
# ============================================================

with tabs[1]:
    st.header("Stock Analytics - Last 3 Months")

    selected_stock = st.selectbox("Select Stock", NIFTY_50_TICKERS)

    # 🔥 3 MONTH PERIOD
    data = safe_download(selected_stock, period="3mo")

    if not data.empty and "Close" in data.columns:

        data = data.dropna().copy()

        close_series = data["Close"]

        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        close_series = pd.Series(close_series).astype(float)
        data["Close"] = close_series

        # Indicators
        rsi_indicator = ta.momentum.RSIIndicator(close_series)
        data["RSI"] = rsi_indicator.rsi()

        macd_indicator = ta.trend.MACD(close_series)
        data["MACD"] = macd_indicator.macd()

        latest = data.iloc[-1]

        ltp = float(latest["Close"])
        volume = int(float(latest["Volume"]))
        high_3m = float(data["High"].max())

        col1, col2, col3 = st.columns(3)
        col1.metric("LTP", f"{ltp:.2f}")
        col2.metric("Volume", f"{volume:,}")
        col3.metric("3M High", f"{high_3m:.2f}")

        st.subheader("Price Chart")
        st.line_chart(data[["Close"]])

        st.subheader("RSI")
        st.line_chart(data[["RSI"]])

        st.subheader("MACD")
        st.line_chart(data[["MACD"]])

    else:
        st.warning("Stock data unavailable.")

# ============================================================
# GARCH SCREENING (3 YEAR DATA FOR STABILITY)
# ============================================================

def get_log_returns(price_series):
    return 100 * np.log(price_series / price_series.shift(1)).dropna()

with tabs[2]:

    st.header("GARCH Quant Screening")

    if run_garch:

        results = []

        for ticker in NIFTY_50_TICKERS:

            data = safe_download(ticker, period="3y")

            if data.empty or "Close" not in data.columns:
                continue

            prices = data["Close"].dropna()

            if len(prices) < 300:
                continue

            returns = get_log_returns(prices)

            jb_stat, jb_p = jarque_bera(returns)
            adf_stat, adf_p, *_ = adfuller(returns)
            arch_stat, arch_p, *_ = het_arch(returns)
            cusum_stat, cusum_p, _ = breaks_cusumolsresid(returns)

            volatility = returns.std() * np.sqrt(252)
            arch_strength = -np.log(arch_p) if arch_p > 0 else 0
            suitable = adf_p < 0.05 and arch_p < 0.05

            results.append({
                "Ticker": ticker,
                "Volatility": round(volatility,4),
                "ARCH p-value": round(arch_p,4),
                "ARCH Strength": round(arch_strength,2),
                "ADF p-value": round(adf_p,4),
                "CUSUM p-value": round(cusum_p,4),
                "GARCH Suitable": suitable
            })

        if results:
            results_df = pd.DataFrame(results)
            results_df.sort_values("ARCH Strength", ascending=False, inplace=True)

            st.dataframe(results_df, use_container_width=True)

            st.download_button(
                "Download GARCH Results",
                results_df.to_csv(index=False),
                file_name="garch_results.csv"
            )

            st.success("GARCH Screening Completed")

        else:
            st.warning("No valid stocks processed.")
