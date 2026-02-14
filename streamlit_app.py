# ============================================================
# NSE INSTITUTIONAL DASHBOARD + GARCH ENGINE
# Cloud Safe Version
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
# SAFE DOWNLOAD FUNCTION
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
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
run_garch = st.sidebar.button("Run GARCH Screening")

tabs = st.tabs([
    "Market Overview",
    "Stock Analytics",
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
            last = data["Close"].iloc[-1]
            prev = data["Close"].iloc[-2]
            change = ((last - prev) / prev) * 100
            st.metric(name, round(last,2), f"{round(change,2)}%")

    bulk = yf.download(NIFTY_50_TICKERS, period="5d", interval="1d",
                       progress=False, auto_adjust=True, threads=False)

    if isinstance(bulk.columns, pd.MultiIndex):

        adv, dec = 0, 0
        movers = []

        for ticker in NIFTY_50_TICKERS:
            try:
                df = bulk.xs(ticker, axis=1, level=1)
                if len(df) >= 2:
                    last = df["Close"].iloc[-1]
                    prev = df["Close"].iloc[-2]
                    pct = ((last - prev) / prev) * 100
                    movers.append((ticker, pct))
                    if pct > 0:
                        adv += 1
                    else:
                        dec += 1
            except:
                continue

        st.metric("Advance / Decline", f"{adv} / {dec}")

        movers_df = pd.DataFrame(movers, columns=["Ticker","% Change"])
        movers_df.sort_values("% Change", ascending=False, inplace=True)

        st.subheader("Top Gainers")
        st.dataframe(movers_df.head(5), use_container_width=True)

        st.subheader("Top Losers")
        st.dataframe(movers_df.tail(5), use_container_width=True)

# ============================================================
# STOCK ANALYTICS (FIXED)
# ============================================================

with tabs[1]:
    st.header("Stock Analytics")

    selected_stock = st.selectbox("Select Stock", NIFTY_50_TICKERS)

    data = safe_download(selected_stock, period="1y")

    if not data.empty and "Close" in data.columns:

        data = data.dropna().copy()

        close_series = data["Close"]

        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(close_series)
        data["RSI"] = rsi_indicator.rsi()

        # MACD
        macd_indicator = ta.trend.MACD(close_series)
        data["MACD"] = macd_indicator.macd()

        latest = data.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("LTP", round(latest["Close"], 2))
        col2.metric("Volume", int(latest["Volume"]))
        col3.metric("52W High", round(data["High"].max(), 2))

        st.line_chart(data[["Close"]])
        st.line_chart(data[["RSI"]])
        st.line_chart(data[["MACD"]])

    else:
        st.warning("Stock data unavailable.")

# ============================================================
# GARCH SCREENING
# ============================================================

def get_log_returns(price_series):
    return 100 * np.log(price_series / price_series.shift(1)).dropna()

with tabs[2]:

    st.header("GARCH Quant Screening")

    if run_garch:

        results = []

        for ticker in NIFTY_50_TICKERS:

            data = safe_download(ticker, start=start_date, end=end_date)

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
