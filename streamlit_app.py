# ============================================================
# NSE DASHBOARD - FINAL CLOUD SAFE VERSION
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.diagnostic import breaks_cusumolsresid

st.set_page_config(page_title="NSE Dashboard", layout="wide")

# ============================================================
# TICKERS
# ============================================================

NIFTY_50_TICKERS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
    "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","SUNPHARMA.NS"
]

# ============================================================
# SAFE DOWNLOAD
# ============================================================

@st.cache_data(ttl=600)
def safe_download(ticker, period):
    try:
        df = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        # Drop MultiIndex completely
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Remove duplicate columns if any
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    except:
        return pd.DataFrame()

# ============================================================
# UI
# ============================================================

tabs = st.tabs(["Stock Analytics (3 Months)", "GARCH Screening"])

# ============================================================
# STOCK ANALYTICS (FINAL FIXED)
# ============================================================

with tabs[0]:

    st.header("Stock Analytics - Last 3 Months")

    selected_stock = st.selectbox("Select Stock", NIFTY_50_TICKERS)

    data = safe_download(selected_stock, "3mo")

    if not data.empty and "Close" in data.columns:

        data = data.dropna().copy()

        # Extract clean numpy arrays (bulletproof method)
        close_array = np.array(data["Close"]).astype(float)
        high_array = np.array(data["High"]).astype(float)
        volume_array = np.array(data["Volume"]).astype(float)

        # Indicators
        rsi_indicator = ta.momentum.RSIIndicator(pd.Series(close_array))
        macd_indicator = ta.trend.MACD(pd.Series(close_array))

        data["RSI"] = rsi_indicator.rsi()
        data["MACD"] = macd_indicator.macd()

        # 🔥 GUARANTEED SCALARS
        ltp = float(close_array[-1])
        volume = int(volume_array[-1])
        high_3m = float(np.max(high_array))

        col1, col2, col3 = st.columns(3)
        col1.metric("LTP", f"{ltp:.2f}")
        col2.metric("Volume", f"{volume:,}")
        col3.metric("3M High", f"{high_3m:.2f}")

        st.subheader("Price")
        st.line_chart(close_array)

        st.subheader("RSI")
        st.line_chart(data["RSI"])

        st.subheader("MACD")
        st.line_chart(data["MACD"])

    else:
        st.warning("Stock data unavailable.")

# ============================================================
# GARCH SCREENING
# ============================================================

def get_log_returns(prices):
    return 100 * np.log(prices[1:] / prices[:-1])

with tabs[1]:

    st.header("GARCH Quant Screening")

    if st.button("Run GARCH"):

        results = []

        for ticker in NIFTY_50_TICKERS:

            df = safe_download(ticker, "3y")

            if df.empty or "Close" not in df.columns:
                continue

            prices = np.array(df["Close"]).astype(float)

            if len(prices) < 300:
                continue

            returns = get_log_returns(prices)

            jb_stat, jb_p = jarque_bera(returns)
            adf_stat, adf_p, *_ = adfuller(returns)
            arch_stat, arch_p, *_ = het_arch(returns)
            cusum_stat, cusum_p, _ = breaks_cusumolsresid(returns)

            volatility = np.std(returns) * np.sqrt(252)
            arch_strength = -np.log(arch_p) if arch_p > 0 else 0
            suitable = adf_p < 0.05 and arch_p < 0.05

            results.append({
                "Ticker": ticker,
                "Volatility": round(volatility,4),
                "ARCH Strength": round(arch_strength,2),
                "GARCH Suitable": suitable
            })

        if results:
            results_df = pd.DataFrame(results)
            results_df.sort_values("ARCH Strength", ascending=False, inplace=True)

            st.dataframe(results_df, use_container_width=True)
            st.success("GARCH Screening Completed")
        else:
            st.warning("No valid stocks processed.")
