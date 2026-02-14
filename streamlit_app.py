# ============================================================
# NSE INSTITUTIONAL DASHBOARD + GARCH ENGINE (Cloud Safe)
# Author: Gokul Thanigaivasan
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
# SIDEBAR
# ============================================================

st.sidebar.header("Configuration")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
run_garch = st.sidebar.button("Run GARCH Screening")

tabs = st.tabs([
    "Market Overview",
    "Sector Performance",
    "Stock Analytics",
    "GARCH Screening"
])

# ============================================================
# SAFE DATA DOWNLOAD FUNCTION
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
# 1️⃣ MARKET OVERVIEW
# ============================================================

@st.cache_data(ttl=600)
def get_market_snapshot():
    snapshot = {}

    for name, ticker in INDEX_SYMBOLS.items():
        data = safe_download(ticker, period="5d", interval="1d")

        if data.empty or "Close" not in data.columns or len(data) < 2:
            continue

        last = data["Close"].iloc[-1]
        prev = data["Close"].iloc[-2]
        change = ((last - prev) / prev) * 100

        snapshot[name] = (last, change)

    return snapshot


@st.cache_data(ttl=600)
def get_nifty_bulk_data():
    return safe_download(NIFTY_50_TICKERS, period="5d", interval="1d")


with tabs[0]:
    st.header("Market Overview")

    snapshot = get_market_snapshot()

    if snapshot:
        cols = st.columns(len(snapshot))
        for i, (name, values) in enumerate(snapshot.items()):
            cols[i].metric(
                label=name,
                value=round(values[0], 2),
                delta=f"{round(values[1], 2)}%"
            )
    else:
        st.warning("Market data unavailable.")

    bulk_data = get_nifty_bulk_data()

    if not bulk_data.empty:
        adv, dec = 0, 0
        movers = []

        for ticker in NIFTY_50_TICKERS:
            try:
                df = bulk_data.xs(ticker, axis=1, level=1)
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
# 2️⃣ SECTOR PERFORMANCE
# ============================================================

SECTORS = {
    "Banking": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS"],
    "IT": ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS"],
    "Pharma": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS"],
    "Auto": ["MARUTI.NS","M&M.NS","HEROMOTOCO.NS","BAJAJ-AUTO.NS"],
    "FMCG": ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS"],
    "Metal": ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS"]
}

with tabs[1]:
    st.header("Sector Performance")

    bulk_data = get_nifty_bulk_data()

    sector_perf = {}

    for sector, stocks in SECTORS.items():
        changes = []
        for s in stocks:
            try:
                df = bulk_data.xs(s, axis=1, level=1)
                if len(df) >= 2:
                    last = df["Close"].iloc[-1]
                    prev = df["Close"].iloc[-2]
                    pct = ((last - prev) / prev) * 100
                    changes.append(pct)
            except:
                continue

        if changes:
            sector_perf[sector] = np.mean(changes)

    df_sector = pd.DataFrame.from_dict(
        sector_perf, orient='index', columns=['% Change']
    )

    if not df_sector.empty:
        fig = px.imshow(
            df_sector.T,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sector Strength Ranking")
        st.dataframe(df_sector.sort_values("% Change", ascending=False))
    else:
        st.warning("Sector data unavailable.")

# ============================================================
# 3️⃣ STOCK ANALYTICS
# ============================================================

with tabs[2]:
    st.header("Stock Analytics")

    selected_stock = st.selectbox("Select Stock", NIFTY_50_TICKERS)

    data = safe_download(selected_stock, period="1y")

    if not data.empty and "Close" in data.columns:

        data.dropna(inplace=True)

        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()

        latest = data.iloc[-1]

        col1, col2, col3 = st.columns(3)
        col1.metric("LTP", round(latest['Close'],2))
        col2.metric("Volume", int(latest['Volume']))
        col3.metric("52W High", round(data['High'].max(),2))

        st.line_chart(data[['Close']])
        st.line_chart(data[['RSI']])
        st.line_chart(data[['MACD']])
    else:
        st.warning("Stock data unavailable.")

# ============================================================
# 4️⃣ GARCH SCREENING
# ============================================================

def get_log_returns(price_series):
    return 100 * np.log(price_series / price_series.shift(1)).dropna()

with tabs[3]:

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
