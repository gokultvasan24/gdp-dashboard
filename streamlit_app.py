# ============================================================
# NSE INSTITUTIONAL DASHBOARD - COMPLETE VERSION
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import ta

st.set_page_config(page_title="NSE Institutional Dashboard", layout="wide")

# ============================================================
# INDEX SYMBOLS
# ============================================================

INDEX_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "NIFTY Bank": "^NSEBANK",
    "India VIX": "^INDIAVIX"
}

# ============================================================
# SECTOR MAPPING
# ============================================================

SECTOR_MAP = {
    "Banking": ["HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS"],
    "IT": ["TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS"],
    "Pharma": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"],
    "Auto": ["MARUTI.NS","M&M.NS","BAJAJ-AUTO.NS"],
    "FMCG": ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS"],
    "Metal": ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS"]
}

ALL_STOCKS = list(set([s for v in SECTOR_MAP.values() for s in v]))

# ============================================================
# SAFE DOWNLOAD
# ============================================================

@st.cache_data(ttl=600)
def safe_download(tickers, period="3mo"):
    df = yf.download(
        tickers,
        period=period,
        progress=False,
        auto_adjust=True,
        threads=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "1️⃣ Market Overview",
    "2️⃣ Sector Performance",
    "3️⃣ Stock Analytics",
    "5️⃣ FII / DII Activity"
])

# ============================================================
# 1️⃣ MARKET OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Market Overview")

    col1, col2, col3 = st.columns(3)

    for i, (name, symbol) in enumerate(INDEX_SYMBOLS.items()):
        data = safe_download(symbol, period="5d")

        if not data.empty and len(data) >= 2:
            close = np.array(data["Close"]).astype(float)
            last = close[-1]
            prev = close[-2]
            change = ((last - prev) / prev) * 100

            if i == 0:
                col1.metric(name, f"{last:.2f}", f"{change:.2f}%")
            elif i == 1:
                col2.metric(name, f"{last:.2f}", f"{change:.2f}%")
            else:
                col3.metric(name, f"{last:.2f}", f"{change:.2f}%")

    # Advance / Decline
    bulk = yf.download(ALL_STOCKS, period="5d",
                       progress=False, auto_adjust=True, threads=False)

    adv, dec = 0, 0
    movers = []

    if isinstance(bulk.columns, pd.MultiIndex):

        for stock in ALL_STOCKS:
            try:
                df = bulk.xs(stock, axis=1, level=1)
                close = np.array(df["Close"]).astype(float)

                pct = ((close[-1] - close[-2]) / close[-2]) * 100
                movers.append((stock, pct))

                if pct > 0:
                    adv += 1
                else:
                    dec += 1
            except:
                continue

    st.metric("Advance / Decline", f"{adv} / {dec}")

    movers_df = pd.DataFrame(movers, columns=["Stock","% Change"])
    movers_df = movers_df.sort_values("% Change", ascending=False)

    colg, coll = st.columns(2)

    colg.subheader("Top Gainers")
    colg.dataframe(movers_df.head(5), use_container_width=True)

    coll.subheader("Top Losers")
    coll.dataframe(movers_df.tail(5), use_container_width=True)

# ============================================================
# 2️⃣ SECTOR PERFORMANCE
# ============================================================

with tabs[1]:

    st.header("Sector Performance")

    sector_perf = {}

    bulk = yf.download(ALL_STOCKS, period="5d",
                       progress=False, auto_adjust=True, threads=False)

    if isinstance(bulk.columns, pd.MultiIndex):

        for sector, stocks in SECTOR_MAP.items():
            changes = []

            for stock in stocks:
                try:
                    df = bulk.xs(stock, axis=1, level=1)
                    close = np.array(df["Close"]).astype(float)
                    pct = ((close[-1] - close[-2]) / close[-2]) * 100
                    changes.append(pct)
                except:
                    continue

            if changes:
                sector_perf[sector] = np.mean(changes)

    df_sector = pd.DataFrame.from_dict(
        sector_perf, orient="index", columns=["% Change"]
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
        st.dataframe(
            df_sector.sort_values("% Change", ascending=False),
            use_container_width=True
        )

# ============================================================
# 3️⃣ STOCK ANALYTICS
# ============================================================

with tabs[2]:

    st.header("Stock-Level Data (3 Months)")

    stock = st.selectbox("Select Stock", ALL_STOCKS)

    df = safe_download(stock, period="3mo")

    if not df.empty and "Close" in df.columns:

        df = df.dropna()

        close = np.array(df["Close"]).astype(float)
        high = np.array(df["High"]).astype(float)
        low = np.array(df["Low"]).astype(float)
        volume = np.array(df["Volume"]).astype(float)

        # VWAP
        typical_price = (high + low + close) / 3
        vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)

        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(pd.Series(close), window=14)
        rsi = rsi_indicator.rsi().fillna(0)

        # MACD
        macd_indicator = ta.trend.MACD(pd.Series(close))
        macd = macd_indicator.macd().fillna(0)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("LTP", f"{close[-1]:.2f}")
        col2.metric("Volume", f"{int(volume[-1]):,}")
        col3.metric("VWAP", f"{vwap[-1]:.2f}")
        col4.metric("52W High", f"{np.max(high):.2f}")

        col5, col6 = st.columns(2)
        col5.metric("RSI", f"{rsi.iloc[-1]:.2f}")
        col6.metric("MACD", f"{macd.iloc[-1]:.2f}")

        st.line_chart(close)

    else:
        st.warning("Data unavailable")

# ============================================================
# 5️⃣ FII / DII ACTIVITY (STRUCTURE READY)
# ============================================================

with tabs[3]:

    st.header("FII / DII Activity")

    st.info("Live NSE FII/DII data requires NSE API integration.")

    # Placeholder institutional data simulation
    data = pd.DataFrame({
        "Participant": ["FII","DII"],
        "Net Buy/Sell (Cr)": [1200, -850]
    })

    st.dataframe(data, use_container_width=True)

    st.subheader("Index Futures Position (Sample)")
    fut = pd.DataFrame({
        "Position": ["Long","Short"],
        "Contracts": [45000, 38000]
    })

    st.dataframe(fut, use_container_width=True)

