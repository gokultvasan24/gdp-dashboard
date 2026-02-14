# ============================================================
# NSE INSTITUTIONAL DASHBOARD - FULL CUSTOM SECTOR VERSION
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import ta

st.set_page_config(page_title="NSE Institutional Dashboard", layout="wide")

# ============================================================
# YOUR CUSTOM SECTOR MAP
# ============================================================

RAW_SECTOR_DATA = [
    ("Metals & Mining","HINDALCO"),
    ("FMCG","HINDUNILVR"),
    ("Services","ETERNAL"),
    ("Metals & Mining","ADANIENT"),
    ("Oil & Gas","ONGC"),
    ("Automobile","HEROMOTOCO"),
    ("Metals & Mining","TATASTEEL"),
    ("Consumer Durables","TITAN"),
    ("Oil & Gas","COALINDIA"),
    ("Services","ADANIPORTS"),
    ("Power","POWERGRID"),
    ("Information Technology","WIPRO"),
    ("FMCG","NESTLEIND"),
    ("Oil & Gas","RELIANCE"),
    ("Information Technology","TCS"),
    ("Captial Goods","BEL"),
    ("Financial Services","HDFCBANK"),
    ("Consumer Durables","ASIANPAINT"),
    ("Financial Services","SHRIRAMFIN"),
    ("Automobile","M&M"),
    ("FMCG","TATACONSUM"),
    ("Construction Materials","GRASIM"),
    ("Metals & Mining","JSWSTEEL"),
    ("Automobile","TMPV"),
    ("Financial Services","JIOFIN"),
    ("Information Technology","HCLTECH"),
    ("Information Technology","INFY"),
    ("FMCG","ITC"),
    ("Power","NTPC"),
    ("Telecommunication","BHARTIARTL"),
    ("Financial Services","KOTAKBANK"),
    ("Financial Services","ICICIBANK"),
    ("Automobile","BAJAJ-AUTO"),
    ("Healthcare","SUNPHARMA"),
    ("Retail","TRENT"),
    ("Financial Services","HDFCLIFE"),
    ("Automobile","MARUTI"),
    ("Financial Services","AXISBANK"),
    ("Financial Services","BAJAJFINSV"),
    ("Healthcare","DRREDDY"),
    ("Construction Materials","ULTRACEMCO"),
    ("Construction","LT"),
    ("Healthcare","APOLLOHOSP"),
    ("Information Technology","TECHM"),
    ("Healthcare","CIPLA"),
    ("Financial Services","SBIN"),
    ("Financial Services","INDUSINDBK"),
    ("Financial Services","SBILIFE"),
    ("Automobile","EICHERMOT"),
    ("Financial Services","BAJFINANCE")
]

# Convert to dict
SECTOR_MAP = {}
for sector, symbol in RAW_SECTOR_DATA:
    symbol_ns = symbol + ".NS"
    SECTOR_MAP.setdefault(sector, []).append(symbol_ns)

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
    "Market Overview",
    "Sector Performance",
    "Stock Analytics"
])

# ============================================================
# MARKET OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Market Overview")

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

    col1, col2 = st.columns(2)

    col1.subheader("Top Gainers")
    col1.dataframe(movers_df.head(5), use_container_width=True)

    col2.subheader("Top Losers")
    col2.dataframe(movers_df.tail(5), use_container_width=True)

# ============================================================
# SECTOR PERFORMANCE
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
# STOCK ANALYTICS
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
        rsi = ta.momentum.RSIIndicator(pd.Series(close), window=14).rsi().fillna(0)

        # MACD
        macd = ta.trend.MACD(pd.Series(close)).macd().fillna(0)

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
        st.warning("Data unavailable.")
