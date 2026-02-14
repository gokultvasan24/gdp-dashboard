# ============================================================
# NSE INSTITUTIONAL DASHBOARD – FULL VERSION
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import ta

st.set_page_config(page_title="NSE Institutional Dashboard", layout="wide")

# ============================================================
# CUSTOM SECTOR MAP (YOUR DATA)
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

SECTOR_MAP = {}
for sector, symbol in RAW_SECTOR_DATA:
    SECTOR_MAP.setdefault(sector, []).append(symbol + ".NS")

ALL_STOCKS = list(set([s for v in SECTOR_MAP.values() for s in v]))

# ============================================================
# CACHE SAFE DOWNLOAD
# ============================================================

@st.cache_data(ttl=600)
def download_data(tickers, period="5d", interval="1d"):
    df = yf.download(
        tickers,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False
    )
    return df

# ============================================================
# TABS
# ============================================================

tabs = st.tabs(["Market Overview", "Sector Performance", "Stock Analytics"])

# ============================================================
# 1️⃣ MARKET OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Market Overview")

    data = download_data(ALL_STOCKS, period="5d")

    adv, dec = 0, 0
    movers = []

    if isinstance(data.columns, pd.MultiIndex):

        for stock in ALL_STOCKS:
            try:
                df = data.xs(stock, axis=1, level=1)
                close = df["Close"].dropna()

                pct = ((close.iloc[-1] - close.iloc[-2]) /
                       close.iloc[-2]) * 100

                movers.append((stock, pct))

                if pct > 0:
                    adv += 1
                else:
                    dec += 1
            except:
                continue

    st.metric("Advance / Decline", f"{adv} / {dec}")

    movers_df = pd.DataFrame(movers, columns=["Stock", "% Change"])
    movers_df = movers_df.sort_values("% Change", ascending=False)

    col1, col2 = st.columns(2)

    col1.subheader("Top Gainers")
    col1.dataframe(movers_df.head(5), use_container_width=True)

    col2.subheader("Top Losers")
    col2.dataframe(movers_df.tail(5), use_container_width=True)

# ============================================================
# 2️⃣ SECTOR PERFORMANCE
# ============================================================

with tabs[1]:

    st.header("Sector Performance")

    data = download_data(ALL_STOCKS, period="5d")

    sector_perf = {}

    if isinstance(data.columns, pd.MultiIndex):

        for sector, stocks in SECTOR_MAP.items():
            changes = []

            for stock in stocks:
                try:
                    df = data.xs(stock, axis=1, level=1)
                    close = df["Close"].dropna()
                    pct = ((close.iloc[-1] - close.iloc[-2]) /
                           close.iloc[-2]) * 100
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

        st.subheader("Sector Ranking")
        st.dataframe(
            df_sector.sort_values("% Change", ascending=False),
            use_container_width=True
        )

# ============================================================
# 3️⃣ STOCK ANALYTICS (5M – 10 DAYS)
# ============================================================

with tabs[2]:

    st.header("Stock-Level Data (5-Min – Last 10 Days)")

    stock = st.selectbox("Select Stock", ALL_STOCKS)

    intraday = download_data(
        stock,
        period="10d",
        interval="5m"
    )

    if not intraday.empty:

        intraday = intraday.dropna()

        close = intraday["Close"]
        high = intraday["High"]
        low = intraday["Low"]
        volume = intraday["Volume"]

        # VWAP
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().fillna(0)

        # MACD
        macd_indicator = ta.trend.MACD(close)
        macd = macd_indicator.macd().fillna(0)

        # 52 Week High
        daily_52w = download_data(stock, period="1y", interval="1d")
        high_52w = daily_52w["High"].max()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("LTP", f"{close.iloc[-1]:.2f}")
        col2.metric("Volume", f"{int(volume.iloc[-1]):,}")
        col3.metric("VWAP", f"{vwap.iloc[-1]:.2f}")
        col4.metric("52W High", f"{high_52w:.2f}")

        col5, col6 = st.columns(2)
        col5.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")
        col6.metric("MACD", f"{macd.iloc[-1]:.2f}")

        st.subheader("5-Min Price Chart")
        st.line_chart(close)

    else:
        st.warning("No intraday data available.")
