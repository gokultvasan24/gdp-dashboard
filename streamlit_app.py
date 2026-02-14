# ============================================================
# NSE INSTITUTIONAL DASHBOARD – FULL PROFESSIONAL VERSION
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import ta
import requests
from datetime import datetime

st.set_page_config(page_title="NSE Institutional Dashboard", layout="wide")

# ============================================================
# CUSTOM SECTOR MAP
# ============================================================

RAW_SECTOR_DATA = [
    ("Metals & Mining","HINDALCO"), ("FMCG","HINDUNILVR"),
    ("Services","ETERNAL"), ("Metals & Mining","ADANIENT"),
    ("Oil & Gas","ONGC"), ("Automobile","HEROMOTOCO"),
    ("Metals & Mining","TATASTEEL"), ("Consumer Durables","TITAN"),
    ("Oil & Gas","COALINDIA"), ("Services","ADANIPORTS"),
    ("Power","POWERGRID"), ("Information Technology","WIPRO"),
    ("FMCG","NESTLEIND"), ("Oil & Gas","RELIANCE"),
    ("Information Technology","TCS"), ("Captial Goods","BEL"),
    ("Financial Services","HDFCBANK"), ("Consumer Durables","ASIANPAINT"),
    ("Financial Services","SHRIRAMFIN"), ("Automobile","M&M"),
    ("FMCG","TATACONSUM"), ("Construction Materials","GRASIM"),
    ("Metals & Mining","JSWSTEEL"), ("Automobile","TMPV"),
    ("Financial Services","JIOFIN"), ("Information Technology","HCLTECH"),
    ("Information Technology","INFY"), ("FMCG","ITC"),
    ("Power","NTPC"), ("Telecommunication","BHARTIARTL"),
    ("Financial Services","KOTAKBANK"), ("Financial Services","ICICIBANK"),
    ("Automobile","BAJAJ-AUTO"), ("Healthcare","SUNPHARMA"),
    ("Retail","TRENT"), ("Financial Services","HDFCLIFE"),
    ("Automobile","MARUTI"), ("Financial Services","AXISBANK"),
    ("Financial Services","BAJAJFINSV"), ("Healthcare","DRREDDY"),
    ("Construction Materials","ULTRACEMCO"), ("Construction","LT"),
    ("Healthcare","APOLLOHOSP"), ("Information Technology","TECHM"),
    ("Healthcare","CIPLA"), ("Financial Services","SBIN"),
    ("Financial Services","INDUSINDBK"), ("Financial Services","SBILIFE"),
    ("Automobile","EICHERMOT"), ("Financial Services","BAJFINANCE")
]

SECTOR_MAP = {}
for sector, symbol in RAW_SECTOR_DATA:
    SECTOR_MAP.setdefault(sector, []).append(symbol + ".NS")

ALL_STOCKS = sorted(list(set([s for v in SECTOR_MAP.values() for s in v])))

# ============================================================
# SAFE DOWNLOAD
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# ============================================================
# NSE FII / DII FETCH
# ============================================================

@st.cache_data(ttl=1800)
def get_fii_dii_data():

    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json"
    }

    try:
        session.get("https://www.nseindia.com", headers=headers)

        # CASH MARKET DATA
        cash_url = "https://www.nseindia.com/api/fiidiiTradeReact"
        cash_resp = session.get(cash_url, headers=headers)

        if cash_resp.status_code != 200:
            return None, None

        data = cash_resp.json()["data"]
        df = pd.DataFrame(data)

        df = df.rename(columns={
            "date": "Date",
            "fiiBuyValue": "FII Buy",
            "fiiSellValue": "FII Sell",
            "fiiNetValue": "FII Net",
            "diiBuyValue": "DII Buy",
            "diiSellValue": "DII Sell",
            "diiNetValue": "DII Net"
        })

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df = df.sort_values("Date", ascending=False).head(5)

        # DERIVATIVES POSITION
        deriv_url = "https://www.nseindia.com/api/fiiDerivativesPosition"
        deriv_resp = session.get(deriv_url, headers=headers)

        futures_position = None

        if deriv_resp.status_code == 200:
            deriv_data = deriv_resp.json()["data"]
            deriv_df = pd.DataFrame(deriv_data)

            index_fut = deriv_df[
                deriv_df["instrument"].str.contains("Index Futures", case=False)
            ]

            if not index_fut.empty:
                futures_position = int(index_fut.iloc[0]["netQty"])

        return df, futures_position

    except:
        return None, None

# ============================================================
# TABS
# ============================================================

tabs = st.tabs(["Market Overview", "Sector Performance", "Stock Analytics"])

# ============================================================
# 1️⃣ MARKET OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Market Overview")

    index_symbols = {
        "NIFTY 50": "^NSEI",
        "NIFTY Bank": "^NSEBANK",
        "India VIX": "^INDIAVIX"
    }

    cols = st.columns(3)

    for i, (name, symbol) in enumerate(index_symbols.items()):
        df = download_data(symbol, period="5d", interval="1h")
        if not df.empty:
            ltp = df["Close"].iloc[-1]
            prev = df["Close"].iloc[-2]
            pct = ((ltp - prev) / prev) * 100
            cols[i].metric(name, f"{ltp:.2f}", f"{pct:.2f}%")
            cols[i].line_chart(df["Close"])

    # Advance Decline
    st.subheader("Advance / Decline Ratio")
    data = yf.download(ALL_STOCKS, period="2d",
                       progress=False, auto_adjust=True, threads=False)

    adv, dec = 0, 0
    if isinstance(data.columns, pd.MultiIndex):
        for stock in ALL_STOCKS:
            try:
                df = data.xs(stock, axis=1, level=1)
                close = df["Close"].dropna()
                if close.iloc[-1] > close.iloc[-2]:
                    adv += 1
                else:
                    dec += 1
            except:
                continue

    st.metric("Advance / Decline", f"{adv} / {dec}")

    # =====================================================
    # FII / DII SECTION (UPDATED – LAST 5 DAYS)
    # =====================================================

    st.subheader("FII / DII Activity – Last 5 Trading Days")

    fii_df, futures_pos = get_fii_dii_data()

    if fii_df is not None:

        st.dataframe(fii_df, use_container_width=True)

        latest_date = fii_df.iloc[0]["Date"].date()
        today = datetime.today().date()

        if latest_date == today:
            st.success("Today's data included ✅")
        else:
            st.warning("Today's data not yet updated on NSE")

        st.subheader("FII Index Futures Net Position")

        if futures_pos is not None:
            st.metric("Net Index Futures Qty", futures_pos)

            if futures_pos > 0:
                st.success("FII Bias: LONG")
            elif futures_pos < 0:
                st.error("FII Bias: SHORT")
            else:
                st.info("FII Bias: Neutral")
        else:
            st.warning("Futures data unavailable")

    else:
        st.error("Unable to fetch FII/DII data from NSE")

# ============================================================
# 2️⃣ SECTOR PERFORMANCE
# ============================================================

with tabs[1]:

    st.header("Sector Performance")

    data = yf.download(ALL_STOCKS, period="2d",
                       progress=False, auto_adjust=True, threads=False)

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
# 3️⃣ STOCK ANALYTICS (1H – 10 DAYS)
# ============================================================

with tabs[2]:

    st.header("Stock-Level Data (30-Min – Last 10 Days)")

    stock = st.selectbox("Select Stock", ALL_STOCKS)

    intraday = download_data(stock, period="10d", interval="1h")

    if not intraday.empty:

        intraday = intraday.dropna()

        close = intraday["Close"].astype(float)
        high = intraday["High"].astype(float)
        low = intraday["Low"].astype(float)
        volume = intraday["Volume"].astype(float)

        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().fillna(0)
        macd = ta.trend.MACD(close).macd().fillna(0)

        daily_52w = download_data(stock, period="1y", interval="1d")
        high_52w = daily_52w["High"].astype(float).max()

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
