# ============================================================
# NSE INSTITUTIONAL DASHBOARD – DROPDOWN VERSION
# 1H + MACD + RSI + ALL STOCKS + INDICES
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NSE Institutional Dashboard", layout="wide")
sns.set_style("darkgrid")

# ============================================================
# RAW SECTOR DATA
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

# Convert to NSE format
STOCK_LIST = sorted(list(set([symbol + ".NS" for _, symbol in RAW_SECTOR_DATA])))

# Add Indices
INDEX_LIST = {
    "NIFTY 50": "^NSEI",
    "NIFTY Bank": "^NSEBANK",
    "India VIX": "^INDIAVIX"
}

# Combine everything
DISPLAY_LIST = list(INDEX_LIST.keys()) + STOCK_LIST


# ============================================================
# DOWNLOAD FUNCTION
# ============================================================

@st.cache_data(ttl=600)
def download_data(ticker, period="6mo", interval="1h"):

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.capitalize() for c in df.columns]

    return df.dropna()


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ============================================================
# MACD
# ============================================================

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


# ============================================================
# UI
# ============================================================

st.title("📊 NSE Institutional Stock Dashboard")

selected = st.selectbox("Select NSE Stock / Index", DISPLAY_LIST)

# Convert display name to actual ticker
if selected in INDEX_LIST:
    ticker = INDEX_LIST[selected]
else:
    ticker = selected

df = download_data(ticker)

if df.empty:
    st.error("No data available.")
    st.stop()

close = df["Close"]

rsi = calculate_rsi(close)
macd, signal, hist = calculate_macd(close)

high_6m = df["High"].max()
low_6m = df["Low"].min()

# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("LTP", f"{close.iloc[-1]:.2f}")
col2.metric("6M High", f"{high_6m:.2f}")
col3.metric("6M Low", f"{low_6m:.2f}")
col4.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")

# ============================================================
# PRICE CHART
# ============================================================

st.subheader("1H Price Chart")

fig1, ax1 = plt.subplots(figsize=(14,6))
sns.lineplot(x=df.index, y=close, ax=ax1)
ax1.set_title(f"{selected} – 1H Price")
ax1.set_ylim(low_6m, high_6m)
st.pyplot(fig1)

# ============================================================
# RSI CHART
# ============================================================

st.subheader("RSI Indicator")

fig2, ax2 = plt.subplots(figsize=(14,4))
sns.lineplot(x=df.index, y=rsi, ax=ax2)
ax2.axhline(70, linestyle="--")
ax2.axhline(30, linestyle="--")
st.pyplot(fig2)

# ============================================================
# MACD CHART
# ============================================================

st.subheader("MACD Indicator")

fig3, ax3 = plt.subplots(figsize=(14,4))
sns.lineplot(x=df.index, y=macd, ax=ax3, label="MACD")
sns.lineplot(x=df.index, y=signal, ax=ax3, label="Signal")
ax3.bar(df.index, hist, alpha=0.3)
ax3.legend()
st.pyplot(fig3)
