# ============================================================
# NSE INSTITUTIONAL DASHBOARD – DROPDOWN + 1H + MACD + RSI
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re

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

# Convert to NSE ticker format
STOCK_LIST = sorted([s + ".NS" for _, s in RAW_SECTOR_DATA])

# Add major indices
INDEX_LIST = {
    "NIFTY 50": "^NSEI",
    "NIFTY Bank": "^NSEBANK",
    "India VIX": "^INDIAVIX"
}

# Dropdown options
DISPLAY_LIST = list(INDEX_LIST.keys()) + STOCK_LIST

# ============================================================
# DATA DOWNLOAD FUNCTION
# ============================================================

@st.cache_data(ttl=600)
def download_data(ticker, period="6mo", interval="1h"):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).capitalize() for c in df.columns]
    return df.dropna()

# ============================================================
# RSI FUNCTION
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
# MACD FUNCTION
# ============================================================

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

# ============================================================
# MANUAL FII PARSER
# ============================================================

def extract_number(value):
    value = value.replace(",", "")
    return float(re.findall(r"-?\d+\.?\d*", value)[0])

def parse_manual_data(text):
    lines = text.split("\n")
    data = []
    row = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if re.search(r"\w{3},", line):
            if row:
                data.append(row)
                row = {}
            row["Date"] = line
        elif "FII Cash Market" in line:
            row["FII Cash"] = extract_number(lines[i+1].strip())
        elif "DII Cash Market" in line:
            row["DII Cash"] = extract_number(lines[i+1].strip())
        elif line == "NIFTY":
            row["NIFTY"] = extract_number(lines[i+1].strip())
    if row:
        data.append(row)
    df = pd.DataFrame(data)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    return df

# ============================================================
# DASHBOARD TABS
# ============================================================

tabs = st.tabs(["Market Overview", "Sector Performance", "Stock Analytics", "Manual FII"])

# ============================================================
# 1️⃣ MARKET OVERVIEW
# ============================================================

with tabs[0]:
    st.header("Market Overview – 52W Range")
    for name, symbol in INDEX_LIST.items():
        df = download_data(symbol, period="1y", interval="1d")
        if df.empty:
            st.warning(f"No data for {name}")
            continue
        high_52w = df["High"].max()
        low_52w = df["Low"].min()
        fig, ax = plt.subplots(figsize=(12,5))
        sns.lineplot(x=df.index, y=df["Close"], ax=ax)
        ax.set_ylim(low_52w, high_52w)
        ax.set_title(name)
        st.pyplot(fig)

# ============================================================
# 2️⃣ SECTOR PERFORMANCE
# ============================================================

with tabs[1]:
    st.header("Sector Performance")
    SECTOR_MAP = {
        "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS"],
        "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS"],
        "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS"],
    }
    ALL_STOCKS_SECTOR = list(set(sum(SECTOR_MAP.values(), [])))
    data = yf.download(ALL_STOCKS_SECTOR, period="2wk", progress=False, auto_adjust=True, threads=False)
    sector_perf = {}
    sector_2week = {}
    if isinstance(data.columns, pd.MultiIndex):
        for sector, stocks in SECTOR_MAP.items():
            changes, two_week_changes = [], []
            for stock in stocks:
                try:
                    df_s = data.xs(stock, axis=1, level=1)
                    close = df_s["Close"].dropna()
                    if len(close) >= 2:
                        changes.append((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
                    if len(close) >= 10:
                        two_week_changes.append((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100)
                except: continue
            if changes: sector_perf[sector] = np.mean(changes)
            if two_week_changes: sector_2week[sector] = np.mean(two_week_changes)
    df_sector = pd.DataFrame.from_dict(sector_perf, orient="index", columns=["1D % Change"])
    df_sector2 = pd.DataFrame.from_dict(sector_2week, orient="index", columns=["2W % Change"])
    if not df_sector.empty:
        fig = px.imshow(df_sector.T, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_sector.join(df_sector2).sort_values("1D % Change", ascending=False), use_container_width=True)

# ============================================================
# 3️⃣ STOCK ANALYTICS
# ============================================================

with tabs[2]:
    st.header("Stock Analytics – 1H + MACD + RSI")
    selected = st.selectbox("Select NSE Stock / Index", DISPLAY_LIST)
    ticker = INDEX_LIST[selected] if selected in INDEX_LIST else selected
    df = download_data(ticker)
    if df.empty:
        st.warning("No data available.")
    else:
        close = df["Close"]
        rsi = calculate_rsi(close)
        macd, signal, hist = calculate_macd(close)
        high_6m, low_6m = df["High"].max(), df["Low"].min()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("LTP", f"{close.iloc[-1]:.2f}")
        col2.metric("6M High", f"{high_6m:.2f}")
        col3.metric("6M Low", f"{low_6m:.2f}")
        col4.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")

        # Price Chart
        st.subheader("1H Price Chart")
        fig1, ax1 = plt.subplots(figsize=(14,6))
        sns.lineplot(x=df.index, y=close, ax=ax1)
        ax1.set_ylim(low_6m, high_6m)
        st.pyplot(fig1)

        # RSI Chart
        st.subheader("RSI Indicator")
        fig2, ax2 = plt.subplots(figsize=(14,4))
        sns.lineplot(x=df.index, y=rsi, ax=ax2)
        ax2.axhline(70, linestyle="--")
        ax2.axhline(30, linestyle="--")
        st.pyplot(fig2)

        # MACD Chart
        st.subheader("MACD Indicator")
        fig3, ax3 = plt.subplots(figsize=(14,4))
        sns.lineplot(x=df.index, y=macd, ax=ax3, label="MACD")
        sns.lineplot(x=df.index, y=signal, ax=ax3, label="Signal")
        ax3.bar(df.index, hist, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3)

# ============================================================
# 4️⃣ MANUAL FII ENTRY
# ============================================================

with tabs[3]:
    st.header("Manual FII Entry")
    raw_text = st.text_area("Paste FII Data Here")
    if raw_text:
        df_manual = parse_manual_data(raw_text)
        if not df_manual.empty:
            st.dataframe(df_manual)
            fig, ax = plt.subplots(figsize=(12,5))
            sns.barplot(x="Date", y="FII Cash", data=df_manual, ax=ax)
            plt.xticks(rotation=45)
            ax.set_title("FII Cash Flow")
            st.pyplot(fig)
