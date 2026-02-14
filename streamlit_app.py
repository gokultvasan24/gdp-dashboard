# ============================================================
# NSE INSTITUTIONAL DASHBOARD – COMPLETE PROFESSIONAL VERSION
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NSE Institutional Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 NSE Institutional Analytics Dashboard")

# ============================================================
# NIFTY 50 STOCK LIST
# ============================================================

NIFTY_50 = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS",
    "KOTAKBANK.NS","ITC.NS","LT.NS","SBIN.NS","AXISBANK.NS",
    "ASIANPAINT.NS","MARUTI.NS","BAJFINANCE.NS","HCLTECH.NS",
    "SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS",
    "NESTLEIND.NS","POWERGRID.NS","ONGC.NS","NTPC.NS",
    "JSWSTEEL.NS","TATASTEEL.NS","HDFCLIFE.NS","BAJAJFINSV.NS",
    "COALINDIA.NS","INDUSINDBK.NS","DRREDDY.NS","CIPLA.NS",
    "BHARTIARTL.NS","M&M.NS","HEROMOTOCO.NS","EICHERMOT.NS",
    "APOLLOHOSP.NS","GRASIM.NS","TECHM.NS","SBILIFE.NS",
    "ADANIPORTS.NS","HINDALCO.NS","TATACONSUM.NS","DIVISLAB.NS",
    "BRITANNIA.NS","BAJAJ-AUTO.NS","SHRIRAMFIN.NS",
    "BPCL.NS","UPL.NS","TATAMOTORS.NS","ADANIENT.NS"
]

# ============================================================
# INDICES
# ============================================================

INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX"
}

# ============================================================
# TECHNICAL FUNCTIONS
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

@st.cache_data(ttl=600)
def download_data(ticker, period="6mo", interval="1h"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c.capitalize() for c in df.columns]
    return df

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Section",
    ["Market Overview", "Sector Heatmap", "Stock Analytics", "FII / DII Data"]
)

# ============================================================
# 1️⃣ MARKET OVERVIEW
# ============================================================

if menu == "Market Overview":

    st.header("📈 Market Overview")

    cols = st.columns(3)

    for i, (name, symbol) in enumerate(INDICES.items()):
        df = download_data(symbol, period="6mo", interval="1d")

        if not df.empty:
            ltp = df["Close"].iloc[-1]
            change = df["Close"].pct_change().iloc[-1] * 100

            cols[i].metric(
                name,
                f"{ltp:.2f}",
                f"{change:.2f}%"
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name=name
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 2️⃣ SECTOR HEATMAP
# ============================================================

elif menu == "Sector Heatmap":

    st.header("🔥 NIFTY 50 Heatmap (1 Day Change)")

    data = yf.download(NIFTY_50, period="5d", progress=False)

    changes = {}
    for stock in NIFTY_50:
        try:
            close = data["Close"][stock]
            pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
            changes[stock.replace(".NS","")] = pct
        except:
            continue

    df_heat = pd.DataFrame.from_dict(changes, orient="index", columns=["1D %"])

    fig = px.imshow(
        df_heat.T,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlGn"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3️⃣ STOCK ANALYTICS
# ============================================================

elif menu == "Stock Analytics":

    st.header("📊 Stock Analytics – 1H + MACD")

    stock = st.selectbox("Select NSE Stock", NIFTY_50)

    df = download_data(stock)

    if df.empty:
        st.warning("No data available")
    else:

        rsi = calculate_rsi(df["Close"])
        macd, signal = calculate_macd(df["Close"])

        col1, col2, col3 = st.columns(3)

        col1.metric("LTP", f"{df['Close'].iloc[-1]:.2f}")
        col2.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
        col3.metric("52W High", f"{df['High'].max():.2f}")

        # Price Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price"
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI"))
        fig_rsi.add_hline(y=70)
        fig_rsi.add_hline(y=30)
        fig_rsi.update_layout(height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=macd, name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=signal, name="Signal"))
        fig_macd.update_layout(height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

# ============================================================
# 4️⃣ FII / DII DATA (MANUAL ENTRY)
# ============================================================

elif menu == "FII / DII Data":

    st.header("🏦 FII / DII Cash Activity")

    raw_text = st.text_area("Paste NSE FII/DII Data Here")

    if raw_text:

        lines = raw_text.split("\n")
        data = []
        row = {}

        for i, line in enumerate(lines):

            line = line.strip()

            if re.search(r"\d{2}-\w{3}-\d{4}", line):
                if row:
                    data.append(row)
                    row = {}
                row["Date"] = line

            elif "FII" in line:
                row["FII"] = float(re.findall(r"-?\d+\.?\d*", line.replace(",",""))[0])

            elif "DII" in line:
                row["DII"] = float(re.findall(r"-?\d+\.?\d*", line.replace(",",""))[0])

        if row:
            data.append(row)

        df_fii = pd.DataFrame(data)

        if not df_fii.empty:
            st.dataframe(df_fii)

            fig = px.bar(df_fii, x="Date", y="FII")
            st.plotly_chart(fig, use_container_width=True)
