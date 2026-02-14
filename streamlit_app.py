# ============================================================
# NSE INSTITUTIONAL DASHBOARD – FINAL STABLE COMBINED VERSION
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="NSE Institutional Dashboard", layout="wide")
sns.set_style("darkgrid")

# ============================================================
# SAFE YFINANCE DOWNLOAD
# ============================================================

@st.cache_data(ttl=600)
def download_data(ticker, period="1y", interval="1d"):

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

    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    return df.dropna()


# ============================================================
# SIMPLE RSI (NO ta LIBRARY)
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================
# 30 DAY PROJECTION
# ============================================================

def project_30_days(df):

    if len(df) < 20:
        return pd.DataFrame()

    x = np.arange(len(df))
    y = df["Close"].values

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(df), len(df) + 30)
    future_y = slope * future_x + intercept

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=30,
        freq="B"
    )

    return pd.DataFrame({"Close": future_y}, index=future_dates)


# ============================================================
# AUTO NSE FII / DII API
# ============================================================

@st.cache_data(ttl=1800)
def get_fii_dii():

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        session.get("https://www.nseindia.com", headers=headers)
        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        resp = session.get(url, headers=headers)

        if resp.status_code != 200:
            return None

        df = pd.DataFrame(resp.json()["data"])
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        return df.sort_values("date", ascending=False).head(5)

    except:
        return None


# ============================================================
# MANUAL FII/DII PARSER (Fallback)
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

        elif line == "INDIA VIX":
            row["VIX"] = extract_number(lines[i+1].strip())

        elif line == "SENSEX":
            row["SENSEX"] = extract_number(lines[i+1].strip())

    if row:
        data.append(row)

    df = pd.DataFrame(data)

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

    return df


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(["Market Overview", "Stock Analytics", "Manual FII Entry"])


# ============================================================
# MARKET OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Market Overview – 52W Range + 30D Projection")

    indices = {
        "NIFTY 50": "^NSEI",
        "NIFTY Bank": "^NSEBANK",
        "India VIX": "^INDIAVIX"
    }

    for name, symbol in indices.items():

        df = download_data(symbol, period="1y", interval="1d")

        if df.empty:
            st.warning(f"No data for {name}")
            continue

        df = df[df.index.dayofweek < 5]

        high_52w = df["High"].max()
        low_52w = df["Low"].min()

        forecast = project_30_days(df)

        fig, ax = plt.subplots(figsize=(12,5))
        sns.lineplot(x=df.index, y=df["Close"], ax=ax, label="Actual")

        if not forecast.empty:
            sns.lineplot(x=forecast.index, y=forecast["Close"], ax=ax, label="30D Projection")

        ax.set_title(name)
        ax.set_ylim(low_52w, high_52w)

        st.pyplot(fig)

    # AUTO FII DII
    st.subheader("FII / DII – Live NSE")

    fii = get_fii_dii()

    if fii is not None:
        st.dataframe(fii)
    else:
        st.warning("NSE API Blocked – Use Manual Entry Tab")


# ============================================================
# STOCK ANALYTICS
# ============================================================

with tabs[1]:

    st.header("Stock Analytics – 1H Data (1 Year)")

    stock = st.text_input("Enter NSE Stock", "RELIANCE.NS")

    df = download_data(stock, period="1y", interval="1h")

    if df.empty:
        st.warning("No data available.")
    else:

        df = df[df.index.dayofweek < 5]

        high_52w = df["High"].max()
        low_52w = df["Low"].min()

        rsi = calculate_rsi(df["Close"])

        col1, col2, col3 = st.columns(3)
        col1.metric("LTP", f"{df['Close'].iloc[-1]:.2f}")
        col2.metric("52W High", f"{high_52w:.2f}")
        col3.metric("52W Low", f"{low_52w:.2f}")

        fig, ax = plt.subplots(figsize=(12,5))
        sns.lineplot(x=df.index, y=df["Close"], ax=ax)
        ax.set_ylim(low_52w, high_52w)
        ax.set_title(f"{stock} Price (1H)")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(12,3))
        sns.lineplot(x=df.index, y=rsi, ax=ax2)
        ax2.axhline(70)
        ax2.axhline(30)
        ax2.set_title("RSI")
        st.pyplot(fig2)


# ============================================================
# MANUAL FII ENTRY TAB
# ============================================================

with tabs[2]:

    st.header("Manual FII / DII Paste")

    raw_text = st.text_area("Paste FII/DII Data Here")

    if raw_text:
        df_manual = parse_manual_data(raw_text)

        if not df_manual.empty:

            st.dataframe(df_manual)

            fig, ax = plt.subplots(figsize=(12,5))
            sns.barplot(x="Date", y="FII Cash", data=df_manual, ax=ax)
            ax.set_title("FII Cash Flow")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        else:
            st.warning("Unable to parse data.")
