# ============================================================
# NSE INSTITUTIONAL DASHBOARD – STABLE CLOUD VERSION
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ta
import requests
from datetime import datetime

st.set_page_config(page_title="NSE Institutional Dashboard", layout="wide")

# ============================================================
# SAFE DOWNLOAD FUNCTION
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
    return df.dropna()

# ============================================================
# SAFE 30 DAY PROJECTION (NO SKLEARN)
# ============================================================

def project_30_days(df):

    df = df.dropna(subset=["Close"])

    if len(df) < 20:
        return pd.DataFrame()

    x = np.arange(len(df))
    y = df["Close"].values

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(df), len(df) + 30)
    future_y = slope * future_x + intercept
    future_y = np.array(future_y).flatten()

    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=30,
        freq="B"
    )

    if len(future_dates) != len(future_y):
        return pd.DataFrame()

    forecast_df = pd.DataFrame(
        {"Close": future_y},
        index=future_dates
    )

    return forecast_df

# ============================================================
# NSE FII / DII FETCH
# ============================================================

@st.cache_data(ttl=1800)
def get_fii_dii_data():

    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    try:
        session.get("https://www.nseindia.com", headers=headers)

        cash_url = "https://www.nseindia.com/api/fiidiiTradeReact"
        cash_resp = session.get(cash_url, headers=headers)

        if cash_resp.status_code != 200:
            return None, None

        df = pd.DataFrame(cash_resp.json()["data"])

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

        deriv_url = "https://www.nseindia.com/api/fiiDerivativesPosition"
        deriv_resp = session.get(deriv_url, headers=headers)

        futures_position = None

        if deriv_resp.status_code == 200:
            deriv_df = pd.DataFrame(deriv_resp.json()["data"])
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

tabs = st.tabs(["Market Overview", "Stock Analytics"])

# ============================================================
# 1️⃣ MARKET OVERVIEW
# ============================================================

with tabs[0]:

    st.header("Market Overview – 52W Fixed Range + 30D Projection")

    index_symbols = {
        "NIFTY 50": "^NSEI",
        "NIFTY Bank": "^NSEBANK",
        "India VIX": "^INDIAVIX"
    }

    for name, symbol in index_symbols.items():

        df = download_data(symbol, period="1y", interval="1d")

        if df.empty:
            st.warning(f"No data for {name}")
            continue

        df = df[df.index.dayofweek < 5]

        high_52w = df["High"].max()
        low_52w = df["Low"].min()

        forecast = project_30_days(df)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Actual"
        ))

        if not forecast.empty:
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast["Close"],
                mode="lines",
                name="30D Projection",
                line=dict(dash="dash")
            ))

        fig.update_layout(
            title=name,
            yaxis=dict(range=[low_52w, high_52w]),
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # FII DII SECTION
    st.subheader("FII / DII – Last 5 Days")

    fii_df, futures_pos = get_fii_dii_data()

    if fii_df is not None:
        st.dataframe(fii_df, use_container_width=True)

        if futures_pos is not None:
            st.metric("FII Index Futures Net Qty", futures_pos)

# ============================================================
# 2️⃣ STOCK ANALYTICS (1H – 1 YEAR)
# ============================================================

with tabs[1]:

    st.header("Stock Analytics – 1H Data (1 Year)")

    stock = st.text_input("Enter NSE Stock (Example: RELIANCE.NS)", "RELIANCE.NS")

    df = download_data(stock, period="1y", interval="1h")

    if df.empty:
        st.warning("No data available.")
    else:

        df = df[df.index.dayofweek < 5]

        high_52w = df["High"].max()
        low_52w = df["Low"].min()

        rsi = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
        macd = ta.trend.MACD(df["Close"]).macd()

        col1, col2, col3 = st.columns(3)
        col1.metric("LTP", f"{df['Close'].iloc[-1]:.2f}")
        col2.metric("52W High", f"{high_52w:.2f}")
        col3.metric("52W Low", f"{low_52w:.2f}")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Price"
        ))

        fig.update_layout(
            yaxis=dict(range=[low_52w, high_52w]),
            template="plotly_dark",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
