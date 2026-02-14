# ===============================
# STREAMLIT FII/DII DASHBOARD
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from nsefin import NSEClient
import plotly.graph_objects as go
from ta.trend import SMAIndicator
from scipy.stats import pearsonr
import statsmodels.api as sm

st.set_page_config(layout="wide")

st.title("📊 FII / DII vs Stock Dashboard")

# -----------------------------------
# Sidebar Inputs
# -----------------------------------

symbol = st.sidebar.text_input("Enter NSE Symbol", "RELIANCE.NS")
years = st.sidebar.selectbox("Years of Data", [1, 2], index=0)

# -----------------------------------
# STEP 1: Get Price Data (1H)
# -----------------------------------

st.subheader("Fetching Price Data...")

price_df = yf.download(
    symbol,
    period=f"{years}y",
    interval="60m",
    progress=False
)

# Remove weekends
price_df = price_df[price_df.index.dayofweek < 5]

# -----------------------------------
# STEP 2: Get 52 Week High/Low
# -----------------------------------

ticker = yf.Ticker(symbol)
info = ticker.info

low_52 = info.get("fiftyTwoWeekLow")
high_52 = info.get("fiftyTwoWeekHigh")

# -----------------------------------
# STEP 3: Add Technical Indicators
# -----------------------------------

price_df["SMA_50"] = SMAIndicator(price_df["Close"], window=50).sma_indicator()
price_df["Returns"] = price_df["Close"].pct_change()

# -----------------------------------
# STEP 4: Get FII/DII Data
# -----------------------------------

st.subheader("Fetching FII/DII Data...")

nse = NSEClient()
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365)

fii_dii_df = nse.get_fii_dii_activity(start_date, end_date)
fii_dii_df["Date"] = pd.to_datetime(fii_dii_df["Date"])
fii_dii_df = fii_dii_df.sort_values("Date")

# -----------------------------------
# STEP 5: Correlation Analysis
# -----------------------------------

daily_price = price_df["Close"].resample("1D").last().dropna()

merged = pd.merge(
    daily_price,
    fii_dii_df,
    left_index=True,
    right_on="Date",
    how="inner"
)

if "FII Net" in merged.columns:
    correlation, _ = pearsonr(merged["Close"], merged["FII Net"])
else:
    correlation = None

# -----------------------------------
# STEP 6: Regression
# -----------------------------------

if "FII Net" in merged.columns:
    X = sm.add_constant(merged["FII Net"])
    model = sm.OLS(merged["Close"], X).fit()
    beta = model.params[1]
else:
    beta = None

# -----------------------------------
# STEP 7: Plotly Chart
# -----------------------------------

fig = go.Figure()

# Price
fig.add_trace(go.Scatter(
    x=price_df.index,
    y=price_df["Close"],
    name="Close Price",
    line=dict(width=2)
))

# SMA
fig.add_trace(go.Scatter(
    x=price_df.index,
    y=price_df["SMA_50"],
    name="SMA 50",
    line=dict(dash="dash")
))

# FII Net
if "FII Net" in fii_dii_df.columns:
    fig.add_trace(go.Scatter(
        x=fii_dii_df["Date"],
        y=fii_dii_df["FII Net"],
        name="FII Net",
        yaxis="y2",
        line=dict(dash="dot")
    ))

# Layout
fig.update_layout(
    title=f"{symbol} - 1H Chart with FII Net",
    yaxis=dict(
        title="Price",
        range=[low_52, high_52] if low_52 and high_52 else None
    ),
    yaxis2=dict(
        title="FII Net",
        overlaying="y",
        side="right"
    ),
    height=700
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# STEP 8: Display Stats
# -----------------------------------

st.subheader("Statistical Analysis")

col1, col2 = st.columns(2)

if correlation:
    col1.metric("Correlation (Price vs FII Net)", round(correlation, 4))

if beta:
    col2.metric("Regression Beta (FII Impact)", round(beta, 4))

# -----------------------------------
# STEP 9: Show Raw Data
# -----------------------------------

with st.expander("Show Price Data"):
    st.dataframe(price_df.tail())

with st.expander("Show FII/DII Data"):
    st.dataframe(fii_dii_df.tail())
