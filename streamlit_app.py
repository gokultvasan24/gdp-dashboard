import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ADX + Stochastic RSI", layout="wide")

# ============================
# Indicator Functions
# ============================

def compute_adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()

    return adx


def compute_stoch_rsi(close, rsi_period=14, stoch_period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    min_rsi = rsi.rolling(stoch_period).min()
    max_rsi = rsi.rolling(stoch_period).max()

    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    return stoch_rsi * 100


# ============================
# UI
# ============================

st.title("📈 ADX (Trend Strength) + Stochastic RSI")

with st.sidebar:
    ticker = st.text_input("Stock Symbol", "RELIANCE.NS")
    start = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end = st.date_input("End Date", pd.to_datetime("today"))
    adx_period = st.slider("ADX Period", 10, 30, 14)

# ============================
# Data
# ============================

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df.dropna()

data = load_data(ticker, start, end)

if data.empty:
    st.error("No data found")
    st.stop()

# ============================
# Indicators
# ============================

adx = compute_adx(data, adx_period)
stoch_rsi = compute_stoch_rsi(data["Close"])

# ============================
# Plot
# ============================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# ADX
ax1.plot(adx.index, adx, label="ADX", linewidth=2)
ax1.axhline(25, linestyle="--", color="red", alpha=0.6, label="Strong Trend")
ax1.set_ylabel("ADX")
ax1.set_title("ADX (Trend Strength)")
ax1.legend()
ax1.grid(alpha=0.3)

# Stochastic RSI
ax2.plot(stoch_rsi.index, stoch_rsi, label="Stochastic RSI", color="green", linewidth=1.8)
ax2.axhline(80, linestyle="--", color="red", alpha=0.6, label="Overbought")
ax2.axhline(20, linestyle="--", color="green", alpha=0.6, label="Oversold")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Stoch RSI")
ax2.set_title("Stochastic RSI")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ============================
# Summary
# ============================

latest_adx = adx.dropna().iloc[-1]
latest_stoch = stoch_rsi.dropna().iloc[-1]

st.subheader("📊 Latest Signal")

col1, col2 = st.columns(2)

with col1:
    st.metric("ADX", f"{latest_adx:.2f}")
    st.write("Strong trend" if latest_adx > 25 else "Weak / Range")

with col2:
    st.metric("Stochastic RSI", f"{latest_stoch:.2f}")
    if latest_stoch > 80:
        st.write("Overbought")
    elif latest_stoch < 20:
        st.write("Oversold")
    else:
        st.write("Neutral")
