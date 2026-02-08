import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Advanced Stock Indicators", layout="wide")

# ----------------------------
# Indicator Functions
# ----------------------------

def compute_adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()

    return adx


def compute_stoch_rsi(series, rsi_period=14, stoch_period=14):
    delta = series.diff()

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


# ----------------------------
# UI
# ----------------------------

st.title("📈 Advanced Stock Technical Indicators")
st.write("ADX (Trend Strength) + Stochastic RSI")

with st.sidebar:
    ticker = st.text_input("Stock Symbol", value="RELIANCE.NS")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    period = st.slider("ADX Period", 10, 30, 14)

# ----------------------------
# Data
# ----------------------------

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df.dropna()

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data found.")
    st.stop()

prices = data["Close"]

# ----------------------------
# Indicators
# ----------------------------

adx = compute_adx(data, period)
stoch_rsi = compute_stoch_rsi(prices)

# ----------------------------
# Plot
# ----------------------------

st.subheader("📊 Technical Indicators")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# ADX
ax1.plot(adx.index, adx, label="ADX", linewidth=2)
ax1.axhline(25, linestyle="--", color="red", alpha=0.7, label="Strong Trend")
ax1.set_ylabel("ADX")
ax1.set_title("ADX (Trend Strength)")
ax1.legend()
ax1.grid(alpha=0.3)

# Stochastic RSI
ax2.plot(stoch_rsi.index, stoch_rsi, label="Stochastic RSI", linewidth=1.8, color="green")
ax2.axhline(80, linestyle="--", color="red", alpha=0.6, label="Overbought")
ax2.axhline(20, linestyle="--", color="green", alpha=0.6, label="Oversold")
ax2.set_ylabel("Stoch RSI")
ax2.set_ylim(0, 100)
ax2.set_title("Stochastic RSI")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# ----------------------------
# Interpretation
# ----------------------------

latest_adx = adx.dropna().iloc[-1]
latest_stoch = stoch_rsi.dropna().iloc[-1]

st.subheader("🧠 Indicator Summary")

col1, col2 = st.columns(2)

with col1:
    st.metric("Latest ADX", f"{latest_adx:.2f}")
    if latest_adx > 25:
        st.success("Strong Trend Detected")
    else:
        st.info("Weak / Range Market")

with col2:
    st.metric("Latest Stoch RSI", f"{latest_stoch:.2f}")
    if latest_stoch > 80:
        st.warning("Overbought")
    elif latest_stoch < 20:
        st.success("Oversold")
    else:
        st.info("Neutral Zone")
