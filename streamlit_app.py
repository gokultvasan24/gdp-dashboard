# streamlit_app.py
# Advanced Stock Forecasting Platform
# Polynomial Regression + GJR-GARCH + Technical Indicators
# SAFE & STABLE VERSION

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------- ARCH ----------------
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(page_title="📈 Advanced Stock Forecasting", layout="wide")
st.title("📈 Advanced Stock Forecasting Platform")

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("🔧 Configuration")

ticker = st.sidebar.text_input("Stock Ticker", "^NSEI").upper()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
with col2:
    end_date = st.sidebar.date_input("End Date", datetime.now())

price_type = st.sidebar.selectbox(
    "Price Type", ["Close", "Open", "High", "Low", "Adj Close"]
)

degree = st.sidebar.slider("Polynomial Degree", 1, 8, 3)

garch_p = st.sidebar.slider("GARCH p", 1, 3, 1)
garch_q = st.sidebar.slider("GARCH q", 1, 3, 1)
garch_o = st.sidebar.slider("GJR o", 1, 3, 1)

run = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

# =============================================================================
# UTILS
# =============================================================================

@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)

def detect_currency(ticker):
    return "₹" if any(x in ticker for x in ["^NSEI", ".NS", ".BO"]) else "$"

def returns(series):
    return series.pct_change().dropna() * 100

# =============================================================================
# MAIN
# =============================================================================

if run:

    if not ARCH_AVAILABLE:
        st.error("ARCH package missing. Install with: pip install arch")
        st.stop()

    data = load_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data found for this ticker.")
        st.stop()

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required):
        st.error("OHLCV data incomplete.")
        st.stop()

    data.dropna(inplace=True)

    currency = detect_currency(ticker)

    # =============================================================================
    # TECHNICAL INDICATORS (SAFE)
    # =============================================================================

    df = data.copy()

    macd = ta.macd(df[price_type])
    adx = ta.adx(df["High"], df["Low"], df["Close"])
    stoch = ta.stochrsi(df[price_type])

    if macd is None or adx is None or stoch is None:
        st.error("Indicator calculation failed.")
        st.stop()

    df = df.join(macd)
    df = df.join(adx)
    df = df.join(stoch)
    df.dropna(inplace=True)

    price_data = df[price_type]
    current_price = float(price_data.iloc[-1])

    # =============================================================================
    # POLYNOMIAL REGRESSION
    # =============================================================================

    st.subheader("📈 Polynomial Regression Forecast")

    dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1)
    X = (dates - dates.mean()) / (dates.max() - dates.min())
    y = price_data.values

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{currency}{rmse:.2f}")
    c2.metric("R²", f"{r2:.4f}")
    c3.metric("MAE", f"{currency}{mae:.2f}")

    next_x = np.array([[X[-1][0] + (1 / len(X))]])
    next_price = float(model.predict(poly.transform(next_x))[0])

    delta = next_price - current_price
    pct = delta / current_price * 100

    st.subheader("🎯 Next Day Forecast")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current", f"{currency}{current_price:.2f}")
    c2.metric("Forecast", f"{currency}{next_price:.2f}", f"{delta:+.2f}")
    c3.metric("Change", f"{pct:+.2f}%")

    # =============================================================================
    # GJR-GARCH
    # =============================================================================

    st.subheader("📊 GJR-GARCH Volatility")

    ret = returns(price_data)

    if len(ret) > 60:
        garch = arch_model(ret, p=garch_p, q=garch_q, o=garch_o, vol="Garch")
        res = garch.fit(disp="off")
        vol = np.sqrt(res.forecast(horizon=5).variance.iloc[-1])

        st.dataframe(pd.DataFrame({
            "Day": range(1, 6),
            "Volatility (%)": vol.round(2)
        }), use_container_width=True)

    # =============================================================================
    # INDICATOR SIGNALS
    # =============================================================================

    st.subheader("📌 Indicator Signals")

    last = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ADX", f"{last['ADX_14']:.2f}",
              "Strong" if last["ADX_14"] > 25 else "Weak")

    c2.metric("MACD",
              "Bullish" if last["MACD_12_26_9"] > last["MACDs_12_26_9"] else "Bearish")

    c3.metric("Stoch RSI",
              "Overbought" if last["STOCHRSIk_14_14_3_3"] > 0.8 else
              "Oversold" if last["STOCHRSIk_14_14_3_3"] < 0.2 else "Neutral")

    c4.metric("Volume", f"{int(last['Volume']):,}")

    # =============================================================================
    # VISUALS
    # =============================================================================

    st.subheader("📊 Charts")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(price_data.index, y, label="Actual")
    ax.plot(price_data.index, y_pred, linestyle="--", label="Regression")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(df.index, df["Volume"])
    ax.set_title("Volume")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["MACD_12_26_9"], label="MACD")
    ax.plot(df.index, df["MACDs_12_26_9"], linestyle="--", label="Signal")
    ax.bar(df.index, df["MACDh_12_26_9"], alpha=0.4)
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["ADX_14"])
    ax.axhline(25, linestyle="--")
    ax.set_title("ADX")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["STOCHRSIk_14_14_3_3"], label="%K")
    ax.plot(df.index, df["STOCHRSId_14_14_3_3"], linestyle="--", label="%D")
    ax.axhline(0.8, linestyle="--")
    ax.axhline(0.2, linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # =============================================================================
    # STATS
    # =============================================================================

    st.subheader("🔬 Statistical Diagnostics")

    resids = y - y_pred
    jb = jarque_bera(resids)
    adf = adfuller(resids)

    st.write(f"Jarque-Bera p-value: **{jb[1]:.4f}**")
    st.write(f"ADF p-value: **{adf[1]:.4f}**")

    st.success("✅ Analysis Complete")
