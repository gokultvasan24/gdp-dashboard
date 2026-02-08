# streamlit_app.py
# 📈 Advanced Stock Forecasting Platform
# Created by Gokul Thanigaivasan

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from arch import arch_model
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Advanced Stock Forecasting Platform",
    layout="wide"
)

st.title("📈 Advanced Stock Forecasting Platform")
st.markdown("👨‍💻 **Created by Gokul Thanigaivasan**")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("🔧 Configuration")

ticker = st.sidebar.text_input("Stock Ticker", "^NSEI").upper()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.now())

price_type = st.sidebar.selectbox(
    "Price Type", ["Close", "Open", "High", "Low", "Adj Close"]
)

degree = st.sidebar.slider("Polynomial Degree", 1, 8, 3)

garch_p = st.sidebar.slider("GARCH p", 1, 3, 1)
garch_q = st.sidebar.slider("GARCH q", 1, 3, 1)
garch_o = st.sidebar.slider("GJR o", 1, 3, 1)

run_btn = st.sidebar.button("🚀 Run Analysis", type="primary")

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def download_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)


def detect_currency(ticker):
    return "₹" if any(x in ticker for x in ["^NSEI", ".NS", "NIFTY"]) else "$"


def calculate_returns(series):
    return series.pct_change().dropna() * 100


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────
def compute_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()

    return adx


def compute_stoch_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    stoch_rsi = (rsi - rsi.rolling(period).min()) / (
        rsi.rolling(period).max() - rsi.rolling(period).min()
    )

    return stoch_rsi * 100


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if run_btn:

    data = download_data(ticker, start_date, end_date)

    if data.empty:
        st.error("No data found")
        st.stop()

    prices = data[price_type].dropna()
    currency = detect_currency(ticker)
    current_price = float(prices.iloc[-1])

    # ─────────────────────────────────────────
    # POLYNOMIAL REGRESSION
    # ─────────────────────────────────────────
    st.subheader("📈 Polynomial Regression")

    X = np.array([d.toordinal() for d in prices.index], dtype=float).reshape(-1, 1)
    y = prices.values.astype(float)

    X_norm = (X - X.mean()) / (X.max() - X.min())

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)

    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    next_x = np.array([[X_norm[-1, 0] + (1 / (X.max() - X.min()))]])
    next_price = model.predict(poly.transform(next_x)).item()

    st.metric(
        "Next-Day Forecast",
        f"{currency}{next_price:.2f}",
        f"{next_price - current_price:+.2f}"
    )

    # ─────────────────────────────────────────
    # GJR-GARCH
    # ─────────────────────────────────────────
    st.subheader("📊 GJR-GARCH Volatility Forecast")

    returns = calculate_returns(prices)

    garch = arch_model(
        returns,
        vol="Garch",
        p=garch_p,
        q=garch_q,
        o=garch_o,
        dist="normal"
    ).fit(disp="off")

    forecast = garch.forecast(horizon=5)
    variance = forecast.variance.iloc[-1].values
    volatility = np.sqrt(variance)

    garch_prices = []
    price = current_price

    for vol in volatility:
        price *= (1 + np.random.normal(0, vol * 0.01))
        garch_prices.append(price)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, 6), garch_prices, marker="o", label="Forecast Price")
    ax.set_title("GJR-GARCH 5-Day Price Forecast")
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel(f"Price ({currency})")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ─────────────────────────────────────────
    # ADX & STOCH RSI
    # ─────────────────────────────────────────
    st.subheader("📉 Technical Indicators")

    adx = compute_adx(data)
    stoch_rsi = compute_stoch_rsi(prices)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(adx, label="ADX", color="blue")
    ax1.axhline(25, linestyle="--", color="red", alpha=0.5)
    ax1.set_title("ADX (Trend Strength)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(stoch_rsi, label="Stochastic RSI", color="green")
    ax2.axhline(80, linestyle="--", color="red", alpha=0.5)
    ax2.axhline(20, linestyle="--", color="green", alpha=0.5)
    ax2.set_title("Stochastic RSI")
    ax2.legend()
    ax2.grid(alpha=0.3)

    st.pyplot(fig)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Polynomial Regression • GJR-GARCH • ADX • Stochastic RSI*\n\n"
    "**Created by Gokul Thanigaivasan**"
)
