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
from datetime import datetime

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Advanced Stock Forecasting Platform",
    layout="wide"
)

st.title("📈 Advanced Stock Forecasting Platform")
st.markdown("👨‍💻 **Created by Gokul Thanigaivasan**")

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Configuration Panel")

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

run_btn = st.sidebar.button("🚀 Run Analysis", type="primary")

# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def download_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)


def detect_currency(ticker):
    if any(x in ticker for x in ["^NSEI", ".NS", "NIFTY"]):
        return "₹"
    return "$"


def performance_metrics(y_true, y_pred, currency):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))

    return {
        "RMSE": f"{currency}{rmse:.2f}",
        "R²": f"{r2:.4f}",
        "MAE": f"{currency}{mae:.2f}",
    }

# ──────────────────────────────────────────────────────────────
# MAIN LOGIC
# ──────────────────────────────────────────────────────────────
if run_btn:

    data = download_data(ticker, start_date, end_date)

    if data.empty or price_type not in data.columns:
        st.error("❌ Invalid ticker or price type.")
        st.stop()

    prices = data[price_type].dropna()

    if len(prices) < 30:
        st.error("❌ Not enough data points.")
        st.stop()

    currency = detect_currency(ticker)
    current_price = float(prices.iloc[-1])

    st.subheader("📊 Data Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"{currency}{current_price:.2f}")
    c2.metric("Data Points", len(prices))
    c3.metric("Currency", currency)

    # ──────────────────────────────────────────────────────────
    # POLYNOMIAL REGRESSION
    # ──────────────────────────────────────────────────────────
    st.subheader("📈 Polynomial Regression")

    # Prepare data
    X = np.array([d.toordinal() for d in prices.index], dtype=float).reshape(-1, 1)
    y = prices.values.astype(float)

    X_mean = X.mean()
    X_range = X.max() - X.min()

    if X_range == 0:
        st.error("❌ Date range invalid.")
        st.stop()

    X_norm = (X - X_mean) / X_range

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_norm)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)

    metrics = performance_metrics(y, y_pred, currency)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", metrics["RMSE"])
    c2.metric("R²", metrics["R²"])
    c3.metric("MAE", metrics["MAE"])

    # ──────────────────────────────────────────────────────────
    # NEXT DAY FORECAST (CLOUD SAFE)
    # ──────────────────────────────────────────────────────────
    last_x = X_norm[-1, 0]
    next_x = np.array([[last_x + (1 / X_range)]])
    next_x_poly = poly.transform(next_x)

    # 🔥 SAFE scalar extraction
    next_price = model.predict(next_x_poly).item()

    delta = next_price - current_price
    delta_pct = (delta / current_price) * 100

    st.subheader("🎯 Next-Day Forecast")
    st.metric(
        "Predicted Price",
        f"{currency}{next_price:.2f}",
        f"{delta:+.2f} ({delta_pct:+.2f}%)"
    )

    # ──────────────────────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────────────────────
    st.subheader("📊 Model Fit")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prices.index, y, label="Actual", linewidth=2)
    ax.plot(prices.index, y_pred, "--", label="Predicted", linewidth=2)
    ax.set_title(f"Polynomial Regression (Degree {degree})")
    ax.set_ylabel(f"Price ({currency})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ──────────────────────────────────────────────────────────
    # RESIDUAL ANALYSIS
    # ──────────────────────────────────────────────────────────
    residuals = y - y_pred

    st.subheader("🔬 Statistical Tests")

    jb_stat, jb_p = jarque_bera(residuals)
    adf_stat, adf_p, *_ = adfuller(residuals)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Jarque-Bera Test**")
        st.write(f"P-value: {jb_p:.4f}")
    with c2:
        st.write("**ADF Test**")
        st.write(f"P-value: {adf_p:.4f}")

    # ──────────────────────────────────────────────────────────
    # ACF / PACF
    # ──────────────────────────────────────────────────────────
    st.subheader("📉 ACF / PACF")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(residuals, ax=ax1, lags=20)
    plot_pacf(residuals, ax=ax2, lags=20)
    plt.tight_layout()
    st.pyplot(fig)

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Built with Streamlit • Polynomial Regression*\n\n"
    "**Created by Gokul Thanigaivasan**"
)
