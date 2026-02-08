# streamlit_app.py
# Created by Gokul Thanigaivasan
# Advanced Stock Forecasting Platform with Polynomial Regression + GJR-GARCH

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# Try to import ARCH models
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    st.warning("ARCH package not available. Install using: pip install arch")

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(
    page_title="📈 Advanced Stock Forecasting Platform",
    layout="wide"
)

# =============================================================================
# SIDEBAR INPUTS
# =============================================================================

st.sidebar.header("🔧 Configuration Panel")

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

run_analysis = st.sidebar.button("🚀 Run Complete Analysis", use_container_width=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def download_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)

def detect_currency(ticker):
    if any(x in ticker for x in ["^NSEI", ".NS", ".BO"]):
        return "₹"
    return "$"

def calculate_returns(prices):
    return prices.pct_change().dropna() * 100

def fit_gjr_garch(returns, p, q, o):
    model = arch_model(returns, vol="Garch", p=p, q=q, o=o, dist="normal")
    return model.fit(disp="off")

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if run_analysis:

    if not ARCH_AVAILABLE:
        st.error("ARCH package not installed")
        st.stop()

    data = download_stock_data(ticker, start_date, end_date)

    if data.empty or price_type not in data.columns:
        st.error("Invalid data or price column")
        st.stop()

    price_data = data[price_type].dropna()

    if len(price_data) < 30:
        st.error("Not enough data")
        st.stop()

    currency = detect_currency(ticker)
    current_price = float(price_data.iloc[-1])

    # =============================================================================
    # POLYNOMIAL REGRESSION
    # =============================================================================

    st.subheader("📈 Polynomial Regression")

    dates = np.array([d.toordinal() for d in price_data.index], dtype=float).reshape(-1, 1)

    dates_mean = dates.mean()
    dates_range = dates.max() - dates.min()

    X = (dates - dates_mean) / dates_range
    y = price_data.values.astype(float)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{currency}{rmse:.4f}")
    col2.metric("R²", f"{r2:.4f}")
    col3.metric("MAE", f"{currency}{mae:.4f}")

    # ===== FIXED SCALAR HANDLING (IMPORTANT PART) =====

    last_normalized_date = float(X[-1].item())
    next_normalized_date = last_normalized_date + (1 / dates_range)

    next_day_features = np.array([[next_normalized_date]])
    next_day_poly = poly.transform(next_day_features)

    next_day_pred = model.predict(next_day_poly)
    forecast_value = float(next_day_pred.item())

    price_change = forecast_value - current_price
    percent_change = (price_change / current_price) * 100

    st.subheader("🎯 Next Day Forecast")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"{currency}{current_price:.2f}")
    c2.metric("Predicted Price", f"{currency}{forecast_value:.2f}", f"{price_change:+.2f}")
    c3.metric("Expected Change", f"{percent_change:+.2f}%")

    # =============================================================================
    # GJR-GARCH
    # =============================================================================

    st.subheader("📊 GJR-GARCH Volatility")

    returns = calculate_returns(price_data)

    if len(returns) > 50:
        garch_model = fit_gjr_garch(returns, garch_p, garch_q, garch_o)

        forecast = garch_model.forecast(horizon=5)
        variance = forecast.variance.iloc[-1].values
        volatility = np.sqrt(variance)

        forecast_prices = []
        temp_price = current_price

        for v in volatility:
            change = np.random.normal(0, v * 0.01)
            temp_price *= (1 + change)
            forecast_prices.append(float(temp_price))

        df = pd.DataFrame({
            "Day": range(1, 6),
            "Forecast Price": [f"{currency}{p:.2f}" for p in forecast_prices],
            "Volatility (%)": [f"{v:.2f}%" for v in volatility]
        })

        st.dataframe(df, use_container_width=True)

    else:
        st.warning("Not enough data for GARCH")

    # =============================================================================
    # STATISTICAL TESTS
    # =============================================================================

    st.subheader("🔬 Statistical Tests")

    residuals = y - y_pred

    jb_stat, jb_p = jarque_bera(residuals)
    adf_stat, adf_p, *_ = adfuller(residuals)

    col1, col2 = st.columns(2)
    col1.write(f"Jarque-Bera p-value: **{jb_p:.4f}**")
    col2.write(f"ADF p-value: **{adf_p:.4f}**")

    # =============================================================================
    # VISUALS
    # =============================================================================

    st.subheader("📊 Visualizations")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(price_data.index, y, label="Actual")
    ax.plot(price_data.index, y_pred, linestyle="--", label="Predicted")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(residuals, ax=ax1, lags=20)
    plot_pacf(residuals, ax=ax2, lags=20)
    st.pyplot(fig)

    st.success("✅ Analysis Complete")
