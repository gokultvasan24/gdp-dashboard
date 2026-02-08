# streamlit_app.py
# ============================================================
# ADVANCED QUANTITATIVE STOCK FORECASTING PLATFORM
# Mean Model      : ARIMA
# Volatility Model: GJR-GARCH (leverage effect)
# No technical indicators (pure statistical modeling)
# Author: Gokul Thanigaivasan
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy.stats import jarque_bera

st.set_page_config(
    page_title="Advanced Quant Forecasting",
    layout="wide"
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Model Configuration")

ticker = st.sidebar.text_input("Ticker", "^NSEI")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

arima_p = st.sidebar.slider("AR Order (p)", 0, 5, 1)
arima_d = st.sidebar.slider("Difference (d)", 0, 2, 1)
arima_q = st.sidebar.slider("MA Order (q)", 0, 5, 1)

forecast_horizon = st.sidebar.slider("Forecast Days", 5, 20, 5)

run = st.sidebar.button("Run Advanced Model")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if run:
    # --------------------------------------------------------
    # Data Loading
    # --------------------------------------------------------
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        st.error("No data downloaded. Check ticker or date range.")
        st.stop()

    prices = data['Close'].dropna()

    st.subheader("Price Series")
    st.line_chart(prices)

    # --------------------------------------------------------
    # Log Returns
    # --------------------------------------------------------
    log_returns = 100 * np.log(prices / prices.shift(1)).dropna()

    # --------------------------------------------------------
    # ARIMA Mean Model
    # --------------------------------------------------------
    st.subheader("ARIMA Mean Model")

    arima_model = ARIMA(prices, order=(arima_p, arima_d, arima_q))
    arima_fit = arima_model.fit()

    arima_forecast = arima_fit.forecast(steps=forecast_horizon)

    st.text(arima_fit.summary())

    # --------------------------------------------------------
    # GJR-GARCH Volatility Model
    # --------------------------------------------------------
    st.subheader("GJR-GARCH Volatility Model")

    garch = arch_model(
        log_returns,
        vol='Garch',
        p=1,
        o=1,
        q=1,
        dist='normal'
    )

    garch_fit = garch.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=forecast_horizon)

    vol_forecast = np.sqrt(garch_forecast.variance.iloc[-1])

    col1, col2 = st.columns(2)

    with col1:
        st.metric("AIC", f"{garch_fit.aic:.2f}")
    with col2:
        st.metric("BIC", f"{garch_fit.bic:.2f}")

    # --------------------------------------------------------
    # Combined Forecast Visualization
    # --------------------------------------------------------
    st.subheader("Forecast: Mean + Risk")

    future_dates = pd.date_range(prices.index[-1], periods=forecast_horizon + 1, freq='B')[1:]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(prices.index, prices, label="Historical Price", color='black')
    ax1.plot(future_dates, arima_forecast, label="ARIMA Forecast", linestyle='--', color='blue')
    ax1.set_ylabel("Price")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(future_dates, vol_forecast, label="Forecast Volatility", color='red')
    ax2.set_ylabel("Volatility (%)")
    ax2.legend(loc='upper right')

    plt.title("Advanced Forecast: Expected Price & Volatility")
    plt.tight_layout()
    st.pyplot(fig)

    # --------------------------------------------------------
    # Residual Diagnostics
    # --------------------------------------------------------
    st.subheader("Model Diagnostics")

    residuals = arima_fit.resid.dropna()
    jb_stat, jb_p = jarque_bera(residuals)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jarque-Bera", f"{jb_stat:.2f}")
    with col2:
        st.metric("Normality p-value", f"{jb_p:.4f}")

    if jb_p > 0.05:
        st.success("Residuals appear normally distributed")
    else:
        st.warning("Residuals deviate from normality")

    st.success("Advanced Quantitative Analysis Complete")

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")
st.markdown("**Advanced Quant Modeling • ARIMA + GJR-GARCH • Risk-Aware Forecasting**")
