# streamlit_app.py
# Created by Gokul Thanigaivasan
# Advanced Stock Forecasting Platform with Polynomial Regression + GJR-GARCH + Technical Indicators

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# ARCH
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    st.warning("ARCH not installed. Run: pip install arch")

# =============================================================================
# STREAMLIT CONFIG
# =============================================================================

st.set_page_config(
    page_title="📈 Advanced Stock Forecasting Platform",
    layout="wide"
)

# =============================================================================
# SIDEBAR
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
# UTILS
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
# MAIN
# =============================================================================

if run_analysis:

    if not ARCH_AVAILABLE:
        st.error("ARCH package missing")
        st.stop()

    data = download_stock_data(ticker, start_date, end_date)

    if data.empty or price_type not in data.columns:
        st.error("Invalid ticker or price type")
        st.stop()

    currency = detect_currency(ticker)

    # =============================================================================
    # TECHNICAL INDICATORS
    # =============================================================================

    df = data.copy()

    # MACD
    macd = ta.macd(df[price_type])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_signal"] = macd["MACDs_12_26_9"]
    df["MACD_hist"] = macd["MACDh_12_26_9"]

    # ADX
    adx = ta.adx(df["High"], df["Low"], df["Close"])
    df["ADX"] = adx["ADX_14"]

    # Stochastic RSI
    stoch = ta.stochrsi(df[price_type])
    df["StochRSI"] = stoch["STOCHRSIk_14_14_3_3"]
    df["StochRSI_signal"] = stoch["STOCHRSId_14_14_3_3"]

    df.dropna(inplace=True)

    price_data = df[price_type]
    current_price = float(price_data.iloc[-1])

    # =============================================================================
    # POLYNOMIAL REGRESSION
    # =============================================================================

    st.subheader("📈 Polynomial Regression Forecast")

    dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1)
    dates_mean = dates.mean()
    dates_range = dates.max() - dates.min()

    X = (dates - dates_mean) / dates_range
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

    next_x = np.array([[X[-1][0] + (1 / dates_range)]])
    next_poly = poly.transform(next_x)
    forecast_price = float(model.predict(next_poly)[0])

    delta = forecast_price - current_price
    pct = delta / current_price * 100

    st.subheader("🎯 Next Day Forecast")
    c1, c2, c3 = st.columns(3)
    c1.metric("Current", f"{currency}{current_price:.2f}")
    c2.metric("Forecast", f"{currency}{forecast_price:.2f}", f"{delta:+.2f}")
    c3.metric("Change %", f"{pct:+.2f}%")

    # =============================================================================
    # GJR-GARCH
    # =============================================================================

    st.subheader("📊 GJR-GARCH Volatility")

    returns = calculate_returns(price_data)

    if len(returns) > 50:
        garch = fit_gjr_garch(returns, garch_p, garch_q, garch_o)
        forecast = garch.forecast(horizon=5)
        vol = np.sqrt(forecast.variance.iloc[-1])

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

    c1.metric("ADX", f"{last['ADX']:.2f}",
              "Strong Trend" if last["ADX"] > 25 else "Weak Trend")

    c2.metric("MACD",
              "Bullish" if last["MACD"] > last["MACD_signal"] else "Bearish")

    c3.metric("Stoch RSI",
              "Overbought" if last["StochRSI"] > 0.8 else
              "Oversold" if last["StochRSI"] < 0.2 else "Neutral")

    c4.metric("Volume", f"{int(last['Volume']):,}")

    # =============================================================================
    # VISUALS
    # =============================================================================

    st.subheader("📊 Charts")

    # Price
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(price_data.index, y, label="Actual")
    ax.plot(price_data.index, y_pred, linestyle="--", label="Regression")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Volume
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(df.index, df["Volume"])
    ax.set_title("Volume")
    st.pyplot(fig)

    # MACD
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df["MACD"], label="MACD")
    ax.plot(df.index, df["MACD_signal"], linestyle="--", label="Signal")
    ax.bar(df.index, df["MACD_hist"], alpha=0.4)
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # ADX
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["ADX"])
    ax.axhline(25, linestyle="--")
    ax.set_title("ADX")
    st.pyplot(fig)

    # Stoch RSI
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df["StochRSI"], label="%K")
    ax.plot(df.index, df["StochRSI_signal"], linestyle="--", label="%D")
    ax.axhline(0.8, linestyle="--")
    ax.axhline(0.2, linestyle="--")
    ax.legend()
    st.pyplot(fig)

    # =============================================================================
    # STAT TESTS
    # =============================================================================

    st.subheader("🔬 Statistical Diagnostics")

    residuals = y - y_pred
    jb_stat, jb_p = jarque_bera(residuals)
    adf_stat, adf_p, *_ = adfuller(residuals)

    st.write(f"Jarque-Bera p-value: **{jb_p:.4f}**")
    st.write(f"ADF p-value: **{adf_p:.4f}**")

    st.success("✅ Analysis Complete")
