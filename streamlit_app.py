import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="📈 Intraday Forecast (Robust Cloud Safe)",
    layout="wide"
)

# ===============================
# SIDEBAR
# ===============================

st.sidebar.header("⚙️ Intraday Settings")
ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Interval", ["5m", "15m"])
run_btn = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

# ===============================
# CONSTANTS (MAX VALUES)
# ===============================

MAX_LOOKBACK = 30
MC_PATHS = 300
LAMBDA_EWMA = 0.94
RIDGE_ALPHA = 1.0

# ===============================
# DATA LOADING
# ===============================

@st.cache_data(ttl=900)
def load_intraday_data(ticker, interval):
    return yf.download(
        ticker,
        period="30d",
        interval=interval,
        progress=False
    )

# ===============================
# FEATURE ENGINEERING
# ===============================

def build_features(df):
    df = df.copy()

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Adaptive rolling windows
    vol_short = min(5, len(df) // 10)
    vol_long = min(20, len(df) // 5)

    df["vol_5"] = df["log_return"].rolling(vol_short).std()
    df["vol_20"] = df["log_return"].rolling(vol_long).std()

    df["volume_z"] = (
        (df["Volume"] - df["Volume"].rolling(vol_long).mean())
        / df["Volume"].rolling(vol_long).std()
    )

    return df.dropna()

def make_sequences(features, target, lookback):
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i:i+lookback].flatten())
        y.append(target[i+lookback])
    return np.array(X), np.array(y)

# ===============================
# RIDGE REGRESSION
# ===============================

def ridge_fit(X, y, alpha):
    I = np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + alpha * I, X.T @ y)

# ===============================
# EWMA VOLATILITY
# ===============================

def ewma_volatility(returns, lam=0.94):
    sigma = np.var(returns)
    for r in returns:
        sigma = lam * sigma + (1 - lam) * r**2
    return np.sqrt(sigma)

# ===============================
# MAIN
# ===============================

if run_btn:

    st.subheader("📊 Loading Intraday Data")

    data = load_intraday_data(ticker, interval)

    if data.empty or len(data) < 80:
        st.error("❌ Not enough raw intraday data")
        st.stop()

    features_df = build_features(data)

    if len(features_df) < 60:
        st.error("❌ Too many missing values after indicators")
        st.stop()

    feature_cols = ["log_return", "vol_5", "vol_20", "volume_z"]
    features = features_df[feature_cols].values
    target = features_df["log_return"].values

    # ===============================
    # AUTO-ADJUST LOOKBACK
    # ===============================

    max_safe_lookback = min(MAX_LOOKBACK, len(features) // 3)
    lookback = max(10, max_safe_lookback)

    X, y = make_sequences(features, target, lookback)

    if len(X) < 30:
        st.error("❌ Still not enough samples for modeling")
        st.stop()

    st.info(f"ℹ️ Adaptive lookback used: {lookback} bars")

    # ===============================
    # MODEL
    # ===============================

    weights = ridge_fit(X, y, RIDGE_ALPHA)
    mean_return = float(X[-1] @ weights)

    # ===============================
    # VOLATILITY
    # ===============================

    sigma = ewma_volatility(target[-lookback:], LAMBDA_EWMA)
    last_price = float(data["Close"].iloc[-1])

    # ===============================
    # MONTE CARLO
    # ===============================

    simulated_prices = last_price * np.exp(
        mean_return + np.random.normal(0, sigma, MC_PATHS)
    )

    expected_price = simulated_prices.mean()
    up_prob = (simulated_prices > last_price).mean() * 100

    # ===============================
    # DISPLAY
    # ===============================

    st.subheader("🔮 Next-Bar Intraday Forecast")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"{last_price:.2f}")
    c2.metric("Expected Price", f"{expected_price:.2f}")
    c3.metric("Upward Probability", f"{up_prob:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(simulated_prices, bins=40, alpha=0.75)
    ax.axvline(last_price, color="black", linestyle="--")
    ax.set_title("Next-Bar Price Distribution")
    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.success("✅ Forecast completed successfully")
