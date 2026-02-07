import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="📈 Intraday Forecast (Streamlit Cloud Safe)",
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
# CONSTANTS
# ===============================

LOOKBACK = 30          # bars
MC_PATHS = 300         # Monte Carlo simulations
LAMBDA_EWMA = 0.94     # volatility decay
RIDGE_ALPHA = 1.0     # regularization

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
    df["vol_5"] = df["log_return"].rolling(5).std()
    df["vol_20"] = df["log_return"].rolling(20).std()
    df["volume_z"] = (
        (df["Volume"] - df["Volume"].rolling(20).mean())
        / df["Volume"].rolling(20).std()
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
# MAIN LOGIC
# ===============================

if run_btn:

    st.subheader("📊 Downloading Intraday Data")

    data = load_intraday_data(ticker, interval)

    if data.empty or len(data) < 150:
        st.error("❌ Not enough intraday data")
        st.stop()

    features_df = build_features(data)

    feature_cols = ["log_return", "vol_5", "vol_20", "volume_z"]

    features = features_df[feature_cols].values
    target = features_df["log_return"].values

    X, y = make_sequences(features, target, LOOKBACK)

    if len(X) < 50:
        st.error("❌ Not enough samples after feature engineering")
        st.stop()

    # ===============================
    # MODEL FIT
    # ===============================

    weights = ridge_fit(X, y, RIDGE_ALPHA)

    last_x = X[-1]
    mean_return = float(last_x @ weights)

    # ===============================
    # VOLATILITY
    # ===============================

    sigma = ewma_volatility(target[-LOOKBACK:], LAMBDA_EWMA)

    last_price = float(data["Close"].iloc[-1])

    # ===============================
    # MONTE CARLO FORECAST
    # ===============================

    simulated_prices = []

    for _ in range(MC_PATHS):
        shock = np.random.normal(0, sigma)
        next_price = last_price * np.exp(mean_return + shock)
        simulated_prices.append(next_price)

    simulated_prices = np.array(simulated_prices)

    # ===============================
    # METRICS
    # ===============================

    expected_price = simulated_prices.mean()
    direction_confidence = (simulated_prices > last_price).mean() * 100

    # ===============================
    # DISPLAY
    # ===============================

    st.subheader("🔮 Next-Bar Intraday Forecast")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"{last_price:.2f}")
    col2.metric("Expected Price", f"{expected_price:.2f}")
    col3.metric("Upward Probability", f"{direction_confidence:.1f}%")

    # ===============================
    # DISTRIBUTION PLOT
    # ===============================

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(simulated_prices, bins=40, alpha=0.75)
    ax.axvline(last_price, color="black", linestyle="--", label="Current Price")
    ax.set_title("Next-Bar Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.success("✅ Intraday Forecast Completed Successfully")
