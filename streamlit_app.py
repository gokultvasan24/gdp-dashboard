import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="📈 Intraday Forecast (Cloud Hardened)",
    layout="wide"
)

st.sidebar.header("⚙️ Intraday Settings")
ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Interval", ["5m", "15m"])
run_btn = st.sidebar.button("🚀 Run Forecast", use_container_width=True)

MAX_LOOKBACK = 30
MC_PATHS = 300
LAMBDA_EWMA = 0.94
RIDGE_ALPHA = 1.0

@st.cache_data(ttl=900)
def load_intraday_data(ticker, interval):
    try:
        df = yf.download(
            ticker,
            period="30d",
            interval=interval,
            progress=False,
            threads=False
        )
        return df
    except Exception:
        return pd.DataFrame()

def build_features(df):
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    short = max(3, min(5, len(df) // 10))
    long = max(10, min(20, len(df) // 5))

    df["vol_s"] = df["log_return"].rolling(short).std()
    df["vol_l"] = df["log_return"].rolling(long).std()
    df["volume_z"] = (
        (df["Volume"] - df["Volume"].rolling(long).mean())
        / df["Volume"].rolling(long).std()
    )

    return df.dropna()

def make_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback].flatten())
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)

def ridge_fit_safe(X, y, alpha):
    I = np.eye(X.shape[1])
    return np.linalg.pinv(X.T @ X + alpha * I) @ X.T @ y

def ewma_vol(returns, lam=0.94):
    var = np.var(returns)
    for r in returns:
        var = lam * var + (1 - lam) * r**2
    return np.sqrt(var)

if run_btn:

    try:
        data = load_intraday_data(ticker, interval)

        if data.empty or len(data) < 80:
            st.error("❌ No usable intraday data (Yahoo throttling or market closed)")
            st.stop()

        feats = build_features(data)

        if len(feats) < 50:
            st.error("❌ Insufficient clean data after indicators")
            st.stop()

        X_raw = feats[["log_return", "vol_s", "vol_l", "volume_z"]].values
        y_raw = feats["log_return"].values

        lookback = min(MAX_LOOKBACK, len(X_raw) // 3)
        lookback = max(10, lookback)

        X, y = make_sequences(X_raw, y_raw, lookback)

        if len(X) < 30:
            st.error("❌ Not enough sequences for modeling")
            st.stop()

        weights = ridge_fit_safe(X, y, RIDGE_ALPHA)
        mean_ret = float(X[-1] @ weights)

        sigma = ewma_vol(y_raw[-lookback:], LAMBDA_EWMA)
        last_price = float(data["Close"].iloc[-1])

        sims = last_price * np.exp(
            mean_ret + np.random.normal(0, sigma, MC_PATHS)
        )

        st.subheader("🔮 Next-Bar Intraday Forecast")

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"{last_price:.2f}")
        c2.metric("Expected Price", f"{sims.mean():.2f}")
        c3.metric("Up Probability", f"{(sims > last_price).mean()*100:.1f}%")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(sims, bins=40, alpha=0.75)
        ax.axvline(last_price, linestyle="--", color="black")
        ax.set_title("Next-Bar Price Distribution")
        ax.grid(alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        st.success("✅ Forecast completed safely")

    except Exception as e:
        st.error("❌ Unexpected cloud error (handled safely)")
        st.exception(e)
