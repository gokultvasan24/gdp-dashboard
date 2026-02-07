import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Intraday GRU Forecast", layout="wide")

# ===============================
# CONFIG
# ===============================

DEVICE = "cpu"   # Streamlit Cloud is CPU-only
LOOKBACK = 30
EPOCHS = 6
MC_PATHS = 300
LAMBDA_EWMA = 0.94

# ===============================
# SIDEBAR
# ===============================

st.sidebar.header("⚙️ Intraday Settings")

ticker = st.sidebar.text_input("Ticker", "^NSEI")
interval = st.sidebar.selectbox("Interval", ["5m", "15m"])
run_btn = st.sidebar.button("🚀 Run Forecast")

# ===============================
# DATA
# ===============================

@st.cache_data(ttl=900)
def load_data(ticker, interval):
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
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    return df.dropna()

def make_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback, 0])  # predict log_return
    return np.array(X), np.array(y)

# ===============================
# GRU MODEL
# ===============================

class GRUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRU(input_size, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

# ===============================
# EWMA VOLATILITY
# ===============================

def ewma_vol(returns, lam=0.94):
    sigma = returns.var()
    for r in returns:
        sigma = lam * sigma + (1 - lam) * r**2
    return np.sqrt(sigma)

# ===============================
# MAIN
# ===============================

if run_btn:

    st.subheader("📊 Downloading Intraday Data")
    data = load_data(ticker, interval)

    if data.empty or len(data) < 100:
        st.error("Not enough intraday data")
        st.stop()

    feats = build_features(data)
    feature_cols = ["log_return", "vol_5", "vol_20", "volume_z"]
    values = feats[feature_cols].values

    X, y = make_sequences(values, LOOKBACK)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = GRUModel(input_size=X.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    st.subheader("🧠 Training GRU (Fast)")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        optimizer.step()

    st.success("GRU training complete")

    # ===============================
    # FORECAST
    # ===============================

    model.eval()
    last_seq = torch.tensor(X[-1:], dtype=torch.float32)

    with torch.no_grad():
        mean_return = model(last_seq).item()

    returns = feats["log_return"].values[-LOOKBACK:]
    sigma = ewma_vol(returns, LAMBDA_EWMA)

    last_price = data["Close"].iloc[-1]

    # Monte Carlo
    prices = []
    for _ in range(MC_PATHS):
        shock = np.random.normal(0, sigma)
        ret = mean_return + shock
        prices.append(last_price * np.exp(ret))

    prices = np.array(prices)

    # ===============================
    # OUTPUT
    # ===============================

    st.subheader("🔮 Next-Bar Intraday Forecast")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"{last_price:.2f}")
    col2.metric("Expected Price", f"{np.mean(prices):.2f}")
    col3.metric("Direction Confidence", f"{(prices > last_price).mean()*100:.1f}%")

    # Fan chart
    fig, ax = plt.subplots(figsize=(10,4))
    ax.hist(prices, bins=40, alpha=0.7)
    ax.axvline(last_price, color="black", linestyle="--", label="Current")
    ax.set_title("Price Distribution (Next Bar)")
    ax.legend()
    st.pyplot(fig)

    st.success("✅ Intraday GRU Forecast Complete")
