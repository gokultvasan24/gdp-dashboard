# streamlit_app.py
# ============================================================
# ADVANCED QUANT WORKFLOW: GARCH-READY STOCK SCREENING PIPELINE
# ------------------------------------------------------------
# This app:
# 1. Screens stocks for return normality
# 2. Tests stationarity (ADF)
# 3. Tests heteroscedasticity (ARCH LM)
# 4. Detects structural breaks (CUSUM)
# 5. Identifies ARCH-suitable equities
# 6. Extracts significant ACF/PACF lags
# 7. Exports results at each stage
# ------------------------------------------------------------
# Author: Gokul Thanigaivasan
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import acf, pacf
from statsmodels.stats.diagnostic import breaks_cusumolsresid

st.set_page_config(page_title="Quant GARCH Screening Pipeline", layout="wide")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Screening Configuration")

tickers = st.sidebar.text_area(
    "Tickers (comma-separated)",
    "^NSEI,RELIANCE.NS,TCS.NS,INFY.NS"
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

run = st.sidebar.button("Run Quant Screening")

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def get_log_returns(price_series):
    return 100 * np.log(price_series / price_series.shift(1)).dropna()

# ------------------------------------------------------------
# Main Logic
# ------------------------------------------------------------

if run:
    tickers_list = [t.strip() for t in tickers.split(',')]

    results = []
    acf_pacf_results = []

    st.subheader("Quant Screening Results")

    for ticker in tickers_list:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            prices = data['Close'].dropna()

            if len(prices) < 300:
                continue

            returns = get_log_returns(prices)
            squared_returns = returns ** 2

            # ------------------------------------------------
            # 1. Normality Test
            # ------------------------------------------------
            jb_stat, jb_p = jarque_bera(returns)
            normal = jb_p > 0.05

            # ------------------------------------------------
            # 2. Stationarity Test
            # ------------------------------------------------
            adf_stat, adf_p, *_ = adfuller(returns)
            stationary = adf_p < 0.05

            # ------------------------------------------------
            # 3. ARCH Effect Test
            # ------------------------------------------------
            arch_stat, arch_p, *_ = het_arch(returns)
            arch_effect = arch_p < 0.05

            # ------------------------------------------------
            # 4. Structural Break Test (CUSUM)
            # ------------------------------------------------
            cusum_stat, cusum_p, _ = breaks_cusumolsresid(returns)
            stable_structure = cusum_p > 0.05

            # ------------------------------------------------
            # 5. ACF/PACF on Squared Returns
            # ------------------------------------------------
            acf_vals = acf(squared_returns, nlags=10, fft=False)
            pacf_vals = pacf(squared_returns, nlags=10)

            sig_acf_lags = [i for i, v in enumerate(acf_vals) if abs(v) > 0.2 and i > 0]
            sig_pacf_lags = [i for i, v in enumerate(pacf_vals) if abs(v) > 0.2 and i > 0]

            # ------------------------------------------------
            # Suitability Decision
            # ------------------------------------------------
            suitable = stationary and arch_effect

            results.append({
                "Ticker": ticker,
                "JB p-value": round(jb_p, 4),
                "ADF p-value": round(adf_p, 4),
                "ARCH p-value": round(arch_p, 4),
                "CUSUM p-value": round(cusum_p, 4),
                "GARCH Suitable": suitable
            })

            acf_pacf_results.append({
                "Ticker": ticker,
                "Significant ACF Lags": sig_acf_lags,
                "Significant PACF Lags": sig_pacf_lags
            })

        except Exception as e:
            st.warning(f"Skipped {ticker}: {e}")

    # ------------------------------------------------------------
    # Display Results
    # ------------------------------------------------------------

    results_df = pd.DataFrame(results)
    acf_pacf_df = pd.DataFrame(acf_pacf_results)

    st.subheader("Screening Summary")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("ACF / PACF Lag Suggestions for GARCH")
    st.dataframe(acf_pacf_df, use_container_width=True)

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    st.download_button(
        "Download Screening Results (CSV)",
        results_df.to_csv(index=False),
        file_name="garch_screening_results.csv"
    )

    st.download_button(
        "Download ACF PACF Lags (CSV)",
        acf_pacf_df.to_csv(index=False),
        file_name="acf_pacf_lags.csv"
    )

    st.success("Quant screening pipeline completed successfully")
