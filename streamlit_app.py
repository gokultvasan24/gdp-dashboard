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
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera, skew, kurtosis, norm
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# Try to import ARCH models
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    st.warning("ARCH package not available. Please install it using: pip install arch")

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

st.set_page_config(
    page_title="📈 Advanced Stock Forecasting Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .creator-credit {
        text-align: center;
        font-size: 1.1rem;
        color: #1f77b4;
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TITLE AND DESCRIPTION
# =============================================================================

st.markdown('<div class="main-header">Advanced Hybrid Stock Forecasting Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="creator-credit">👨‍💻 Created by <strong>Gokul Thanigaivasan</strong></div>', unsafe_allow_html=True)

with st.expander("📖 About this App", expanded=False):
    st.markdown("""
    **Created by:** Gokul Thanigaivasan
    
    **Features:**
    - 📊 Polynomial Regression + GJR-GARCH hybrid modeling
    - 🔍 Comprehensive statistical testing
    - 📈 Multi-timeframe analysis and forecasting
    - 💹 Volatility modeling with GJR-GARCH
    - 🎯 Interactive parameter tuning
    
    **Investment Disclaimer:**
    - This tool is for educational purposes only
    - Past performance doesn't guarantee future results
    - Always do your own research before investing
    """)

# =============================================================================
# SIDEBAR - UNIFIED INPUT SECTION
# =============================================================================

st.sidebar.header("🔧 Configuration Panel")

# Stock Selection with default ^NSEI (Nifty 50)
ticker = st.sidebar.text_input("Stock Ticker", "^NSEI").upper()

# Date Range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime(2020, 1, 1).date(),
        min_value=datetime(2010, 1, 1).date(),
        max_value=datetime.now().date()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now().date(),
        min_value=datetime(2010, 1, 1).date(),
        max_value=datetime.now().date()
    )

# Price Type Selection
st.sidebar.subheader("Price Configuration")
price_type = st.sidebar.selectbox("Price Type", ["Close", "High", "Low", "Open", "Adj Close"])

# Model Parameters
st.sidebar.subheader("Model Parameters")
degree = st.sidebar.slider("Polynomial Degree", 1, 8, 3)

# GJR-GARCH Parameters
st.sidebar.subheader("GJR-GARCH Configuration")
garch_p = st.sidebar.slider("GARCH p (Lag Order)", 1, 3, 1)
garch_q = st.sidebar.slider("GARCH q (Volatility Lags)", 1, 3, 1)
garch_o = st.sidebar.slider("GJR o (Asymmetric Terms)", 1, 3, 1)

# Analysis Control
run_analysis_btn = st.sidebar.button("🚀 Run Complete Analysis", type="primary", use_container_width=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def download_stock_data(ticker, start_date, end_date):
    """Download stock data with caching for performance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def detect_currency(ticker):
    """Detect currency based on ticker symbols"""
    indian_indicators = ['.NS', '.BO', '.NSE', '.BSE', '^NSEI', 'NIFTY']
    if any(indicator in ticker.upper() for indicator in indian_indicators):
        return "₹"
    else:
        return "$"

def fit_gjr_garch_model(returns, p=1, q=1, o=1):
    """Fit GJR-GARCH model with error handling"""
    try:
        model = arch_model(returns, vol='Garch', p=p, q=q, o=o, dist='normal')
        fitted_model = model.fit(disp='off')
        return fitted_model, None
    except Exception as e:
        return None, f"GJR-GARCH fitting failed: {str(e)}"

def create_performance_metrics(y_true, y_pred, currency_symbol):
    """Create comprehensive performance metrics"""
    y_true_flat = np.ravel(y_true)
    y_pred_flat = np.ravel(y_pred)
    
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    r2 = float(r2_score(y_true_flat, y_pred_flat))
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    
    return {
        'RMSE': f"{currency_symbol}{rmse:.4f}",
        'R²': f"{r2:.4f}",
        'MAE': f"{currency_symbol}{mae:.4f}"
    }

def calculate_returns(prices):
    """Calculate percentage returns from price series"""
    try:
        returns = prices.pct_change().dropna() * 100
        return returns
    except Exception as e:
        return None

def create_simple_garch_plot(price_forecasts, volatility_forecasts, current_price, currency_symbol):
    """Create a simple but robust GARCH forecast plot"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Price forecast as a simple line chart
        days = list(range(len(price_forecasts) + 1))
        prices_to_plot = [current_price] + price_forecasts
        
        ax1.plot(days, prices_to_plot, marker='o', linewidth=2, color='red', markersize=6)
        ax1.set_title("Price Forecast", fontsize=14, fontweight='bold')
        ax1.set_ylabel(f"Price ({currency_symbol})", fontweight='bold')
        ax1.set_xlabel("Days", fontweight='bold')
        ax1.set_xticks(days)
        ax1.set_xticklabels(['Today'] + [f'Day {i+1}' for i in range(len(price_forecasts))])
        ax1.grid(True, alpha=0.3)
        
        # Add price labels on points
        for i, (day, price) in enumerate(zip(days, prices_to_plot)):
            ax1.annotate(f'{price:.2f}', (day, price), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        # Plot 2: Volatility forecast as bar chart
        volatility_bars = ax2.bar(range(1, len(volatility_forecasts) + 1), 
                                volatility_forecasts, color='orange', alpha=0.7)
        ax2.set_title("Volatility Forecast", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Volatility (%)", fontweight='bold')
        ax2.set_xlabel("Forecast Days", fontweight='bold')
        ax2.set_xticks(range(1, len(volatility_forecasts) + 1))
        ax2.set_xticklabels([f'Day {i}' for i in range(1, len(volatility_forecasts) + 1)])
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, vol in zip(volatility_bars, volatility_forecasts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{vol:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig, None
        
    except Exception as e:
        return None, f"Simple plotting failed: {str(e)}"

def safe_forecast_generation(garch_model, current_price, returns, forecast_steps=5):
    """Safely generate forecasts from GARCH model"""
    try:
        # Generate forecasts
        forecast = garch_model.forecast(horizon=forecast_steps)
        
        # Get variance forecasts - handle different arch package versions
        if hasattr(forecast, 'variance') and hasattr(forecast.variance, 'iloc'):
            variance_forecasts = forecast.variance.iloc[-1].values
        else:
            # Use the last conditional volatility and project forward
            last_vol = garch_model.conditional_volatility.iloc[-1]
            variance_forecasts = np.full(forecast_steps, last_vol ** 2)
        
        # Calculate volatility (standard deviation)
        volatility_forecasts = np.sqrt(variance_forecasts)
        
        # Generate simple but realistic price forecasts
        price_forecasts = []
        current_forecast_price = current_price
        
        for i in range(forecast_steps):
            # Simple price projection based on random walk with volatility
            volatility = float(volatility_forecasts[i])
            random_change = np.random.normal(0, volatility * 0.01)  # Small random changes
            next_price = current_forecast_price * (1 + random_change)
            price_forecasts.append(float(next_price))
            current_forecast_price = next_price
        
        # Ensure all outputs are simple Python lists
        return {
            'price_forecasts': price_forecasts,
            'volatility_forecasts': [float(v) for v in volatility_forecasts],
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# =============================================================================
# MAIN ANALYSIS LOGIC
# =============================================================================

if run_analysis_btn:
    if not ARCH_AVAILABLE:
        st.error("ARCH package is not available. Please install it using: pip install arch")
        st.stop()
    
    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Data Loading
    status_text.text("📥 Downloading stock data...")
    data = download_stock_data(ticker, start_date, end_date)
    progress_bar.progress(20)
    
    if data is None or data.empty:
        st.error("❌ No data found. Please check the ticker symbol and date range.")
        st.stop()
    
    # Validate price columns
    if price_type not in data.columns:
        st.error(f"❌ Price column '{price_type}' not found. Available: {list(data.columns)}")
        st.stop()
    
    # Data preprocessing
    status_text.text("🔧 Processing data...")
    price_data = data[price_type].dropna()
    
    if len(price_data) < 30:
        st.error("❌ Insufficient data points after cleaning. Please expand date range.")
        st.stop()
    
    currency_symbol = detect_currency(ticker)
    progress_bar.progress(40)
    
    # =============================================================================
    # DATA OVERVIEW SECTION
    # =============================================================================
    
    st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_price = float(price_data.iloc[-1])
        st.metric(f"Current {price_type}", f"{currency_symbol}{current_price:.2f}")
    with col2:
        st.metric("Data Points", len(price_data))
    with col3:
        period_days = (price_data.index[-1] - price_data.index[0]).days
        st.metric("Analysis Period", f"{period_days} days")
    with col4:
        st.metric("Currency", currency_symbol)
    
    # Price chart
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(price_data.index, price_data.values, linewidth=1, alpha=0.8, color='steelblue')
        ax.set_title(f"{ticker} {price_type} Price History")
        ax.set_ylabel(f"Price ({currency_symbol})")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display price chart: {str(e)}")
    
    progress_bar.progress(50)
    
    # =============================================================================
    # POLYNOMIAL REGRESSION SECTION
    # =============================================================================
    
    st.markdown('<div class="section-header">📈 Polynomial Regression Analysis</div>', unsafe_allow_html=True)
    
    status_text.text("🔮 Running polynomial regression...")
    
    try:
        # Prepare features
        dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1).astype(float)
        dates_mean = float(dates.mean())
        dates_range = float(dates.max() - dates.min())
        
        if dates_range == 0:
            st.error("All dates are identical. Please check date range.")
            st.stop()
        
        X = (dates - dates_mean) / dates_range
        y = price_data.values.astype(float)
        
        # Polynomial regression
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        # Performance metrics
        metrics = create_performance_metrics(y, y_pred, currency_symbol)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", metrics['RMSE'])
        col2.metric("R² Score", metrics['R²'])
        col3.metric("MAE", metrics['MAE'])
        
        # Forecast next day
        last_normalized_date = X[-1][0]
        next_normalized_date = last_normalized_date + (1 / dates_range)
        next_day_features = np.array([[next_normalized_date]])
        next_day_poly = poly.transform(next_day_features)
        next_day_pred = model.predict(next_day_poly)
        
        forecast_value = float(next_day_pred[0])
        price_change = forecast_value - current_price
        percent_change = (price_change / current_price) * 100
        
        # Display forecast
        st.subheader("🎯 Next Day Forecast")
        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
        with forecast_col1:
            st.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
        with forecast_col2:
            st.metric("Predicted Price", f"{currency_symbol}{forecast_value:.2f}", 
                     f"{price_change:+.2f}")
        with forecast_col3:
            st.metric("Expected Change", f"{percent_change:+.2f}%")
            
    except Exception as e:
        st.error(f"Polynomial regression failed: {str(e)}")
        st.stop()
    
    progress_bar.progress(70)
    
    # =============================================================================
    # GJR-GARCH VOLATILITY FORECASTING SECTION
    # =============================================================================
    
    st.markdown('<div class="section-header">📊 GJR-GARCH Volatility Analysis</div>', unsafe_allow_html=True)
    
    status_text.text("🔍 Running GJR-GARCH analysis...")
    
    # Calculate returns for GARCH modeling
    returns = calculate_returns(price_data)
    
    forecast_data_stored = None
    
    if returns is not None and len(returns) > 50:
        try:
            # Fit GJR-GARCH model
            with st.spinner("Fitting GJR-GARCH model..."):
                garch_model, garch_error = fit_gjr_garch_model(
                    returns, p=garch_p, q=garch_q, o=garch_o
                )
            
            if garch_error:
                st.error(f"GJR-GARCH modeling failed: {garch_error}")
            else:
                # Display model summary
                st.subheader("🏆 GJR-GARCH Model Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Model", f"GJR-GARCH({garch_p},{garch_q},{garch_o})")
                col2.metric("AIC", f"{garch_model.aic:.2f}")
                col3.metric("BIC", f"{garch_model.bic:.2f}")
                
                # Generate forecasts
                forecast_steps = 5
                forecast_result = safe_forecast_generation(garch_model, current_price, returns, forecast_steps)
                
                if forecast_result['success']:
                    price_forecasts = forecast_result['price_forecasts']
                    volatility_forecasts = forecast_result['volatility_forecasts']
                    
                    # Display GJR-GARCH forecast
                    st.subheader("📅 GJR-GARCH 5-Day Forecast")
                    forecast_dates = [price_data.index[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
                    
                    forecast_data = []
                    for i in range(forecast_steps):
                        forecast_price = price_forecasts[i]
                        volatility = volatility_forecasts[i]
                        change = forecast_price - current_price
                        change_pct = (change / current_price) * 100
                        
                        forecast_data.append({
                            'Day': i + 1,
                            'Date': forecast_dates[i].strftime('%Y-%m-%d'),
                            'Price': f"{currency_symbol}{forecast_price:.2f}",
                            'Volatility': f"{volatility:.2f}%",
                            'Change': f"{change:+.2f}",
                            'Change %': f"{change_pct:+.2f}%"
                        })
                    
                    # Display as dataframe
                    forecast_df = pd.DataFrame(forecast_data)
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Volatility insights
                    st.subheader("💡 Volatility Insights")
                    avg_volatility = np.mean(volatility_forecasts)
                    max_volatility = np.max(volatility_forecasts)
                    
                    vol_col1, vol_col2 = st.columns(2)
                    with vol_col1:
                        st.metric("Average Forecast Volatility", f"{avg_volatility:.2f}%")
                    with vol_col2:
                        st.metric("Maximum Forecast Volatility", f"{max_volatility:.2f}%")
                    
                    if avg_volatility > 3.0:
                        st.warning("🔴 High volatility expected - Consider risk management")
                    elif avg_volatility > 1.5:
                        st.info("🟡 Moderate volatility expected - Normal market conditions")
                    else:
                        st.success("🟢 Low volatility expected - Stable market conditions")
                    
                    # Store forecasts for visualization
                    forecast_data_stored = {
                        'price_forecasts': price_forecasts,
                        'volatility_forecasts': volatility_forecasts
                    }
                else:
                    st.error(f"Forecast generation failed: {forecast_result['error']}")
                    
        except Exception as e:
            st.error(f"GJR-GARCH analysis failed: {str(e)}")
    else:
        st.warning("Insufficient return data for GJR-GARCH modeling. Need at least 50 data points.")
    
    progress_bar.progress(90)
    
    # =============================================================================
    # STATISTICAL TESTS SECTION
    # =============================================================================
    
    st.markdown('<div class="section-header">🔬 Statistical Tests</div>', unsafe_allow_html=True)
    
    # Residuals analysis
    residuals = y - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Normality Test (Jarque-Bera)")
        try:
            jb_stat, jb_p = jarque_bera(residuals)
            st.write(f"Statistic: {jb_stat:.4f}")
            st.write(f"P-value: {jb_p:.4f}")
            if jb_p > 0.05:
                st.success("✓ Residuals are normally distributed")
            else:
                st.warning("✗ Residuals are not normally distributed")
        except Exception as e:
            st.error(f"Test failed: {str(e)}")
    
    with col2:
        st.subheader("Stationarity Test (ADF)")
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(residuals)
            st.write(f"Statistic: {adf_stat:.4f}")
            st.write(f"P-value: {adf_p:.4f}")
            if adf_p <= 0.05:
                st.success("✓ Residuals are stationary")
            else:
                st.warning("✗ Residuals are not stationary")
        except Exception as e:
            st.error(f"Test failed: {str(e)}")
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    # =============================================================================
    # VISUALIZATION SECTION
    # =============================================================================
    
    st.markdown('<div class="section-header">📊 Visualizations</div>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Model Fit", "Residuals", "ACF/PACF", "GARCH Forecast"])
    
    with tab1:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(price_data.index, y, label='Actual', linewidth=2, alpha=0.8, color='blue')
            ax.plot(price_data.index, y_pred, label='Predicted', linestyle='--', linewidth=2, color='red')
            ax.set_title(f"Polynomial Regression Fit (Degree {degree})")
            ax.set_ylabel(f"Price ({currency_symbol})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not create model fit plot: {str(e)}")
    
    with tab2:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Residuals time series
            ax1.plot(price_data.index, residuals, color='red', linewidth=1)
            ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax1.set_title("Residuals Over Time")
            ax1.set_ylabel("Residual")
            ax1.grid(True, alpha=0.3)
            
            # Residuals histogram
            ax2.hist(residuals, bins=50, alpha=0.7, color='red', edgecolor='black', density=True)
            ax2.set_title("Residuals Distribution")
            ax2.set_xlabel("Residual")
            ax2.set_ylabel("Density")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not create residuals plot: {str(e)}")
    
    with tab3:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            plot_acf(residuals, ax=ax1, lags=20, alpha=0.05)
            ax1.set_title("Autocorrelation Function (ACF)")
            plot_pacf(residuals, ax=ax2, lags=20, alpha=0.05)
            ax2.set_title("Partial Autocorrelation Function (PACF)")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not create ACF/PACF plots: {str(e)}")
    
    with tab4:
        if forecast_data_stored is not None:
            try:
                # Create simple GARCH forecast plot
                fig, plot_error = create_simple_garch_plot(
                    forecast_data_stored['price_forecasts'],
                    forecast_data_stored['volatility_forecasts'],
                    current_price,
                    currency_symbol
                )
                
                if plot_error:
                    st.error(plot_error)
                else:
                    st.pyplot(fig)
                    st.success("✅ GJR-GARCH Forecast Visualization Complete!")
                    
            except Exception as e:
                st.error(f"Could not create GARCH forecast plot: {str(e)}")
        else:
            st.info("No GJR-GARCH forecast available to display. This may be due to insufficient data or model fitting issues.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><em>Built with Streamlit • Powered by Python • GJR-GARCH Volatility Modeling</em></p>
    <p><strong>Created by Gokul Thanigaivasan</strong></p>
</div>
""", unsafe_allow_html=True)
