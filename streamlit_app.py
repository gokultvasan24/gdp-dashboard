# streamlit_app.py
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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# Try to import ARCH models, install if not available
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    st.warning("ARCH package not available. Installing required packages...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "arch"])
    from arch import arch_model
    ARCH_AVAILABLE = True

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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TITLE AND DESCRIPTION
# =============================================================================

st.markdown('<div class="main-header">Advanced Hybrid Stock Forecasting Platform</div>', unsafe_allow_html=True)

with st.expander("📖 About this App", expanded=False):
    st.markdown("""
    **Features:**
    - 📊 Polynomial Regression + GJR-GARCH hybrid modeling
    - 🔍 Comprehensive statistical testing (ADF, KPSS, Jarque-Bera, Ljung-Box)
    - 📈 Multi-timeframe analysis and forecasting
    - 💹 Volatility modeling with GJR-GARCH
    - 🎯 Interactive parameter tuning
    - 📱 Responsive design for all devices
    
    **Mathematical Insights:**
    - "True understanding of markets lies in the art of mathematical analysis"
    - "Live your Life as an Exclamation rather than an Explanation" - Isaac Newton
    - "Earning in the face of Risk" - STOCK MARKET
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
price_type = st.sidebar.selectbox("Dependent Price (Y)", ["Close", "High", "Low", "Open", "Adj Close"])
price_type1 = st.sidebar.selectbox("Independent Price (X)", ["Open", "High", "Low", "Close"])

# Model Parameters
st.sidebar.subheader("Model Parameters")
degree = st.sidebar.slider("Polynomial Degree", 1, 8, 3, 
                          help="Higher degrees can capture more complex patterns but may overfit")

# GJR-GARCH Parameters
st.sidebar.subheader("GJR-GARCH Configuration")
garch_p = st.sidebar.slider("GARCH p (Lag Order)", 1, 3, 1)
garch_q = st.sidebar.slider("GARCH q (Volatility Lags)", 1, 3, 1)
garch_o = st.sidebar.slider("GJR o (Asymmetric Terms)", 1, 3, 1)

# Forecast Input with default 26100
st.sidebar.subheader("Forecast Input")
today_open_input = st.sidebar.number_input(
    "Today's Open Price",
    value=26100.0,
    min_value=0.0,
    step=100.0,
    help="Enter the current open price for prediction"
)

# Analysis Control
run_analysis_btn = st.sidebar.button("🚀 Run Complete Analysis", type="primary", use_container_width=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)  # Cache data for 1 hour
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
    # Ensure 1D arrays
    y_true_flat = np.ravel(y_true)
    y_pred_flat = np.ravel(y_pred)
    
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    r2 = float(r2_score(y_true_flat, y_pred_flat))
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    
    # Safe MAPE calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / np.where(y_true_flat != 0, y_true_flat, 1))) * 100
    mape = float(mape) if np.isfinite(mape) else float(0)
    
    return {
        'RMSE': f"{currency_symbol}{rmse:.4f}",
        'R²': f"{r2:.4f}",
        'MAE': f"{currency_symbol}{mae:.4f}",
        'MAPE': f"{mape:.2f}%"
    }

def safe_descriptive_stats(data, test_name=""):
    """Calculate descriptive statistics with proper error handling"""
    try:
        # Ensure data is 1D array and handle NaN/inf
        data_flat = np.ravel(data)
        data_clean = data_flat[np.isfinite(data_flat)]
        
        if len(data_clean) == 0:
            return None, f"No valid data points for {test_name}"
        
        stats = {
            'mean': float(np.mean(data_clean)),
            'std': float(np.std(data_clean)),
            'skewness': float(skew(data_clean)),
            'kurtosis': float(kurtosis(data_clean))
        }
        return stats, None
    except Exception as e:
        return None, f"{test_name} failed: {str(e)}"

def safe_stat_test(test_func, data, test_name=""):
    """Safe wrapper for statistical tests"""
    try:
        # Ensure data is 1D array
        data_flat = np.ravel(data)
        data_clean = data_flat[np.isfinite(data_flat)]
        
        if len(data_clean) == 0:
            return None, f"No valid data for {test_name}"
            
        return test_func(data_clean), None
    except Exception as e:
        return None, f"{test_name} failed: {str(e)}"

def calculate_returns(prices):
    """Calculate percentage returns from price series"""
    try:
        returns = prices.pct_change().dropna() * 100
        return returns
    except Exception as e:
        return None

def safe_plot_forecast(historical_dates, historical_prices, forecast_dates, forecast_values, volatility_forecast=None):
    """Safe plotting function for forecasts"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Price forecast
        ax1.plot(historical_dates, historical_prices, label='Historical Prices', 
               linewidth=2, color='blue')
        
        # Ensure forecast data is properly formatted
        forecast_dates_flat = list(forecast_dates)
        forecast_values_flat = [float(x) for x in np.ravel(forecast_values)]
        
        ax1.plot(forecast_dates_flat, forecast_values_flat, label='Price Forecast', 
               linewidth=2, marker='o', color='red')
        
        ax1.set_title("GJR-GARCH Price Forecast")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.sca(ax1)
        plt.xticks(rotation=45)
        
        # Plot 2: Volatility forecast
        if volatility_forecast is not None:
            volatility_flat = [float(x) for x in np.ravel(volatility_forecast)]
            ax2.plot(forecast_dates_flat, volatility_flat, label='Volatility Forecast', 
                   linewidth=2, marker='s', color='orange')
            ax2.set_title("GJR-GARCH Volatility Forecast")
            ax2.set_ylabel("Volatility (%)")
            ax2.set_xlabel("Date")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.sca(ax2)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig, None
        
    except Exception as e:
        return None, f"Plotting failed: {str(e)}"

# =============================================================================
# MAIN ANALYSIS LOGIC
# =============================================================================

if run_analysis_btn:
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
    required_columns = [price_type, price_type1]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"❌ Missing price columns: {missing_columns}. Available: {list(data.columns)}")
        st.stop()
    
    # Data preprocessing
    status_text.text("🔧 Processing data...")
    price_data = data[price_type].dropna()
    price_data1 = data[price_type1].dropna()
    
    # Align both series
    common_dates = price_data.index.intersection(price_data1.index)
    price_data = price_data.loc[common_dates]
    price_data1 = price_data1.loc[common_dates]
    
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
        # Prepare features - ensure proper array shapes
        dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1).astype(float)
        dates_mean = float(dates.mean())
        dates_range = float(dates.max() - dates.min())
        
        if dates_range == 0:
            st.error("All dates are identical. Please check date range.")
            st.stop()
        
        X = (dates - dates_mean) / dates_range
        y = price_data.values.astype(float)
        
        # Ensure y is 1D
        y_flat = np.ravel(y)
        
        # Polynomial regression
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y_flat)
        y_pred = model.predict(X_poly)
        
        # Ensure predictions are 1D
        y_pred_flat = np.ravel(y_pred)
        
        # Performance metrics
        metrics = create_performance_metrics(y_flat, y_pred_flat, currency_symbol)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", metrics['RMSE'])
        col2.metric("R² Score", metrics['R²'])
        col3.metric("MAE", metrics['MAE'])
        col4.metric("MAPE", metrics['MAPE'])
        
        # Forecast next day
        last_normalized_date = X[-1][0]
        next_normalized_date = last_normalized_date + (1 / dates_range)
        next_day_features = np.array([[next_normalized_date]])
        next_day_poly = poly.transform(next_day_features)
        next_day_pred = model.predict(next_day_poly)
        
        forecast_value = float(np.ravel(next_day_pred)[0])
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
    
    if returns is not None and len(returns) > 50:
        try:
            # Fit GJR-GARCH model
            with st.spinner("Fitting GJR-GARCH model..."):
                garch_model, garch_error = fit_gjr_garch_model(
                    returns, p=garch_p, q=garch_q, o=garch_o
                )
            
            if garch_error:
                st.error(f"GJR-GARCH modeling failed: {garch_error}")
                garch_forecast = None
            else:
                # Display model summary
                st.subheader("🏆 GJR-GARCH Model Summary")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Model", f"GJR-GARCH({garch_p},{garch_q},{garch_o})")
                col1.metric("AIC", f"{garch_model.aic:.2f}")
                col2.metric("BIC", f"{garch_model.bic:.2f}")
                col3.metric("Log Likelihood", f"{garch_model.loglikelihood:.2f}")
                
                # Display parameters
                st.subheader("📋 Model Parameters")
                params_df = pd.DataFrame({
                    'Parameter': garch_model.params.index,
                    'Value': garch_model.params.values,
                    'Std Error': garch_model.std_errors.values
                })
                st.dataframe(params_df, use_container_width=True)
                
                # Generate forecasts
                forecast_steps = 5
                garch_forecast = garch_model.forecast(horizon=forecast_steps, reindex=False)
                
                # Get volatility forecasts (standard deviation)
                conditional_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
                
                # Create price forecasts based on volatility and recent trend
                last_return = returns.iloc[-1]
                volatility_scale = conditional_volatility / 100  # Convert to decimal
                
                # Generate realistic price forecasts
                price_forecasts = []
                current_forecast_price = current_price
                
                for i in range(forecast_steps):
                    # Use volatility to generate realistic price movements
                    expected_return = last_return * 0.7  # Dampen the last return
                    random_component = np.random.normal(0, volatility_scale[i] * 2)
                    daily_return = (expected_return + random_component) / 100
                    
                    next_price = current_forecast_price * (1 + daily_return)
                    price_forecasts.append(next_price)
                    current_forecast_price = next_price
                
                # Display GJR-GARCH forecast
                st.subheader("📅 GJR-GARCH 5-Day Forecast")
                forecast_dates = [price_data.index[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
                
                forecast_data = []
                for i in range(forecast_steps):
                    forecast_price = float(price_forecasts[i])
                    volatility = float(conditional_volatility[i])
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
                avg_volatility = np.mean(conditional_volatility)
                max_volatility = np.max(conditional_volatility)
                
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
                    
        except Exception as e:
            st.error(f"GJR-GARCH analysis failed: {str(e)}")
            garch_forecast = None
    else:
        st.warning("Insufficient return data for GJR-GARCH modeling.")
        garch_forecast = None
    
    progress_bar.progress(90)
    
    # =============================================================================
    # STATISTICAL TESTS SECTION
    # =============================================================================
    
    st.markdown('<div class="section-header">🔬 Statistical Tests</div>', unsafe_allow_html=True)
    
    # Residuals analysis - ensure 1D array
    residuals = y_flat - y_pred_flat
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Normality Test (Jarque-Bera)")
        jb_result, jb_error = safe_stat_test(
            lambda x: jarque_bera(x), 
            residuals, 
            "Jarque-Bera"
        )
        
        if jb_error:
            st.error(jb_error)
        else:
            jb_stat, jb_p = jb_result
            st.write(f"Statistic: {jb_stat:.4f}")
            st.write(f"P-value: {jb_p:.4f}")
            if jb_p > 0.05:
                st.success("✓ Residuals are normally distributed")
            else:
                st.warning("✗ Residuals are not normally distributed")
    
    with col2:
        st.subheader("Stationarity Test (ADF)")
        adf_result, adf_error = safe_stat_test(
            lambda x: adfuller(x), 
            residuals, 
            "ADF"
        )
        
        if adf_error:
            st.error(adf_error)
        else:
            adf_stat, adf_p, _, _, _, _ = adf_result
            st.write(f"Statistic: {adf_stat:.4f}")
            st.write(f"P-value: {adf_p:.4f}")
            if adf_p <= 0.05:
                st.success("✓ Residuals are stationary")
            else:
                st.warning("✗ Residuals are not stationary")
    
    # Additional tests
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Autocorrelation Test (Ljung-Box)")
        lb_result, lb_error = safe_stat_test(
            lambda x: acorr_ljungbox(x, lags=10, return_df=True), 
            residuals, 
            "Ljung-Box"
        )
        
        if lb_error:
            st.error(lb_error)
        else:
            lb_stat = lb_result['lb_stat'].iloc[-1]
            lb_p = lb_result['lb_pvalue'].iloc[-1]
            st.write(f"Statistic: {lb_stat:.4f}")
            st.write(f"P-value: {lb_p:.4f}")
            if lb_p > 0.05:
                st.success("✓ No significant autocorrelation")
            else:
                st.warning("✗ Significant autocorrelation present")
    
    with col4:
        st.subheader("Descriptive Statistics")
        stats, stats_error = safe_descriptive_stats(residuals, "Descriptive statistics")
        
        if stats_error:
            st.error(stats_error)
        else:
            st.write(f"Mean: {stats['mean']:.6f}")
            st.write(f"Std Dev: {stats['std']:.6f}")
            st.write(f"Skewness: {stats['skewness']:.4f}")
            st.write(f"Kurtosis: {stats['kurtosis']:.4f}")
    
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
            ax.plot(price_data.index, y_flat, label='Actual', linewidth=2, alpha=0.8, color='blue')
            ax.plot(price_data.index, y_pred_flat, label='Predicted', linestyle='--', linewidth=2, color='red')
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
            xmin, xmax = ax2.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, np.mean(residuals), np.std(residuals))
            ax2.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
            ax2.set_title("Residuals Distribution")
            ax2.set_xlabel("Residual")
            ax2.set_ylabel("Density")
            ax2.legend()
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
        if garch_forecast is not None and 'price_forecasts' in locals():
            try:
                # Historical data (last 60 days)
                plot_days = min(60, len(price_data))
                historical_dates = price_data.index[-plot_days:]
                historical_prices = price_data.values[-plot_days:]
                
                # Prepare forecast data
                forecast_dates = [price_data.index[-1]] + [price_data.index[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
                forecast_values = [price_data.values[-1]] + price_forecasts
                
                # Get volatility forecasts
                conditional_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
                volatility_values = [0] + list(conditional_volatility)  # Add 0 for current day
                
                # Create plot
                fig, plot_error = safe_plot_forecast(
                    historical_dates, historical_prices, 
                    forecast_dates, forecast_values,
                    volatility_values
                )
                
                if plot_error:
                    st.error(plot_error)
                else:
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Could not create GARCH forecast plot: {str(e)}")
        else:
            st.info("No GJR-GARCH forecast available to display.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><em>"True understanding of markets lies in the art of mathematical analysis"</em></p>
    <p><em>"Live your Life as an Exclamation rather than an Explanation" - Isaac Newton</em></p>
    <p><em>Built with Streamlit • Powered by Python • GJR-GARCH Volatility Modeling</em></p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Investment Disclaimer:**
- This tool is for educational purposes only
- Past performance doesn't guarantee future results
- Always do your own research before investing
- Consult with financial advisors for investment decisions

**GJR-GARCH Model:**
- Captures volatility clustering and leverage effects
- Better for financial time series than ARIMA
- Accounts for asymmetric volatility responses
""")
