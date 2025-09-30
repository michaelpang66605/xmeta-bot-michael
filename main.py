import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import smtplib, ssl
from email.mime.text import MIMEText
from datetime import datetime
import json
import os
import time

# ==========================
# CONFIGURATION
# ==========================
ARCHIVE_ID = 43128
PLATFORM_ID = 942
LOG_FILE = "xmeta_price_log.csv"
CHECK_INTERVAL = 300  # seconds
ALERT_LOW = 60.0
ALERT_HIGH = 70.0

# Email setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "输入邮箱"
RECEIVER_EMAIL = "输入邮箱"
EMAIL_PASSWORD = "enter password"

# XMeta API Authentication 
XMETA_API_KEY = "输入API" 
XMETA_TOKEN = "如果有输入TOKEN"      
XMETA_USER_ID = "USER_ID"      

# Monte Carlo 
N_SIMULATIONS = 1000  
FORECAST_HORIZON = 30  # days

# ==========================
# FUNCTIONS (ORIGINAL)
# ==========================
def fetch_price():
    """Fetch latest price from XMeta API with authentication"""
    try:
        url = "https://api.x-metash.cn/h5/goods/archiveGoods"
        payload = {
            "archiveId": ARCHIVE_ID,
            "platformId": PLATFORM_ID,
            "page": 1,
            "pageSize": 1
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://x-metash.cn/',
            'Origin': 'https://x-metash.cn'
        }
        
        # Add authentication if credentials are provided
        if XMETA_API_KEY != "输入API":
            headers['Authorization'] = f'Bearer {XMETA_API_KEY}'
            headers['X-API-Key'] = XMETA_API_KEY
            headers['X-Token'] = XMETA_TOKEN
            headers['X-User-ID'] = XMETA_USER_ID
            print("Using XMeta API authentication")
        else:
            print("No XMeta API credentials found - trying without authentication")
        
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        print(f"API Response: {data}")
        
        # Check if data structure is valid
        if not data or "data" not in data or not data["data"] or "list" not in data["data"] or not data["data"]["list"]:
            print("Warning: API returned invalid data structure")
            return None
            
        price = float(data["data"]["list"][0]["price"])
        print(f"Successfully fetched XMeta price: ${price}")
        return price
        
    except requests.exceptions.Timeout:
        print("API timeout - XMeta API is slow or down")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if e.response.status_code == 401:
            print("Authentication required - need XMeta API credentials")
        return None
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

def log_price(price):
    ts = datetime.now()
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if not exists:
            f.write("timestamp,price\n")
        f.write(f"{ts},{price}\n")

def send_email_alert(price):
    message = MIMEText(f"XMeta Price Alert Current price: {price:.2f}")
    message["Subject"] = "XMeta Price Alert"
    message["From"] = SENDER_EMAIL
    message["To"] = RECEIVER_EMAIL

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message.as_string())

def monte_carlo_forecast(price_series):
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu, sigma = log_returns.mean(), log_returns.std()
    last_price = price_series.iloc[-1]

    simulations = np.zeros((FORECAST_HORIZON, N_SIMULATIONS))
    for i in range(N_SIMULATIONS):
        prices = [last_price]
        for t in range(FORECAST_HORIZON):
            prices.append(prices[-1] * np.exp(np.random.normal(mu, sigma)))
        simulations[:, i] = prices[1:]
    return simulations

def compute_risk_metrics(price_series):
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    annual_vol = log_returns.std() * np.sqrt(252)
    sharpe = (log_returns.mean() * 252) / annual_vol if annual_vol > 0 else 0
    var_95 = np.percentile(log_returns, 5)
    return annual_vol, sharpe, var_95

def build_plot(price_series, forecast):
    future_dates = pd.date_range(start=price_series.index[-1], periods=FORECAST_HORIZON+1, freq="D")[1:]
    forecast_mean = forecast.mean(axis=1)
    lower_bound = np.percentile(forecast, 5, axis=1)
    upper_bound = np.percentile(forecast, 95, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_series.index, y=price_series.values,
                             mode="lines", name="Historical", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_mean,
                             mode="lines", name="Forecast Mean", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=future_dates, y=lower_bound,
                             mode="lines", name="5% CI", line=dict(color="red", dash="dot")))
    fig.add_trace(go.Scatter(x=future_dates, y=upper_bound,
                             mode="lines", name="95% CI", line=dict(color="green", dash="dot")))
    fig.update_layout(title="XMeta Price Forecast", xaxis_title="Date", yaxis_title="Price",
                      template="plotly_dark")
    return fig

# ==========================
# 主要功能
# ==========================

def compute_moving_averages(price_series, windows=[5, 10, 20, 50, 100]):
    """Return a DataFrame of moving averages for the given windows."""
    ma = pd.DataFrame(index=price_series.index)
    for w in windows:
        ma[f"SMA_{w}"] = price_series.rolling(window=w, min_periods=1).mean()
        ma[f"EMA_{w}"] = price_series.ewm(span=w, adjust=False).mean()
    return ma

def compute_RSI(price_series, period=14):
    """Compute Relative Strength Index (RSI)"""
    delta = price_series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(price_series, span_short=12, span_long=26, span_signal=9):
    ema_short = price_series.ewm(span=span_short, adjust=False).mean()
    ema_long = price_series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    hist = macd - signal
    df = pd.DataFrame({"MACD": macd, "MACD_Signal": signal, "MACD_Hist": hist}, index=price_series.index)
    return df

def compute_bollinger_bands(price_series, window=20, num_std=2):
    sma = price_series.rolling(window=window, min_periods=1).mean()
    std = price_series.rolling(window=window, min_periods=1).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return pd.DataFrame({"BB_Middle": sma, "BB_Upper": upper, "BB_Lower": lower}, index=price_series.index)

def compute_ATR_like(price_series, window=14):
    """
    Approximate Average True Range (since we only have price not OHLC).
    We'll approximate using rolling mean of absolute daily returns scaled to levels.
    """
    daily_diff = price_series.diff().abs()
    atr_like = daily_diff.rolling(window=window, min_periods=1).mean()
    return atr_like

def compute_all_technical_indicators(price_series):
    """Return a single DataFrame with all indicators for convenience."""
    df_ind = pd.DataFrame(index=price_series.index)
    df_ind["price"] = price_series
    ma = compute_moving_averages(price_series)
    df_ind = df_ind.join(ma)
    df_ind["RSI_14"] = compute_RSI(price_series)
    macd = compute_MACD(price_series)
    df_ind = df_ind.join(macd)
    bb = compute_bollinger_bands(price_series)
    df_ind = df_ind.join(bb)
    df_ind["ATR_Like_14"] = compute_ATR_like(price_series)
    return df_ind

# ==========================
# ADVANCED ANALYTICS
# ==========================
def compute_correlation(price_series, other_series):
    """Compute correlation between price_series and other_series (aligned on index)."""
    joined = pd.concat([price_series, other_series], axis=1).dropna()
    if joined.shape[0] < 2:
        return np.nan
    return joined.iloc[:, 0].corr(joined.iloc[:, 1])

def compute_beta(price_series, benchmark_series):
    """
    Compute beta regression coefficient (covariance / variance) between stock and benchmark.
    benchmark_series should be the benchmark price series (e.g., S&P 500).
    """
    joined = pd.concat([price_series, benchmark_series], axis=1).dropna()
    if joined.shape[0] < 2:
        return np.nan
    logr = np.log(joined / joined.shift(1)).dropna()
    cov = np.cov(logr.iloc[:,0], logr.iloc[:,1])[0,1]
    var = np.var(logr.iloc[:,1])
    if var == 0:
        return np.nan
    beta = cov / var
    return beta

def detect_seasonality(price_series, period=7):
    """
    Simple seasonality detection: compute average weekday pattern or weekly seasonality.
    Returns a small summary dict with autocorrelation at lag equal to period and monthly averages.
    """
    result = {}
    if price_series.shape[0] < period * 2:
        result["autocorr_lag"] = np.nan
        result["monthly_mean"] = {}
        return result
    autocorr = price_series.autocorr(lag=period)
    result["autocorr_lag"] = autocorr
    # monthly/weekly pattern (if we have dates)
    try:
        idx = price_series.index
        if hasattr(idx, "month"):
            monthly = price_series.groupby(price_series.index.month).mean().to_dict()
            result["monthly_mean"] = monthly
        else:
            result["monthly_mean"] = {}
    except Exception:
        result["monthly_mean"] = {}
    return result

def regime_detection(price_series, vol_window=20, vol_multiplier=1.5):
    """
    Very simple regime detection based on rolling volatility:
    - 'low' if vol < threshold_low
    - 'high' if vol > threshold_high
    """
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    rolling_vol = log_returns.rolling(window=vol_window, min_periods=1).std()
    median_vol = rolling_vol.median()
    threshold_high = median_vol * vol_multiplier
    current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else np.nan
    if np.isnan(current_vol):
        regime = "unknown"
    elif current_vol > threshold_high:
        regime = "high_volatility"
    else:
        regime = "low_volatility"
    return {"regime": regime, "current_vol": current_vol, "threshold_high": threshold_high, "median_vol": median_vol}

def scenario_analysis_hook(price_series, shock_pct=-0.1, days=7):
    """
    Simple scenario hook: apply an instant shock (e.g., -10%) and produce a short deterministic path
    for 'days' based on recent drift. This is a hook you can expand to run alternate MC sims.
    """
    last_price = price_series.iloc[-1]
    shocked_price = last_price * (1 + shock_pct)
    # compute recent drift from log returns
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    mu = log_returns.mean() if not log_returns.empty else 0
    scenario = [shocked_price]
    for _ in range(days-1):
        scenario.append(scenario[-1] * np.exp(mu))
    future_index = pd.date_range(start=price_series.index[-1], periods=days+1, freq="D")[1:]
    return pd.Series(scenario, index=future_index)

# ==========================
# 仪表盘 
# ==========================
def main():
    st.title("XMeta 仪表盘 - PZM")
    st.markdown("Real-time XMeta price tracking, Monte Carlo simulation, and risk metrics.")
    
    # Sidebar toggles 
    st.sidebar.header("Extras / Visualizations")
    show_technical = st.sidebar.checkbox("Show Technical Indicators (MA, RSI, MACD, BB)", value=True)
    show_advanced = st.sidebar.checkbox("Show Advanced Analytics (beta, seasonality, regime)", value=False)
    benchmark_csv = st.sidebar.file_uploader("Optional: Upload benchmark CSV (timestamp,price) for beta/correlation", type=["csv"])
    scenario_shock = st.sidebar.slider("Scenario shock (%)", min_value=-50, max_value=50, value=-10)
    scenario_days = st.sidebar.number_input("Scenario days", min_value=1, max_value=90, value=7)

    # Fetch & log live price
    with st.spinner("Fetching XMeta price..."):
        price = fetch_price()
    if price is None:
        st.error("Failed to fetch current price from XMeta API")
        st.warning("The XMeta API is currently down or requires authentication")
        st.info("To get real data, you need XMeta API credentials")
        st.code("""
# Add these to your code:
XMETA_API_KEY = "输入API"
XMETA_TOKEN = "如果有输入TOKEN"
XMETA_USER_ID = "USER_ID"
        """)
        return
    else:
        st.success(f"Successfully fetched XMeta price: ${price:.2f}")
        log_price(price)

    # Load historical data
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
    else:
        st.warning("No historical data found. Creating sample data for demonstration.")
        # Create sample data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        sample_prices = np.random.normal(65, 5, 30)  # Sample prices around 65
        df = pd.DataFrame({'price': sample_prices}, index=dates)

    # Compute metrics & forecast (unchanged)
    forecast = monte_carlo_forecast(df["price"])
    vol, sharpe, var_95 = compute_risk_metrics(df["price"])

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"{price:.2f}")
    col2.metric("Annualized Volatility", f"{vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    st.markdown(f"**Value-at-Risk (95%)**: {var_95:.2%}")

    # Plot chart (original fig)
    fig = build_plot(df["price"], forecast)

    # === Add technical indicators and overlay traces if requested ===
    if show_technical:
        with st.spinner("Computing technical indicators..."):
            ind = compute_all_technical_indicators(df["price"])
        
        # Add a few indicator traces to the plot: SMA_20, EMA_20 (if present), Bollinger Bands
        if "SMA_20" in ind.columns:
            fig.add_trace(go.Scatter(x=ind.index, y=ind["SMA_20"], mode="lines", name="SMA 20", line=dict(width=1, dash="dash")))
        if "EMA_20" in ind.columns:
            fig.add_trace(go.Scatter(x=ind.index, y=ind["EMA_20"], mode="lines", name="EMA 20", line=dict(width=1, dash="dot")))
        if {"BB_Upper", "BB_Lower"}.issubset(ind.columns):
            fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Upper"], mode="lines", name="BB Upper", line=dict(width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_Lower"], mode="lines", name="BB Lower", line=dict(width=1, dash="dot")))
        # Add RSI & MACD as small separate charts below the main chart using Plotly subplots would be nicer,
        # but to keep it simple we display them as separate small figures.
        rsi_fig = None
        macd_fig = None
        try:
            rsi_series = ind["RSI_14"]
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values, mode="lines", name="RSI (14)"))
            rsi_fig.update_layout(height=200, title="RSI (14)", template="plotly_dark", yaxis=dict(range=[0, 100]))
        except Exception:
            rsi_fig = None

        try:
            macd_df = ind[["MACD", "MACD_Signal", "MACD_Hist"]].dropna()
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["MACD"], mode="lines", name="MACD"))
            macd_fig.add_trace(go.Scatter(x=macd_df.index, y=macd_df["MACD_Signal"], mode="lines", name="MACD Signal"))
            macd_fig.update_layout(height=200, title="MACD", template="plotly_dark")
        except Exception:
            macd_fig = None

        # Display a compact table of recent indicator values
        recent_ind = ind.tail(1).T
        recent_ind.columns = ["last_value"]
        st.markdown("**Latest Technical Indicator Values**")
        st.dataframe(recent_ind)

    # === Advanced analytics ===
    adv_res = {}
    if show_advanced:
        with st.spinner("Running advanced analytics..."):
            # If user supplied benchmark CSV, attempt to compute beta/correlation
            benchmark_series = None
            if benchmark_csv is not None:
                try:
                    bench_df = pd.read_csv(benchmark_csv, parse_dates=["timestamp"])
                    bench_df.set_index("timestamp", inplace=True)
                    # Expect a 'price' column in the uploaded CSV
                    if "price" in bench_df.columns:
                        benchmark_series = bench_df["price"]
                        adv_res["correlation_with_benchmark"] = compute_correlation(df["price"], benchmark_series)
                        adv_res["beta_vs_benchmark"] = compute_beta(df["price"], benchmark_series)
                    else:
                        st.warning("Benchmark CSV does not have 'price' column. Skipping beta/correlation.")
                except Exception as e:
                    st.warning(f"Failed to read benchmark CSV: {e}")
            else:
                adv_res["correlation_with_benchmark"] = np.nan
                adv_res["beta_vs_benchmark"] = np.nan

            # Seasonality and regime
            adv_res["seasonality"] = detect_seasonality(df["price"])
            adv_res["regime"] = regime_detection(df["price"])
            # Provide a short scenario analysis series
            adv_res["scenario_series"] = scenario_analysis_hook(df["price"], shock_pct=(scenario_shock/100.0), days=scenario_days)

        # Display advanced analytics in Streamlit
        st.markdown("### Advanced Analytics")
        colA, colB = st.columns(2)
        colA.metric("Correlation vs Benchmark", f"{adv_res.get('correlation_with_benchmark', np.nan):.3f}")
        beta_val = adv_res.get("beta_vs_benchmark", np.nan)
        colB.metric("Beta vs Benchmark", f"{beta_val:.3f}" if not np.isnan(beta_val) else "n/a")
        st.write("**Seasonality summary:**")
        st.json(adv_res["seasonality"])
        st.write("**Regime detection:**")
        st.json(adv_res["regime"])

        # Show scenario analysis plot
        st.markdown("**Scenario analysis (shock path)**")
        scen = adv_res["scenario_series"]
        scen_fig = go.Figure()
        scen_fig.add_trace(go.Scatter(x=scen.index, y=scen.values, mode="lines+markers", name="Scenario Path"))
        scen_fig.update_layout(title=f"Scenario: shock {scenario_shock}% then drift", template="plotly_dark")
        st.plotly_chart(scen_fig, use_container_width=True)

    # Finalize charts: show main figure and possible extra small charts for RSI/MACD
    st.plotly_chart(fig, use_container_width=True)
    if show_technical:
        if rsi_fig:
            st.plotly_chart(rsi_fig, use_container_width=True)
        if macd_fig:
            st.plotly_chart(macd_fig, use_container_width=True)

    # Price alert 
    if price <= ALERT_LOW or price >= ALERT_HIGH:
        send_email_alert(price)
        st.warning("Price crossed alert threshold! Email sent.")

def run_command_line():
    """Command line interface for the dashboard"""
    print("=" * 50)
    print("    XMeta Quant Bot - Command Line")
    print("=" * 50)
    print()
    print("Choose how to run:")
    print("1. Streamlit Dashboard (Interactive Web App)")
    print("2. Python Script (Command Line Analysis)")
    print()
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        print("\nStarting Streamlit dashboard...")
        print("The app will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        print()
        import subprocess
        import sys
        # Use the current Python executable (virtualenv)
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    elif choice == "2":
        print("\nRunning XMeta analysis...")
        print("Fetching XMeta price data...")
        # Run the analysis without Streamlit
        price = fetch_price()
        if price:
            print(f"XMeta Price: ${price:.2f}")
            log_price(price)
            print("Price logged to CSV file")
        else:
            print("Failed to fetch XMeta price")
            print("The XMeta API may be down or require authentication")
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    # Always show command line menu first
    run_command_line()
