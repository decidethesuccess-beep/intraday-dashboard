# intraday_dashboard_GPT.py
# This script creates a Streamlit dashboard for the DTS Intraday AI Strategy.
# It allows users to configure strategy parameters, run backtests,
# and visualize trading performance.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time as dt_time, timedelta
import os
import json
import logging
from dotenv import load_dotenv
from typing import Dict, Any
import threading # Import threading for background tasks

# Import your core trading system components
from redis_store import RedisStore
from angelone_api_patch import AngelOneAPI
from backtester import Backtester
from strategy import StrategyManager # Import StrategyManager to access its parameters
from llm_client import LLMClient # Import LLMClient
from ai_webhook import AIWebhook # Import AIWebhook
from sentiment_analyzer import SentimentAnalyzer # Import SentimentAnalyzer
from live_stream import LiveStreamManager # Import LiveStreamManager

# Configure logging for the Streamlit app
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

# --- Global Instances (initialized once) ---
@st.cache_resource
def get_angel_api():
    """Caches and returns a single instance of AngelOneAPI for historical data."""
    # Use the historical API key for data fetching in backtester
    angel_api = AngelOneAPI(api_key_to_use=os.getenv("ANGELONE_HISTORICAL_API_KEY"))
    if not angel_api.login():
        # Display the specific error message from AngelOneAPI
        st.error(f"Failed to log in to Angel One Historical API. Check credentials. Details: {angel_api.login_error_message}")
        st.stop()
    return angel_api

@st.cache_resource
def get_redis_store(_angel_api_instance: AngelOneAPI): # Added _angel_api_instance parameter
    """Caches and returns a single instance of RedisStore."""
    redis_store = RedisStore(angel_api=_angel_api_instance) # Pass angel_api_instance here
    if not redis_store.connect():
        st.error("Failed to connect to Redis. Please check Redis configuration.")
        st.stop() # Stop the app if Redis connection fails
    return redis_store

@st.cache_resource
def get_llm_client():
    """Caches and returns a single instance of LLMClient."""
    return LLMClient()

@st.cache_resource
def get_ai_webhook(_llm_client_instance: LLMClient): # Fixed: Added underscore to prevent hashing
    """Caches and returns a single instance of AIWebhook."""
    return AIWebhook(_llm_client_instance)

@st.cache_resource
def get_sentiment_analyzer(_llm_client_instance: LLMClient, _redis_store_instance: RedisStore):
    """Caches and returns a single instance of SentimentAnalyzer."""
    return SentimentAnalyzer(_llm_client_instance, _redis_store_instance)


# Initialize AngelOneAPI first, then Redis and others
angel_api = get_angel_api()
redis_store = get_redis_store(angel_api) # Pass the initialized angel_api
llm_client = get_llm_client() # Initialize LLMClient
ai_webhook = get_ai_webhook(llm_client) # Initialize AIWebhook with LLMClient
sentiment_analyzer = get_sentiment_analyzer(llm_client, redis_store) # Initialize SentimentAnalyzer

backtester = Backtester(angel_api, redis_store)
strategy_manager = StrategyManager(redis_store, ai_webhook) # Pass ai_webhook to StrategyManager

# --- Dashboard Configuration and State Management ---

st.set_page_config(layout="wide", page_title="DTS Intraday AI Strategy Dashboard")

st.title("📈 DTS Intraday AI Strategy Dashboard")

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Live Status", "Strategy Settings", "Backtesting", "Trade History"])

# --- Helper function to load/save settings from Redis ---
def load_settings_from_redis():
    """Loads all configurable settings from Redis, applying defaults if not found."""
    settings = {
        "min_ai_score_buy": redis_store.get_setting("min_ai_score_buy", strategy_manager.MIN_AI_SCORE_BUY),
        "min_ai_score_sell": redis_store.get_setting("min_ai_score_sell", strategy_manager.MIN_AI_SCORE_SELL),
        "sl_percent": redis_store.get_setting("sl_percent", strategy_manager.SL_PERCENT * 100),
        "target_percent": redis_store.get_setting("target_percent", strategy_manager.TARGET_PERCENT * 100),
        "tsl_percent": redis_store.get_setting("tsl_percent", strategy_manager.TSL_PERCENT * 100),
        "tsl_activation_buffer_percent": redis_store.get_setting("tsl_activation_buffer_percent", strategy_manager.TSL_ACTIVATION_BUFFER_PERCENT * 100),
        "max_active_positions": redis_store.get_setting("max_active_positions", strategy_manager.MAX_ACTIVE_POSITIONS),
        "cooldown_period_seconds": redis_store.get_setting("cooldown_period_seconds", strategy_manager.COOLDOWN_PERIOD_SECONDS),
        "market_open_time": redis_store.get_setting("market_open_time", strategy_manager.MARKET_OPEN_TIME.strftime("%H:%M")),
        "market_close_time": redis_store.get_setting("market_close_time", strategy_manager.MARKET_CLOSE_TIME.strftime("%H:%M")),
        "auto_exit_time": redis_store.get_setting("auto_exit_time", strategy_manager.AUTO_EXIT_TIME.strftime("%H:%M")),
        "leverage_enabled": redis_store.get_setting("leverage_enabled", False), # Default OFF (dashboard specific)
        "ai_auto_leverage": redis_store.get_setting("ai_auto_leverage", True), # Default ON (dashboard specific)
        "tsl_enabled": redis_store.get_setting("tsl_enabled", True), # Default ON (dashboard specific)
        "ai_tsl_enabled": redis_store.get_setting("ai_tsl_enabled", True), # Default ON (dashboard specific)
        "default_leverage_multiplier": redis_store.get_setting("default_leverage_multiplier", strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER),
        # Capital Tiers Settings
        "small_capital_threshold": redis_store.get_setting("small_capital_threshold", strategy_manager.SMALL_CAPITAL_THRESHOLD),
        "mid_capital_threshold": redis_store.get_setting("mid_capital_threshold", strategy_manager.MID_CAPITAL_THRESHOLD),
        "small_capital_allocation_pct": redis_store.get_setting("small_capital_allocation_pct", strategy_manager.SMALL_CAPITAL_ALLOCATION_PCT * 100),
        "mid_capital_allocation_pct": redis_store.get_setting("mid_capital_allocation_pct", strategy_manager.MID_CAPITAL_ALLOCATION_PCT * 100),
        "large_capital_allocation_pct": redis_store.get_setting("large_capital_allocation_pct", strategy_manager.LARGE_CAPITAL_ALLOCATION_PCT * 100),
        # New Trading Mode Setting
        "trade_mode": redis_store.get_setting("trade_mode", "paper") # Default to 'paper' (dashboard specific)
    }
    return settings

def save_settings_to_redis(settings: Dict[str, Any]):
    """Saves all configurable settings to Redis."""
    for key, value in settings.items():
        # Convert percentages back to decimals for strategy manager
        if key in ["sl_percent", "target_percent", "tsl_percent", "tsl_activation_buffer_percent",
                   "small_capital_allocation_pct", "mid_capital_allocation_pct", "large_capital_allocation_pct"]:
            redis_store.set_setting(key, value / 100.0)
        else:
            redis_store.set_setting(key, value)
    st.success("Settings saved to Redis!")
    logger.info("Strategy settings updated and saved to Redis.")

def reset_all_settings_to_default():
    """Resets ALL dashboard settings to their default values and saves them to Redis."""
    # Reset AI Score Thresholds
    st.session_state.settings["min_ai_score_buy"] = strategy_manager.MIN_AI_SCORE_BUY
    st.session_state.settings["min_ai_score_sell"] = strategy_manager.MIN_AI_SCORE_SELL

    # Reset Risk Management (Percentages)
    st.session_state.settings["sl_percent"] = strategy_manager.SL_PERCENT * 100
    st.session_state.settings["target_percent"] = strategy_manager.TARGET_PERCENT * 100
    st.session_state.settings["tsl_percent"] = strategy_manager.TSL_PERCENT * 100
    st.session_state.settings["tsl_activation_buffer_percent"] = strategy_manager.TSL_ACTIVATION_BUFFER_PERCENT * 100
    
    # Reset Position Management
    st.session_state.settings["max_active_positions"] = strategy_manager.MAX_ACTIVE_POSITIONS
    st.session_state.settings["cooldown_period_seconds"] = strategy_manager.COOLDOWN_PERIOD_SECONDS

    # Reset Market Timing
    st.session_state.settings["market_open_time"] = strategy_manager.MARKET_OPEN_TIME.strftime("%H:%M")
    st.session_state.settings["market_close_time"] = strategy_manager.MARKET_CLOSE_TIME.strftime("%H:%M")
    st.session_state.settings["auto_exit_time"] = strategy_manager.AUTO_EXIT_TIME.strftime("%H:%M")

    # Reset Leverage Settings
    st.session_state.settings["leverage_enabled"] = False # Dashboard default
    st.session_state.settings["ai_auto_leverage"] = True # Dashboard default
    st.session_state.settings["default_leverage_multiplier"] = strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER

    # Reset TSL Toggles
    st.session_state.settings["tsl_enabled"] = True # Dashboard default
    st.session_state.settings["ai_tsl_enabled"] = True # Dashboard default

    # Reset Capital Tier Settings
    st.session_state.settings["small_capital_threshold"] = strategy_manager.SMALL_CAPITAL_THRESHOLD
    st.session_state.settings["mid_capital_threshold"] = strategy_manager.MID_CAPITAL_THRESHOLD
    st.session_state.settings["small_capital_allocation_pct"] = strategy_manager.SMALL_CAPITAL_ALLOCATION_PCT * 100
    st.session_state.settings["mid_capital_allocation_pct"] = strategy_manager.MID_CAPITAL_ALLOCATION_PCT * 100
    st.session_state.settings["large_capital_allocation_pct"] = strategy_manager.LARGE_CAPITAL_ALLOCATION_PCT * 100

    # Reset Trading Mode
    st.session_state.settings["trade_mode"] = "paper" # Dashboard default

    # Save all these default values to Redis
    save_settings_to_redis(st.session_state.settings)
    st.success("✅ All settings have been reset to default values.")
    logger.info("All dashboard settings restored to default values and saved to Redis.")
    st.rerun() # Rerun to reflect the changes immediately

# Helper function to get allocation percentage based on capital
def get_current_allocation_percentage(capital: float, settings: Dict[str, Any]) -> float:
    """Determines the allocation percentage based on current capital and tier settings."""
    if capital <= settings["small_capital_threshold"]:
        return settings["small_capital_allocation_pct"] / 100.0
    elif capital <= settings["mid_capital_threshold"]:
        return settings["mid_capital_allocation_pct"] / 100.0
    else:
        return settings["large_capital_allocation_pct"] / 100.0

# Helper function to get leverage tier based on capital
def get_leverage_tier(capital: float, settings: Dict[str, Any]) -> str:
    """Determines the leverage tier based on current capital and tier settings."""
    if capital <= settings["small_capital_threshold"]:
        return "1X" # Small Cap
    elif capital <= settings["mid_capital_threshold"]:
        return "2X" # Mid Cap
    else:
        return "5X" # Large Cap (using 5X as per example, adjust if strategy has a different default)


# Load initial settings when the app starts
if 'settings' not in st.session_state:
    st.session_state.settings = load_settings_from_redis()

# --- Start background threads if not already running ---
# This ensures these loops run continuously in the background
if not any(isinstance(t, threading.Thread) and t.name == 'live_stream_thread' for t in threading.enumerate()):
    live_stream_thread = threading.Thread(target=LiveStreamManager(angel_api, redis_store).run_live_stream, daemon=True, name='live_stream_thread')
    live_stream_thread.start()
    logger.info("Live stream manager thread started.")

if not any(isinstance(t, threading.Thread) and t.name == 'sentiment_analyzer_thread' for t in threading.enumerate()):
    sentiment_analyzer_thread = threading.Thread(target=sentiment_analyzer.run_sentiment_analysis_loop, daemon=True, name='sentiment_analyzer_thread')
    sentiment_analyzer_thread.start()
    logger.info("Sentiment analyzer thread started.")

# The paper trading loop is typically started by the main application runner,
# but if this dashboard is meant to be the primary entry point for paper trading,
# you might start it here. However, it's often better to have a dedicated `main.py`
# or similar for the core trading system loops.
# For now, assuming paper_trade_system is started elsewhere or will be integrated.


# --- Live Status Page ---
if page == "Live Status":
    st.header("Live Trading Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System Control")
        is_sync_paused = redis_store.get_setting("is_sync_paused", False)
        new_sync_paused = st.toggle("Pause Trading System Sync", value=is_sync_paused, key="sync_pause_toggle")
        if new_sync_paused != is_sync_paused:
            redis_store.set_setting("is_sync_paused", new_sync_paused)
            st.session_state.settings['is_sync_paused'] = new_sync_paused # Update session state
            st.rerun() # Rerun to reflect the change immediately

        if new_sync_paused:
            st.warning("Trading system sync is PAUSED. No new trades will be entered, and existing trades will not be managed.")
        else:
            st.success("Trading system sync is ACTIVE.")

        st.subheader("Current Capital & PnL")
        available_capital = redis_store.get_available_capital()
        total_realized_pnl = redis_store.get_total_realized_pnl()
        st.metric(label="Available Capital", value=f"₹ {available_capital:,.2f}")
        st.metric(label="Total Realized PnL", value=f"₹ {total_realized_pnl:,.2f}")

    with col2:
        st.subheader("Active Positions")
        # Changed get_active_trades() to get_all_active_trades()
        active_trades = redis_store.get_all_active_trades()
        if active_trades:
            active_trades_df = pd.DataFrame(active_trades.values())
            # Convert timestamp to readable format
            active_trades_df['timestamp'] = active_trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(active_trades_df[['symbol', 'direction', 'qty', 'entry_price', 'sl', 'tgt', 'tsl', 'timestamp']])
        else:
            st.info("No active positions.")

    st.subheader("Real-time Market Data (from Redis)")
    # Fetch some sample LTPs from Redis
    sample_symbols = ["RELIANCE", "INFY", "TCS", "SBIN", "HDFC_BANK"] # Example symbols
    ltp_data = {}
    for sym in sample_symbols:
        ltp = redis_store.read_ltp(sym)
        volume = redis_store.redis_client.get(f"VOLUME:{sym}")
        last_update = redis_store.redis_client.get(f"LAST_UPDATE:{sym}")
        ltp_data[sym] = {
            "LTP": f"₹ {ltp:,.2f}" if ltp is not None else "N/A",
            # Corrected: Decode volume bytes, convert to float, then to int
            "Volume": int(float(volume.decode('utf-8'))) if volume else "N/A",
            "Last Update": last_update.decode('utf-8') if last_update else "N/A"
        }
    st.dataframe(pd.DataFrame.from_dict(ltp_data, orient='index'))


# --- Strategy Settings Page ---
elif page == "Strategy Settings":
    st.header("Strategy Configuration")

    with st.form("strategy_settings_form"):
        # New Trading Mode Toggle
        st.subheader("Trading Mode")
        current_trade_mode = st.session_state.settings["trade_mode"]
        new_trade_mode = st.radio(
            "Select Trading Mode",
            ["paper", "live"],
            index=0 if current_trade_mode == "paper" else 1,
            help="In 'paper' mode, trades are simulated. In 'live' mode, actual orders are placed."
        )
        st.session_state.settings["trade_mode"] = new_trade_mode
        if new_trade_mode == "live":
            st.warning("⚠️ Live trading mode is ACTIVE. Be extremely careful!")
        else:
            st.info("Currently in Paper trading mode.")

        st.subheader("AI Score Thresholds")
        st.session_state.settings["min_ai_score_buy"] = st.slider(
            "Minimum AI Score for BUY",
            min_value=0.0, max_value=1.0, value=st.session_state.settings["min_ai_score_buy"], step=0.01
        )
        st.session_state.settings["min_ai_score_sell"] = st.slider(
            "Minimum AI Score for SELL (e.g., -0.7 means score must be <= -0.7)",
            min_value=-1.0, max_value=0.0, value=st.session_state.settings["min_ai_score_sell"], step=0.01
        )

        st.subheader("Risk Management (Percentages)")
        st.session_state.settings["sl_percent"] = st.slider(
            "Stop Loss Percentage (%)",
            min_value=0.5, max_value=10.0, value=st.session_state.settings["sl_percent"], step=0.1
        )
        st.session_state.settings["target_percent"] = st.slider(
            "Target Profit Percentage (%)",
            min_value=1.0, max_value=20.0, value=st.session_state.settings["target_percent"], step=0.1
        )
        st.session_state.settings["tsl_percent"] = st.slider(
            "Trailing Stop Loss Percentage (%)",
            min_value=0.1, max_value=5.0, value=st.session_state.settings["tsl_percent"], step=0.1
        )
        st.session_state.settings["tsl_activation_buffer_percent"] = st.slider(
            "TSL Activation Buffer Percentage (%) (Profit needed to activate TSL)",
            min_value=0.0, max_value=5.0, value=st.session_state.settings["tsl_activation_buffer_percent"], step=0.1
        )
        
        st.subheader("Position Management")
        st.session_state.settings["max_active_positions"] = st.number_input(
            "Maximum Active Positions",
            min_value=1, max_value=20, value=st.session_state.settings["max_active_positions"], step=1
        )
        st.session_state.settings["cooldown_period_seconds"] = st.number_input(
            "Cooldown Period (seconds) after Exit",
            min_value=60, max_value=3600, value=st.session_state.settings["cooldown_period_seconds"], step=60
        )

        st.subheader("Market Timing")
        st.session_state.settings["market_open_time"] = st.text_input(
            "Market Open Time (HH:MM)", value=st.session_state.settings["market_open_time"]
        )
        st.session_state.settings["market_close_time"] = st.text_input(
            "Market Close Time (HH:MM)", value=st.session_state.settings["market_close_time"]
        )
        st.session_state.settings["auto_exit_time"] = st.text_input(
            "Auto Exit Time (HH:MM)", value=st.session_state.settings["auto_exit_time"]
        )

        st.subheader("Leverage Settings")
        st.session_state.settings["leverage_enabled"] = st.toggle(
            "Enable Manual Leverage", value=st.session_state.settings["leverage_enabled"]
        )
        st.session_state.settings["default_leverage_multiplier"] = st.slider(
            "Default Leverage Multiplier (if enabled)",
            min_value=1.0, max_value=10.0, value=st.session_state.settings["default_leverage_multiplier"], step=0.5
        )
        st.session_state.settings["ai_auto_leverage"] = st.toggle(
            "Enable AI Auto-Leverage (amplifies position size for high-score signals)",
            value=st.session_state.settings["ai_auto_leverage"]
        )

        st.subheader("Trailing Stop Loss (TSL) Toggles")
        st.session_state.settings["tsl_enabled"] = st.toggle(
            "Enable Trailing Stop Loss (TSL)", value=st.session_state.settings["tsl_enabled"]
        )
        st.session_state.settings["ai_tsl_enabled"] = st.toggle(
            "Enable AI-Driven TSL (TSL activates after profit buffer, trails based on momentum)",
            value=st.session_state.settings["ai_tsl_enabled"]
        )

        st.subheader("Capital Tier Settings")
        st.session_state.settings["small_capital_threshold"] = st.number_input(
            "Small Capital Threshold (₹)",
            min_value=1000.0, max_value=500000.0, value=st.session_state.settings["small_capital_threshold"], step=1000.0
        )
        st.session_state.settings["mid_capital_threshold"] = st.number_input(
            "Mid Capital Threshold (₹)",
            min_value=50000.0, max_value=5000000.0, value=st.session_state.settings["mid_capital_threshold"], step=10000.0
        )
        st.session_state.settings["small_capital_allocation_pct"] = st.slider(
            "Small Capital Allocation % (per trade)",
            min_value=1.0, max_value=20.0, value=st.session_state.settings["small_capital_allocation_pct"], step=0.5
        )
        st.session_state.settings["mid_capital_allocation_pct"] = st.slider(
            "Mid Capital Allocation % (per trade)",
            min_value=0.5, max_value=10.0, value=st.session_state.settings["mid_capital_allocation_pct"], step=0.1
        )
        st.session_state.settings["large_capital_allocation_pct"] = st.slider(
            "Large Capital Allocation % (per trade)",
            min_value=0.1, max_value=5.0, value=st.session_state.settings["large_capital_allocation_pct"], step=0.1
        )

        col_save, col_reset = st.columns([1, 1]) # Use columns to place buttons side-by-side
        with col_save:
            submitted = st.form_submit_button("Save Settings")
            if submitted:
                save_settings_to_redis(st.session_state.settings)
                # Reload settings to ensure consistency after saving
                st.session_state.settings = load_settings_from_redis()
                st.rerun() # Rerun to reflect updated values in inputs (if any formatting changes)
        with col_reset:
            if st.form_submit_button("Reset to Default"):
                reset_all_settings_to_default()


# --- Backtesting Page ---
elif page == "Backtesting":
    st.header("Backtesting Strategy Performance")

    # Backtest parameters
    st.subheader("Backtest Parameters")
    
    # Example symbols for dropdown. In a real app, this might come from a scrip master.
    # Ensure these match your SYMBOL_DHAN_MAP or Angel One's symbol tokens
    # For now, using Angel One compatible tokens and symbols for `HistoricalDataManager`
    backtest_symbols = {
        "RELIANCE": {"token": "11536", "exchangeType": "NSE"},
        "INFY": {"token": "10604", "exchangeType": "NSE"},
        "TCS": {"token": "3045", "exchangeType": "NSE"},
        "SBIN": {"token": "1333", "exchangeType": "NSE"},
        "HDFC_BANK": {"token": "3432", "exchangeType": "NSE"},
    }
    selected_symbol_name = st.selectbox("Select Symbol", list(backtest_symbols.keys()))
    selected_symbol_info = backtest_symbols[selected_symbol_name]

    col_date1, col_date2 = st.columns(2)
    with col_date1:
        from_date = st.date_input("From Date", value=datetime.now() - timedelta(days=7))
        from_time = st.time_input("From Time", value=dt_time(9, 15))
    with col_date2:
        to_date = st.date_input("To Date", value=datetime.now())
        to_time = st.time_input("To Time", value=dt_time(15, 20)) # Auto-exit time for backtest

    interval = st.selectbox("Candle Interval", ["ONE_MINUTE", "FIFTEEN_MINUTE", "DAY"])

    full_from_datetime = datetime.combine(from_date, from_time)
    full_to_datetime = datetime.combine(to_date, to_time)

    if st.button("Run Backtest"):
        if full_from_datetime >= full_to_datetime:
            st.error("From Date/Time must be before To Date/Time.")
        else:
            with st.spinner(f"Running backtest for {selected_symbol_name}... This may take a moment."):
                report = backtester.run_backtest(
                    symbol=selected_symbol_name,
                    symbol_token=selected_symbol_info["token"],
                    exchange_type=selected_symbol_info["exchangeType"],
                    from_date=full_from_datetime,
                    to_date=full_to_datetime,
                    interval="ONE_MINUTE", # Backtest with 1-minute data
                    leverage_enabled=False, # Example: Disable leverage for this backtest
                    ai_auto_leverage=False
                )

            if report:
                st.subheader("Backtest Results")
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Total PnL", f"₹ {report['total_pnl']:,.2f}")
                col_res2.metric("Number of Trades", report['num_trades'])
                col_res3.metric("Win Rate", f"{report['win_rate']:.2f}%")
                col_res1.metric("Max Drawdown", f"{report['max_drawdown_percent']:.2f}%")
                col_res2.metric("Final Capital", f"₹ {report['final_capital']:,.2f}")
                
                st.info(f"Backtest used: Manual Leverage: {report['leverage_used_in_backtest']}, "
                        f"AI Auto-Leverage: {report['ai_auto_leverage_in_backtest']}, "
                        f"TSL Enabled: {report['tsl_enabled_in_backtest']}, "
                        f"AI-TSL Enabled: {report['ai_tsl_enabled_in_backtest']}")

                st.subheader("Trade Log")
                trade_log_df = pd.DataFrame(report['trade_logs'])
                if not trade_log_df.empty:
                    # Convert timestamp to datetime and then to readable string
                    trade_log_df['timestamp'] = pd.to_datetime(trade_log_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(trade_log_df)

                    # Plot Capital History (Equity Curve)
                    st.subheader("Equity Curve")
                    capital_history_df = pd.DataFrame(backtester.capital_history, columns=['timestamp', 'capital'])
                    capital_history_df['timestamp'] = pd.to_datetime(capital_history_df['timestamp'])
                    
                    fig = px.line(capital_history_df, x='timestamp', y='capital', 
                                  title='Capital Over Time (Equity Curve)',
                                  labels={'capital': 'Capital (₹)', 'timestamp': 'Time'})
                    fig.update_layout(hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No trades were executed during this backtest period.")
            else:
                st.error("Backtest failed. Please check logs for details.")

# --- Trade History Page ---
elif page == "Trade History":
    st.header("Closed Trade History")

    closed_trades = redis_store.get_all_closed_trades()
    if closed_trades:
        closed_trades_df = pd.DataFrame(closed_trades)
        # Convert timestamp columns to readable format
        closed_trades_df['timestamp'] = pd.to_datetime(closed_trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        closed_trades_df['exit_timestamp'] = pd.to_datetime(closed_trades_df['exit_timestamp']) # Keep as datetime for filtering later
        
        # Calculate Capital Used
        closed_trades_df['capital_used'] = closed_trades_df['entry_price'] * closed_trades_df['qty']
        
        # Determine Leverage Tier
        # Assuming `total_capital` is the initial capital used for the overall strategy,
        # or the capital at the time the trade was entered. For simplicity, we'll use
        # the current available capital from Redis to determine the tier.
        # In a more precise backtest, `allocated_capital_per_trade` would be stored with each trade.
        # For live display, we'll use the current tier logic.
        
        # Get the initial capital from Redis, or a default if not set
        initial_capital_for_tiering = redis_store.get_setting("initial_capital", 100000.0) # Default from .env or strategy
        closed_trades_df['leverage_tier'] = closed_trades_df['capital_used'].apply(
            lambda x: get_leverage_tier(initial_capital_for_tiering, st.session_state.settings)
        )

        # Format columns for display
        closed_trades_df['Capital Used'] = closed_trades_df['capital_used'].apply(lambda x: f"₹ {x:,.2f}")
        closed_trades_df['Leverage Tier'] = closed_trades_df['leverage_tier']


        st.dataframe(closed_trades_df[[
            'symbol', 'direction', 'qty', 'entry_price', 'exit_price', 'pnl',
            'Capital Used', 'Leverage Tier', 'timestamp', 'exit_timestamp' # Include new columns
        ]])

        # Basic performance metrics for closed trades
        total_pnl = closed_trades_df['pnl'].sum()
        num_trades = len(closed_trades_df)
        num_wins = closed_trades_df[closed_trades_df['pnl'] > 0].shape[0]
        
        # New Win Rate (Trades)
        win_rate_trades = (num_wins / num_trades * 100) if num_trades > 0 else 0.0

        # New Profitability Based on Capital Used
        total_capital_used_overall = closed_trades_df['capital_used'].sum()
        profit_pct_of_capital_used = (total_pnl / total_capital_used_overall * 100) if total_capital_used_overall > 0 else 0.0

        st.subheader("Summary of Closed Trades")
        col_hist1, col_hist2, col_hist3 = st.columns(3)
        col_hist1.metric("Total Realized PnL", f"₹ {total_pnl:,.2f}")
        col_hist2.metric("Total Closed Trades", num_trades)
        col_hist3.metric("Win Rate (Trades)", f"{num_wins} / {num_trades} = {win_rate_trades:.2f}%")
        st.metric("Profit % of Capital Used", f"{profit_pct_of_capital_used:.2f}%")


        # Metrics for "today"
        today_date = datetime.now().date()
        trades_today = closed_trades_df[closed_trades_df['exit_timestamp'].dt.date == today_date]

        total_capital_used_today = trades_today['capital_used'].sum()
        
        # Calculate average leverage tier for today's trades
        # This requires mapping tiers back to numerical values for averaging
        leverage_tier_map = {"1X": 1, "2X": 2, "5X": 5} # Map tiers to their multipliers
        trades_today['leverage_multiplier'] = trades_today['leverage_tier'].map(leverage_tier_map)
        average_leverage_used_today = trades_today['leverage_multiplier'].mean() if not trades_today.empty else 0.0


        st.metric("Total Capital Used Today", f"₹ {total_capital_used_today:,.2f}")
        st.metric("Average Leverage Used Today", f"{average_leverage_used_today:.2f}x")

        # Plot PnL distribution
        st.subheader("PnL Distribution")
        fig_pnl = px.histogram(closed_trades_df, x='pnl', nbins=20, title='Distribution of PnL per Trade')
        st.plotly_chart(fig_pnl, use_container_width=True)

    else:
        st.info("No closed trades found in history.")
