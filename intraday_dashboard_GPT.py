# intraday_dashboard_GPT.py
# This script creates a Streamlit dashboard for the DTS Intraday AI Strategy.
# It allows users to monitor live trades, view performance metrics, and control system settings.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time as dt_time
import os
import time
import threading
import json

# Import core components
from redis_store import RedisStore
from angelone_api_patch import AngelOneAPI
from dhan_api_patch import DhanAPI
from llm_client import LLMClient
from sentiment_analyzer import SentimentAnalyzer
from strategy import StrategyManager
from paper_trade_system import PaperTradeSystem
from live_stream import LiveStreamManager
from backtester import Backtester # Import the Backtester
from ai_webhook import AIWebhook # NEW: Import AIWebhook

# Configure logging for the dashboard (optional, Streamlit handles its own logging)
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("dashboard_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)


# --- Global Initialization (Run once) ---
# Use Streamlit's session state to store expensive-to-initialize objects
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    st.session_state.angel_api = AngelOneAPI()
    st.session_state.angel_api.login() # Ensure login on dashboard startup

    st.session_state.redis_store = RedisStore(st.session_state.angel_api)
    st.session_state.redis_store.connect()

    st.session_state.dhan_api = DhanAPI() # Initialize DhanAPI

    st.session_state.llm_client = LLMClient() # Initialize LLMClient
    
    # NEW: Initialize AIWebhook
    st.session_state.ai_webhook = AIWebhook(st.session_state.llm_client)

    st.session_state.sentiment_analyzer = SentimentAnalyzer(st.session_state.llm_client, st.session_state.redis_store)
    
    # NEW: Pass ai_webhook to StrategyManager
    st.session_state.strategy_manager = StrategyManager(st.session_state.redis_store, st.session_state.ai_webhook)

    # UPDATED: Pass ai_webhook to PaperTradeSystem
    st.session_state.paper_trade_system = PaperTradeSystem(
        st.session_state.redis_store, 
        st.session_state.strategy_manager, 
        st.session_state.dhan_api, 
        st.session_state.angel_api,
        st.session_state.ai_webhook # NEW: Pass ai_webhook
    )

    st.session_state.live_stream_manager = LiveStreamManager(st.session_state.angel_api, st.session_state.redis_store)

    # Start live stream and paper trading loops in separate threads if not already running
    if not any(isinstance(t, threading.Thread) and t.name == 'live_stream_thread' for t in threading.enumerate()):
        live_stream_thread = threading.Thread(target=st.session_state.live_stream_manager.run_live_stream, daemon=True, name='live_stream_thread')
        live_stream_thread.start()
        logger.info("Live stream manager thread started.")
    
    if not any(isinstance(t, threading.Thread) and t.name == 'paper_trade_thread' for t in threading.enumerate()):
        paper_trade_thread = threading.Thread(target=st.session_state.paper_trade_system.run_paper_trading_loop, daemon=True, name='paper_trade_thread')
        paper_trade_thread.start()
        logger.info("Paper trading loop thread started.")

    # Initialize Backtester
    st.session_state.backtester = Backtester(st.session_state.angel_api, st.session_state.redis_store)
    logger.info("Backtester initialized.")

    st.session_state.initialized = True
    logger.info("Dashboard global initialization complete.")


# --- Dashboard UI ---
st.set_page_config(layout="wide", page_title="DTS AI Trading Dashboard")

st.title("DTS Intraday AI Trading System Dashboard")

# Tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Trading", "Performance Metrics", "System Settings", "Backtesting", "AI Feedback (WIP)"])

with tab1:
    st.header("Live Trading Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Current Capital & PnL")
        metrics = st.session_state.paper_trade_system.get_dashboard_metrics()
        st.metric(label="Available Capital", value=f"₹{metrics['available_capital']:,}")
        st.metric(label="Total Realized PnL", value=f"₹{metrics['total_realized_pnl']:,}")
        st.metric(label="Total Unrealized PnL", value=f"₹{metrics['total_unrealized_pnl']:,}")
        st.metric(label="Overall PnL (Realized + Unrealized)", value=f"₹{metrics['total_pnl']:,}")

    with col2:
        st.subheader("Active Positions")
        st.metric(label="Active Positions Count", value=f"{metrics['active_positions_count']}/{metrics['max_active_positions']}")
        
        st.write("---")
        st.subheader("System Control")
        if st.button("Toggle System Sync (Pause/Resume Strategy)"):
            st.session_state.paper_trade_system.toggle_sync()
            st.rerun() # Rerun to update button state
        st.info(f"System Sync Status: {'PAUSED' if st.session_state.paper_trade_system.is_sync_paused else 'ACTIVE'}")

    with col3:
        st.subheader("Trade Metrics")
        st.metric(label="Win Rate", value=f"{metrics['win_rate']:.2f}%")
        st.metric(label="Capital Efficiency", value=f"{metrics['capital_efficiency']:.2f}%")
        st.metric(label="Leverage Tier", value=metrics['leverage_tier']) # Placeholder


    st.subheader("Detailed Active Trades")
    active_trades_df = pd.DataFrame(st.session_state.paper_trade_system.active_trades.values())
    if not active_trades_df.empty:
        # Select and reorder columns for display
        display_cols = [
            'symbol', 'direction', 'entry_price', 'qty', 'ltp', 'unrealized_pnl',
            'sl', 'tgt', 'tsl', 'ai_score_at_entry', 'sentiment_at_entry', 'timestamp'
        ]
        # Ensure 'ltp' and 'unrealized_pnl' are calculated and present for display
        # They are updated in paper_trade_system.run_paper_trading_loop
        
        # Manually add LTP and Unrealized PnL for display purposes, as they are dynamic
        # and not directly stored in the active_trades dict in the same way.
        # This is a simplified approach. In a real app, you'd calculate these on demand.
        active_trades_for_display = []
        for trade_id, trade in st.session_state.paper_trade_system.active_trades.items():
            current_ltp = st.session_state.redis_store.read_ltp(trade['symbol'])
            display_trade = trade.copy()
            display_trade['ltp'] = round(current_ltp, 2) if current_ltp is not None else 'N/A'
            
            unrealized_pnl = 0.0
            if current_ltp is not None:
                if trade['direction'] == 'BUY':
                    unrealized_pnl = (current_ltp - trade['entry_price']) * trade['qty']
                elif trade['direction'] == 'SELL':
                    unrealized_pnl = (trade['entry_price'] - current_ltp) * trade['qty']
            display_trade['unrealized_pnl'] = round(unrealized_pnl, 2)
            active_trades_for_display.append(display_trade)

        active_trades_df_display = pd.DataFrame(active_trades_for_display)
        
        # Handle potential missing columns if data structure changes
        for col in ['ltp', 'unrealized_pnl', 'ai_score_at_entry', 'sentiment_at_entry']:
            if col not in active_trades_df_display.columns:
                active_trades_df_display[col] = 'N/A'

        st.dataframe(active_trades_df_display[display_cols])
    else:
        st.info("No active trades.")

    st.subheader("AI Sentiment Scores (Latest from Redis)")
    # Get all subscribed symbols
    subscribed_symbols_config = []
    try:
        with open('subscribed_symbols.json', 'r') as f:
            subscribed_symbols_config = json.load(f)
    except FileNotFoundError:
        st.warning("subscribed_symbols.json not found. Cannot display AI sentiment scores.")
    except json.JSONDecodeError:
        st.error("Error decoding subscribed_symbols.json. Please check its format.")

    sentiment_data = []
    for item in subscribed_symbols_config:
        symbol = item['symbol']
        ai_score = st.session_state.strategy_manager.get_ai_score(symbol)
        sentiment = st.session_state.strategy_manager.get_sentiment(symbol)
        ltp = st.session_state.redis_store.read_ltp(symbol)
        sentiment_data.append({
            "Symbol": symbol,
            "LTP": f"₹{ltp:.2f}" if ltp is not None else "N/A",
            "AI Score": f"{ai_score:.2f}" if ai_score is not None else "N/A",
            "Sentiment": sentiment if sentiment is not None else "N/A"
        })
    sentiment_df = pd.DataFrame(sentiment_data)
    st.dataframe(sentiment_df)


with tab2:
    st.header("Performance Metrics & History")

    st.subheader("Realized PnL Over Time")
    closed_trades_df = pd.DataFrame(st.session_state.paper_trade_system.closed_trades)
    if not closed_trades_df.empty:
        closed_trades_df['exit_timestamp'] = pd.to_datetime(closed_trades_df['exit_timestamp'])
        closed_trades_df = closed_trades_df.sort_values('exit_timestamp')
        closed_trades_df['cumulative_pnl'] = closed_trades_df['pnl'].cumsum()

        fig_pnl = px.line(closed_trades_df, x='exit_timestamp', y='cumulative_pnl', 
                          title='Cumulative Realized PnL',
                          labels={'exit_timestamp': 'Time', 'cumulative_pnl': 'Cumulative PnL (₹)'})
        st.plotly_chart(fig_pnl, use_container_width=True)

        st.subheader("Trade Details")
        st.dataframe(closed_trades_df[['symbol', 'direction', 'entry_price', 'exit_price', 'qty', 'pnl', 'exit_reason', 'exit_timestamp']])
    else:
        st.info("No closed trades yet to display performance.")

with tab3:
    st.header("System Settings")
    st.write("Adjust strategy parameters and system behavior.")

    st.subheader("Strategy Parameters")
    # Using st.session_state to persist input values
    min_ai_buy = st.number_input("Min AI Score (BUY)", value=st.session_state.strategy_manager.MIN_AI_SCORE_BUY, step=0.01)
    min_ai_sell = st.number_input("Min AI Score (SELL)", value=st.session_state.strategy_manager.MIN_AI_SCORE_SELL, step=0.01)
    sl_pct = st.number_input("Stop Loss %", value=st.session_state.strategy_manager.SL_PERCENT * 100, step=0.1)
    tgt_pct = st.number_input("Target %", value=st.session_state.strategy_manager.TARGET_PERCENT * 100, step=0.1)
    tsl_pct = st.number_input("Trailing Stop Loss %", value=st.session_state.strategy_manager.TSL_PERCENT * 100, step=0.1)
    tsl_activation_buffer_pct = st.number_input("TSL Activation Buffer %", value=st.session_state.strategy_manager.TSL_ACTIVATION_BUFFER_PERCENT * 100, step=0.1)
    cooldown_sec = st.number_input("Cooldown Period (seconds)", value=st.session_state.strategy_manager.COOLDOWN_PERIOD_SECONDS, step=10)
    max_positions = st.number_input("Max Active Positions", value=st.session_state.strategy_manager.MAX_ACTIVE_POSITIONS, step=1)
    default_leverage = st.number_input("Default Leverage Multiplier", value=st.session_state.strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER, step=0.5)

    # Dashboard specific settings
    tsl_enabled = st.checkbox("Enable Trailing Stop Loss", value=st.session_state.redis_store.get_setting("tsl_enabled", True))
    ai_tsl_enabled = st.checkbox("Enable AI-Enhanced TSL", value=st.session_state.redis_store.get_setting("ai_tsl_enabled", True))
    leverage_enabled = st.checkbox("Enable Leverage", value=st.session_state.redis_store.get_setting("leverage_enabled", False))
    ai_auto_leverage = st.checkbox("Enable AI Auto-Leverage", value=st.session_state.redis_store.get_setting("ai_auto_leverage", True))


    if st.button("Save Strategy Settings"):
        st.session_state.strategy_manager.MIN_AI_SCORE_BUY = min_ai_buy
        st.session_state.strategy_manager.MIN_AI_SCORE_SELL = min_ai_sell
        st.session_state.strategy_manager.SL_PERCENT = sl_pct / 100
        st.session_state.strategy_manager.TARGET_PERCENT = tgt_pct / 100
        st.session_state.strategy_manager.TSL_PERCENT = tsl_pct / 100
        st.session_state.strategy_manager.TSL_ACTIVATION_BUFFER_PERCENT = tsl_activation_buffer_pct / 100
        st.session_state.strategy_manager.COOLDOWN_PERIOD_SECONDS = cooldown_sec
        st.session_state.strategy_manager.MAX_ACTIVE_POSITIONS = max_positions
        st.session_state.strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER = default_leverage

        # Save dashboard specific settings to Redis
        st.session_state.redis_store.set_setting("tsl_enabled", tsl_enabled)
        st.session_state.redis_store.set_setting("ai_tsl_enabled", ai_tsl_enabled)
        st.session_state.redis_store.set_setting("leverage_enabled", leverage_enabled)
        st.session_state.redis_store.set_setting("ai_auto_leverage", ai_auto_leverage)

        st.success("Strategy settings saved!")
        st.rerun() # Rerun to reflect updated settings

    st.subheader("Market Timings (Read-Only)")
    st.write(f"Market Open: {st.session_state.strategy_manager.MARKET_OPEN_TIME.strftime('%H:%M')}")
    st.write(f"Market Close: {st.session_state.strategy_manager.MARKET_CLOSE_TIME.strftime('%H:%M')}")
    st.write(f"Auto Exit Time: {st.session_state.strategy_manager.AUTO_EXIT_TIME.strftime('%H:%M')}")

with tab4:
    st.header("Backtesting")
    st.write("Run your strategy against historical data.")

    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End Date", value=datetime(2024, 1, 31))

    if st.button("Run Backtest"):
        with st.spinner("Running backtest... This may take a while."):
            st.session_state.backtester.run_backtest(
                datetime.combine(start_date, dt_time.min),
                datetime.combine(end_date, dt_time.max)
            )
        st.success("Backtest completed! See results below.")
        
        # Display backtest results
        st.subheader("Backtest Summary")
        backtest_metrics = st.session_state.paper_trade_system.get_dashboard_metrics() # Re-use metrics after backtest
        st.json(backtest_metrics) # Display as JSON for now

        st.subheader("Backtest Closed Trades")
        backtest_closed_trades_df = pd.DataFrame(st.session_state.paper_trade_system.closed_trades)
        if not backtest_closed_trades_df.empty:
            st.dataframe(backtest_closed_trades_df)
        else:
            st.info("No trades were closed during the backtest.")

with tab5:
    st.header("AI Feedback (Work In Progress)")
    st.write("This section will display real-time AI suggestions, cautions, and insights.")
    st.info("AI feedback will appear here as the system processes trading events.")

    # Placeholder for displaying AI feedback
    # In a real implementation, you'd fetch these from Redis or a dedicated queue
    # populated by the AIWebhook.
    st.subheader("Latest AI Insights")
    # For demonstration, let's just show a dummy message or a few recent ones
    # This would ideally be pulled from Redis where AIWebhook stores its responses.
    
    # Example: Fetching a dummy AI response from Redis (if stored by AIWebhook)
    # You would need to modify AIWebhook to store its responses in a retrievable way.
    # For now, let's just show a static placeholder.
    st.write("No AI feedback available yet. Run the main system to generate insights.")
    st.code("""
    [AI Suggestion]: RELIANCE looks strong on volume. Consider a cautious entry.
    [AI Feedback]: Trade for INFY rejected due to cooldown. Good risk management.
    [AI Review]: Successful TSL exit for TCS. Profit captured effectively.
    """)

# Auto-refresh the dashboard every few seconds
time.sleep(5)
st.rerun()

