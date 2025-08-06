# backtester.py
# This module provides a backtesting framework for the DTS Intraday AI Trading System.
# It simulates trading over historical data to evaluate strategy performance.

import logging
import pandas as pd
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Any, List, Optional
import os

# Import necessary components
from angelone_api_patch import AngelOneAPI
from redis_store import RedisStore
from strategy import StrategyManager
from paper_trade_system import PaperTradeSystem # Re-use PaperTradeSystem for backtesting logic
from ai_webhook import AIWebhook # NEW: Import AIWebhook
from llm_client import LLMClient # NEW: Import LLMClient for AIWebhook

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

class Backtester:
    """
    Simulates trading on historical data to evaluate the performance of the strategy.
    """
    def __init__(self, angel_api: AngelOneAPI, redis_store: RedisStore):
        """
        Initializes the Backtester.

        Args:
            angel_api (AngelOneAPI): An instance of AngelOneAPI for historical data (mocked).
            redis_store (RedisStore): An instance of RedisStore for data interaction (mocked).
        """
        self.angel_api = angel_api
        self.redis_store = redis_store

        # NEW: Initialize LLMClient and AIWebhook for backtester
        # In a backtest, these will likely interact with mocks or be bypassed.
        self.llm_client = LLMClient()
        self.ai_webhook = AIWebhook(self.llm_client)

        # Initialize StrategyManager and PaperTradeSystem (re-using core logic)
        self.strategy_manager = StrategyManager(redis_store, self.ai_webhook) # NEW: Pass ai_webhook
        self.paper_trade_system = PaperTradeSystem(redis_store, self.strategy_manager, None, angel_api) # DhanAPI is None for backtesting


        self.historical_data: Dict[str, pd.DataFrame] = {} # {symbol: DataFrame of historical data}
        self.load_historical_data() # Load data on init

        logger.info("Backtester initialized.")

    def load_historical_data(self):
        """
        Loads historical data from a CSV file (e.g., api-scrip-master.csv).
        This is a placeholder for actual historical data fetching.
        """
        try:
            # Assuming api-scrip-master.csv contains historical data for symbols
            # For a real backtest, this would be proper OHLCV data over time.
            # For this simulation, we'll use it to get a "mock" LTP for backtesting.
            file_path = 'api-scrip-master.csv'
            if not os.path.exists(file_path):
                logger.warning(f"Historical data file not found at {file_path}. Backtesting will be limited.")
                return

            df = pd.read_csv(file_path)
            # Filter for NSE Equity (EQ) if applicable
            df = df[df['exch_seg'] == 'NSECM'] # Assuming NSECM for equity cash segment
            
            # Create a simplified mapping for symbols to mock LTP
            # In a real backtest, you'd iterate through time-series data
            for index, row in df.iterrows():
                symbol = row['symbol']
                # Using 'close' as a mock LTP for backtesting purposes
                # In a real backtest, this would be a time-series of prices
                self.historical_data[symbol] = pd.DataFrame([{'timestamp': datetime.now(), 'ltp': row['close']}])
            
            logger.info(f"Loaded mock historical data for {len(self.historical_data)} symbols.")

        except Exception as e:
            logger.error(f"Error loading historical data: {e}", exc_info=True)

    def get_mock_ltp_for_backtest(self, symbol: str) -> Optional[float]:
        """
        Retrieves a mock LTP for a given symbol from the loaded historical data.
        In a real backtest, this would advance through time-series data.
        """
        data = self.historical_data.get(symbol)
        if data is not None and not data.empty:
            # For simplicity, return the last known LTP from the mock data
            return data['ltp'].iloc[-1]
        return None

    def run_backtest(self, start_date: datetime, end_date: datetime):
        """
        Runs the backtest over the specified date range.
        This is a simplified backtest that iterates through symbols and
        simulates trade decisions based on mock LTP and AI scores.
        """
        logger.info(f"Starting backtest from {start_date.date()} to {end_date.date()}...")

        # Reset paper trading system for a clean backtest run
        self.paper_trade_system.available_capital = self.paper_trade_system.initial_capital
        self.paper_trade_system.total_realized_pnl = 0.0
        self.paper_trade_system.active_trades = {}
        self.paper_trade_system.closed_trades = []
        self.paper_trade_system._save_state() # Save reset state

        current_date = start_date
        while current_date <= end_date:
            logger.info(f"Backtesting for date: {current_date.date()}")
            
            # Simulate market open and close times within the day
            # For simplicity, we'll just run one "snapshot" per day
            # In a real backtest, you'd iterate through minute/hourly data

            # Simulate sentiment analysis for each symbol (mocked)
            # In a real backtest, sentiment would be based on historical news for that day
            symbols_to_backtest = list(self.historical_data.keys()) # Use all loaded symbols

            for symbol in symbols_to_backtest:
                # Mock AI score and sentiment for backtesting
                # In a real backtest, this would come from a historical AI model run
                mock_ai_score = round(random.uniform(-0.9, 0.9), 2)
                mock_sentiment = random.choice(['positive', 'negative', 'neutral'])

                # Store mock sentiment in Redis for strategy to pick up
                self.redis_store.redis_client.set(f"score:{symbol}", str(mock_ai_score))
                self.redis_store.redis_client.set(f"sentiment:{symbol}", mock_sentiment)
                self.redis_store.redis_client.set(f"sentiment_last_update:{symbol}", datetime.now().isoformat()) # Mock timestamp

                current_ltp = self.get_mock_ltp_for_backtest(symbol)

                if current_ltp:
                    # Simulate trade entry conditions
                    direction = 'BUY' if mock_ai_score >= self.strategy_manager.MIN_AI_SCORE_BUY else \
                                'SELL' if mock_ai_score <= self.strategy_manager.MIN_AI_SCORE_SELL else None
                    
                    if direction:
                        # Pass dashboard settings as they are part of the trade entry logic
                        tsl_enabled = self.redis_store.get_setting("tsl_enabled", True)
                        ai_tsl_enabled = self.redis_store.get_setting("ai_tsl_enabled", True)
                        leverage_enabled = self.redis_store.get_setting("leverage_enabled", False)
                        ai_auto_leverage = self.redis_store.get_setting("ai_auto_leverage", True)

                        self.paper_trade_system.enter_trade(
                            symbol=symbol,
                            direction=direction,
                            ltp=current_ltp,
                            ai_score=mock_ai_score,
                            sentiment=mock_sentiment,
                            tsl_enabled=tsl_enabled,
                            ai_tsl_enabled=ai_tsl_enabled,
                            leverage_enabled=leverage_enabled,
                            ai_auto_leverage=ai_auto_leverage
                        )
                    else:
                        logger.info(f"BACKTEST: No strong signal for {symbol} (AI Score: {mock_ai_score}).")
                else:
                    logger.warning(f"BACKTEST: No mock LTP found for {symbol}. Skipping for this date.")

            # Simulate end-of-day exit for all active trades
            trades_to_exit_eod = list(self.paper_trade_system.active_trades.keys())
            for trade_id in trades_to_exit_eod:
                trade = self.paper_trade_system.active_trades.get(trade_id)
                if trade:
                    current_ltp = self.get_mock_ltp_for_backtest(trade['symbol'])
                    if current_ltp:
                        self.paper_trade_system.exit_trade(trade_id, current_ltp, 'AUTO_EXIT_EOD_BACKTEST')
                    else:
                        logger.warning(f"BACKTEST: Could not get LTP for {trade['symbol']} for EOD exit.")

            current_date += timedelta(days=1) # Move to next day

        logger.info("Backtest completed.")
        self.display_backtest_results()

    def display_backtest_results(self):
        """Displays the overall results of the backtest."""
        logger.info("\n--- Backtest Results ---")
        metrics = self.paper_trade_system.get_dashboard_metrics()
        for key, value in metrics.items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
        
        logger.info(f"Total Closed Trades: {len(self.paper_trade_system.closed_trades)}")
        logger.info(f"Final Available Capital: {self.paper_trade_system.available_capital:.2f}")
        logger.info(f"Total Realized PnL: {self.paper_trade_system.total_realized_pnl:.2f}")

