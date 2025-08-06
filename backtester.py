# backtester.py
# This module provides a backtesting framework for the DTS Intraday AI Trading System.
# It simulates trading over historical data to evaluate strategy performance.

import logging
import pandas as pd
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Any, List, Optional
import os
import random # Import random for mock AI scores and sentiment

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
        self.strategy_manager = StrategyManager(redis_store, self.ai_webhook) # Pass ai_webhook
        # Pass ai_webhook to PaperTradeSystem as well
        self.paper_trade_system = PaperTradeSystem(redis_store, self.strategy_manager, None, angel_api, self.ai_webhook) # NEW: Pass ai_webhook


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

    def run_backtest(self,
                     symbol: str, # Added symbol for specific backtest
                     symbol_token: str, # Added symbol_token
                     exchange_type: str, # Added exchange_type
                     from_date: datetime,
                     to_date: datetime,
                     interval: str,
                     leverage_enabled: bool,
                     ai_auto_leverage: bool
                     ) -> Optional[Dict[str, Any]]: # Changed return type to Dict for report
        """
        Runs the backtest over the specified date range for a single symbol.
        This is a simplified backtest that iterates through time and
        simulates trade decisions based on mock LTP and AI scores.

        Args:
            symbol (str): The trading symbol to backtest.
            symbol_token (str): Angel One token for the symbol.
            exchange_type (str): Exchange type for the symbol.
            from_date (datetime): Start datetime for backtest.
            to_date (datetime): End datetime for backtest.
            interval (str): Candle interval for historical data (e.g., "ONE_MINUTE").
            leverage_enabled (bool): Whether leverage is enabled for this backtest.
            ai_auto_leverage (bool): Whether AI auto-leverage is enabled for this backtest.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing backtest report metrics and trade logs,
                                      or None if backtest fails.
        """
        logger.info(f"Starting backtest for {symbol} from {from_date.strftime('%Y-%m-%d %H:%M')} to {to_date.strftime('%Y-%m-%d %H:%M')}...")

        # Reset paper trading system for a clean backtest run
        self.paper_trade_system.available_capital = self.paper_trade_system.initial_capital
        self.paper_trade_system.total_realized_pnl = 0.0
        self.paper_trade_system.active_trades = {}
        self.paper_trade_system.closed_trades = []
        self.paper_trade_system._save_state() # Save reset state

        # Store capital history for equity curve plotting
        self.capital_history = []
        self.capital_history.append((from_date, self.paper_trade_system.initial_capital))

        # Fetch actual historical data for the symbol
        historical_candles = self.angel_api.get_candle_data({
            "exchange": exchange_type,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date.strftime('%Y-%m-%d %H:%M'),
            "todate": to_date.strftime('%Y-%m-%d %H:%M')
        })

        if not historical_candles or not historical_candles.get('data'):
            logger.warning(f"No historical data found for {symbol} for the backtest period. Aborting backtest.")
            return None

        candles_data = historical_candles['data']
        logger.info(f"Fetched {len(candles_data)} candles for backtest of {symbol}.")

        for candle in candles_data:
            try:
                # Parse candle data
                timestamp_str = candle[0]
                ltp = float(candle[4]) # Close price as LTP for simulation
                
                # Convert timestamp to datetime object
                # Angel One timestamp format: "YYYY-MM-DDTHH:MM:SS+HH:MM"
                current_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S%z")

                # Mock AI score and sentiment for backtesting
                # In a real backtest, this would come from a historical AI model run
                mock_ai_score = round(random.uniform(-0.9, 0.9), 2)
                mock_sentiment = random.choice(['positive', 'negative', 'neutral'])

                # Store mock sentiment in Redis for strategy to pick up
                # Ensure Redis is connected. If not, this will fail silently in mock mode.
                if self.redis_store.redis_client:
                    self.redis_store.redis_client.set(f"score:{symbol}", str(mock_ai_score))
                    self.redis_store.redis_client.set(f"sentiment:{symbol}", mock_sentiment)
                    self.redis_store.redis_client.set(f"sentiment_last_update:{symbol}", current_timestamp.isoformat()) # Mock timestamp
                    self.redis_store.write_ltp(symbol, ltp) # Update LTP in Redis for strategy/dashboard

                # Simulate market open/close for entry logic
                # Pass current_timestamp for market timing checks in strategy
                current_time_for_strategy = current_timestamp.time()
                is_market_open_for_entry = self.strategy_manager._is_market_open_for_entry(current_time_for_strategy)

                # 1. Manage Active Trades (Exit Logic)
                trades_to_check = list(self.paper_trade_system.active_trades.keys())
                for trade_id in trades_to_check:
                    trade = self.paper_trade_system.active_trades.get(trade_id)
                    if not trade:
                        continue

                    # Update TSL if enabled for this trade
                    if trade['tsl_enabled']: # Use trade-specific TSL setting
                        updated_tsl = self.strategy_manager.update_trailing_sl(
                            trade,
                            ltp, # Use current candle's LTP
                            ai_tsl_enabled=trade['ai_tsl_enabled'] # Use trade-specific AI-TSL setting
                        )
                        if updated_tsl is not None and updated_tsl != trade.get('tsl'):
                            trade['tsl'] = updated_tsl
                            self.redis_store.save_trade_state(trade_id, trade) # Save updated TSL

                    # Check exit conditions
                    exit_reason = self.strategy_manager.should_exit_trade(
                        trade,
                        ltp, # Use current candle's LTP
                        current_timestamp,
                        tsl_enabled=trade['tsl_enabled'], # Pass trade-specific TSL flag
                        ai_tsl_enabled=trade['ai_tsl_enabled'] # Pass trade-specific AI-TSL flag
                    )

                    if exit_reason:
                        self.paper_trade_system.exit_trade(trade_id, ltp, exit_reason)
                
                # 2. Look for New Trade Opportunities (Entry Logic)
                if is_market_open_for_entry:
                    direction = None
                    if mock_ai_score >= self.strategy_manager.MIN_AI_SCORE_BUY:
                        direction = 'BUY'
                    elif mock_ai_score <= self.strategy_manager.MIN_AI_SCORE_SELL:
                        direction = 'SELL'

                    if direction:
                        # Check should_enter_trade with dashboard-level settings
                        if self.strategy_manager.should_enter_trade(
                            symbol,
                            ltp,
                            direction,
                            len(self.paper_trade_system.active_trades),
                            tsl_enabled=leverage_enabled, # Use backtest-specific settings
                            ai_tsl_enabled=ai_auto_leverage # Use backtest-specific settings
                        ):
                            self.paper_trade_system.enter_trade(
                                symbol,
                                direction,
                                ltp,
                                mock_ai_score,
                                mock_sentiment,
                                tsl_enabled=leverage_enabled, # Pass backtest-specific settings
                                ai_tsl_enabled=ai_auto_leverage,
                                leverage_enabled=leverage_enabled,
                                ai_auto_leverage=ai_auto_leverage
                            )
                        else:
                            # Log why trade was not entered (e.g., sentiment filter, cooldown already logged by strategy)
                            pass # Specific rejection reasons are logged by strategy_manager.should_enter_trade

                # Update capital history at each step
                self.capital_history.append((current_timestamp, self.paper_trade_system.available_capital))

            except Exception as e:
                logger.error(f"Error processing candle for {symbol} at {candle[0]}: {e}", exc_info=True)
                continue

        logger.info("Backtest completed for %s.", symbol)
        
        # Calculate final metrics for the report
        final_metrics = self.paper_trade_system.get_dashboard_metrics()
        
        # Calculate Max Drawdown
        capital_series = pd.Series([c[1] for c in self.capital_history])
        peak = capital_series.expanding(min_periods=1).max()
        drawdown = (capital_series - peak) / peak
        max_drawdown_percent = abs(drawdown.min()) * 100 if not drawdown.empty else 0.0

        report = {
            "symbol": symbol,
            "total_pnl": final_metrics['total_realized_pnl'],
            "num_trades": len(self.paper_trade_system.closed_trades),
            "win_rate": final_metrics['win_rate'],
            "final_capital": final_metrics['available_capital'],
            "max_drawdown_percent": max_drawdown_percent,
            "trade_logs": self.paper_trade_system.closed_trades,
            "leverage_used_in_backtest": leverage_enabled,
            "ai_auto_leverage_in_backtest": ai_auto_leverage,
            "tsl_enabled_in_backtest": leverage_enabled, # Assuming tsl_enabled is linked to leverage_enabled for backtest parameters
            "ai_tsl_enabled_in_backtest": ai_auto_leverage # Assuming ai_tsl_enabled is linked to ai_auto_leverage for backtest parameters
        }
        return report

