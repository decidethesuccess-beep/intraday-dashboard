# strategy.py
# This module defines the core trading strategy logic, including entry, exit,
# stop-loss (SL), target (TGT), and trailing stop-loss (TSL) mechanisms.
# It uses Redis for fetching real-time data and AI scores.

import logging
import os
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Import RedisStore for data access
from redis_store import RedisStore
# Import AIWebhook for sending AI feedback
from ai_webhook import AIWebhook # NEW: Import AIWebhook

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manages the trading strategy logic, including entry, exit,
    and stop-loss/target/trailing stop-loss calculations.
    """
    def __init__(self, redis_store: RedisStore, ai_webhook: AIWebhook): # NEW: Added ai_webhook
        """
        Initializes the StrategyManager with Redis connection and strategy parameters.

        Args:
            redis_store (RedisStore): An instance of RedisStore for data interaction.
            ai_webhook (AIWebhook): An instance of AIWebhook for sending AI feedback. # NEW
        """
        load_dotenv() # Load environment variables

        self.redis_store = redis_store
        self.ai_webhook = ai_webhook # NEW: Store AIWebhook instance

        # Strategy Parameters from .env
        self.MIN_AI_SCORE_BUY = float(os.getenv("MIN_AI_SCORE_BUY", 0.7)) # Minimum AI score for BUY entry
        self.MIN_AI_SCORE_SELL = float(os.getenv("MIN_AI_SCORE_SELL", -0.7)) # Minimum AI score for SELL entry
        self.SL_PERCENT = float(os.getenv("SL_PERCENT", 2)) / 100 # Stop Loss % (e.g., 2% -> 0.02)
        self.TARGET_PERCENT = float(os.getenv("TARGET_PERCENT", 10)) / 100 # Target % (e.g., 10% -> 0.10)
        self.TSL_PERCENT = float(os.getenv("TSL_PERCENT", 1)) / 100 # Trailing Stop Loss % (e.g., 1% -> 0.01)
        self.TSL_ACTIVATION_BUFFER_PERCENT = float(os.getenv("TSL_ACTIVATION_BUFFER_PERCENT", 1)) / 100 # TSL activates after 1% profit

        self.MAX_ACTIVE_POSITIONS = int(os.getenv("MAX_ACTIVE_POSITIONS", 10))
        self.MARKET_OPEN_TIME = dt_time(int(os.getenv("MARKET_OPEN_TIME_HOUR", 9)), int(os.getenv("MARKET_OPEN_TIME_MINUTE", 15)))
        self.MARKET_CLOSE_TIME = dt_time(int(os.getenv("MARKET_CLOSE_TIME_HOUR", 15)), int(os.getenv("MARKET_CLOSE_TIME_MINUTE", 30)))
        self.AUTO_EXIT_TIME = dt_time(int(os.getenv("AUTO_EXIT_TIME_HOUR", 15)), int(os.getenv("AUTO_EXIT_TIME_MINUTE", 20)))

        # Cooldown period from .env
        self.COOLDOWN_PERIOD_SECONDS = int(os.getenv("COOLDOWN_PERIOD_SECONDS", 300)) # Default 5 minutes

        # Leverage parameter from .env
        self.DEFAULT_LEVERAGE_MULTIPLIER = float(os.getenv("DEFAULT_LEVERAGE_MULTIPLIER", 5.0)) # Default 5x leverage

        # Capital Tiers for dynamic capital allocation (new parameters)
        self.INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 100000.0)) # Initial capital for paper trading/backtesting
        self.SMALL_CAPITAL_THRESHOLD = float(os.getenv("SMALL_CAPITAL_THRESHOLD", 50000.0))
        self.MID_CAPITAL_THRESHOLD = float(os.getenv("MID_CAPITAL_THRESHOLD", 500000.0))
        self.SMALL_CAPITAL_ALLOCATION_PCT = float(os.getenv("SMALL_CAPITAL_ALLOCATION_PCT", 10)) / 100 # 10%
        self.MID_CAPITAL_ALLOCATION_PCT = float(os.getenv("MID_CAPITAL_ALLOCATION_PCT", 5)) / 100   # 5%
        self.LARGE_CAPITAL_ALLOCATION_PCT = float(os.getenv("LARGE_CAPITAL_ALLOCATION_PCT", 2)) / 100 # 2%

        # Placeholder for Sentiment Analysis module (will be initialized externally or passed in)
        # For now, it's None. You would integrate a real sentiment analysis tool here.
        self.sentiment_analysis = None # TODO: Initialize with a proper sentiment analysis client

        logger.info("StrategyManager initialized with parameters:")
        logger.info(f"  Min AI Score (BUY): {self.MIN_AI_SCORE_BUY}")
        logger.info(f"  Min AI Score (SELL): {self.MIN_AI_SCORE_SELL}")
        logger.info(f"  SL %: {self.SL_PERCENT*100}%, TGT %: {self.TARGET_PERCENT*100}%, TSL %: {self.TSL_PERCENT*100}%")
        logger.info(f"  TSL Activation Buffer %: {self.TSL_ACTIVATION_BUFFER_PERCENT*100}%")
        logger.info(f"  Max Active Positions: {self.MAX_ACTIVE_POSITIONS}")
        logger.info(f"  Market Open: {self.MARKET_OPEN_TIME}, Market Close: {self.MARKET_CLOSE_TIME}, Auto Exit: {self.AUTO_EXIT_TIME}")
        logger.info(f"  Cooldown Period: {self.COOLDOWN_PERIOD_SECONDS} seconds")
        logger.info(f"  Default Leverage Multiplier: {self.DEFAULT_LEVERAGE_MULTIPLIER}x")
        logger.info(f"  Capital Tiers: Small < {self.SMALL_CAPITAL_THRESHOLD}, Mid < {self.MID_CAPITAL_THRESHOLD}")
        logger.info(f"  Allocation %: Small={self.SMALL_CAPITAL_ALLOCATION_PCT*100}%, Mid={self.MID_CAPITAL_ALLOCATION_PCT*100}%, Large={self.LARGE_CAPITAL_ALLOCATION_PCT*100}%")


    def _is_market_open_for_entry(self, current_time: Optional[dt_time] = None) -> bool:
        """Checks if the market is open for new entries."""
        if current_time is None:
            current_time = datetime.now().time()
        
        # Allow entries only after market open and before auto-exit time
        return self.MARKET_OPEN_TIME <= current_time < self.AUTO_EXIT_TIME

    def get_ai_score(self, symbol: str) -> Optional[float]:
        """
        Retrieves the latest AI score for a given symbol from Redis.
        Assumes scores are stored under 'score:{symbol}' key.
        """
        score_str = self.redis_store.redis_client.get(f"score:{symbol}")
        if score_str:
            try:
                return float(score_str)
            except ValueError:
                logger.warning(f"Invalid AI score format for {symbol}: {score_str}")
                return None
        return None

    def get_sentiment(self, symbol: str) -> Optional[str]:
        """
        Retrieves the latest sentiment for a given symbol from Redis.
        Assumes sentiment is stored under 'sentiment:{symbol}' key.
        Possible values: 'positive', 'negative', 'neutral', or None.
        """
        # The sentiment analysis module is now integrated via main.py and populates Redis.
        # So, we just need to read from Redis.
        sentiment_str = self.redis_store.redis_client.get(f"sentiment:{symbol}")
        if sentiment_str:
            return sentiment_str.decode('utf-8').lower() # Decode bytes and convert to lowercase
        return None

    def get_leverage_tier(self, capital: float, settings: Dict[str, Any]) -> float:
        """
        Determines the leverage tier based on current capital and system settings.
        For now, it returns the default leverage multiplier.
        This can be extended to implement dynamic tiers based on capital.

        Args:
            capital (float): The current available capital.
            settings (Dict[str, Any]): Dictionary of current system settings from Redis.

        Returns:
            float: The determined leverage multiplier.
        """
        # In the future, you could implement more complex logic here
        # based on capital ranges or other settings.
        # Example:
        # if settings.get("ai_auto_leverage_enabled", False):
        #     # Logic for AI-driven leverage tier based on capital or other factors
        #     if capital > 500000: return 5.0
        #     elif capital > 100000: return 3.0
        #     else: return 1.0
        # else:
        #     return settings.get("default_leverage_multiplier", self.DEFAULT_LEVERAGE_MULTIPLIER)

        # For now, simply return the default multiplier from strategy settings
        # or the class attribute if not found in settings.
        return settings.get("default_leverage_multiplier", self.DEFAULT_LEVERAGE_MULTIPLIER)

    def get_capital_allocation_pct(self, current_capital: float) -> float:
        """
        Determines the capital allocation percentage based on the current total capital.
        Implements the 'Capital tiers' logic.
        """
        if current_capital < self.SMALL_CAPITAL_THRESHOLD:
            return self.SMALL_CAPITAL_ALLOCATION_PCT
        elif current_capital < self.MID_CAPITAL_THRESHOLD:
            return self.MID_CAPITAL_ALLOCATION_PCT
        else:
            return self.LARGE_CAPITAL_ALLOCATION_PCT


    def should_enter_trade(self,
                           symbol: str,
                           ltp: float,
                           direction: str,
                           current_active_positions: int,
                           tsl_enabled: bool = True, # New parameter from dashboard
                           ai_tsl_enabled: bool = True # New parameter from dashboard
                           ) -> bool:
        """
        Determines if a new trade should be entered based on strategy rules.

        Args:
            symbol (str): The trading symbol.
            ltp (float): Latest Traded Price.
            direction (str): "BUY" or "SELL".
            current_active_positions (int): Number of currently active trades.
            tsl_enabled (bool): Flag from dashboard if TSL is enabled.
            ai_tsl_enabled (bool): Flag from dashboard if AI-TSL is enabled.

        Returns:
            bool: True if a trade should be entered, False otherwise.
        """
        current_time = datetime.now().time()

        # 1. Market Open Check
        if not self._is_market_open_for_entry(current_time):
            logger.debug(f"Cannot enter {direction} {symbol}: Market not open for entries at {current_time}.")
            return False

        # 2. Max Positions Check
        if current_active_positions >= self.MAX_ACTIVE_POSITIONS:
            logger.debug(f"Cannot enter {direction} {symbol}: Max active positions ({self.MAX_ACTIVE_POSITIONS}) reached.")
            return False

        # 3. Cooldown Check
        if self.redis_store.is_on_cooldown(symbol):
            logger.info(f"Cannot enter {direction} {symbol}: Symbol is on cooldown.")
            return False

        # 4. AI Score Check
        ai_score = self.get_ai_score(symbol)
        if ai_score is None:
            logger.debug(f"Cannot enter {direction} {symbol}: No AI score available.")
            return False

        # 5. Sentiment Filter Check (NEW)
        sentiment = self.get_sentiment(symbol)
        if sentiment is None:
            logger.warning(f"No sentiment data available for {symbol}. Proceeding with trade based on AI score only.")
            # Decide if you want to block trades if sentiment is missing or proceed.
            # For now, we'll proceed but log a warning.
        
        # Check for borderline AI score and send feedback (NEW)
        if (self.MIN_AI_SCORE_BUY - 0.10 <= ai_score <= self.MIN_AI_SCORE_BUY + 0.10 and direction == 'BUY') or \
           (self.MIN_AI_SCORE_SELL - 0.10 <= ai_score <= self.MIN_AI_SCORE_SELL + 0.10 and direction == 'SELL'):
            logger.info(f"AI Score for {symbol} ({ai_score:.2f}) is borderline for {direction} trade. Sending feedback.")
            # Send AI feedback for borderline sentiment
            self.ai_webhook.send_entry_suggestion_feedback(
                symbol=symbol,
                direction=direction,
                ltp=ltp,
                ai_score=ai_score,
                sentiment=sentiment,
                current_active_positions=current_active_positions,
                max_active_positions=self.MAX_ACTIVE_POSITIONS
            )


        if direction == 'BUY':
            if ai_score < self.MIN_AI_SCORE_BUY:
                logger.debug(f"Cannot enter BUY {symbol}: AI score {ai_score:.2f} below min BUY score {self.MIN_AI_SCORE_BUY:.2f}.")
                return False
            # Sentiment filter for BUY: Block if sentiment is negative
            if sentiment == 'negative':
                logger.info(f"Cannot enter BUY {symbol}: Negative sentiment detected.")
                return False
        elif direction == 'SELL':
            if ai_score > self.MIN_AI_SCORE_SELL: # Note: SELL score is negative, so greater means less negative (worse)
                logger.debug(f"Cannot enter SELL {symbol}: AI score {ai_score:.2f} above min SELL score {self.MIN_AI_SCORE_SELL:.2f}.")
                return False
            # Sentiment filter for SELL: Block if sentiment is positive
            if sentiment == 'positive':
                logger.info(f"Cannot enter SELL {symbol}: Positive sentiment detected.")
                return False
        else:
            logger.warning(f"Invalid direction specified: {direction}")
            return False

        # If all checks pass
        logger.info(f"ENTRY SIGNAL: {direction} for {symbol} at {ltp:.2f} with AI Score: {ai_score:.2f}, Sentiment: {sentiment}")
        return True

    def calculate_sl_target(self, entry_price: float, direction: str) -> Dict[str, float]:
        """
        Calculates initial Stop Loss (SL) and Target (TGT) prices.
        """
        if direction == 'BUY':
            sl = entry_price * (1 - self.SL_PERCENT)
            tgt = entry_price * (1 + self.TARGET_PERCENT)
        elif direction == 'SELL':
            sl = entry_price * (1 + self.SL_PERCENT)
            tgt = entry_price * (1 - self.TARGET_PERCENT)
        else:
            raise ValueError("Direction must be 'BUY' or 'SELL'")
        
        # Round to 2 decimal places for practical trading
        return {"sl": round(sl, 2), "tgt": round(tgt, 2)}

    def update_trailing_sl(self, trade: Dict[str, Any], current_ltp: float, ai_tsl_enabled: bool = True) -> Optional[float]:
        """
        Updates the Trailing Stop Loss (TSL) for an active trade.
        AI-TSL logic: TSL activates after a profit buffer and trails based on momentum.

        Args:
            trade (Dict[str, Any]): The active trade dictionary.
            current_ltp (float): The current Latest Traded Price.
            ai_tsl_enabled (bool): Flag from dashboard if AI-TSL is enabled.

        Returns:
            Optional[float]: The updated TSL price, or None if TSL is not active/updated.
        """
        direction = trade['direction']
        entry_price = trade['entry_price']
        current_tsl = trade.get('tsl')
        
        # Calculate current profit percentage
        profit_percent = 0.0
        if direction == 'BUY':
            profit_percent = ((current_ltp - entry_price) / entry_price) * 100
        elif direction == 'SELL':
            # Corrected variable name from 'current_price' to 'current_ltp'
            profit_percent = ((entry_price - current_ltp) / entry_price) * 100 

        # AI-TSL Activation Buffer
        if profit_percent < self.TSL_ACTIVATION_BUFFER_PERCENT:
            # If not yet reached activation buffer, TSL remains inactive or at initial SL
            return current_tsl # No update to TSL yet

        # TSL is now active (or was already active)
        if direction == 'BUY':
            # Update peak price for BUY trades
            if trade['peak_price'] is None or current_ltp > trade['peak_price']:
                trade['peak_price'] = current_ltp
                logger.debug(f"Strategy: Updated peak price for {trade['symbol']} (BUY) to {trade['peak_price']:.2f}")
            
            # Calculate new TSL based on peak price
            new_tsl = trade['peak_price'] * (1 - self.TSL_PERCENT)
            
            # TSL should only move up (for BUY) or stay the same
            if current_tsl is None or new_tsl > current_tsl:
                logger.info(f"Strategy: TSL updated for {trade['symbol']} (BUY) to {round(new_tsl, 2)}.")
                return round(new_tsl, 2)
        
        elif direction == 'SELL':
            # Update trough price for SELL trades
            if trade['trough_price'] is None or current_ltp < trade['trough_price']:
                trade['trough_price'] = current_ltp
                logger.debug(f"Strategy: Updated trough price for {trade['symbol']} (SELL) to {trade['trough_price']:.2f}")
            
            # Calculate new TSL based on trough price
            new_tsl = trade['trough_price'] * (1 + self.TSL_PERCENT)
            
            # TSL should only move down (for SELL) or stay the same
            if current_tsl is None or new_tsl < current_tsl:
                logger.info(f"Strategy: TSL updated for {trade['symbol']} (SELL) to {round(new_tsl, 2)}.")
                return round(new_tsl, 2)
        
        return current_tsl # TSL not updated in this iteration

    def should_exit_trade(self,
                          trade: Dict[str, Any],
                          current_ltp: float,
                          current_timestamp: datetime,
                          tsl_enabled: bool = True, # New parameter from dashboard
                          ai_tsl_enabled: bool = True # New parameter from dashboard
                          ) -> Optional[str]:
        """
        Determines if an active trade should be exited based on various conditions.

        Args:
            trade (Dict[str, Any]): The active trade dictionary.
            current_ltp (float): The current Latest Traded Price.
            current_timestamp (datetime): The current timestamp (for EOD exit).
            tsl_enabled (bool): Flag from dashboard if TSL is enabled.
            ai_tsl_enabled (bool): Flag from dashboard if AI-TSL is enabled.

        Returns:
            Optional[str]: Reason for exit ('SL', 'TGT', 'TSL', 'TREND_FLIP', 'AUTO_EXIT_EOD'),
                           or None if no exit condition is met.
        """
        direction = trade['direction']
        entry_price = trade['entry_price']
        initial_sl = trade['sl']
        target = trade['tgt']
        current_tsl = trade.get('tsl')

        # 1. Stop Loss (SL) Check
        if direction == 'BUY' and current_ltp <= initial_sl:
            logger.info(f"EXIT SIGNAL: SL hit for {trade['symbol']} at {current_ltp:.2f} (Initial SL: {initial_sl:.2f}).")
            return 'SL'
        elif direction == 'SELL' and current_ltp >= initial_sl:
            logger.info(f"EXIT SIGNAL: SL hit for {trade['symbol']} at {current_ltp:.2f} (Initial SL: {initial_sl:.2f}).")
            return 'SL'

        # 2. Target (TGT) Check
        if direction == 'BUY' and current_ltp >= target:
            logger.info(f"EXIT SIGNAL: Target hit for {trade['symbol']} at {current_ltp:.2f} (Target: {target:.2f}).")
            return 'TGT'
        elif direction == 'SELL' and current_ltp <= target:
            logger.info(f"EXIT SIGNAL: Target hit for {trade['symbol']} at {current_ltp:.2f} (Target: {target:.2f}).")
            return 'TGT'

        # 3. Trailing Stop Loss (TSL) Check (if enabled)
        # Only apply TSL check if tsl_enabled is True AND current_tsl is set
        if tsl_enabled and current_tsl is not None:
            if direction == 'BUY' and current_ltp <= current_tsl:
                logger.info(f"EXIT SIGNAL: TSL hit for {trade['symbol']} at {current_ltp:.2f} (TSL: {current_tsl:.2f}).")
                return 'TSL'
            elif direction == 'SELL' and current_ltp >= current_tsl:
                logger.info(f"EXIT SIGNAL: TSL hit for {trade['symbol']} at {current_ltp:.2f} (TSL: {current_tsl:.2f}).")
                return 'TSL'

        # 4. Trend-Flip Exit (AI Score based)
        ai_score = self.get_ai_score(trade['symbol'])
        if ai_score is not None:
            if direction == 'BUY' and ai_score < 0: # If BUY trade, and AI score turns negative
                logger.info(f"EXIT SIGNAL: Trend-flip (AI score negative) for {trade['symbol']} at {current_ltp:.2f} (AI Score: {ai_score:.2f}).")
                return 'TREND_FLIP'
            elif direction == 'SELL' and ai_score > 0: # If SELL trade, and AI score turns positive
                logger.info(f"EXIT SIGNAL: Trend-flip (AI score positive) for {trade['symbol']} at {current_ltp:.2f} (AI Score: {ai_score:.2f}).")
                return 'TREND_FLIP'

        # 5. Auto Exit at End of Day (EOD)
        if current_timestamp.time() >= self.AUTO_EXIT_TIME:
            logger.info(f"EXIT SIGNAL: Auto-exit at EOD for {trade['symbol']} at {current_ltp:.2f} (Time: {current_timestamp.time()}).")
            return 'AUTO_EXIT_EOD'

        # 6. Sentiment-based Exit (NEW)
        sentiment = self.get_sentiment(trade['symbol'])
        if sentiment is not None:
            if direction == 'BUY' and sentiment == 'negative':
                logger.info(f"EXIT SIGNAL: Negative sentiment detected for BUY trade {trade['symbol']} at {current_ltp:.2f}.")
                return 'SENTIMENT_FLIP'
            elif direction == 'SELL' and sentiment == 'positive':
                logger.info(f"EXIT SIGNAL: Positive sentiment detected for SELL trade {trade['symbol']} at {current_ltp:.2f}.")
                return 'SENTIMENT_FLIP'


        return None # No exit condition met

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    print("--- Starting StrategyManager Module Test ---")
    
    # Mock RedisStore for testing purposes
    class MockRedisStoreForStrategy:
        def __init__(self):
            self.data = {} # Simulates Redis data
            logger.info("MockRedisStoreForStrategy initialized.")

        def redis_client(self):
            # Provide a mock redis_client with get and setex methods
            class MockRedisClient:
                def __init__(self, parent_data):
                    self.parent_data = parent_data
                
                def get(self, key):
                    logger.debug(f"Mock Redis GET: {key}")
                    return self.parent_data.get(key)
                
                def setex(self, key, expiry, value):
                    logger.debug(f"Mock Redis SETEX: {key} -> {value} (Expiry: {expiry})")
                    self.parent_data[key] = value # Store for now, expiry not fully simulated
                
                def exists(self, key):
                    logger.debug(f"Mock Redis EXISTS: {key}")
                    return 1 if key in self.parent_data else 0

                def delete(self, key):
                    logger.debug(f"Mock Redis DELETE: {key}")
                    if key in self.parent_data:
                        del self.parent_data[key]


            return MockRedisClient(self.data)

        def is_on_cooldown(self, symbol: str) -> bool:
            # Simulate cooldown by checking a specific key
            cooldown_key = f"cooldown:{symbol}"
            return self.redis_client().exists(cooldown_key) == 1

        def set_cooldown_timer(self, symbol: str, duration_seconds: int):
            cooldown_key = f"cooldown:{symbol}"
            self.redis_client().setex(cooldown_key, duration_seconds, "active")
        
        def load_settings_from_redis(self) -> Dict[str, Any]: # Mock this method
            return {
                "min_ai_score_buy": 0.7,
                "min_ai_score_sell": -0.7,
                "sl_percent": 0.02,
                "target_percent": 0.10,
                "tsl_percent": 0.01,
                "tsl_activation_buffer_percent": 0.01,
                "cooldown_period_seconds": 300,
                "max_active_positions": 10,
                "market_open_time": "09:15",
                "market_close_time": "15:30",
                "auto_exit_time": "15:20",
                "default_leverage_multiplier": 5.0,
                "tsl_enabled": True,
                "ai_tsl_enabled": True,
                "leverage_enabled": True,
                "ai_auto_leverage": True,
                "is_sync_paused": False
            }


    # Mock AIWebhook for testing purposes
    class MockAIWebhookForStrategy:
        def send_entry_suggestion_feedback(self, *args, **kwargs):
            logger.info(f"Mock AIWebhook: send_entry_suggestion_feedback called with args: {args}, kwargs: {kwargs}")
        def send_trade_rejection_feedback(self, *args, **kwargs):
            logger.info(f"Mock AIWebhook: send_trade_rejection_feedback called with args: {args}, kwargs: {kwargs}")
        # Add other mock methods if needed by StrategyManager


    mock_redis_store = MockRedisStoreForStrategy()
    mock_ai_webhook = MockAIWebhookForStrategy() # NEW: Instantiate mock AIWebhook
    strategy_manager = StrategyManager(mock_redis_store, mock_ai_webhook) # NEW: Pass mock AIWebhook

    # --- Test Capital Allocation Tiers ---
    print("\n--- Testing Capital Allocation Tiers ---")
    print(f"Capital 40000: Allocation % = {strategy_manager.get_capital_allocation_pct(40000) * 100:.2f}%") # Should be SMALL
    print(f"Capital 100000: Allocation % = {strategy_manager.get_capital_allocation_pct(100000) * 100:.2f}%") # Should be MID
    print(f"Capital 600000: Allocation % = {strategy_manager.get_capital_allocation_pct(600000) * 100:.2f}%") # Should be LARGE

    # --- Test Leverage Tier (NEW) ---
    print("\n--- Testing Leverage Tier ---")
    mock_settings = mock_redis_store.load_settings_from_redis()
    print(f"Leverage Tier for 100000 capital: {strategy_manager.get_leverage_tier(100000.0, mock_settings)}x") # Should be 5.0x


    # --- Test Entry Conditions ---
    print("\n--- Testing Entry Conditions ---")
    symbol_test = "INFY"
    ltp_test = 1500.0
    mock_redis_store.redis_client().set(f"score:{symbol_test}", "0.80") # Set AI score for INFY
    mock_redis_store.redis_client().set(f"sentiment:{symbol_test}", "neutral") # Set sentiment for INFY

    # Test BUY entry (should pass)
    if strategy_manager.should_enter_trade(symbol_test, ltp_test, 'BUY', 0):
        print(f"✅ Should enter BUY for {symbol_test}.")
    else:
        print(f"❌ Should NOT enter BUY for {symbol_test}.")

    # Test BUY entry with low score (should fail)
    mock_redis_store.redis_client().set(f"score:{symbol_test}", "0.60")
    if not strategy_manager.should_enter_trade(symbol_test, ltp_test, 'BUY', 0):
        print(f"✅ Correctly prevented BUY for {symbol_test} (low score).")
    else:
        print(f"❌ Incorrectly allowed BUY for {symbol_test} (low score).")

    # Test BUY entry with negative sentiment (should fail)
    mock_redis_store.redis_client().set(f"score:{symbol_test}", "0.80") # Reset score
    mock_redis_store.redis_client().set(f"sentiment:{symbol_test}", "negative")
    if not strategy_manager.should_enter_trade(symbol_test, ltp_test, 'BUY', 0):
        print(f"✅ Correctly prevented BUY for {symbol_test} (negative sentiment).")
    else:
        print(f"❌ Incorrectly allowed BUY for {symbol_test} (negative sentiment).")
    mock_redis_store.redis_client().set(f"sentiment:{symbol_test}", "neutral") # Reset sentiment


    # Test SELL entry (should pass)
    mock_redis_store.redis_client().set(f"score:{symbol_test}", "-0.80") # Set AI score for INFY for SELL
    if strategy_manager.should_enter_trade(symbol_test, ltp_test, 'SELL', 0):
        print(f"✅ Should enter SELL for {symbol_test}.")
    else:
        print(f"❌ Should NOT enter SELL for {symbol_test}.")

    # Test SELL entry with low score (should fail)
    mock_redis_store.redis_client().set(f"score:{symbol_test}", "-0.60") # Less negative, worse for SELL
    if not strategy_manager.should_enter_trade(symbol_test, ltp_test, 'SELL', 0):
        print(f"✅ Correctly prevented SELL for {symbol_test} (low score).")
    else:
        print(f"❌ Incorrectly allowed SELL for {symbol_test} (low score).")

    # Test SELL entry with positive sentiment (should fail)
    mock_redis_store.redis_client().set(f"score:{symbol_test}", "-0.80") # Reset score
    mock_redis_store.redis_client().set(f"sentiment:{symbol_test}", "positive")
    if not strategy_manager.should_enter_trade(symbol_test, ltp_test, 'SELL', 0):
        print(f"✅ Correctly prevented SELL for {symbol_test} (positive sentiment).")
    else:
        print(f"❌ Incorrectly allowed SELL for {symbol_test} (positive sentiment).")
    mock_redis_store.redis_client().set(f"sentiment:{symbol_test}", "neutral") # Reset sentiment


    # Test entry when max positions reached (should fail)
    mock_redis_store.redis_client().set(f"score:{symbol_test}", "0.80") # Reset score
    if not strategy_manager.should_enter_trade(symbol_test, ltp_test, 'BUY', strategy_manager.MAX_ACTIVE_POSITIONS):
        print(f"✅ Correctly prevented BUY for {symbol_test} (max positions).")
    else:
        print(f"❌ Incorrectly allowed BUY for {symbol_test} (max positions).")

    # Test entry when on cooldown (should fail)
    mock_redis_store.set_cooldown_timer(symbol_test, 60) # Set cooldown for 60 seconds
    if not strategy_manager.should_enter_trade(symbol_test, ltp_test, 'BUY', 0):
        print(f"✅ Correctly prevented BUY for {symbol_test} (on cooldown).")
    else:
        print(f"❌ Incorrectly allowed BUY for {symbol_test} (on cooldown).")
    mock_redis_store.redis_client().delete(f"cooldown:{symbol_test}") # Clear cooldown for next tests

    # Test BUY entry with borderline AI score (should trigger webhook)
    mock_redis_store.redis_client().set(f"score:{symbol_test}", "0.68") # Borderline BUY score
    if strategy_manager.should_enter_trade(symbol_test, ltp_test, 'BUY', 0):
        print(f"✅ Should enter BUY for {symbol_test} (borderline score).")
    else:
        print(f"❌ Should NOT enter BUY for {symbol_test} (borderline score).")


    # --- Test SL/TGT Calculation ---
    print("\n--- Testing SL/TGT Calculation ---")
    entry_price_calc = 100.0
    sl_tgt_buy = strategy_manager.calculate_sl_target(entry_price_calc, 'BUY')
    print(f"BUY Entry: {entry_price_calc}, SL: {sl_tgt_buy['sl']:.2f}, TGT: {sl_tgt_buy['tgt']:.2f}")
    # Expected SL: 100 * (1 - 0.02) = 98.0, TGT: 100 * (1 + 0.10) = 110.0
    
    sl_tgt_sell = strategy_manager.calculate_sl_target(entry_price_calc, 'SELL')
    print(f"SELL Entry: {entry_price_calc}, SL: {sl_tgt_sell['sl']:.2f}, TGT: {sl_tgt_sell['tgt']:.2f}")
    # Expected SL: 100 * (1 + 0.02) = 102.0, TGT: 100 * (1 - 0.10) = 90.0


    # --- Test TSL Update and Exit Conditions ---
    print("\n--- Testing TSL Update and Exit Conditions ---")
    
    # Scenario 1: BUY Trade, Price moves up, TSL activates and trails
    trade_buy = {
        'trade_id': 'buy1', 'symbol': 'RELIANCE', 'entry_price': 2000.0, 'qty': 10,
        'direction': 'BUY', 'sl': 1960.0, 'tgt': 2200.0, 'tsl': None,
        'peak_price': 2000.0, 'trough_price': None
    }
    mock_redis_store.redis_client().set(f"score:{trade_buy['symbol']}", "0.80")
    mock_redis_store.redis_client().set(f"sentiment:{trade_buy['symbol']}", "neutral")

    # Price moves slightly up, not enough for TSL activation
    current_ltp = 2010.0 # +0.5%
    updated_tsl = strategy_manager.update_trailing_sl(trade_buy, current_ltp)
    exit_reason = strategy_manager.should_exit_trade(trade_buy, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, TSL: {updated_tsl}, Exit: {exit_reason}") # TSL should be None, Exit None

    # Price moves further up, TSL activates and moves
    current_ltp = 2030.0 # +1.5% (above 1% buffer)
    updated_tsl = strategy_manager.update_trailing_sl(trade_buy, current_ltp)
    exit_reason = strategy_manager.should_exit_trade(trade_buy, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, TSL: {updated_tsl}, Exit: {exit_reason}") # TSL should be 2030 * (1-0.01) = 2009.7, Exit None

    # Price drops, hits TSL
    current_ltp = 2000.0 # Drops below TSL (2009.7)
    trade_buy['tsl'] = updated_tsl # Manually update trade's TSL for next check
    exit_reason = strategy_manager.should_exit_trade(trade_buy, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, TSL: {trade_buy['tsl']}, Exit: {exit_reason}") # Exit should be TSL

    # Scenario 2: SELL Trade, Price moves down, TSL activates and trails
    trade_sell = {
        'trade_id': 'sell1', 'symbol': 'TCS', 'entry_price': 3500.0, 'qty': 5,
        'direction': 'SELL', 'sl': 3570.0, 'tgt': 3150.0, 'tsl': None,
        'peak_price': None, 'trough_price': 3500.0
    }
    mock_redis_store.redis_client().set(f"score:{trade_sell['symbol']}", "-0.85")
    mock_redis_store.redis_client().set(f"sentiment:{trade_sell['symbol']}", "neutral")

    # Price moves slightly down, not enough for TSL activation
    current_ltp = 3480.0 # -0.57%
    updated_tsl = strategy_manager.update_trailing_sl(trade_sell, current_ltp)
    exit_reason = strategy_manager.should_exit_trade(trade_sell, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, TSL: {updated_tsl}, Exit: {exit_reason}") # TSL should be None, Exit None

    # Price moves further down, TSL activates and moves
    current_ltp = 3450.0 # -1.42% (above 1% buffer)
    updated_tsl = strategy_manager.update_trailing_sl(trade_sell, current_ltp)
    exit_reason = strategy_manager.should_exit_trade(trade_sell, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, TSL: {updated_tsl}, Exit: {exit_reason}") # TSL should be 3450 * (1+0.01) = 3484.5, Exit None

    # Price goes up, hits TSL
    current_ltp = 3490.0 # Goes above TSL (3484.5)
    trade_sell['tsl'] = updated_tsl # Manually update trade's TSL for next check
    exit_reason = strategy_manager.should_exit_trade(trade_sell, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, TSL: {trade_sell['tsl']}, Exit: {exit_reason}") # Exit should be TSL

    # Test Trend-Flip Exit
    print("\n--- Testing Trend-Flip Exit ---")
    trade_buy_flip = {
        'trade_id': 'buy_flip', 'symbol': 'SBIN', 'entry_price': 500.0, 'qty': 20,
        'direction': 'BUY', 'sl': 490.0, 'tgt': 550.0, 'tsl': None,
        'peak_price': 500.0, 'trough_price': None
    }
    mock_redis_store.redis_client().set(f"score:{trade_buy_flip['symbol']}", "0.75")
    mock_redis_store.redis_client().set(f"sentiment:{trade_buy_flip['symbol']}", "neutral")
    current_ltp = 505.0
    exit_reason = strategy_manager.should_exit_trade(trade_buy_flip, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, AI Score (positive), Exit: {exit_reason}") # Should be None

    mock_redis_store.redis_client().set(f"score:{trade_buy_flip['symbol']}", "-0.10") # AI score turns negative
    exit_reason = strategy_manager.should_exit_trade(trade_buy_flip, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, AI Score (negative), Exit: {exit_reason}") # Should be TREND_FLIP

    # Test Auto-Exit EOD
    print("\n--- Testing Auto-Exit EOD ---")
    trade_eod = {
        'trade_id': 'eod1', 'symbol': 'HDFC_BANK', 'entry_price': 1500.0, 'qty': 10,
        'direction': 'BUY', 'sl': 1480.0, 'tgt': 1550.0, 'tsl': None,
        'peak_price': 1500.0, 'trough_price': None
    }
    mock_redis_store.redis_client().set(f"score:{trade_eod['symbol']}", "0.70")
    mock_redis_store.redis_client().set(f"sentiment:{trade_eod['symbol']}", "neutral")
    current_ltp = 1505.0

    # Before auto-exit time
    exit_reason = strategy_manager.should_exit_trade(trade_eod, current_ltp, datetime.now().replace(hour=15, minute=19, second=59))
    print(f"LTP: {current_ltp}, Time: 15:19:59, Exit: {exit_reason}") # Should be None

    # At or after auto-exit time
    exit_reason = strategy_manager.should_exit_trade(trade_eod, current_ltp, datetime.now().replace(hour=15, minute=20, second=0))
    print(f"LTP: {current_ltp}, Time: 15:20:00, Exit: {exit_reason}") # Should be AUTO_EXIT_EOD

    # Test Sentiment-based Exit (NEW)
    print("\n--- Testing Sentiment-based Exit ---")
    trade_sentiment_exit_buy = {
        'trade_id': 'sentiment_buy', 'symbol': 'RELIANCE', 'entry_price': 2100.0, 'qty': 10,
        'direction': 'BUY', 'sl': 2050.0, 'tgt': 2200.0, 'tsl': None,
        'peak_price': 2150.0, 'trough_price': None
    }
    mock_redis_store.redis_client().set(f"score:{trade_sentiment_exit_buy['symbol']}", "0.80")
    mock_redis_store.redis_client().set(f"sentiment:{trade_sentiment_exit_buy['symbol']}", "neutral")
    current_ltp = 2140.0 # In profit, but not at TGT/SL

    exit_reason = strategy_manager.should_exit_trade(trade_sentiment_exit_buy, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, Sentiment: neutral, Exit: {exit_reason}") # Should be None

    mock_redis_store.redis_client().set(f"sentiment:{trade_sentiment_exit_buy['symbol']}", "negative")
    exit_reason = strategy_manager.should_exit_trade(trade_sentiment_exit_buy, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, Sentiment: negative, Exit: {exit_reason}") # Should be SENTIMENT_FLIP

    trade_sentiment_exit_sell = {
        'trade_id': 'sentiment_sell', 'symbol': 'TCS', 'entry_price': 3400.0, 'qty': 5,
        'direction': 'SELL', 'sl': 3450.0, 'tgt': 3300.0, 'tsl': None,
        'peak_price': None, 'trough_price': 3350.0
    }
    mock_redis_store.redis_client().set(f"score:{trade_sentiment_exit_sell['symbol']}", "-0.85")
    mock_redis_store.redis_client().set(f"sentiment:{trade_sentiment_exit_sell['symbol']}", "neutral")
    current_ltp = 3360.0 # In profit, but not at TGT/SL

    exit_reason = strategy_manager.should_exit_trade(trade_sentiment_exit_sell, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, Sentiment: neutral, Exit: {exit_reason}") # Should be None

    mock_redis_store.redis_client().set(f"sentiment:{trade_sentiment_exit_sell['symbol']}", "positive")
    exit_reason = strategy_manager.should_exit_trade(trade_sentiment_exit_sell, current_ltp, datetime.now())
    print(f"LTP: {current_ltp}, Sentiment: positive, Exit: {exit_reason}") # Should be SENTIMENT_FLIP


    print("\n--- StrategyManager Module Test End ---")
