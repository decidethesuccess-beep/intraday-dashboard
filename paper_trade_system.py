# paper_trade_system.py
# This module simulates a paper trading system, managing virtual capital,
# executing trades based on strategy signals, and tracking PnL.
# It interacts with Redis for state persistence and Angel One for LTP.

import logging
import os
import json
import uuid # For generating unique trade IDs
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Import necessary modules
from redis_store import RedisStore
from strategy import StrategyManager
from dhan_api_patch import DhanAPI # For potential future Dhan order placement
from angelone_api_patch import AngelOneAPI # For fetching LTP fallback
from ai_webhook import AIWebhook # NEW: Import AIWebhook for sending feedback

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

class PaperTradeSystem:
    """
    Simulates a trading system, managing virtual capital, executing trades,
    and tracking profit/loss.
    """
    def __init__(self,
                 redis_store: RedisStore,
                 strategy_manager: StrategyManager,
                 dhan_api: Optional[DhanAPI], # Optional Dhan API for future live trading
                 angel_api: AngelOneAPI, # Angel One API for LTP fallback
                 ai_webhook: AIWebhook # NEW: AIWebhook for sending feedback
                 ):
        """
        Initializes the PaperTradeSystem.

        Args:
            redis_store (RedisStore): An instance of RedisStore for data persistence.
            strategy_manager (StrategyManager): An instance of StrategyManager for trade logic.
            dhan_api (Optional[DhanAPI]): An optional instance of DhanAPI for live order placement.
            angel_api (AngelOneAPI): An instance of AngelOneAPI for LTP fallback.
            ai_webhook (AIWebhook): An instance of AIWebhook for sending AI feedback. # NEW
        """
        load_dotenv()

        self.redis_store = redis_store
        self.strategy_manager = strategy_manager
        self.dhan_api = dhan_api # Store DhanAPI instance
        self.angel_api = angel_api # Store AngelOneAPI instance
        self.ai_webhook = ai_webhook # NEW: Store AIWebhook instance

        self.initial_capital = float(os.getenv("INITIAL_CAPITAL", 100000.0))
        self.available_capital = self.initial_capital
        self.total_realized_pnl = 0.0
        self.active_trades: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.is_sync_paused = False # Flag to pause/resume system sync

        # Load state from Redis on startup
        self._load_state()

        # Define mapping for symbols to their Angel One tokens (for LTP fetching)
        # This should be consistent with your subscribed_symbols.json or similar
        self.ANGEL_ONE_SYMBOL_MAP = {
            "NIFTY_50": {"token": "26009", "exchangeType": "NSE"},
            "RELIANCE": {"token": "11536", "exchangeType": "NSE"},
            "TCS": {"token": "3045", "exchangeType": "NSE"},
            "HDFC_BANK": {"token": "3432", "exchangeType": "NSE"},
            "SBIN": {"token": "1333", "exchangeType": "NSE"},
            "ICICI_BANK": {"token": "1660", "exchangeType": "NSE"},
            "INFY": {"token": "10604", "exchangeType": "NSE"},
            "ITC": {"token": "10606", "exchangeType": "NSE"},
            "MARUTI": {"token": "20374", "exchangeType": "NSE"},
            "AXISBANK": {"token": "1348", "exchangeType": "NSE"},
            # Add more as needed, ensure tokens are correct
        }

        # Define mapping for symbols to their Dhan Security IDs (for Dhan API calls)
        # IMPORTANT: These are placeholders. You MUST replace these with the actual
        # Security IDs, Exchange Segments, and Instrument Types from Dhan's instrument master
        # or discovery APIs once you have access to them.
        self.SYMBOL_DHAN_MAP = {
            "INFY": {"securityId": "1594", "exchangeSegment": "NSE_EQ", "instrumentType": "EQUITY"}, # Example INFY ID
            "TCS": {"securityId": "11536", "exchangeSegment": "NSE_EQ", "instrumentType": "EQUITY"}, # Example TCS ID
            "RELIANCE": {"securityId": "2885", "exchangeSegment": "NSE_EQ", "instrumentType": "EQUITY"},
            "SBIN": {"securityId": "3045", "exchangeSegment": "NSE_EQ", "instrumentType": "EQUITY"},
            "HDFC_BANK": {"securityId": "3432", "exchangeSegment": "NSE_EQ", "instrumentType": "EQUITY"},
            "ICICI_BANK": {"securityId": "4963", "exchangeSegment": "NSE_EQ", "instrumentType": "EQUITY"},
            "ITC": {"securityId": "10606", "exchangeSegment": "NSE_EQ", "instrumentType": "EQUITY"},
            "MARUTI": {"securityId": "20374", "exchangeType": "NSE_EQ", "instrumentType": "EQUITY"},
            "AXISBANK": {"securityId": "1348", "exchangeType": "NSE_EQ", "instrumentType": "EQUITY"},
            "NIFTY_50": {"securityId": "999260000", "exchangeSegment": "NSE_INDEX", "instrumentType": "INDEX"}, # Example NIFTY ID
            # Add more symbols as needed
        }

        logger.info("PaperTradeSystem initialized. Initial Capital: ₹%.2f", self.initial_capital)

    def _load_state(self):
        """Loads the trading system's state from Redis."""
        if not self.redis_store.connect():
            logger.critical("Failed to connect to Redis during state load. Using default state.")
            return

        state = self.redis_store.load_system_state()
        if state:
            self.available_capital = state.get("available_capital", self.initial_capital)
            self.total_realized_pnl = state.get("total_pnl", 0.0)
            self.is_sync_paused = self.redis_store.get_setting("is_sync_paused", False)
            logger.info("Loaded system state from Redis. Available Capital: ₹%.2f, Total PnL: ₹%.2f, Sync Paused: %s",
                        self.available_capital, self.total_realized_pnl, self.is_sync_paused)
        else:
            logger.info("No existing system state found in Redis. Initializing with default values.")
            self._save_state() # Save initial state

        # Load active trades
        self.active_trades = self.redis_store.get_all_active_trades()
        logger.info(f"Loaded {len(self.active_trades)} active trades from Redis.")

        # Load closed trades
        self.closed_trades = self.redis_store.get_all_closed_trades()
        logger.info(f"Loaded {len(self.closed_trades)} closed trades from Redis.")

    def _save_state(self):
        """Saves the trading system's state to Redis."""
        if self.redis_store.connect():
            self.redis_store.save_system_state(self.available_capital, self.total_realized_pnl, self.is_sync_paused)
            # Active trades are saved/updated individually by save_trade_state/remove_active_trade
            # Closed trades are added to a list by add_closed_trade
        else:
            logger.warning("Redis not connected. Could not save system state.")

    def toggle_sync(self):
        """Toggles the system sync (pause/resume)."""
        self.is_sync_paused = not self.is_sync_paused
        self.redis_store.set_setting("is_sync_paused", self.is_sync_paused)
        logger.info(f"Trading system sync toggled to: {'PAUSED' if self.is_sync_paused else 'ACTIVE'}")

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Calculates and returns key metrics for the dashboard.
        """
        # Ensure latest state is loaded for metrics
        self._load_state() 

        # Calculate unrealized PnL
        total_unrealized_pnl = 0.0
        for trade_id, trade in self.active_trades.items():
            current_ltp = self.redis_store.read_ltp(trade['symbol'])
            if current_ltp is not None:
                if trade['direction'] == 'BUY':
                    unrealized_pnl = (current_ltp - trade['entry_price']) * trade['qty']
                elif trade['direction'] == 'SELL':
                    unrealized_pnl = (trade['entry_price'] - current_ltp) * trade['qty']
                total_unrealized_pnl += unrealized_pnl
            else:
                logger.warning(f"Could not get LTP for active trade {trade['symbol']} for unrealized PnL calculation.")

        total_pnl = self.total_realized_pnl + total_unrealized_pnl
        
        num_trades = len(self.closed_trades)
        num_wins = sum(1 for trade in self.closed_trades if trade.get('pnl', 0) > 0)
        win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0.0

        # Capital Efficiency: (Total PnL / Initial Capital) * 100
        capital_efficiency = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0.0

        # Leverage Tier (based on current available capital)
        leverage_tier = self.strategy_manager.get_leverage_tier(self.available_capital, self.redis_store.load_settings_from_redis())


        return {
            "available_capital": self.available_capital,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_pnl": total_pnl,
            "active_positions_count": len(self.active_trades),
            "max_active_positions": self.strategy_manager.MAX_ACTIVE_POSITIONS,
            "win_rate": win_rate,
            "capital_efficiency": capital_efficiency,
            "leverage_tier": leverage_tier # Include leverage tier
        }

    def calculate_position_size(self, symbol: str, ltp: float, ai_score: float, leverage_enabled: bool, ai_auto_leverage: bool) -> int:
        """
        Calculates the quantity of shares to trade based on available capital,
        strategy allocation, and leverage settings.
        """
        if ltp <= 0:
            logger.warning(f"Invalid LTP ({ltp}) for {symbol}. Cannot calculate position size.")
            return 0

        # Get current capital allocation percentage from strategy manager (based on capital tiers)
        capital_allocation_pct = self.strategy_manager.get_capital_allocation_pct(self.available_capital)
        
        # Calculate capital to allocate for this trade
        capital_to_allocate = self.available_capital * capital_allocation_pct
        
        # Apply leverage if enabled
        effective_leverage = 1.0
        if leverage_enabled:
            if ai_auto_leverage:
                # AI Auto-Leverage: Amplify based on AI score confidence
                # Example: score 0.75 -> 1x, 0.9 -> 2x, 0.95 -> 3x, 0.99 -> 5x
                # This is a simplified example, adjust as per your AI model's confidence mapping
                if abs(ai_score) >= 0.95:
                    effective_leverage = self.strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER # Max leverage
                elif abs(ai_score) >= 0.90:
                    effective_leverage = self.strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER / 2 # Half leverage
                else:
                    effective_leverage = 1.0 # No leverage for lower confidence
                logger.debug(f"AI Auto-Leverage: AI Score {ai_score:.2f}, Effective Leverage: {effective_leverage:.1f}x")
            else:
                effective_leverage = self.strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER
                logger.debug(f"Manual Leverage: {effective_leverage:.1f}x")
        
        leveraged_capital = capital_to_allocate * effective_leverage
        
        # Calculate quantity
        quantity = int(leveraged_capital / ltp)

        logger.info(f"Calculated position size for {symbol}: Capital to allocate: ₹{capital_to_allocate:,.2f}, "
                    f"Leveraged Capital: ₹{leveraged_capital:,.2f} (x{effective_leverage:.1f}), Quantity: {quantity}")
        
        return quantity

    def enter_trade(self,
                    symbol: str,
                    direction: str,
                    ltp: float,
                    ai_score: float,
                    sentiment: str,
                    tsl_enabled: bool,
                    ai_tsl_enabled: bool,
                    leverage_enabled: bool,
                    ai_auto_leverage: bool):
        """
        Simulates entering a trade.
        """
        if self.is_sync_paused:
            logger.info(f"System sync is PAUSED. Skipping trade entry for {symbol}.")
            self.ai_webhook.send_trade_rejection_feedback(
                symbol=symbol,
                direction=direction,
                reason="System sync paused",
                ai_score=ai_score,
                sentiment=sentiment
            )
            return

        trade_id = str(uuid.uuid4()) # Generate a unique trade ID
        
        # Calculate position size
        quantity = self.calculate_position_size(symbol, ltp, ai_score, leverage_enabled, ai_auto_leverage)

        if quantity <= 0:
            logger.warning(f"Cannot enter {direction} trade for {symbol}: Calculated quantity is zero or negative.")
            self.ai_webhook.send_trade_rejection_feedback(
                symbol=symbol,
                direction=direction,
                reason="Calculated quantity is zero",
                ai_score=ai_score,
                sentiment=sentiment
            )
            return

        # Check if enough capital is available for the trade (even if leveraged, initial capital is blocked)
        required_capital_for_trade = quantity * ltp
        if required_capital_for_trade > self.available_capital:
            logger.warning(f"Cannot enter {direction} trade for {symbol}: Insufficient capital. "
                           f"Required: ₹{required_capital_for_trade:,.2f}, Available: ₹{self.available_capital:,.2f}")
            self.ai_webhook.send_trade_rejection_feedback(
                symbol=symbol,
                direction=direction,
                reason="Insufficient capital",
                ai_score=ai_score,
                sentiment=sentiment
            )
            return

        # Calculate SL and TGT
        sl_tgt = self.strategy_manager.calculate_sl_target(ltp, direction)
        initial_sl = sl_tgt['sl']
        target = sl_tgt['tgt']

        trade_details = {
            "trade_id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "qty": quantity,
            "entry_price": ltp,
            "sl": initial_sl,
            "tgt": target,
            "tsl": None, # Trailing Stop Loss, initially None
            "peak_price": ltp if direction == 'BUY' else None, # For TSL calculation
            "trough_price": ltp if direction == 'SELL' else None, # For TSL calculation
            "ai_score_at_entry": ai_score,
            "sentiment_at_entry": sentiment,
            "timestamp": datetime.now().isoformat(),
            "status": "ACTIVE",
            "leverage_enabled": leverage_enabled, # Store settings for this trade
            "ai_auto_leverage": ai_auto_leverage,
            "tsl_enabled": tsl_enabled,
            "ai_tsl_enabled": ai_tsl_enabled
        }

        self.active_trades[trade_id] = trade_details
        self.available_capital -= required_capital_for_trade # Deduct capital
        self._save_state() # Save updated capital and active trades

        self.redis_store.save_trade_state(trade_id, trade_details) # Save individual trade state
        self.redis_store.set_cooldown_timer(symbol, self.strategy_manager.COOLDOWN_PERIOD_SECONDS) # Set cooldown

        logger.info(f"Entered {direction} trade for {symbol} (Qty: {quantity}) at ₹{ltp:.2f}. "
                    f"SL: ₹{initial_sl:.2f}, TGT: ₹{target:.2f}. Available Capital: ₹{self.available_capital:,.2f}")

        # Send AI feedback for trade entry
        self.ai_webhook.send_entry_suggestion_feedback(
            symbol=symbol,
            direction=direction,
            ltp=ltp,
            ai_score=ai_score,
            sentiment=sentiment,
            current_active_positions=len(self.active_trades),
            max_active_positions=self.strategy_manager.MAX_ACTIVE_POSITIONS
        )


        # --- Live Order Placement (Placeholder) ---
        trade_mode = os.getenv("TRADE_MODE", "paper")
        if trade_mode == "live" and self.dhan_api:
            logger.info(f"Attempting LIVE order placement for {symbol} (Trade ID: {trade_id})...")
            dhan_info = self.SYMBOL_DHAN_MAP.get(symbol)
            if dhan_info:
                order_payload = {
                    "symbol": symbol,
                    "securityId": dhan_info["securityId"],
                    "exchangeSegment": dhan_info["exchangeSegment"],
                    "transactionType": direction, # BUY/SELL
                    "productType": "INTRADAY", # Or "CNC" for delivery
                    "orderType": "MARKET", # Or "LIMIT"
                    "quantity": quantity,
                    # "price": ltp # For LIMIT orders
                }
                response = self.dhan_api.place_order_dry_run(order_payload) # Using dry_run for now
                if response and response.get('status') == 'success':
                    logger.info(f"LIVE Order for {symbol} placed successfully (Dry Run). Order ID: {response.get('simulatedOrderId')}")
                else:
                    logger.error(f"Failed to place LIVE order for {symbol}: {response}")
            else:
                logger.error(f"No Dhan mapping for symbol {symbol}. Cannot place live order.")
        elif trade_mode == "live" and not self.dhan_api:
            logger.warning("Live trading mode enabled, but DhanAPI client is not initialized. Cannot place live orders.")
        # --- End Live Order Placement Placeholder ---


    def exit_trade(self, trade_id: str, exit_price: float, exit_reason: str):
        """
        Simulates exiting an active trade.
        """
        trade = self.active_trades.pop(trade_id, None)
        if trade:
            direction = trade['direction']
            entry_price = trade['entry_price']
            qty = trade['qty']

            pnl = 0.0
            if direction == 'BUY':
                pnl = (exit_price - entry_price) * qty
            elif direction == 'SELL':
                pnl = (entry_price - exit_price) * qty

            self.total_realized_pnl += pnl
            self.available_capital += (entry_price * qty) + pnl # Return blocked capital + PnL
            
            trade["exit_price"] = exit_price
            trade["pnl"] = pnl
            trade["exit_reason"] = exit_reason
            trade["exit_timestamp"] = datetime.now().isoformat()
            trade["status"] = "CLOSED"

            self.closed_trades.append(trade)
            self._save_state() # Save updated capital and PnL
            self.redis_store.remove_active_trade(trade_id) # Remove from active trades in Redis
            self.redis_store.add_closed_trade(trade) # Add to closed trades list in Redis

            logger.info(f"Exited {direction} trade for {trade['symbol']} (ID: {trade_id}) at ₹{exit_price:.2f}. "
                        f"PnL: ₹{pnl:.2f}. Reason: {exit_reason}. Available Capital: ₹{self.available_capital:,.2f}")
            
            # Send AI feedback for trade exit
            self.ai_webhook.send_trade_exit_feedback(
                symbol=trade['symbol'],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                exit_reason=exit_reason
            )

        else:
            logger.warning(f"Attempted to exit non-existent trade with ID: {trade_id}")

    def run_paper_trading_loop(self):
        """
        The main loop for the paper trading system.
        Continuously checks for trade opportunities and manages active trades.
        """
        logger.info("Paper trading loop started.")
        while True:
            # Ensure Redis is connected
            if not self.redis_store.connect():
                logger.critical("Redis connection lost in paper trading loop. Retrying in 10 seconds...")
                time.sleep(10)
                continue

            # Load latest settings from Redis (dashboard can update these)
            settings = self.redis_store.load_settings_from_redis()
            if settings:
                # Update strategy manager's parameters from Redis settings
                self.strategy_manager.MIN_AI_SCORE_BUY = settings.get("min_ai_score_buy", self.strategy_manager.MIN_AI_SCORE_BUY)
                self.strategy_manager.MIN_AI_SCORE_SELL = settings.get("min_ai_score_sell", self.strategy_manager.MIN_AI_SCORE_SELL)
                self.strategy_manager.SL_PERCENT = settings.get("sl_percent", self.strategy_manager.SL_PERCENT)
                self.strategy_manager.TARGET_PERCENT = settings.get("target_percent", self.strategy_manager.TARGET_PERCENT)
                self.strategy_manager.TSL_PERCENT = settings.get("tsl_percent", self.strategy_manager.TSL_PERCENT)
                self.strategy_manager.TSL_ACTIVATION_BUFFER_PERCENT = settings.get("tsl_activation_buffer_percent", self.strategy_manager.TSL_ACTIVATION_BUFFER_PERCENT)
                self.strategy_manager.MAX_ACTIVE_POSITIONS = settings.get("max_active_positions", self.strategy_manager.MAX_ACTIVE_POSITIONS)
                self.strategy_manager.COOLDOWN_PERIOD_SECONDS = settings.get("cooldown_period_seconds", self.strategy_manager.COOLDOWN_PERIOD_SECONDS)
                self.strategy_manager.MARKET_OPEN_TIME = dt_time.fromisoformat(settings["market_open_time"]) if isinstance(settings["market_open_time"], str) else settings["market_open_time"]
                self.strategy_manager.MARKET_CLOSE_TIME = dt_time.fromisoformat(settings["market_close_time"]) if isinstance(settings["market_close_time"], str) else settings["market_close_time"]
                self.strategy_manager.AUTO_EXIT_TIME = dt_time.fromisoformat(settings["auto_exit_time"]) if isinstance(settings["auto_exit_time"], str) else settings["auto_exit_time"]
                self.strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER = settings.get("default_leverage_multiplier", self.strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER)
                
                # Dashboard specific settings (passed to strategy manager for exit logic)
                tsl_enabled_dashboard = settings.get("tsl_enabled", True)
                ai_tsl_enabled_dashboard = settings.get("ai_tsl_enabled", True)
                leverage_enabled_dashboard = settings.get("leverage_enabled", False)
                ai_auto_leverage_dashboard = settings.get("ai_auto_leverage", True)
                self.is_sync_paused = settings.get("is_sync_paused", False) # Update pause status

            current_time = datetime.now()
            
            # Update market status in Redis
            market_status = "OPEN" if self.strategy_manager.MARKET_OPEN_TIME <= current_time.time() < self.strategy_manager.MARKET_CLOSE_TIME else "CLOSED"
            self.redis_store.set_setting("market_status", market_status)

            # 1. Manage Active Trades (Exit Logic)
            trades_to_check = list(self.active_trades.keys()) # Create a copy to allow modification during iteration
            for trade_id in trades_to_check:
                trade = self.active_trades.get(trade_id)
                if not trade:
                    continue # Trade might have been removed by another process

                current_ltp = self.redis_store.read_ltp(trade['symbol'])
                if current_ltp is None:
                    logger.warning(f"Could not get current LTP for {trade['symbol']}. Skipping exit check for this trade.")
                    continue

                # Update TSL if enabled
                if tsl_enabled_dashboard:
                    updated_tsl = self.strategy_manager.update_trailing_sl(
                        trade,
                        current_ltp,
                        ai_tsl_enabled=ai_tsl_enabled_dashboard # Pass AI-TSL flag
                    )
                    if updated_tsl is not None and updated_tsl != trade.get('tsl'):
                        trade['tsl'] = updated_tsl
                        self.redis_store.save_trade_state(trade_id, trade) # Save updated TSL

                # Check exit conditions
                exit_reason = self.strategy_manager.should_exit_trade(
                    trade,
                    current_ltp,
                    current_time,
                    tsl_enabled=tsl_enabled_dashboard, # Pass TSL enabled flag
                    ai_tsl_enabled=ai_tsl_enabled_dashboard # Pass AI-TSL enabled flag
                )

                if exit_reason:
                    self.exit_trade(trade_id, current_ltp, exit_reason)

            # 2. Look for New Trade Opportunities (Entry Logic)
            if not self.is_sync_paused and market_status == "OPEN":
                # Get symbols with recent AI scores from Redis
                # Assuming AI scores are stored as 'score:{symbol}'
                ai_score_keys = self.redis_store.redis_client.keys("score:*")
                symbols_with_scores = [key.decode('utf-8').split(':')[1] for key in ai_score_keys]

                # Filter out symbols that are already active or on cooldown
                eligible_symbols = [
                    s for s in symbols_with_scores
                    if s not in [t['symbol'] for t in self.active_trades.values()] and
                       not self.redis_store.is_on_cooldown(s)
                ]
                
                # Sort symbols by absolute AI score (highest conviction first)
                eligible_symbols.sort(key=lambda s: abs(self.strategy_manager.get_ai_score(s) or 0), reverse=True)

                for symbol in eligible_symbols:
                    if len(self.active_trades) >= self.strategy_manager.MAX_ACTIVE_POSITIONS:
                        logger.info("Max active positions reached. Skipping new entries.")
                        self.ai_webhook.send_missed_opportunity_feedback(
                            symbol=symbol,
                            direction="N/A", # Direction unknown at this point
                            ltp=self.redis_store.read_ltp(symbol) or 0.0,
                            ai_score=self.strategy_manager.get_ai_score(symbol),
                            sentiment=self.strategy_manager.get_sentiment(symbol),
                            missed_reason="Max active positions reached"
                        )
                        break # Stop checking for new trades if max positions are reached

                    ltp = self.redis_store.read_ltp(symbol)
                    ai_score = self.strategy_manager.get_ai_score(symbol)
                    sentiment = self.strategy_manager.get_sentiment(symbol)

                    if ltp is None or ai_score is None:
                        logger.debug(f"Skipping {symbol}: Missing LTP or AI score.")
                        continue

                    direction = None
                    if ai_score >= self.strategy_manager.MIN_AI_SCORE_BUY:
                        direction = 'BUY'
                    elif ai_score <= self.strategy_manager.MIN_AI_SCORE_SELL:
                        direction = 'SELL'

                    if direction:
                        if self.strategy_manager.should_enter_trade(
                            symbol,
                            ltp,
                            direction,
                            len(self.active_trades),
                            tsl_enabled=tsl_enabled_dashboard, # Pass TSL enabled flag
                            ai_tsl_enabled=ai_tsl_enabled_dashboard # Pass AI-TSL enabled flag
                        ):
                            self.enter_trade(
                                symbol,
                                direction,
                                ltp,
                                ai_score,
                                sentiment,
                                tsl_enabled=tsl_enabled_dashboard,
                                ai_tsl_enabled=ai_tsl_enabled_dashboard,
                                leverage_enabled=leverage_enabled_dashboard,
                                ai_auto_leverage=ai_auto_leverage_dashboard
                            )
                        else:
                            # Log why trade was not entered (e.g., sentiment filter, cooldown already logged by strategy)
                            # AIWebhook feedback for rejection is handled within should_enter_trade if borderline.
                            pass # Specific rejection reasons are logged by strategy_manager.should_enter_trade

            # Sleep for a short interval before the next iteration
            time.sleep(1) # Check every 1 second (adjust as needed)

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    print("--- Starting PaperTradeSystem Module Test ---")
    
    # Ensure .env variables are loaded
    load_dotenv()

    # Initialize mock or real dependencies for testing
    angel_api_instance = AngelOneAPI()
    if not angel_api_instance.login():
        print("❌ Failed to log in to Angel One API. Cannot run PaperTradeSystem test.")
        exit()

    redis_store_instance = RedisStore(angel_api=angel_api_instance)
    if not redis_store_instance.connect():
        print("❌ Failed to connect to Redis. Cannot run PaperTradeSystem test.")
        angel_api_instance.logout()
        exit()

    dhan_api_instance = DhanAPI() # Initialize DhanAPI for testing

    llm_client_instance = LLMClient() # Initialize LLMClient
    ai_webhook_instance = AIWebhook(llm_client_instance) # Initialize AIWebhook

    strategy_manager_instance = StrategyManager(redis_store_instance, ai_webhook_instance) # Pass AIWebhook

    paper_trade_system = PaperTradeSystem(
        redis_store_instance,
        strategy_manager_instance,
        dhan_api_instance, # Pass DhanAPI instance
        angel_api_instance, # Pass AngelOneAPI instance
        ai_webhook_instance # NEW: Pass AIWebhook instance
    )

    # Simulate some initial data in Redis for testing
    redis_store_instance.write_ltp("TEST_SYMBOL_BUY", 100.0)
    redis_store_instance.redis_client.set("score:TEST_SYMBOL_BUY", "0.8")
    redis_store_instance.redis_client.set("sentiment:TEST_SYMBOL_BUY", "positive")

    redis_store_instance.write_ltp("TEST_SYMBOL_SELL", 200.0)
    redis_store_instance.redis_client.set("score:TEST_SYMBOL_SELL", "-0.8")
    redis_store_instance.redis_client.set("sentiment:TEST_SYMBOL_SELL", "negative")

    # Set some initial settings in Redis for the dashboard to pick up
    redis_store_instance.set_setting("tsl_enabled", True)
    redis_store_instance.set_setting("ai_tsl_enabled", True)
    redis_store_instance.set_setting("leverage_enabled", True)
    redis_store_instance.set_setting("ai_auto_leverage", True)
    redis_store_instance.set_setting("trade_mode", "paper") # Ensure it's in paper mode for testing

    # Test entry
    print("\n--- Testing Trade Entry ---")
    paper_trade_system.enter_trade(
        "TEST_SYMBOL_BUY", "BUY", 100.0, 0.8, "positive", True, True, True, True
    )
    paper_trade_system.enter_trade(
        "TEST_SYMBOL_SELL", "SELL", 200.0, -0.8, "negative", True, True, True, True
    )

    print("\nActive Trades after entry:")
    print(paper_trade_system.active_trades)

    # Simulate LTP change and check for TSL/TGT/SL
    print("\n--- Simulating LTP Change and Exit Check ---")
    redis_store_instance.write_ltp("TEST_SYMBOL_BUY", 105.0) # Price moves up for BUY
    redis_store_instance.write_ltp("TEST_SYMBOL_SELL", 190.0) # Price moves down for SELL

    # Manually trigger exit check (normally done by the loop)
    for trade_id, trade in list(paper_trade_system.active_trades.items()):
        current_ltp = redis_store_instance.read_ltp(trade['symbol'])
        if current_ltp:
            # Update TSL (normally done by the loop)
            updated_tsl = paper_trade_system.strategy_manager.update_trailing_sl(
                trade, current_ltp, ai_tsl_enabled=True
            )
            if updated_tsl is not None:
                trade['tsl'] = updated_tsl
                redis_store_instance.save_trade_state(trade_id, trade)

            exit_reason = paper_trade_system.strategy_manager.should_exit_trade(
                trade, current_ltp, datetime.now(), tsl_enabled=True, ai_tsl_enabled=True
            )
            if exit_reason:
                paper_trade_system.exit_trade(trade_id, current_ltp, exit_reason)

    print("\nActive Trades after exit check:")
    print(paper_trade_system.active_trades)
    print("\nClosed Trades:")
    print(paper_trade_system.closed_trades)
    print(f"\nAvailable Capital: {paper_trade_system.available_capital}")
    print(f"Total Realized PnL: {paper_trade_system.total_realized_pnl}")

    # Test dashboard metrics
    print("\n--- Dashboard Metrics ---")
    metrics = paper_trade_system.get_dashboard_metrics()
    print(json.dumps(metrics, indent=2))

    # Clean up
    angel_api_instance.logout()
    redis_store_instance.disconnect()
    print("\n--- PaperTradeSystem Module Test End ---")
