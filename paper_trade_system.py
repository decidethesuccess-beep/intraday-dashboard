# paper_trade_system.py
# This module simulates a paper trading environment for the DTS Intraday AI Strategy.
# It manages active and closed trades, calculates PnL, and interacts with Redis
# to persist state and fetch real-time data.

import logging
import uuid
import json
import time
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict  # To manage trade counts per symbol
import os

# Import necessary components
from redis_store import RedisStore
from strategy import StrategyManager
# Removed: from dhan_api_patch import DhanAPI # Removed DhanAPI import
from angelone_api_patch import AngelOneAPI  # For fetching instrument details if needed
from ai_webhook import AIWebhook  # NEW: Import AIWebhook

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_log.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ])
logger = logging.getLogger(__name__)


class PaperTradeSystem:
    """
    Simulates a paper trading environment for the DTS Intraday AI Strategy.
    Manages trades, calculates PnL, and persists state to Redis.
    """

    # Removed dhan_api from constructor as it's no longer used
    def __init__(self, redis_store: RedisStore,
                 strategy_manager: StrategyManager, angel_api: AngelOneAPI):
        """
        Initializes the PaperTradeSystem.

        Args:
            redis_store (RedisStore): An instance of RedisStore for data persistence.
            strategy_manager (StrategyManager): An instance of StrategyManager for trade logic.
            angel_api (AngelOneAPI): An instance of AngelOneAPI for instrument lookup.
        """
        self.redis_store = redis_store
        self.strategy_manager = strategy_manager
        # Removed: self.dhan_api = dhan_api
        self.angel_api = angel_api  # Store AngelOneAPI instance

        # NEW: Initialize AIWebhook
        # Assuming LLMClient is accessible or can be passed here if needed,
        # but for simplicity, AIWebhook will get its own LLMClient instance.
        # In a more complex setup, LLMClient might be passed from main.py
        from llm_client import LLMClient
        self.ai_webhook = AIWebhook(LLMClient())

        self.initial_capital = float(os.getenv("INITIAL_CAPITAL", 100000.0))
        self.trade_mode = os.getenv("TRADE_MODE", "paper")  # 'paper' or 'live'

        self.available_capital: float = self.initial_capital
        self.total_realized_pnl: float = 0.0
        self.active_trades: Dict[str,
                                 Dict[str,
                                      Any]] = {}  # {trade_id: trade_details}
        self.closed_trades: List[Dict[str, Any]] = []
        self.is_sync_paused: bool = False  # Flag to pause/resume system sync from dashboard

        self.max_active_positions = self.strategy_manager.MAX_ACTIVE_POSITIONS  # From StrategyManager

        self._load_state()  # Load previous state from Redis on startup

        # Dictionary to track active trade count per symbol for cooldown management
        self.active_trade_count_per_symbol = defaultdict(int)
        for trade in self.active_trades.values():
            self.active_trade_count_per_symbol[trade['symbol']] += 1

        logger.info(
            f"PaperTradeSystem initialized with Initial Capital: {self.initial_capital}, Max Positions: {self.max_active_positions}. Trade Mode: {self.trade_mode}."
        )

    def _load_state(self):
        """Loads the trading system's state from Redis."""
        logger.info("Loading trading system state from Redis.")

        # Load active trades
        self.active_trades = self.redis_store.get_all_active_trades()
        logger.info(
            f"Loaded {len(self.active_trades)} active trades from Redis.")

        # Load closed trades
        self.closed_trades = self.redis_store.get_all_closed_trades()
        logger.info(
            f"Loaded {len(self.closed_trades)} closed trades from Redis.")

        # Load overall system state
        system_state = self.redis_store.load_system_state()
        if system_state:
            self.available_capital = system_state.get("available_capital",
                                                      self.initial_capital)
            self.total_realized_pnl = system_state.get("total_pnl", 0.0)
            self.is_sync_paused = system_state.get("sync_paused", False)
            logger.info(
                f"Loaded state: Available Capital={self.available_capital:.2f}, Total PnL={self.total_realized_pnl:.2f}, Sync Paused={self.is_sync_paused}"
            )
        else:
            logger.info(
                "No previous system state found in Redis. Starting with initial capital."
            )
            self.redis_store.save_system_state(
                self.available_capital, self.total_realized_pnl,
                self.is_sync_paused)  # Save initial state

    def _save_state(self):
        """Saves the current trading system's state to Redis."""
        self.redis_store.save_system_state(self.available_capital,
                                           self.total_realized_pnl,
                                           self.is_sync_paused)
        # Individual active trades are saved/updated/removed by their respective methods.
        # Closed trades are appended to a list.

    def _calculate_unrealized_pnl(self, trade: Dict[str, Any],
                                  current_ltp: float) -> float:
        """Calculates the unrealized PnL for a single trade."""
        if trade['direction'] == 'BUY':
            return (current_ltp - trade['entry_price']) * trade['qty']
        elif trade['direction'] == 'SELL':
            return (trade['entry_price'] -
                    current_ltp) * trade['qty']  # Corrected: use current_ltp
        return 0.0

    def _get_angel_one_instrument_details(
            self, symbol: str) -> Optional[Dict[str, str]]:
        """
        Fetches Angel One instrument details (token, exchangeType) for a given symbol.
        This is a placeholder. In a real system, you'd use a comprehensive instrument master.
        """
        # This mapping should ideally come from a loaded instrument master file (e.g., from fetch_top_volume_gainers)
        # For now, a small hardcoded map for the test symbols.
        ANGEL_ONE_INSTRUMENTS = {
            "NIFTY_50": {
                "exchangeType": "NSE",
                "token": "999260000"
            },  # Corrected token
            "RELIANCE": {
                "exchangeType": "NSE",
                "token": "2885"
            },  # Corrected token
            "TCS": {
                "exchangeType": "NSE",
                "token": "3045"
            },
            "HDFC_BANK": {
                "exchangeType": "NSE",
                "token": "3432"
            },
            "SBIN": {
                "exchangeType": "NSE",
                "token": "1333"
            },
            "ICICI_BANK": {
                "exchangeType": "NSE",
                "token": "1660"
            },  # Corrected token
            "INFY": {
                "exchangeType": "NSE",
                "token": "10604"
            },
            "ITC": {
                "exchangeType": "NSE",
                "token": "10606"
            },
            "MARUTI": {
                "exchangeType": "NSE",
                "token": "20374"
            },
            "AXISBANK": {
                "exchangeType": "NSE",
                "token": "1348"
            },
        }
        return ANGEL_ONE_INSTRUMENTS.get(symbol)

    def enter_trade(
        self,
        symbol: str,
        direction: str,
        ltp: float,
        ai_score: float,
        sentiment: str,
        tsl_enabled: bool = True,  # From dashboard settings
        ai_tsl_enabled: bool = True,  # From dashboard settings
        leverage_enabled: bool = False,  # From dashboard settings
        ai_auto_leverage: bool = False  # From dashboard settings
    ):
        """
        Attempts to enter a new trade based on strategy signals.
        """
        current_time = datetime.now()

        # Check if symbol is on cooldown
        if self.redis_store.is_on_cooldown(symbol):
            logger.info(
                f"ENTRY BLOCKED: {symbol} is on cooldown. Skipping trade entry."
            )
            # NEW: Send AI feedback for rejection
            self.ai_webhook.send_trade_rejection_feedback(
                symbol=symbol,
                direction=direction,
                reason="Symbol on cooldown",
                ai_score=ai_score,
                sentiment=sentiment)
            return

        # Check if max active positions reached
        if len(self.active_trades) >= self.max_active_positions:
            logger.warning(
                f"ENTRY BLOCKED: Max active positions ({self.max_active_positions}) reached. Cannot enter {symbol}."
            )
            # NEW: Send AI feedback for missed opportunity
            self.ai_webhook.send_missed_opportunity_feedback(
                symbol=symbol,
                direction=direction,
                ltp=ltp,
                ai_score=ai_score,
                sentiment=sentiment,
                missed_reason="Max active positions reached")
            return

        # Check strategy entry conditions
        # The strategy manager now handles AI score and sentiment checks internally
        if not self.strategy_manager.should_enter_trade(
                symbol, ltp, direction, len(self.active_trades), tsl_enabled,
                ai_tsl_enabled):
            logger.info(
                f"ENTRY BLOCKED: Strategy rules prevent {direction} for {symbol} at {ltp:.2f}."
            )
            # NEW: Send AI feedback for rejection based on strategy rules
            # We need to get the specific reason from strategy_manager if possible
            # For now, a generic "Strategy rules" reason.
            self.ai_webhook.send_trade_rejection_feedback(
                symbol=symbol,
                direction=direction,
                reason="Strategy rules not met (e.g., AI score, sentiment)",
                ai_score=ai_score,
                sentiment=sentiment)
            return

        # Calculate quantity based on capital allocation and leverage
        capital_allocation_pct = self.strategy_manager.get_capital_allocation_pct(
            self.available_capital)
        allocated_capital_for_trade = self.available_capital * capital_allocation_pct

        # Calculate leverage based on settings and AI score
        effective_leverage = 1.0
        if leverage_enabled:
            effective_leverage = self.strategy_manager.DEFAULT_LEVERAGE_MULTIPLIER

        if ai_auto_leverage and ai_score is not None:
            # Simple AI auto-leverage logic (can be refined)
            ai_multiplier = 1.0
            if direction == 'BUY' and ai_score > self.strategy_manager.MIN_AI_SCORE_BUY:
                ai_multiplier = 1.0 + (
                    ai_score - self.strategy_manager.MIN_AI_SCORE_BUY) * 2
            elif direction == 'SELL' and ai_score < self.strategy_manager.MIN_AI_SCORE_SELL:
                ai_multiplier = 1.0 + (
                    self.strategy_manager.MIN_AI_SCORE_SELL - ai_score) * 2
            ai_multiplier = min(ai_multiplier, 2.0)  # Cap AI multiplier
            effective_leverage *= ai_multiplier

        effective_leverage = min(effective_leverage, 10.0)  # Overall cap

        quantity = int(leveraged_capital / ltp) if ltp > 0 else 0

        if quantity == 0:
            logger.warning(
                f"ENTRY BLOCKED: Calculated quantity for {symbol} is 0. Not enough capital or invalid LTP."
            )
            # NEW: Send AI feedback for rejection due to quantity
            self.ai_webhook.send_trade_rejection_feedback(
                symbol=symbol,
                direction=direction,
                reason="Calculated quantity is zero",
                ai_score=ai_score,
                sentiment=sentiment)
            return

        # Calculate initial SL and TGT
        sl_tgt = self.strategy_manager.calculate_sl_target(ltp, direction)

        trade_id = f"PAPER_{uuid.uuid4().hex[:8]}"  # Unique ID for paper trades

        new_trade = {
            'trade_id': trade_id,
            'symbol': symbol,
            'entry_price': ltp,
            'qty': quantity,
            'direction': direction,
            'timestamp': current_time.isoformat(),
            'sl': sl_tgt['sl'],
            'tgt': sl_tgt['tgt'],
            'tsl': None,  # TSL starts as None
            'peak_price': ltp if direction == 'BUY' else None,  # For BUY TSL
            'trough_price':
            ltp if direction == 'SELL' else None,  # For SELL TSL
            'status': 'ACTIVE',
            'ai_score_at_entry': ai_score,
            'sentiment_at_entry': sentiment,
            'capital_allocated': allocated_capital_for_trade,
            'leverage_applied': effective_leverage,
            'unrealized_pnl': 0.0  # Initial unrealized PnL
        }

        # --- SIMULATE ORDER PLACEMENT (REMOVED DHAN DRY RUN) ---
        # In a paper trading system, we just log the "order" and add it to active trades.
        # No external API call for "dry run" is strictly necessary.
        logger.info(
            f"SIMULATING ORDER: {direction} {quantity} x {symbol} @ {ltp:.2f} (ID: {trade_id[-8:]})"
        )

        self.active_trades[trade_id] = new_trade
        self.redis_store.save_trade_state(trade_id,
                                          new_trade)  # Persist to Redis
        self.active_trade_count_per_symbol[symbol] += 1

        logger.info(
            f"TRADE ENTERED: {direction} {quantity} x {symbol} @ {ltp:.2f} (ID: {trade_id[-8:]}) | "
            f"SL: {new_trade['sl']:.2f}, TGT: {new_trade['tgt']:.2f} | "
            f"Capital Alloc: {allocated_capital_for_trade:.2f}, Leverage: {effective_leverage:.2f}x"
        )
        self._save_state()  # Save overall system state after trade

        # NEW: Send AI feedback for successful entry suggestion
        self.ai_webhook.send_entry_suggestion_feedback(
            symbol=symbol,
            direction=direction,
            ltp=ltp,
            ai_score=ai_score,
            sentiment=sentiment,
            current_active_positions=len(self.active_trades),
            max_active_positions=self.max_active_positions)
        # --- END SIMULATED ORDER PLACEMENT ---

    def exit_trade(self, trade_id: str, current_ltp: float, exit_reason: str):
        """
        Exits an active trade and calculates realized PnL.
        """
        trade = self.active_trades.get(trade_id)
        if not trade:
            logger.warning(
                f"Attempted to exit non-existent trade ID: {trade_id}")
            return

        entry_price = trade['entry_price']
        quantity = trade['qty']
        direction = trade['direction']
        ai_score_at_entry = trade.get('ai_score_at_entry')
        sentiment_at_entry = trade.get('sentiment_at_entry')

        pnl = 0.0
        if direction == 'BUY':
            pnl = (current_ltp - entry_price) * quantity
        elif direction == 'SELL':
            pnl = (entry_price - current_ltp) * quantity

        # Update capital and total PnL
        self.available_capital += pnl
        self.total_realized_pnl += pnl

        # Mark trade as closed
        trade['exit_price'] = current_ltp
        trade['exit_reason'] = exit_reason
        trade['exit_timestamp'] = datetime.now().isoformat()
        trade['pnl'] = pnl
        trade['status'] = 'CLOSED'

        self.closed_trades.append(trade)
        self.redis_store.add_closed_trade(trade)  # Persist closed trade

        self.active_trades.pop(trade_id)  # Remove from active trades
        self.redis_store.remove_active_trade(
            trade_id)  # Remove from Redis active trades
        self.active_trade_count_per_symbol[trade['symbol']] -= 1
        if self.active_trade_count_per_symbol[trade['symbol']] == 0:
            del self.active_trade_count_per_symbol[trade[
                'symbol']]  # Clean up if no more active trades for symbol

        # Set cooldown for the exited symbol
        self.redis_store.set_cooldown_timer(
            trade['symbol'], self.strategy_manager.COOLDOWN_PERIOD_SECONDS)

        logger.info(
            f"TRADE EXITED: {direction} {quantity} x {trade['symbol']} @ {current_ltp:.2f} (ID: {trade_id[-8:]}) | "
            f"Reason: {exit_reason} | PnL: {pnl:.2f} | Current Capital: {self.available_capital:.2f}"
        )
        self._save_state()  # Save overall system state after trade

        # NEW: Send AI feedback for trade exit
        self.ai_webhook.send_trade_exit_feedback(symbol=trade['symbol'],
                                                 direction=direction,
                                                 entry_price=entry_price,
                                                 exit_price=current_ltp,
                                                 pnl=pnl,
                                                 exit_reason=exit_reason)

    def run_paper_trading_loop(self):
        """
        The main loop for the paper trading system.
        It continuously monitors active trades and checks for exit conditions.
        """
        logger.info("Starting paper trading loop...")
        while True:
            logger.info("\n--- Paper Trading Loop Iteration ---")
            current_time = datetime.now()

            # Dashboard settings for TSL and Leverage (fetched from Redis)
            tsl_enabled = self.redis_store.get_setting("tsl_enabled", True)
            ai_tsl_enabled = self.redis_store.get_setting(
                "ai_tsl_enabled", True)
            leverage_enabled = self.redis_store.get_setting(
                "leverage_enabled", False)
            ai_auto_leverage = self.redis_store.get_setting(
                "ai_auto_leverage", True)

            trades_to_exit_in_this_iteration = []

            for trade_id, trade in list(self.active_trades.items(
            )):  # Iterate over a copy to allow modification
                symbol = trade['symbol']

                # Fetch current LTP from Redis (or fallback to REST API)
                current_ltp = self.redis_store.read_ltp(symbol)

                if current_ltp is None:
                    logger.warning(
                        f"Could not get LTP for active trade {symbol}. Skipping exit check for this trade."
                    )
                    # Log current values for debugging
                    logger.debug(
                        f"Trade {trade_id[-8:]} - Symbol: {symbol}, Entry: {trade['entry_price']:.2f}, SL: {trade['sl']:.2f}, TGT: {trade['tgt']:.2f}, TSL: {trade['tsl'] if trade['tsl'] is not None else 'N/A'}"
                    )
                    continue  # Skip this trade if no LTP is available

                # Update TSL for the trade (if TSL is enabled)
                if tsl_enabled:
                    updated_tsl = self.strategy_manager.update_trailing_sl(
                        trade, current_ltp, ai_tsl_enabled=ai_tsl_enabled)
                    if updated_tsl is not None:
                        trade['tsl'] = updated_tsl
                        self.redis_store.save_trade_state(
                            trade_id, trade)  # Persist updated TSL

                # Calculate unrealized PnL for logging
                unrealized_pnl = self._calculate_unrealized_pnl(
                    trade, current_ltp)
                trade[
                    'unrealized_pnl'] = unrealized_pnl  # Update in trade object

                # Check for exit conditions
                exit_reason = self.strategy_manager.should_exit_trade(
                    trade, current_ltp, current_time, tsl_enabled,
                    ai_tsl_enabled)

                # --- Enhanced Logging for Active Trades ---
                logger.info(
                    f"Active Trade: ID: {trade_id[-8:]}, Symbol: {symbol}, Dir: {trade['direction']}, "
                    f"Entry: {trade['entry_price']:.2f}, Qty: {trade['qty']}, Current LTP: {current_ltp:.2f}, "
                    f"SL: {trade['sl']:.2f}, TGT: {trade['tgt']:.2f}, TSL: {trade['tsl'] if trade['tsl'] is not None else 'N/A'}, "
                    f"Unrealized PnL: {unrealized_pnl:.2f}")
                # --- End Enhanced Logging ---

                if exit_reason:
                    trades_to_exit_in_this_iteration.append(
                        (trade_id, current_ltp, exit_reason))

            # Execute exits for identified trades
            for trade_id, ltp, reason in trades_to_exit_in_this_iteration:
                self.exit_trade(trade_id, ltp, reason)

            # Log current active positions and capital
            self._log_current_status()

            # Sleep for a short interval before the next iteration
            time.sleep(1)  # Check every 1 second for active trades

    def _log_current_status(self):
        """Logs the current state of active positions and capital."""
        logger.info(f"Current Active Positions: {len(self.active_trades)}")
        for trade_id, trade in self.active_trades.items():
            # Recalculate unrealized PnL just before logging for accuracy
            current_ltp = self.redis_store.read_ltp(trade['symbol'])
            unrealized_pnl = self._calculate_unrealized_pnl(
                trade, current_ltp) if current_ltp is not None else 0.0

            logger.info(
                f"  ID: {trade_id[-8:]}, Symbol: {trade['symbol']}, Dir: {trade['direction']}, "
                f"Entry: {trade['entry_price']:.2f}, Qty: {trade['qty']}, "
                f"SL: {trade['sl']:.2f}, TGT: {trade['tgt']:.2f}, "
                f"TSL: {trade['tsl'] if trade['tsl'] is not None else 'N/A'}, Unrealized PnL: {unrealized_pnl:.2f}"
            )

        logger.info(
            f"Available Capital: {self.available_capital:.2f}, Total Realized PnL: {self.total_realized_pnl:.2f}"
        )

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Gathers key metrics for the dashboard display.
        """
        total_unrealized_pnl = 0.0
        for trade_id, trade in self.active_trades.items():
            current_ltp = self.redis_store.read_ltp(trade['symbol'])
            if current_ltp is not None:
                total_unrealized_pnl += self._calculate_unrealized_pnl(
                    trade, current_ltp)

        total_pnl = self.total_realized_pnl + total_unrealized_pnl

        num_trades = len(self.closed_trades)
        num_wins = sum(1 for trade in self.closed_trades if trade['pnl'] > 0)
        win_rate = (num_wins / num_trades * 100) if num_trades > 0 else 0.0

        # Calculate Capital Efficiency (Total Profit / Total Capital Used)
        # This requires tracking "capital used" which is complex in paper trading with leverage.
        # For simplicity, let's use initial capital for now.
        capital_efficiency = (self.total_realized_pnl / self.initial_capital *
                              100) if self.initial_capital > 0 else 0.0

        # Leverage Tier (Placeholder - needs actual implementation based on symbol category)
        # For now, just return a default or based on a simple rule.
        leverage_tier = "N/A"  # This needs to be determined based on the actual trade's symbol category (Large/Mid/Small Cap)

        return {
            "available_capital": round(self.available_capital, 2),
            "total_realized_pnl": round(self.total_realized_pnl, 2),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "active_positions_count": len(self.active_trades),
            "max_active_positions": self.max_active_positions,
            "win_rate": round(win_rate, 2),
            "capital_efficiency": round(capital_efficiency, 2),
            "leverage_tier": leverage_tier,  # Placeholder
            "is_sync_paused": self.is_sync_paused
        }

    def toggle_sync(self):
        """Toggles the system sync pause status."""
        self.is_sync_paused = not self.is_sync_paused
        self.redis_store.save_system_state(self.available_capital,
                                           self.total_realized_pnl,
                                           self.is_sync_paused)
        logger.info(
            f"System sync paused status toggled to: {self.is_sync_paused}")

    def __del__(self):
        """Ensures state is saved on object destruction."""
        logger.info("PaperTradeSystem shutting down. Saving final state...")
        # Check if redis_store attribute exists and is not None before using it
        if hasattr(self, 'redis_store') and self.redis_store is not None:
            self._save_state()
        else:
            logger.warning(
                "redis_store not available during PaperTradeSystem.__del__ for saving state."
            )
        logger.info("PaperTradeSystem shutdown complete.")
