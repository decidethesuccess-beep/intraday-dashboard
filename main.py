# main.py
# This script is the central entry point for the DTS Intraday AI Trading System.
# It initializes and runs the core components (LiveStreamManager, SentimentAnalyzer,
# PaperTradeSystem) in separate background threads to ensure continuous operation
# for cloud deployment on Replit, connecting to an external Upstash Redis.

import os
import logging
import threading
import time
import json
from dotenv import load_dotenv

# Import keep_alive function
from keep_alive import keep_alive  # NEW: Import keep_alive

# Import core components
from angelone_api_patch import AngelOneAPI
from redis_store import RedisStore
from sentiment_analyzer import SentimentAnalyzer
from live_stream import LiveStreamManager
from paper_trade_system import PaperTradeSystem
from strategy import StrategyManager
from llm_client import LLMClient
from ai_webhook import AIWebhook

# Configure logging for the main application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot_log.log"),
              logging.StreamHandler()])
logger = logging.getLogger(__name__)


def load_subscribed_symbols(
        file_path: str = "subscribed_symbols.json") -> list:
    """Loads the list of symbols to subscribe to from a JSON file."""
    if not os.path.exists(file_path):
        logger.error(
            f"Symbol subscription file not found: {file_path}. Please create it."
        )
        return []
    try:
        with open(file_path, 'r') as f:
            symbols = json.load(f)
            logger.info(f"Loaded {len(symbols)} symbols from {file_path}.")
            return symbols
    except json.JSONDecodeError as e:
        logger.critical(
            f"Error decoding JSON from {file_path}: {e}. Ensure it's valid JSON."
        )
        return []
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred loading symbols from {file_path}: {e}"
        )
        return []


def main():
    """
    Initializes and starts all core trading system components in background threads.
    """
    load_dotenv()  # Load environment variables at the very beginning

    logger.info("Starting DTS Intraday AI Trading System...")

    # --- Start Keep-Alive Server ---
    keep_alive()  # NEW: Call keep_alive to start the Flask server
    logger.info("Keep-alive server initiated.")

    # --- Initialize Core Services ---
    # 1. AngelOneAPI (for historical data and potentially REST fallbacks in Redis)
    angel_api = AngelOneAPI(
        api_key_to_use=os.getenv("ANGELONE_HISTORICAL_API_KEY"))
    if not angel_api.login():
        logger.critical(
            f"Failed to log in to Angel One API. Exiting. Details: {angel_api.login_error_message}"
        )
        return

    # 2. RedisStore (will now connect to Upstash Redis using env vars)
    redis_store = RedisStore(
        angel_api=angel_api)  # Pass angel_api for RedisStore's fallback
    if not redis_store.connect():
        logger.critical("Failed to connect to Redis. Exiting.")
        angel_api.logout()
        return

    # 3. LLMClient (for sentiment analysis)
    llm_client = LLMClient()

    # 4. AIWebhook (for strategy manager)
    ai_webhook = AIWebhook(llm_client)

    # 5. SentimentAnalyzer
    sentiment_analyzer = SentimentAnalyzer(llm_client, redis_store)

    # 6. StrategyManager
    strategy_manager = StrategyManager(redis_store, ai_webhook)

    # 7. PaperTradeSystem
    # Pass angel_api to PaperTradeSystem as it now requires it
    paper_trade_system = PaperTradeSystem(redis_store, strategy_manager,
                                          angel_api)

    # 8. LiveStreamManager
    subscribed_symbols = load_subscribed_symbols()
    if not subscribed_symbols:
        logger.critical(
            "No symbols loaded for live streaming. Please populate subscribed_symbols.json."
        )
        angel_api.logout()
        redis_store.disconnect()
        return

    live_stream_manager = LiveStreamManager(angel_api, redis_store)
    live_stream_manager.subscribed_symbols_config = subscribed_symbols

    logger.info("All core services initialized successfully.")

    # --- Start Background Threads ---
    live_stream_thread = threading.Thread(
        target=live_stream_manager.run_live_stream,
        daemon=True,
        name='live_stream_thread')
    live_stream_thread.start()
    logger.info("Live stream manager started in a separate thread.")

    sentiment_analyzer_thread = threading.Thread(
        target=sentiment_analyzer.run_sentiment_analysis_loop,
        daemon=True,
        name='sentiment_analyzer_thread')
    sentiment_analyzer_thread.start()
    logger.info("Sentiment analyzer started in a separate thread.")

    paper_trade_system_thread = threading.Thread(
        target=paper_trade_system.run_paper_trading_loop,
        daemon=True,
        name='paper_trade_system_thread')
    paper_trade_system_thread.start()
    logger.info("Paper trading system started in a separate thread.")

    logger.info("Main application running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Main application interrupted. Shutting down...")
    finally:
        # Perform cleanup
        if live_stream_manager.websocket:
            live_stream_manager.websocket.close_websocket()
        angel_api.logout()
        redis_store.disconnect()
        logger.info("DTS Intraday AI Trading System stopped.")


if __name__ == "__main__":
    main()
