# live_stream.py
# This module handles real-time market data streaming from Angel One's SmartWebSocketV2
# and updates the Redis store with LTP, Volume, and Timestamp.

import os
import logging
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# Import necessary modules
from angelone_api_patch import AngelOneAPI # For Angel One login and SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2 # The actual WebSocket client
from redis_store import RedisStore # For interacting with Redis

# Configure logging for the module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

class LiveStreamManager:
    """
    Manages the live market data streaming from Angel One and updates Redis.
    """
    def __init__(self, angel_api: AngelOneAPI, redis_store: RedisStore):
        """
        Initializes the LiveStreamManager.

        Args:
            angel_api (AngelOneAPI): An instance of AngelOneAPI for login.
            redis_store (RedisStore): An instance of RedisStore for data storage.
        """
        load_dotenv() # Load environment variables

        self.angel_api = angel_api
        self.redis_store = redis_store
        self.websocket = None
        self.is_connected = False
        self.subscribed_symbols_config: List[Dict[str, Any]] = []
        self.token_to_symbol_map: Dict[str, str] = {}
        self.last_subscription_config_hash: Optional[str] = None # To detect changes in symbol list

        self.SYMBOLS_CONFIG_FILE = "subscribed_symbols.json" # New: File to load symbols from

        # --- Load subscribed symbols from JSON file, then .env, then default ---
        self._load_subscribed_symbols()

        self.RECONNECT_ATTEMPTS = 5
        self.RECONNECT_DELAY_SECONDS = 10
        self.SUBSCRIPTION_CHECK_INTERVAL = 60 # Check for new symbols every 60 seconds

        logger.info("LiveStreamManager initialized.")

    def _set_default_subscribed_symbols(self):
        """Sets a default list of subscribed symbols if not configured via .env or JSON."""
        self.subscribed_symbols_config = [
            {"exchangeType": 1, "tokens": ["26009"], "symbol": "NIFTY_50"},
            {"exchangeType": 1, "tokens": ["11536"], "symbol": "RELIANCE"},
            {"exchangeType": 1, "tokens": ["3045"], "symbol": "TCS"},
            {"exchangeType": 1, "tokens": ["3432"], "symbol": "HDFC_BANK"},
            {"exchangeType": 1, "tokens": ["1333"], "symbol": "SBIN"},
        ]
        logger.info("Using default hardcoded symbols for live stream.")

    def _load_symbols_from_json_file(self) -> bool:
        """
        Attempts to load subscribed symbols from a JSON file.
        Returns True if successful, False otherwise.
        """
        if os.path.exists(self.SYMBOLS_CONFIG_FILE):
            try:
                with open(self.SYMBOLS_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.subscribed_symbols_config = data
                        logger.info(f"Loaded {len(self.subscribed_symbols_config)} symbols from {self.SYMBOLS_CONFIG_FILE}.")
                        return True
                    else:
                        logger.error(f"Invalid format in {self.SYMBOLS_CONFIG_FILE}: Expected a list.")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {self.SYMBOLS_CONFIG_FILE}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error reading {self.SYMBOLS_CONFIG_FILE}: {e}", exc_info=True)
        return False

    def _load_subscribed_symbols(self):
        """
        Loads subscribed symbols, prioritizing JSON file, then .env, then defaults.
        Also updates the token_to_symbol_map and last_subscription_config_hash.
        """
        if self._load_symbols_from_json_file():
            pass # Successfully loaded from JSON file
        else:
            subscribed_symbols_json_str = os.getenv("SUBSCRIBED_SYMBOLS_JSON")
            if subscribed_symbols_json_str:
                try:
                    self.subscribed_symbols_config = json.loads(subscribed_symbols_json_str)
                    logger.info(f"Loaded {len(self.subscribed_symbols_config)} symbols from SUBSCRIBED_SYMBOLS_JSON (from .env).")
                except json.JSONDecodeError as e:
                    logger.critical(f"Error decoding SUBSCRIBED_SYMBOLS_JSON from .env: {e}. Using default symbols.", exc_info=True)
                    self._set_default_subscribed_symbols()
            else:
                logger.warning("SUBSCRIBED_SYMBOLS_JSON not found in .env. Using default symbols.")
                self._set_default_subscribed_symbols()

        # Build token to symbol map dynamically
        self.token_to_symbol_map = {}
        for config in self.subscribed_symbols_config:
            for token_str in config.get('tokens', []):
                symbol_name = config.get('symbol', f"TOKEN_{token_str}")
                self.token_to_symbol_map[token_str] = symbol_name
        
        # Update hash to detect changes
        self.last_subscription_config_hash = self._generate_config_hash(self.subscribed_symbols_config)

    def _generate_config_hash(self, config_list: List[Dict[str, Any]]) -> str:
        """Generates a hash of the current subscription configuration for change detection."""
        # Sort the list of dictionaries and then dump to JSON to ensure consistent hashing
        sorted_config = sorted(config_list, key=lambda x: (x.get('exchangeType', ''), tuple(x.get('tokens', [])), x.get('symbol', '')))
        return str(hash(json.dumps(sorted_config, sort_keys=True)))


    def _on_open(self, ws):
        """Callback when WebSocket connection is opened."""
        logger.info("WebSocket connection opened. Subscribing to symbols...")
        self.is_connected = True
        self._subscribe_to_symbols() # Call the subscription method

    def _subscribe_to_symbols(self):
        """Sends subscription requests to the WebSocket based on current config."""
        token_list_for_subscribe = []
        for config in self.subscribed_symbols_config:
            token_list_for_subscribe.append({
                "exchangeType": config["exchangeType"],
                "tokens": config["tokens"]
            })
        
        try:
            self.websocket.subscribe(
                correlation_id=f"subscribe_{datetime.now().timestamp()}",
                mode=1, # Mode 1 for LTP only
                token_list=token_list_for_subscribe
            )
            logger.info(f"Subscribed to {len(self.subscribed_symbols_config)} symbol groups for LTP updates.")
        except Exception as e:
            logger.error(f"Error during WebSocket subscription: {e}", exc_info=True)

    def _on_message(self, ws, message):
        """Callback when a message (tick data) is received from WebSocket."""
        try:
            # Enhanced Logging: Log the raw incoming message
            logger.debug(f"Received raw WebSocket message: {message}")
            tick_data = json.loads(message)

            # Angel One ticks usually come as a list of dictionaries, even for single tick
            if isinstance(tick_data, list):
                for tick in tick_data:
                    exchange = tick.get('e')
                    token = tick.get('tk') # Token is a string from the API
                    ltp = tick.get('ltp')
                    volume = tick.get('v') # Volume is total traded volume for the day
                    ltt = tick.get('ltt') # Last Traded Time

                    if token and ltp is not None:
                        # Enhanced Logging: Log parsed tick data
                        logger.debug(f"Parsed tick: Token='{token}', LTP={ltp}, Volume={volume}")

                        symbol = self.token_to_symbol_map.get(token, f"UNKNOWN_TOKEN_{token}")
                        
                        # Convert LTP and Volume to appropriate types
                        try:
                            ltp_float = float(ltp)
                            volume_int = int(float(volume)) if volume is not None else 0 # Ensure volume is float before int conversion
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Failed to parse LTP/Volume for token '{token}': {e}. Skipping update.", exc_info=True)
                            continue

                        # Update Redis
                        self.redis_store.write_ltp(symbol, ltp_float)
                        self.redis_store.redis_client.set(f"VOLUME:{symbol}", volume_int) # Direct Redis client usage
                        self.redis_store.redis_client.set(f"LAST_UPDATE:{symbol}", datetime.now().isoformat()) # Store current timestamp

                        logger.debug(f"Redis updated for {symbol}: LTP={ltp_float}, Volume={volume_int}, Time={datetime.now().isoformat()}")
                    else:
                        logger.warning(f"Incomplete tick data received: {tick}")
            else:
                logger.warning(f"Unexpected message format from WebSocket: {message}")

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding WebSocket message JSON: {e}. Message: {message}", exc_info=True)
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}. Message: {message}", exc_info=True)

    def _on_close(self, ws, *args):
        """Callback when WebSocket connection is closed."""
        logger.warning("WebSocket connection closed. Attempting to reconnect...")
        self.is_connected = False
        # Reconnect logic will be handled by the run_live_stream loop

    def _on_error(self, ws, error):
        """Callback when a WebSocket error occurs."""
        logger.error(f"WebSocket error: {error}", exc_info=True)
        self.is_connected = False
        # Reconnect logic will be handled by the run_live_stream loop

    def run_live_stream(self):
        """
        Starts the live streaming process, including Angel One login and WebSocket connection.
        Includes reconnect logic and periodic symbol list updates.
        """
        logger.info("Starting live stream manager...")

        # Ensure Redis is connected
        if not self.redis_store.connect():
            logger.critical("Failed to connect to Redis. Cannot start live stream.")
            return

        # Attempt Angel One login
        if not self.angel_api.login():
            logger.critical("Failed to log in to Angel One. Cannot start live stream.")
            self.redis_store.disconnect()
            return

        # Get necessary tokens for WebSocket
        feed_token = self.angel_api.get_feed_token()
        client_code = self.angel_api.client_code
        api_key = self.angel_api.api_key
        auth_token = self.angel_api.get_jwt_token() # Correctly retrieve JWT token

        if not all([feed_token, client_code, api_key, auth_token]):
            logger.critical("Missing required tokens for WebSocket connection. Aborting live stream.")
            self.angel_api.logout()
            self.redis_store.disconnect()
            return

        reconnect_attempts = 0
        last_subscription_check_time = time.time()
        while reconnect_attempts < self.RECONNECT_ATTEMPTS:
            try:
                logger.info(f"Attempting to connect to SmartWebSocketV2 (Attempt {reconnect_attempts + 1}/{self.RECONNECT_ATTEMPTS})...")
                
                # Initialize SmartWebSocketV2
                self.websocket = SmartWebSocketV2(
                    auth_token=auth_token,
                    api_key=api_key,
                    client_code=client_code,
                    feed_token=feed_token
                )

                # Set callbacks
                self.websocket.on_open = self._on_open
                self.websocket.on_message = self._on_message
                self.websocket.on_error = self._on_error
                self.websocket.on_close = self._on_close

                # Start the WebSocket connection in a non-blocking way
                self.websocket.connect() # This call is blocking and manages its own loop
                
                # This loop runs while the WebSocket is connected
                while self.is_connected:
                    # Periodically check for updated symbol list from JSON file
                    if time.time() - last_subscription_check_time >= self.SUBSCRIPTION_CHECK_INTERVAL:
                        logger.info("Checking for updated symbol list from JSON file...")
                        self._load_subscribed_symbols() # Reload symbols
                        new_config_hash = self._generate_config_hash(self.subscribed_symbols_config)
                        if new_config_hash != self.last_subscription_config_hash:
                            logger.info("Symbol list changed. Re-subscribing to new symbols.")
                            self.last_subscription_config_hash = new_config_hash
                            # Close and re-open WebSocket to apply new subscriptions reliably
                            # Or, implement granular unsubscribe/subscribe logic if SmartWebSocketV2 supports it
                            self.websocket.close_websocket() # Force close to trigger reconnect and re-subscribe
                            self.is_connected = False # Mark as disconnected to break this inner loop
                        last_subscription_check_time = time.time()

                    time.sleep(1) # Keep the main thread alive while WebSocket runs in background

                logger.warning("WebSocket connection lost. Retrying...")
                reconnect_attempts += 1
                time.sleep(self.RECONNECT_DELAY_SECONDS) # Wait before retrying

            except Exception as e:
                logger.error(f"Error during WebSocket connection attempt: {e}", exc_info=True)
                reconnect_attempts += 1
                time.sleep(self.RECONNECT_DELAY_SECONDS)

        logger.critical(f"Failed to establish WebSocket connection after {self.RECONNECT_ATTEMPTS} attempts. Aborting live stream.")
        self.angel_api.logout()
        self.redis_store.disconnect()

# Main execution block for testing
if __name__ == "__main__":
    print("--- Starting LiveStreamManager Test ---")
    
    # Initialize AngelOneAPI and RedisStore
    angel_api_instance = AngelOneAPI()
    redis_store_instance = RedisStore()

    # Create and run the LiveStreamManager
    live_stream_manager = LiveStreamManager(angel_api_instance, redis_store_instance)
    
    # Run the stream. This will attempt to connect to Angel One and then WebSocket.
    # It will run indefinitely until interrupted (Ctrl+C).
    try:
        live_stream_manager.run_live_stream()
    except KeyboardInterrupt:
        logger.info("Live stream interrupted by user.")
    finally:
        # Ensure logout and disconnection on exit
        if live_stream_manager.websocket:
            live_stream_manager.websocket.close_websocket() # Ensure websocket is properly closed
        angel_api_instance.logout()
        redis_store_instance.disconnect()
        logger.info("Live stream manager stopped.")
    
print("--- LiveStreamManager Test End ---")
