# redis_store.py
# This module manages interactions with Redis, serving as a central
# data store for real-time market data (LTP, Volume), strategy settings,
# trade states, and cooldown timers.

import redis
import logging
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Optional, Any, Dict, List

# Import AngelOneAPI for LTP fallback (will be passed during initialization)
# This import is here for type hinting and clarity, but the instance will be provided.
from angelone_api_patch import AngelOneAPI

# Configure logging for the module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

class RedisStore:
    """
    Manages connections and operations with a Redis database.
    Used for storing real-time data, strategy settings, and trade states.
    """
    def __init__(self, angel_api: Optional[AngelOneAPI] = None):
        """
        Initializes the RedisStore.

        Args:
            angel_api (Optional[AngelOneAPI]): An optional instance of AngelOneAPI for LTP fallback.
        """
        load_dotenv() # Load environment variables

        self.host = os.getenv("REDIS_HOST")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.password = os.getenv("REDIS_PASSWORD")
        self.db = int(os.getenv("REDIS_DB", 0))
        self.redis_client: Optional[redis.Redis] = None
        self.angel_api = angel_api # Store the AngelOneAPI instance

        logger.info(f"RedisStore initialized for {self.host}:{self.port}")

    def connect(self) -> bool:
        """
        Establishes a connection to the Redis server.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        if self.redis_client and self.redis_client.ping():
            logger.info("Already connected to Redis.")
            return True
        
        try:
            logger.info(f"Connecting to Redis at Host: {self.host}, Port: {self.port}, Password set: {'Yes' if self.password else 'No'}")
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                socket_connect_timeout=5, # Timeout for initial connection
                socket_timeout=5 # Timeout for subsequent operations
            )
            self.redis_client.ping() # Test the connection
            logger.info("Successfully connected to Redis.")
            return True
        except redis.exceptions.ConnectionError as e:
            logger.critical(f"Could not connect to Redis: {e}", exc_info=True)
            self.redis_client = None
            return False
        except Exception as e:
            logger.critical(f"An unexpected error occurred during Redis connection: {e}", exc_info=True)
            self.redis_client = None
            return False

    def disconnect(self):
        """
        Closes the connection to the Redis server.
        """
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Disconnected from Redis.")
                self.redis_client = None
            except Exception as e:
                logger.error(f"Error disconnecting from Redis: {e}", exc_info=True)
        else:
            logger.info("Redis client not connected. No disconnection needed.")

    def write_ltp(self, symbol: str, ltp: float):
        """
        Writes the Latest Traded Price (LTP) for a symbol to Redis.
        """
        if self.redis_client:
            try:
                self.redis_client.set(f"LTP:{symbol}", str(ltp))
                logger.debug(f"Wrote LTP for {symbol}: {ltp}")
            except Exception as e:
                logger.error(f"Error writing LTP for {symbol} to Redis: {e}", exc_info=True)
        else:
            logger.warning(f"Redis client not connected. Cannot write LTP for {symbol}.")

    def _fetch_ltp_from_rest_api(self, symbol: str) -> Optional[float]:
        """
        Fetches LTP for a symbol using Angel One's REST API as a fallback.
        This method assumes `self.angel_api` is initialized and logged in.
        """
        if not self.angel_api:
            logger.warning("AngelOneAPI instance not provided to RedisStore. Cannot fetch LTP via REST API fallback.")
            return None
        
        # You need a mapping from symbol (e.g., "RELIANCE") to Angel One's
        # symboltoken and exchangeType for the REST API call.
        # For simplicity, we'll use a hardcoded map for common symbols for now.
        # In a real system, you'd fetch this from a master instrument list.
        ANGEL_ONE_REST_SYMBOLS = {
            "NIFTY_50": {"exchangeType": "NSE", "token": "999260000", "tradingsymbol": "NIFTY"}, # Corrected token for NIFTY_50
            "RELIANCE": {"exchangeType": "NSE", "token": "2885", "tradingsymbol": "RELIANCE"}, # Corrected token for RELIANCE
            "TCS": {"exchangeType": "NSE", "token": "3045", "tradingsymbol": "TCS"},
            "HDFC_BANK": {"exchangeType": "NSE", "token": "3432", "tradingsymbol": "HDFCBANK"},
            "SBIN": {"exchangeType": "NSE", "token": "1333", "tradingsymbol": "SBIN"},
            "ICICI_BANK": {"exchangeType": "NSE", "token": "1660", "tradingsymbol": "ICICIBANK"}, # Corrected token for ICICI_BANK
            "INFY": {"exchangeType": "NSE", "token": "10604", "tradingsymbol": "INFY"},
            "ITC": {"exchangeType": "NSE", "token": "10606", "tradingsymbol": "ITC"},
            "MARUTI": {"exchangeType": "NSE", "token": "20374", "tradingsymbol": "MARUTI"},
            "AXISBANK": {"exchangeType": "NSE", "token": "1348", "tradingsymbol": "AXISBANK"},
            # Add more as needed
        }

        # Define expected LTP ranges for sanity checks (approximate values)
        # This is a basic example; for a real system, these would be dynamic or more robust.
        EXPECTED_LTP_RANGES = {
            "NIFTY_50": (20000.0, 25000.0), # Example range for NIFTY 50 Index
            "RELIANCE": (2200.0, 3000.0),  # Example range for Reliance Industries
            "TCS": (3000.0, 4000.0),
            "HDFC_BANK": (1400.0, 1800.0),
            "SBIN": (600.0, 800.0),
            "ICICI_BANK": (900.0, 1200.0),
            "INFY": (1300.0, 1700.0),
            "ITC": (400.0, 500.0),
            "MARUTI": (8000.0, 12000.0),
            "AXISBANK": (900.0, 1200.0),
        }


        symbol_info = ANGEL_ONE_REST_SYMBOLS.get(symbol)
        if not symbol_info:
            logger.warning(f"No Angel One REST API mapping found for symbol: {symbol}. Cannot fetch LTP via REST.")
            return None

        try:
            logger.info(f"Attempting to fetch LTP for {symbol} via Angel One REST API fallback...")
            
            # CORRECTED: Directly call ltpData on the smart_api_client
            smart_api_client = self.angel_api.get_smart_api_client()
            if not smart_api_client:
                logger.error("Angel One SmartAPI client not available. Cannot fetch LTP via REST API.")
                return None

            response = smart_api_client.ltpData( # Direct call to ltpData
                exchange=symbol_info["exchangeType"],
                tradingsymbol=symbol_info["tradingsymbol"],
                symboltoken=symbol_info["token"]
            )

            if response and response.get('status') and response.get('data'):
                ltp = response['data'].get('ltp')
                if ltp is not None:
                    ltp_float = float(ltp)
                    logger.info(f"Successfully fetched LTP for {symbol} via REST API: {ltp_float}")

                    # Sanity check for LTP value
                    expected_range = EXPECTED_LTP_RANGES.get(symbol)
                    if expected_range:
                        min_val, max_val = expected_range
                        if not (min_val <= ltp_float <= max_val):
                            logger.warning(f"LTP for {symbol} ({ltp_float:.2f}) is outside expected range ({min_val:.2f}-{max_val:.2f}). "
                                           "This might indicate stale data or an incorrect instrument.")
                    else:
                        logger.debug(f"No expected LTP range defined for {symbol}. Skipping sanity check.")

                    return ltp_float
                else:
                    logger.warning(f"LTP not found in REST API response for {symbol}: {response}")
            else:
                error_message = response.get('message', 'No data or unknown error.') if response else "No response from REST API."
                logger.warning(f"Failed to fetch LTP for {symbol} via REST API: {error_message}")
            return None
        except Exception as e:
            logger.error(f"Error fetching LTP for {symbol} via Angel One REST API: {e}", exc_info=True)
            return None


    def read_ltp(self, symbol: str) -> Optional[float]:
        """
        Reads the Latest Traded Price (LTP) for a symbol from Redis.
        If not found, attempts to fetch from Angel One REST API as a fallback.
        """
        if not self.redis_client:
            logger.critical(f"Redis client not connected. Cannot read LTP for {symbol}.")
            return None

        try:
            ltp_str = self.redis_client.get(f"LTP:{symbol}")
            if ltp_str:
                return float(ltp_str)
            else:
                logger.warning(f"LTP for {symbol} not found in Redis. Attempting REST API fallback...")
                # Fallback to REST API if LTP is not in Redis
                rest_ltp = self._fetch_ltp_from_rest_api(symbol)
                if rest_ltp is not None:
                    # Optionally, write the fetched REST LTP to Redis for future reads
                    self.write_ltp(symbol, rest_ltp)
                    return rest_ltp
                else:
                    logger.critical(f"LTP for {symbol} is None from both Redis and REST API. Data missing!")
                    return None
        except ValueError as e:
            logger.error(f"Error converting LTP for {symbol} to float: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error reading LTP for {symbol} from Redis: {e}", exc_info=True)
            return None

    def is_on_cooldown(self, symbol: str) -> bool:
        """
        Checks if a symbol is currently on cooldown.
        """
        if self.redis_client:
            try:
                return self.redis_client.exists(f"cooldown:{symbol}") == 1
            except Exception as e:
                logger.error(f"Error checking cooldown for {symbol}: {e}", exc_info=True)
                return False
        logger.warning(f"Redis client not connected. Cannot check cooldown for {symbol}.")
        return False

    def set_cooldown_timer(self, symbol: str, duration_seconds: int):
        """
        Sets a cooldown timer for a symbol in Redis.
        """
        if self.redis_client:
            try:
                self.redis_client.setex(f"cooldown:{symbol}", duration_seconds, "active")
                logger.info(f"Set cooldown for {symbol} for {duration_seconds} seconds.")
            except Exception as e:
                logger.error(f"Error setting cooldown for {symbol}: {e}", exc_info=True)
        else:
            logger.warning(f"Redis client not connected. Cannot set cooldown for {symbol}.")

    def save_trade_state(self, trade_id: str, trade_data: Dict[str, Any]):
        """
        Saves the state of an individual trade to Redis.
        """
        if self.redis_client:
            try:
                self.redis_client.hset("active_trades", trade_id, json.dumps(trade_data))
                logger.debug(f"Saved active trade {trade_id} state to Redis.")
            except Exception as e:
                logger.error(f"Error saving trade {trade_id} state to Redis: {e}", exc_info=True)
        else:
            logger.warning(f"Redis client not connected. Cannot save trade state for {trade_id}.")

    def load_trade_state(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads the state of an individual trade from Redis.
        """
        if self.redis_client:
            try:
                trade_json = self.redis_client.hget("active_trades", trade_id)
                if trade_json:
                    return json.loads(trade_json)
            except Exception as e:
                logger.error(f"Error loading trade {trade_id} state from Redis: {e}", exc_info=True)
        else:
            logger.warning(f"Redis client not connected. Cannot load trade state for {trade_id}.")
        return None

    def get_all_active_trades(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves all active trades from Redis.
        """
        active_trades = {}
        if self.redis_client:
            try:
                all_trades_hash = self.redis_client.hgetall("active_trades")
                for trade_id_bytes, trade_json_bytes in all_trades_hash.items():
                    try:
                        trade_id = trade_id_bytes.decode('utf-8')
                        trade_data = json.loads(trade_json_bytes)
                        active_trades[trade_id] = trade_data
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.error(f"Error decoding active trade from Redis: {e}. Data: {trade_json_bytes}", exc_info=True)
                        continue
            except Exception as e:
                logger.error(f"Error retrieving all active trades from Redis: {e}", exc_info=True)
        else:
            logger.warning("Redis client not connected. Cannot get all active trades.")
        return active_trades

    def remove_active_trade(self, trade_id: str):
        """
        Removes an active trade from Redis.
        """
        if self.redis_client:
            try:
                self.redis_client.hdel("active_trades", trade_id)
                logger.debug(f"Removed active trade {trade_id} from Redis.")
            except Exception as e:
                logger.error(f"Error removing active trade {trade_id} from Redis: {e}", exc_info=True)
        else:
            logger.warning(f"Redis client not connected. Cannot remove active trade {trade_id}.")

    def add_closed_trade(self, trade_data: Dict[str, Any]):
        """
        Adds a closed trade to a list in Redis.
        """
        if self.redis_client:
            try:
                # Use RPUSH to add to the end of a list
                self.redis_client.rpush("closed_trades_list", json.dumps(trade_data))
                logger.debug(f"Added closed trade {trade_data.get('trade_id', 'N/A')} to Redis.")
            except Exception as e:
                logger.error(f"Error adding closed trade to Redis: {e}", exc_info=True)
        else:
            logger.warning(f"Redis client not connected. Cannot add closed trade.")

    def get_all_closed_trades(self) -> List[Dict[str, Any]]:
        """
        Retrieves all closed trades from Redis.
        """
        closed_trades = []
        if self.redis_client:
            try:
                # Use LRANGE to get all items from the list
                all_trades_bytes = self.redis_client.lrange("closed_trades_list", 0, -1)
                for trade_json_bytes in all_trades_bytes:
                    try:
                        closed_trades.append(json.loads(trade_json_bytes))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        logger.error(f"Error decoding closed trade from Redis: {e}. Data: {trade_json_bytes}", exc_info=True)
                        continue
            except Exception as e:
                logger.error(f"Error retrieving all closed trades from Redis: {e}", exc_info=True)
        else:
            logger.warning("Redis client not connected. Cannot get all closed trades.")
        return closed_trades

    def save_system_state(self, available_capital: float, total_pnl: float, sync_paused: bool):
        """
        Saves the overall system state (capital, PnL, sync status) to Redis.
        """
        if self.redis_client:
            try:
                state = {
                    "available_capital": available_capital,
                    "total_pnl": total_pnl,
                    "sync_paused": sync_paused,
                    "last_saved": datetime.now().isoformat()
                }
                self.redis_client.set("system_state", json.dumps(state))
                logger.debug("Saved system state to Redis.")
            except Exception as e:
                logger.error(f"Error saving system state to Redis: {e}", exc_info=True)
        else:
            logger.warning("Redis client not connected. Cannot save system state.")

    def load_system_state(self) -> Optional[Dict[str, Any]]:
        """
        Loads the overall system state from Redis.
        """
        if self.redis_client:
            try:
                state_json = self.redis_client.get("system_state")
                if state_json:
                    return json.loads(state_json)
            except Exception as e:
                logger.error(f"Error loading system state from Redis: {e}", exc_info=True)
        else:
            logger.warning("Redis client not connected. Cannot load system state.")
        return {} # Changed from None to empty dictionary

    def get_available_capital(self) -> float:
        """
        Retrieves the available capital from the system state stored in Redis.
        Returns 0.0 if not found or an error occurs.
        """
        state = self.load_system_state()
        if state:
            return state.get("available_capital", 0.0)
        return 0.0

    def get_total_realized_pnl(self) -> float:
        """
        Retrieves the total realized PnL from the system state stored in Redis.
        Returns 0.0 if not found or an error occurs.
        """
        state = self.load_system_state()
        if state:
            return state.get("total_pnl", 0.0)
        return 0.0

    def get_setting(self, key: str, default_value: Any = None) -> Any:
        """
        Retrieves a strategy setting from Redis.
        If not found, returns a default value.
        """
        if self.redis_client:
            try:
                setting_value = self.redis_client.get(f"setting:{key}")
                if setting_value:
                    # Attempt to deserialize if it was stored as JSON, otherwise return raw
                    try:
                        return json.loads(setting_value)
                    except json.JSONDecodeError:
                        return setting_value.decode('utf-8') # Return as string
                else:
                    logger.debug(f"Setting '{key}' not found in Redis. Returning default value: {default_value}")
                    return default_value
            except Exception as e:
                logger.error(f"Error retrieving setting '{key}' from Redis: {e}", exc_info=True)
                return default_value
        else:
            logger.warning("Redis client not connected. Cannot get setting.")
            return default_value

    def set_setting(self, key: str, value: Any):
        """
        Sets a strategy setting in Redis.
        """
        if self.redis_client:
            try:
                # Store as JSON string to handle various data types
                self.redis_client.set(f"setting:{key}", json.dumps(value))
                logger.info(f"Set setting '{key}' to '{value}' in Redis.")
            except Exception as e:
                logger.error(f"Error setting '{key}' in Redis: {e}", exc_info=True)
        else:
            logger.warning("Redis client not connected. Cannot set setting.")

    def get_all_settings(self) -> Dict[str, Any]:
        """
        Retrieves all strategy settings stored in Redis under the 'setting:*' pattern.

        Returns:
            Dict[str, Any]: A dictionary of all settings.
        """
        all_settings = {}
        if self.redis_client:
            try:
                # Use 'keys' to find all keys matching the pattern
                setting_keys = self.redis_client.keys("setting:*")
                for key_bytes in setting_keys:
                    key = key_bytes.decode('utf-8')
                    # Extract the actual setting name (remove "setting:")
                    setting_name = key.replace("setting:", "")
                    value = self.redis_client.get(key)
                    if value:
                        try:
                            # Attempt to deserialize if it was stored as JSON
                            all_settings[setting_name] = json.loads(value)
                        except json.JSONDecodeError:
                            all_settings[setting_name] = value.decode('utf-8')
            except Exception as e:
                logger.error(f"Error retrieving all settings from Redis: {e}", exc_info=True)
        else:
            logger.warning("Redis client not connected. Cannot get all settings.")
        return all_settings


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    print("--- Starting RedisStore Module Test ---")
    
    # For testing RedisStore directly, we need a mock AngelOneAPI
    class MockAngelOneAPIForRedisStore:
        def get_smart_api_client(self):
            class MockSmartApiClient:
                def ltpData(self, exchange: str, tradingsymbol: str, symboltoken: str) -> Dict[str, Any]:
                    # Simulate a response based on the corrected tokens
                    if tradingsymbol == "RELIANCE" and symboltoken == "2885":
                        return {"status": True, "data": {"ltp": 2400.00}} # Updated mock LTP to be more realistic
                    elif tradingsymbol == "INFY" and symboltoken == "10604":
                        return {"status": True, "data": {"ltp": 1500.75}}
                    elif tradingsymbol == "NIFTY" and symboltoken == "999260000":
                        return {"status": True, "data": {"ltp": 23000.00}} # Mock NIFTY LTP
                    elif tradingsymbol == "ICICIBANK" and symboltoken == "1660":
                        return {"status": True, "data": {"ltp": 1000.00}} # Mock ICICI Bank LTP
                    else:
                        return {"status": False, "message": f"Symbol {tradingsymbol} with token {symboltoken} not found in mock."}
            return MockSmartApiClient()

    mock_angel_api = MockAngelOneAPIForRedisStore()
    redis_store = RedisStore(angel_api=mock_angel_api) # Pass the mock AngelOneAPI

    if redis_store.connect():
        # --- Test LTP Read/Write ---
        print("\n--- Testing LTP Read/Write ---")
        redis_store.write_ltp("TEST_SYMBOL_1", 100.50)
        ltp1 = redis_store.read_ltp("TEST_SYMBOL_1")
        print(f"LTP for TEST_SYMBOL_1: {ltp1}") # Should be 100.50

        ltp_non_existent = redis_store.read_ltp("NON_EXISTENT_SYMBOL")
        print(f"LTP for NON_EXISTENT_SYMBOL (from Redis, then REST fallback): {ltp_non_existent}") # Should be None or from REST if mapped

        ltp_reliance_rest = redis_store.read_ltp("RELIANCE")
        print(f"LTP for RELIANCE (from Redis, then REST fallback): {ltp_reliance_rest}") # Should be 2400.00 (from mock)

        ltp_nifty_rest = redis_store.read_ltp("NIFTY_50")
        print(f"LTP for NIFTY_50 (from Redis, then REST fallback): {ltp_nifty_rest}") # Should be 23000.00 (from mock)

        ltp_icici_rest = redis_store.read_ltp("ICICI_BANK")
        print(f"LTP for ICICI_BANK (from Redis, then REST fallback): {ltp_icici_rest}") # Should be 1000.00 (from mock)

        # --- Test Cooldown ---
        print("\n--- Testing Cooldown ---")
        redis_store.set_cooldown_timer("TEST_SYMBOL_2", 5) # 5 seconds cooldown
        print(f"TEST_SYMBOL_2 on cooldown: {redis_store.is_on_cooldown('TEST_SYMBOL_2')}") # Should be True
        # time.sleep(6) # Uncomment to test cooldown expiry
        # print(f"TEST_SYMBOL_2 on cooldown after 6s: {redis_store.is_on_cooldown('TEST_SYMBOL_2')}") # Should be False

        # --- Test Trade State Management ---
        print("\n--- Testing Trade State Management ---")
        test_trade_id = "TRADE_XYZ_123"
        test_trade_data = {"symbol": "INFY", "entry_price": 1500.0, "qty": 10, "status": "ACTIVE"}
        redis_store.save_trade_state(test_trade_id, test_trade_data)
        loaded_trade = redis_store.load_trade_state(test_trade_id)
        print(f"Loaded trade {test_trade_id}: {loaded_trade}") # Should match test_trade_data

        all_active = redis_store.get_all_active_trades()
        print(f"All active trades: {all_active}") # Should include TEST_TRADE_XYZ_123

        redis_store.remove_active_trade(test_trade_id)
        all_active_after_remove = redis_store.get_all_active_trades()
        print(f"All active trades after removal: {all_active_after_remove}") # Should be empty

        # Test closed trades
        closed_trade_data = {"symbol": "INFY", "pnl": 50.0, "status": "CLOSED"}
        redis_store.add_closed_trade(closed_trade_data)
        all_closed = redis_store.get_all_closed_trades()
        print(f"All closed trades: {all_closed}")

        # --- Test System State Management ---
        print("\n--- Testing System State Management ---")
        redis_store.save_system_state(99999.50, 123.45, True)
        loaded_state = redis_store.load_system_state()
        print(f"Loaded system state: {loaded_state}")
        
        # Test new getters
        print(f"Available Capital from getter: {redis_store.get_available_capital()}")
        print(f"Total Realized PnL from getter: {redis_store.get_total_realized_pnl()}")


        # --- Test Settings Management ---
        print("\n--- Testing Settings Management ---")
        redis_store.set_setting("my_test_setting", "some_value")
        setting_val = redis_store.get_setting("my_test_setting")
        print(f"Setting 'my_test_setting': {setting_val}")

        redis_store.set_setting("another_setting", {"key": "value", "num": 123})
        another_setting_val = redis_store.get_setting("another_setting")
        print(f"Setting 'another_setting': {another_setting_val}")

        # Test get_all_settings (NEW)
        print("\n--- Testing get_all_settings (NEW) ---")
        all_settings = redis_store.get_all_settings()
        print(f"All retrieved settings: {all_settings}")
        # Expected output should include 'my_test_setting' and 'another_setting'

        redis_store.disconnect()
    else:
        print("Failed to connect to Redis, skipping tests.")
    
    print("--- RedisStore Module Test End ---")
