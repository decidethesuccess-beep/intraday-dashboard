# historical_data_manager.py
# This module is responsible for fetching and managing historical OHLCV data.
# It now leverages the Angel One API for more granular intraday data.

import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Import necessary components
from redis_store import RedisStore
from angelone_api_patch import AngelOneAPI # Import AngelOneAPI for data fetching

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalDataManager:
    """
    Manages fetching and caching of historical OHLCV data, now using Angel One API.
    """
    def __init__(self, angel_api: AngelOneAPI, redis_store: RedisStore):
        """
        Initializes the HistoricalDataManager.

        Args:
            angel_api (AngelOneAPI): An authenticated instance of AngelOneAPI.
            redis_store (RedisStore): An instance of RedisStore for caching.
        """
        self.angel_api = angel_api
        self.redis_store = redis_store
        self.CACHE_EXPIRY_SECONDS = 3600 # Cache historical data for 1 hour (adjust as needed)
        logger.info("HistoricalDataManager initialized.")

    def _fetch_from_angel_one(self,
                              symbol_token: str,
                              exchange_type: str,
                              from_date: datetime,
                              to_date: datetime,
                              interval: str) -> Optional[List[List[Any]]]:
        """
        Fetches historical candle data from Angel One API.
        """
        try:
            smart_api_client = self.angel_api.get_smart_api_client()
            if not smart_api_client:
                logger.error("Angel One SmartAPI client not available. Cannot fetch historical data.")
                return None

            # Angel One's getCandleData expects parameters in a dictionary
            params = {
                "exchange": exchange_type,
                "symboltoken": symbol_token,
                "interval": interval, # e.g., "ONE_MINUTE", "FIFTEEN_MINUTE", "DAY"
                "fromdate": from_date.strftime('%Y-%m-%d %H:%M'),
                "todate": to_date.strftime('%Y-%m-%d %H:%M')
            }
            logger.info(f"Fetching historical data from Angel One for token {symbol_token} with params: {params}")

            response = smart_api_client.getCandleData(params)
            logger.debug(f"Raw Angel One historical data response for token {symbol_token}: {json.dumps(response, indent=2)}")

            if response and response.get('status') and response.get('data'):
                parsed_candles = []
                for candle in response['data']:
                    try:
                        # Angel One timestamp format: "YYYY-MM-DDTHH:MM:SS+HH:MM" (ISO 8601 with timezone)
                        # Convert to datetime object, then to ISO string for consistency
                        timestamp_dt = datetime.strptime(candle[0], "%Y-%m-%dT%H:%M:%S%z")
                        timestamp_iso_str = timestamp_dt.isoformat()

                        open_price = float(candle[1])
                        high_price = float(candle[2])
                        low_price = float(candle[3])
                        close_price = float(candle[4])
                        volume = int(float(candle[5])) # Convert to float first, then int

                        parsed_candles.append([
                            timestamp_iso_str,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            volume
                        ])
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing Angel One candle data row: {candle}. Error: {e}. Skipping candle.")
                        continue
                
                if parsed_candles:
                    logger.info(f"Successfully parsed {len(parsed_candles)} candles from Angel One.")
                    return parsed_candles
                else:
                    logger.warning(f"Angel One returned data but no valid candles could be parsed for token {symbol_token}.")
                    return None
            else:
                error_message = response.get('message', 'No data or unknown error.')
                logger.warning(f"Angel One historical data fetch failed for token {symbol_token}: {error_message}. Full response: {response}")
                return None
        except Exception as e:
            logger.error(f"An error occurred during Angel One historical data fetch for token {symbol_token}: {e}", exc_info=True)
            return None

    def get_historical_data(self,
                            symbol: str,
                            symbol_token: str, # Angel One's token
                            exchange_type: str, # Angel One's exchange type (e.g., "NSE")
                            from_date: datetime,
                            to_date: datetime,
                            interval: str) -> Optional[List[List[Any]]]:
        """
        Retrieves historical OHLCV data for a given symbol and period.
        Prioritizes cache, then fetches from Angel One API if not in cache.

        Args:
            symbol (str): Human-readable symbol (e.g., "INFY").
            symbol_token (str): Angel One's unique token for the instrument.
            exchange_type (str): Angel One's exchange type (e.g., "NSE").
            from_date (datetime): Start datetime for data.
            to_date (datetime): End datetime for data.
            interval (str): Candle interval (e.g., "ONE_MINUTE", "FIFTEEN_MINUTE", "DAY").

        Returns:
            Optional[List[List[Any]]]: A list of lists, where each inner list is a candle
                                       [timestamp_str, open, high, low, close, volume],
                                       or None if the request fails.
        """
        # Create a unique cache key based on symbol, dates, and interval
        cache_key = f"historical_data:{symbol}:{from_date.strftime('%Y%m%d%H%M')}:{to_date.strftime('%Y%m%d%H%M')}:{interval}"

        # 1. Try to fetch from Redis cache
        cached_data = self.redis_store.redis_client.get(cache_key)
        if cached_data:
            try:
                data = json.loads(cached_data)
                logger.info(f"Fetched historical data for {symbol} from Redis cache ({len(data)} candles).")
                return data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode cached historical data for {symbol}: {e}. Fetching from API.")
                # If cache is corrupted, delete it
                self.redis_store.redis_client.delete(cache_key)

        # 2. If not in cache, fetch from Angel One API
        logger.info(f"Fetching historical data for {symbol} (Token: {symbol_token}) from Angel One API (not from cache)...")
        data = self._fetch_from_angel_one(symbol_token, exchange_type, from_date, to_date, interval)

        if data:
            # 3. Store in Redis cache
            try:
                self.redis_store.redis_client.setex(cache_key, self.CACHE_EXPIRY_SECONDS, json.dumps(data))
                logger.info(f"Successfully cached historical data for {symbol} in Redis.")
            except Exception as e:
                logger.error(f"Error caching historical data for {symbol}: {e}", exc_info=True)
            return data
        else:
            logger.warning(f"No historical data available for {symbol} from Angel One API in the specified period.")
            return None

# Example usage (for testing purposes)
if __name__ == "__main__":
    print("--- Starting HistoricalDataManager Module Test ---")

    # Ensure .env variables are loaded for AngelOneAPI and RedisStore
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize AngelOneAPI (ensure your ANGELONE_API_KEY for historical data is set in .env)
    # This might be a different API key than your trading one if Angel One separates them.
    angel_api_instance = AngelOneAPI(api_key_to_use=os.getenv("ANGELONE_HISTORICAL_API_KEY"))
    if not angel_api_instance.login():
        print("❌ Failed to log in to Angel One API. Cannot run historical data manager test.")
        exit()

    # Initialize RedisStore
    redis_store_instance = RedisStore()
    if not redis_store_instance.connect():
        print("❌ Failed to connect to Redis. Cannot run historical data manager test.")
        angel_api_instance.logout()
        exit()

    historical_data_manager = HistoricalDataManager(angel_api_instance, redis_store_instance)

    # Define test parameters for a symbol (e.g., RELIANCE)
    # IMPORTANT: Use correct Angel One symbol token and exchange type
    # You will need to get these from Angel One's scrip master or documentation.
    # Example: RELIANCE (NSE_CM) token 11536
    test_symbol = "RELIANCE"
    test_symbol_token = "11536" # Example token for RELIANCE
    test_exchange_type = "NSE" # Example exchange type

    # Define a date range for intraday data (e.g., one full trading day)
    # Ensure this is a recent trading day where data is expected.
    test_from_date = datetime(2025, 7, 1, 9, 15) # Start of market hours
    test_to_date = datetime(2025, 7, 1, 15, 30)   # End of market hours for a single day

    test_interval = "ONE_MINUTE" # Request 1-minute candles

    print(f"\n--- Testing fetch for {test_symbol} ({test_interval}) ---")
    historical_candles = historical_data_manager.get_historical_data(
        symbol=test_symbol,
        symbol_token=test_symbol_token,
        exchange_type=test_exchange_type,
        from_date=test_from_date,
        to_date=test_to_date,
        interval=test_interval
    )

    if historical_candles:
        print(f"✅ Fetched {len(historical_candles)} candles for {test_symbol}.")
        print("First 5 candles:")
        for i, candle in enumerate(historical_candles[:5]):
            print(f"  {i+1}: {candle}")
        if len(historical_candles) > 5:
            print("...")
        print("Last 5 candles:")
        for i, candle in enumerate(historical_candles[-5:]):
            print(f"  {len(historical_candles) - 5 + i + 1}: {candle}")
    else:
        print(f"❌ Failed to fetch historical data for {test_symbol}.")

    # Clean up
    angel_api_instance.logout()
    redis_store_instance.disconnect()
    print("\n--- HistoricalDataManager Module Test End ---")
