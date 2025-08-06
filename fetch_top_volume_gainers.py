# fetch_top_volume_gainers.py
# This module is responsible for identifying top volume gainers from the market
# for backtesting or live trading.

import logging
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
from typing import List, Dict, Any, Optional
import requests # Added for fetching JSON from URL
import json
import time # Added for rate limiting

# Assuming AngelOneAPI and RedisStore are available for historical data fetching
from angelone_api_patch import AngelOneAPI
from redis_store import RedisStore
from historical_data_manager import HistoricalDataManager # Import HistoricalDataManager

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Ensure this URL is a plain string, without any markdown link formatting like []()
ANGEL_ONE_INSTRUMENT_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
SCRIP_MASTER_CSV_PATH = "api-scrip-master.csv" # Your DHAN CSV file
DHAN_SYMBOL_COLUMN_NAME = "SEM_TRADING_SYMBOL" # The correct column name for symbols in your DHAN CSV

# --- Global variable for cached Angel One instrument data ---
# This will store the filtered Angel One instrument dump once fetched for the day
_angel_one_filtered_instruments_df: Optional[pd.DataFrame] = None
_last_instrument_dump_fetch_date: Optional[datetime] = None

def fetch_angel_one_instrument_dump(url: str = ANGEL_ONE_INSTRUMENT_URL) -> Optional[pd.DataFrame]:
    """
    Fetches the full Angel One instrument dump from the provided URL,
    filters it for NSE Equity symbols, and returns a DataFrame.
    Caches the result for the current day to avoid multiple fetches.
    """
    global _angel_one_filtered_instruments_df
    global _last_instrument_dump_fetch_date

    current_date = datetime.now().date()

    # Check if we already fetched and cached for today
    if _angel_one_filtered_instruments_df is not None and \
       _last_instrument_dump_fetch_date == current_date:
        logger.info("Using cached Angel One instrument dump for today.")
        return _angel_one_filtered_instruments_df

    logger.info(f"Fetching Angel One instrument dump from: {url}")
    try:
        response = requests.get(url, timeout=10) # 10 second timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        full_dump = response.json()
        logger.info(f"Successfully downloaded Angel One instrument dump. Total records: {len(full_dump)}")

        # Convert to DataFrame for easier filtering
        df_angel = pd.DataFrame(full_dump)

        # --- UPDATED FILTERING LOGIC ---
        # Filter for NSE Equity symbols based on:
        # 1. Exchange segment is 'NSE'
        # 2. 'symbol' column ends with '-EQ' (common for equities in this dump)
        # 3. 'name' and 'token' are not null
        df_filtered = df_angel[
            (df_angel['exch_seg'] == 'NSE') &
            (df_angel['symbol'].str.endswith('-EQ', na=False)) & # Check for '-EQ' suffix for equities
            (df_angel['name'].notna()) & # Ensure name is not null (this is the actual symbol name)
            (df_angel['token'].notna()) # Ensure token is not null
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Rename 'token' column to 'SEM_INSTRUMENT_TOKEN' for consistency
        # and select only necessary columns. Also rename 'name' to 'ANGEL_SYMBOL'
        # as it seems to be the cleaner base symbol for merging.
        df_filtered = df_filtered.rename(columns={
            'token': 'SEM_INSTRUMENT_TOKEN',
            'exch_seg': 'EXCHANGE',
            'name': 'ANGEL_SYMBOL' # Use 'name' as the cleaner symbol for merging
        })
        df_filtered = df_filtered[['ANGEL_SYMBOL', 'SEM_INSTRUMENT_TOKEN', 'EXCHANGE']]
        
        # Ensure symbol and token are strings
        df_filtered['ANGEL_SYMBOL'] = df_filtered['ANGEL_SYMBOL'].astype(str)
        df_filtered['SEM_INSTRUMENT_TOKEN'] = df_filtered['SEM_INSTRUMENT_TOKEN'].astype(str)
        df_filtered['EXCHANGE'] = df_filtered['EXCHANGE'].astype(str)

        logger.info(f"Filtered Angel One instrument dump to {len(df_filtered)} NSE Equity symbols.")
        
        # Cache the result
        _angel_one_filtered_instruments_df = df_filtered
        _last_instrument_dump_fetch_date = current_date
        
        return df_filtered

    except requests.exceptions.RequestException as e:
        logger.critical(f"Error fetching Angel One instrument dump from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.critical(f"Error decoding JSON from Angel One instrument dump: {e}")
        return None
    except KeyError as e:
        logger.critical(f"Missing expected column in Angel One instrument dump: {e}. Please check JSON structure.")
        return None
    except Exception as e:
        logger.critical(f"An unexpected error occurred while fetching/processing Angel One instrument dump: {e}", exc_info=True)
        return None

def get_top_100_volume_gainers(angel_api: AngelOneAPI, redis_store: RedisStore, trade_date: str, top_n: int = 100) -> List[Dict[str, Any]]:
    """
    Identifies the top N volume gainers for a given trade date by:
    1. Loading symbols from api-scrip-master.csv (DHAN data).
    2. Fetching Angel One's full instrument dump and merging to get SEM_INSTRUMENT_TOKENs.
    3. Fetching 1-minute historical data for the first 15 minutes of the market open (9:15-9:30).
    4. Calculating total volume for this period.
    5. Sorting by volume and returning the top N symbols.

    Args:
        angel_api (AngelOneAPI): An authenticated AngelOneAPI instance.
        redis_store (RedisStore): An instance of RedisStore for caching.
        trade_date (str): The date for which to fetch gainers (YYYY-MM-DD).
        top_n (int): The number of top volume gainers to return.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing 'symbol', 'token', 'exchange', and 'volume'.
    """
    logger.info(f"Identifying top {top_n} volume gainers for {trade_date}...")

    # --- Step 1: Load DHAN symbols and merge with Angel One tokens ---
    try:
        df_dhan = pd.read_csv(SCRIP_MASTER_CSV_PATH)
        # Use the correct column name for symbols from DHAN CSV
        if DHAN_SYMBOL_COLUMN_NAME not in df_dhan.columns:
            logger.critical(f"Column '{DHAN_SYMBOL_COLUMN_NAME}' not found in '{SCRIP_MASTER_CSV_PATH}'. Cannot proceed.")
            return []
        df_dhan[DHAN_SYMBOL_COLUMN_NAME] = df_dhan[DHAN_SYMBOL_COLUMN_NAME].astype(str)
        logger.info(f"Loaded {len(df_dhan)} symbols from DHAN CSV using column '{DHAN_SYMBOL_COLUMN_NAME}'.")
    except FileNotFoundError:
        logger.critical(f"'{SCRIP_MASTER_CSV_PATH}' not found. Please ensure it exists.")
        return []
    except Exception as e:
        logger.critical(f"Error loading '{SCRIP_MASTER_CSV_PATH}': {e}", exc_info=True)
        return []

    # Fetch and filter Angel One instrument dump
    df_angel_instruments = fetch_angel_one_instrument_dump()
    if df_angel_instruments is None or df_angel_instruments.empty:
        logger.critical("Failed to fetch or filter Angel One instrument dump. Cannot proceed with volume gainers.")
        return []

    # Merge DHAN symbols with Angel One tokens
    # Merge on the 'symbol' column from DHAN and 'ANGEL_SYMBOL' from Angel One dump
    df_combined = pd.merge(
        df_dhan,
        df_angel_instruments,
        left_on=DHAN_SYMBOL_COLUMN_NAME, # Use the correct column from DHAN
        right_on='ANGEL_SYMBOL',         # Use the renamed column from Angel One dump
        how='inner'                      # Use inner join to only keep symbols present in both
    )
    
    # Filter out symbols that don't have a matching Angel One token (though inner join should handle this)
    df_combined_valid = df_combined[df_combined['SEM_INSTRUMENT_TOKEN'].notna()].copy()
    
    if df_combined_valid.empty:
        logger.critical("No valid NSE Equity symbols with Angel One tokens found after merging. Check your DHAN CSV and Angel One dump data.")
        return []

    logger.info(f"Combined and filtered down to {len(df_combined_valid)} NSE Equity symbols with valid Angel One tokens.")

    # --- Step 2: Fetch 1-min Historical Data (09:15-09:30) and Sum Volumes ---
    volume_data = []
    
    # Define the 15-minute window for volume calculation
    from_date_time = datetime.strptime(f"{trade_date} 09:15", "%Y-%m-%d %H:%M")
    to_date_time = datetime.strptime(f"{trade_date} 09:30", "%Y-%m-%d %H:%M")

    # Initialize HistoricalDataManager
    historical_data_manager = HistoricalDataManager(angel_api, redis_store)

    # Implement rate limiting for historical data calls
    # Reverted to 0.5 seconds as per user's request
    request_delay_seconds = 0.5

    for index, row in df_combined_valid.iterrows():
        symbol = row[DHAN_SYMBOL_COLUMN_NAME] # Use the symbol from the DHAN CSV
        token = row['SEM_INSTRUMENT_TOKEN']
        exchange = row['EXCHANGE'] # Use the exchange from the Angel One dump

        try:
            # Fetch historical data for the 15-minute window using HistoricalDataManager
            candles = historical_data_manager.get_historical_data(
                symbol=symbol,
                symbol_token=token,
                exchange_type=exchange,
                from_date=from_date_time,
                to_date=to_date_time,
                interval="ONE_MINUTE"
            )

            if candles:
                # Sum volumes for the first 15 minutes
                total_volume = sum(float(c[5]) for c in candles) # c[5] is volume
                volume_data.append({
                    "symbol": symbol, # Use the DHAN symbol
                    "token": token,
                    "exchange": exchange,
                    "volume": total_volume
                })
            else:
                logger.warning(f"No historical data for {symbol} ({token}) for {trade_date} 09:15-09:30. Skipping.")
        except Exception as e:
            logger.warning(f"Error fetching 15-min volume for {symbol} ({token}): {e}. Skipping symbol.", exc_info=True)
        
        time.sleep(request_delay_seconds) # Pause to respect API rate limits

    if not volume_data:
        logger.warning(f"No volume data collected for any symbols for {trade_date}. Check API connectivity or data availability.")
        return []

    # --- Step 3: Sort by Volume and Select Top N ---
    df_volume = pd.DataFrame(volume_data)
    df_volume_sorted = df_volume.sort_values(by='volume', ascending=False)
    
    top_n_symbols = df_volume_sorted.head(top_n).to_dict(orient='records')

    logger.info(f"Successfully identified top {len(top_n_symbols)} volume gainers for {trade_date}.")
    return top_n_symbols

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed output during testing
    print("--- Testing fetch_top_volume_gainers module ---")
    
    # Mock AngelOneAPI and RedisStore for testing purposes
    class MockAngelOneAPIForFetcher:
        def __init__(self):
            logger.info("MockAngelOneAPIForFetcher initialized.")
            self._mock_historical_data = {
                "1660": [ # ITC
                    ["2025-07-01T09:15:00+05:30", "417.0", "418.0", "416.5", "417.5", "100000"],
                    ["2025-07-01T09:16:00+05:30", "417.5", "418.5", "417.0", "418.0", "150000"],
                    ["2025-07-01T09:17:00+05:30", "418.0", "419.0", "417.5", "418.5", "200000"]
                ],
                "3045": [ # SBIN
                    ["2025-07-01T09:15:00+05:30", "820.0", "821.0", "819.5", "820.5", "200000"],
                    ["2025-07-01T09:16:00+05:30", "820.5", "821.5", "820.0", "821.0", "250000"],
                    ["2025-07-01T09:17:00+05:30", "821.0", "822.0", "820.5", "821.5", "300000"]
                ],
                "2885": [ # RELIANCE
                    ["2025-07-01T09:15:00+05:30", "1500.0", "1501.0", "1499.5", "1500.5", "500000"],
                    ["2025-07-01T09:16:00+05:30", "1500.5", "1501.5", "1500.0", "1501.0", "600000"],
                    ["2025-07-01T09:17:00+05:30", "1501.0", "1502.0", "1500.5", "1501.5", "700000"]
                ],
                # Add more mock data for other symbols if needed for comprehensive testing
                "1594": [ # INFY
                    ["2025-07-01T09:15:00+05:30", "1600.0", "1601.0", "1599.5", "1600.5", "120000"],
                    ["2025-07-01T09:16:00+05:30", "1600.5", "1601.5", "1600.0", "1601.0", "130000"],
                    ["2025-07-01T09:17:00+05:30", "1601.0", "1602.0", "1600.5", "1601.5", "140000"]
                ]
            }

        def get_historical_data(self, symbol_token: str, exchange_type: str, from_date: datetime, to_date: datetime, interval: str) -> List[List[str]]:
            # Mock historical data response for a few symbols
            logger.debug(f"Mock: Fetching historical data for token {symbol_token} from {from_date.strftime('%H:%M')} to {to_date.strftime('%H:%M')}")
            # Simulate only returning data for the specified 15-min window
            if from_date.time() == dt_time(9, 15) and to_date.time() == dt_time(9, 30):
                return self._mock_historical_data.get(symbol_token, [])
            return []

    class MockRedisStoreForFetcher:
        def __init__(self):
            logger.info("MockRedisStoreForFetcher initialized.")
        def connect(self): return True
        def disconnect(self): pass
        def get_historical_data(self, *args, **kwargs): return None # Not used directly by this module for historical
        def set_historical_data(self, *args, **kwargs): pass # Not used directly by this module for historical
        def redis_client(self): # Mock redis_client attribute
            class MockRedisClient:
                def get(self, key): return None
                def setex(self, key, expiry, value): pass
                def delete(self, key): pass
            return MockRedisClient()


    # Create a dummy api-scrip-master.csv for testing
    dummy_dhan_data = {
        DHAN_SYMBOL_COLUMN_NAME: ['ITC', 'SBIN', 'RELIANCE', 'INFY', 'TCS', 'HDFCBANK', 'BHARTIARTL', 'MARUTI', 'ASIANPAINT', 'BAJFINANCE'],
        'instrument_type': ['EQ', 'EQ', 'EQ', 'EQ', 'EQ', 'EQ', 'EQ', 'EQ', 'EQ', 'EQ'],
        'exchange': ['NSE', 'NSE', 'NSE', 'NSE', 'NSE', 'NSE', 'NSE', 'NSE', 'NSE', 'NSE'],
        'segment': ['E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E', 'E'],
        'some_other_dhan_column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    df_dummy_dhan = pd.DataFrame(dummy_dhan_data)
    df_dummy_dhan.to_csv(SCRIP_MASTER_CSV_PATH, index=False)
    print(f"Created dummy '{SCRIP_MASTER_CSV_PATH}' for testing.")

    mock_angel_api = MockAngelOneAPIForFetcher()
    mock_redis_store = MockRedisStoreForFetcher()

    test_date = "2025-07-01"
    top_symbols = get_top_100_volume_gainers(mock_angel_api, mock_redis_store, test_date, top_n=3)
    
    print(f"\nTop 3 Volume Gainers for {test_date}:")
    for s in top_symbols:
        print(f"  Symbol: {s['symbol']}, Token: {s['token']}, Volume: {s['volume']:.0f}")

    # Clean up dummy CSV
    import os
    if os.path.exists(SCRIP_MASTER_CSV_PATH):
        os.remove(SCRIP_MASTER_CSV_PATH)
        print(f"Cleaned up dummy '{SCRIP_MASTER_CSV_PATH}'.")

    print("\n--- fetch_top_volume_gainers module Test End ---")
