# sentiment_analyzer.py
# This module is responsible for fetching real-time news, analyzing its sentiment
# and generating an AI score using an LLM, and then storing these results in Redis.

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import time  # For simulating delays/rate limiting
import re  # Still needed for heuristic fallback if LLM doesn't return perfect JSON
import os  # Added to check for subscribed_symbols.json

# Import necessary components
from llm_client import LLMClient
from redis_store import RedisStore
import google_search  # Import the mock Google Search tool
# NEW: Import AIWebhook for direct testing in __main__
from ai_webhook import AIWebhook

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_log.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ])
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes market sentiment and generates AI scores for symbols based on news.
    It uses Google Search to fetch news and an LLM to process it.
    """

    def __init__(self, llm_client: LLMClient, redis_store: RedisStore):
        """
        Initializes the SentimentAnalyzer.

        Args:
            llm_client (LLMClient): An instance of the LLMClient for AI processing.
            redis_store (RedisStore): An instance of RedisStore for data persistence.
        """
        self.llm_client = llm_client
        self.redis_store = redis_store
        self.sentiment_cache_expiry_seconds = 3600  # Cache sentiment for 1 hour
        self.ANALYSIS_INTERVAL_SECONDS = int(
            os.getenv("SENTIMENT_ANALYSIS_INTERVAL_SECONDS",
                      300))  # Default 5 minutes

        logger.info("SentimentAnalyzer initialized.")

    def _get_news_snippets(self, symbol: str) -> List[str]:
        """
        Fetches relevant news snippets for a given symbol using the Google Search tool.
        """
        query = f"{symbol} stock news"
        logger.info(f"Fetching news for {symbol} with query: '{query}'")
        try:
            search_results = google_search.search(queries=[query])
            snippets = []
            if search_results and search_results[0].results:
                for res in search_results[0].results:
                    if res.snippet:
                        snippets.append(res.snippet)
            logger.info(f"Found {len(snippets)} news snippets for {symbol}.")
            return snippets
        except Exception as e:
            logger.error(
                f"Error fetching news for {symbol} using Google Search: {e}",
                exc_info=True)
            return []

    def _analyze_with_llm(self, symbol: str,
                          news_snippets: List[str]) -> Dict[str, Any]:
        """
        Sends news snippets to the LLM to get an AI score and sentiment.

        Args:
            symbol (str): The trading symbol.
            news_snippets (List[str]): A list of news snippets related to the symbol.

        Returns:
            Dict[str, Any]: A dictionary containing 'ai_score' (float) and 'sentiment' (str),
                            or default values if analysis fails.
        """
        if not news_snippets:
            logger.warning(
                f"No news snippets provided for LLM analysis for {symbol}. Returning neutral."
            )
            return {"ai_score": 0.0, "sentiment": "neutral"}

        # Construct a prompt for the LLM
        prompt = (
            f"Analyze the following news snippets for {symbol} and provide a single AI score "
            f"between -1.0 (strongly bearish) and 1.0 (strongly bullish), and a sentiment "
            f"('positive', 'negative', or 'neutral').\n\n"
            f"News Snippets for {symbol}:\n" + "\n".join(news_snippets) +
            f"\n\nRespond in JSON format with 'ai_score' (float) and 'sentiment' (string)."
        )

        # Define the expected JSON schema for the LLM response
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "ai_score": {
                    "type": "NUMBER"
                },
                "sentiment": {
                    "type": "STRING",
                    "enum": ["positive", "negative", "neutral"]
                }
            },
            "required": ["ai_score", "sentiment"]
        }

        try:
            logger.info(
                f"Sending news for {symbol} to LLM for sentiment analysis...")
            # Pass the response_schema to the LLMClient's generate_text method
            llm_raw_response = self.llm_client.generate_text(
                prompt, response_schema=response_schema)

            if llm_raw_response:
                # Attempt to parse the JSON response
                try:
                    llm_output = json.loads(llm_raw_response)
                    ai_score = float(llm_output.get("ai_score", 0.0))
                    sentiment = llm_output.get("sentiment", "neutral").lower()

                    # Basic validation of sentiment
                    if sentiment not in ["positive", "negative", "neutral"]:
                        sentiment = "neutral"  # Default to neutral if invalid

                    logger.info(
                        f"LLM analysis for {symbol}: AI Score={ai_score:.2f}, Sentiment={sentiment}"
                    )
                    return {"ai_score": ai_score, "sentiment": sentiment}
                except json.JSONDecodeError:
                    logger.error(
                        f"LLM response for {symbol} was not valid JSON: {llm_raw_response}. Attempting regex parse as fallback."
                    )
                    # Fallback for non-JSON response: try to extract score/sentiment heuristically
                    ai_score_match = re.search(
                        r"\"ai_score\":\s*(-?\d+\.?\d*)", llm_raw_response)
                    sentiment_match = re.search(
                        r"\"sentiment\":\s*\"(positive|negative|neutral)\"",
                        llm_raw_response)

                    ai_score = float(
                        ai_score_match.group(1)) if ai_score_match else 0.0
                    sentiment = sentiment_match.group(
                        1) if sentiment_match else "neutral"

                    logger.warning(
                        f"Heuristic analysis for {symbol}: AI Score={ai_score:.2f}, Sentiment={sentiment}"
                    )
                    return {"ai_score": ai_score, "sentiment": sentiment}
            else:
                logger.warning(
                    f"LLM returned no response for {symbol}. Returning neutral."
                )
                return {"ai_score": 0.0, "sentiment": "neutral"}
        except Exception as e:
            logger.error(f"Error during LLM analysis for {symbol}: {e}",
                         exc_info=True)
            return {"ai_score": 0.0, "sentiment": "neutral"}

    def analyze_and_store_sentiment(self, symbol: str):
        """
        Orchestrates fetching news, analyzing sentiment/score, and storing in Redis.
        """
        logger.info(f"Starting sentiment analysis for {symbol}...")

        # Check if sentiment data is already cached and fresh
        cached_score = self.redis_store.redis_client.get(f"score:{symbol}")
        cached_sentiment = self.redis_store.redis_client.get(
            f"sentiment:{symbol}")
        last_update_time_bytes = self.redis_store.redis_client.get(
            f"sentiment_last_update:{symbol}")  # Get as bytes

        # Decode bytes to string, or set to None if not found
        # This is the line that caused the TypeError in main.py
        last_update_time_str = last_update_time_bytes.decode(
            'utf-8') if last_update_time_bytes else None

        if cached_score and cached_sentiment and last_update_time_str:
            try:
                # last_update_time_str is now guaranteed to be a string or None
                last_update_dt = datetime.fromisoformat(last_update_time_str)
                if (datetime.now() - last_update_dt
                    ).total_seconds() < self.sentiment_cache_expiry_seconds:
                    logger.info(
                        f"Sentiment for {symbol} found in cache and is fresh. Skipping re-analysis."
                    )
                    return  # No need to re-analyze
                else:
                    logger.info(
                        f"Cached sentiment for {symbol} is stale. Re-analyzing."
                    )
            except (ValueError, UnicodeDecodeError):
                logger.warning(
                    f"Invalid timestamp in sentiment cache for {symbol}. Re-analyzing."
                )
            # If any of cached_score, cached_sentiment, last_update_time_str is None/False,
            # or if the cache is stale/invalid, the code continues here to re-analyze.
        else:
            logger.info(
                f"No fresh sentiment data in cache for {symbol}. Proceeding with analysis."
            )

        news_snippets = self._get_news_snippets(symbol)
        analysis_result = self._analyze_with_llm(symbol, news_snippets)

        ai_score = analysis_result.get("ai_score")
        sentiment = analysis_result.get("sentiment")

        if ai_score is not None and sentiment is not None:
            # Store in Redis with an expiry
            self.redis_store.redis_client.setex(
                f"score:{symbol}", self.sentiment_cache_expiry_seconds,
                str(ai_score))
            self.redis_store.redis_client.setex(
                f"sentiment:{symbol}", self.sentiment_cache_expiry_seconds,
                sentiment)
            self.redis_store.redis_client.setex(
                f"sentiment_last_update:{symbol}",
                self.sentiment_cache_expiry_seconds,
                datetime.now().isoformat())
            logger.info(
                f"Stored AI Score ({ai_score:.2f}) and Sentiment ({sentiment}) for {symbol} in Redis."
            )
        else:
            logger.error(
                f"Failed to get valid AI score or sentiment for {symbol}. Not storing in Redis."
            )

    def run_sentiment_analysis_loop(self):
        """
        Continuously runs the sentiment analysis for all subscribed symbols.
        """
        logger.info("Starting sentiment analysis loop...")
        while True:
            logger.info("\n--- Sentiment Analysis Loop Iteration ---")

            # Load symbols from subscribed_symbols.json
            subscribed_symbols = []
            symbols_file_path = "subscribed_symbols.json"
            if os.path.exists(symbols_file_path):
                try:
                    with open(symbols_file_path, 'r') as f:
                        # Expecting a list of dictionaries with 'symbol' key
                        config_data = json.load(f)
                        subscribed_symbols = [
                            item['symbol'] for item in config_data
                            if 'symbol' in item
                        ]
                    logger.debug(
                        f"Loaded {len(subscribed_symbols)} symbols for sentiment analysis."
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Error decoding JSON from {symbols_file_path}: {e}. Skipping sentiment analysis for this iteration."
                    )
                    subscribed_symbols = []
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred loading symbols from {symbols_file_path}: {e}. Skipping sentiment analysis for this iteration."
                    )
                    subscribed_symbols = []
            else:
                logger.warning(
                    f"'{symbols_file_path}' not found. Cannot perform sentiment analysis without subscribed symbols."
                )

            if not subscribed_symbols:
                logger.info(
                    "No symbols to analyze. Waiting for next iteration.")
            else:
                for symbol in subscribed_symbols:
                    self.analyze_and_store_sentiment(symbol)
                    time.sleep(
                        1
                    )  # Small delay between symbol analyses to avoid API rate limits

            logger.info(
                f"Sentiment analysis loop sleeping for {self.ANALYSIS_INTERVAL_SECONDS} seconds..."
            )
            time.sleep(self.ANALYSIS_INTERVAL_SECONDS
                       )  # Wait before the next full cycle


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    print("--- Starting SentimentAnalyzer Module Test ---")

    # Mock LLMClient for testing
    class MockLLMClient:

        def generate_text(
                self,
                prompt: str,
                response_schema: Optional[Dict[str, Any]] = None) -> str:
            # Simulate LLM response based on prompt content
            if "RELIANCE" in prompt:
                if "surge" in prompt:
                    return '{"ai_score": 0.9, "sentiment": "positive"}'
                elif "slowdown" in prompt:
                    return '{"ai_score": -0.8, "sentiment": "negative"}'
                else:
                    return '{"ai_score": 0.1, "sentiment": "neutral"}'
            elif "INFY" in prompt:
                return '{"ai_score": 0.75, "sentiment": "positive"}'
            elif "TCS" in prompt:  # Added for TCS
                return '{"ai_score": 0.6, "sentiment": "positive"}'
            elif "HDFC_BANK" in prompt:  # Added for HDFC_BANK
                return '{"ai_score": -0.4, "sentiment": "negative"}'
            else:
                return '{"ai_score": 0.0, "sentiment": "neutral"}'

    # Mock RedisStore for testing
    class MockRedisStore:

        def __init__(self):
            self.data = {}
            logger.info("MockRedisStore initialized.")

        def connect(self):
            return True

        def disconnect(self):
            pass

        @property  # Use property decorator to make redis_client accessible like an attribute
        def redis_client(self):

            class MockRedisClient:

                def __init__(self, parent_data):
                    self.parent_data = parent_data

                def get(self, key):
                    # Simulate real redis-py behavior: return bytes if it exists, else None
                    val = self.parent_data.get(key)
                    return val.encode('utf-8') if isinstance(val, str) else val

                def setex(self, key, expiry, value):
                    self.parent_data[
                        key] = value  # Simplified, no actual expiry
                    logger.debug(
                        f"Mock Redis SETEX: {key} = {value} (Expiry: {expiry})"
                    )

                def delete(self, key):
                    if key in self.parent_data:
                        del self.parent_data[key]

            return MockRedisClient(self.data)

        # Mock specific RedisStore methods used by other modules
        def write_ltp(self, symbol: str, ltp: float):
            pass

        def read_ltp(self, symbol: str) -> float | None:
            return 100.0  # Dummy LTP

        def is_on_cooldown(self, symbol: str) -> bool:
            return False

        def set_cooldown_timer(self, symbol: str, duration_seconds: int):
            pass

    mock_llm_client = MockLLMClient()
    mock_redis_store = MockRedisStore()
    sentiment_analyzer = SentimentAnalyzer(mock_llm_client, mock_redis_store)

    # Create a dummy subscribed_symbols.json for testing the loop
    dummy_symbols_config = [
        {
            "exchangeType": 1,
            "token": "11536",
            "symbol": "RELIANCE"
        },
        {
            "exchangeType": 1,
            "token": "10604",
            "symbol": "INFY"
        },
        {
            "exchangeType": 1,
            "token": "3045",
            "symbol": "TCS"
        },
        {
            "exchangeType": 1,
            "token": "3432",
            "symbol": "HDFC_BANK"
        },
    ]
    with open("subscribed_symbols.json", "w") as f:
        json.dump(dummy_symbols_config, f, indent=4)
    print("Created dummy 'subscribed_symbols.json' for testing.")

    # Test analysis for RELIANCE (positive scenario)
    print("\n--- Testing RELIANCE (Positive News) ---")
    # To simulate positive news, the mock google_search needs to return it.
    # For this test, we rely on the mock_llm_client to interpret "RELIANCE" in prompt as positive.
    sentiment_analyzer.analyze_and_store_sentiment("RELIANCE")
    # Decode bytes for printing (as mock_redis_store now returns bytes)
    print(
        f"RELIANCE AI Score in Redis: {mock_redis_store.redis_client.get('score:RELIANCE').decode('utf-8')}"
    )
    print(
        f"RELIANCE Sentiment in Redis: {mock_redis_store.redis_client.get('sentiment:RELIANCE').decode('utf-8')}"
    )

    # Test analysis for INFY
    print("\n--- Testing INFY ---")
    sentiment_analyzer.analyze_and_store_sentiment("INFY")
    print(
        f"INFY AI Score in Redis: {mock_redis_store.redis_client.get('score:INFY').decode('utf-8')}"
    )
    print(
        f"INFY Sentiment in Redis: {mock_redis_store.redis_client.get('sentiment:INFY').decode('utf-8')}"
    )

    # Test cache mechanism
    print("\n--- Testing RELIANCE again (should be from cache) ---")
    sentiment_analyzer.analyze_and_store_sentiment(
        "RELIANCE")  # Should log "found in cache"

    # NEW: Direct test call to AIWebhook
    print("\n--- Testing direct AIWebhook call ---")
    # Initialize AIWebhook with the mock LLMClient
    test_ai_webhook = AIWebhook(mock_llm_client)
    test_ai_webhook.send_entry_suggestion_feedback(symbol="INFY",
                                                   direction="BUY",
                                                   ltp=1600.0,
                                                   ai_score=0.72,
                                                   sentiment="neutral",
                                                   current_active_positions=1,
                                                   max_active_positions=10)
    print("--- Direct AIWebhook call completed ---")

    # Test the run_sentiment_analysis_loop (will run for a few iterations)
    print(
        "\n--- Testing run_sentiment_analysis_loop (will run for a few seconds) ---"
    )
    # To avoid indefinite loop during test, we'll run it in a separate thread and stop it
    import threading
    stop_event = threading.Event()

    def run_loop_for_test():
        sentiment_analyzer.ANALYSIS_INTERVAL_SECONDS = 5  # Shorten interval for test
        sentiment_analyzer.run_sentiment_analysis_loop()

    loop_thread = threading.Thread(target=run_loop_for_test, daemon=True)
    loop_thread.start()

    # Let it run for a couple of iterations
    time.sleep(15)  # Wait for 2-3 iterations (5s interval + 1s per symbol)

    # In a real scenario, the main application loop would manage stopping this thread.
    # For this test, we'll just let the daemon thread terminate with the main process.

    # Clean up dummy CSV
    if os.path.exists("subscribed_symbols.json"):
        os.remove("subscribed_symbols.json")
        print("Cleaned up dummy 'subscribed_symbols.json'.")

    print("--- SentimentAnalyzer Module Test End ---")
