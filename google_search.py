# google_search.py
# This module provides a mock implementation for a Google Search tool.
# In a real-world scenario, this would integrate with a search API (e.g., Google Custom Search API, SerpApi).
# For the purpose of this project, it simulates search results.

import logging
import time
from typing import List, Dict, Any, Optional, Union
import random  # For simulating diverse results
from datetime import datetime, timedelta
import re  # Added: Import re for regex parsing in SentimentAnalyzer fallback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot_log.log"),
              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --- Data Classes for Search Results (Matching Google Search Tool API) ---


class PerQueryResult:
    """Represents a single search result item."""

    def __init__(self,
                 index: Optional[str] = None,
                 publication_time: Optional[str] = None,
                 snippet: Optional[str] = None,
                 source_title: Optional[str] = None,
                 url: Optional[str] = None):
        self.index = index
        self.publication_time = publication_time
        self.snippet = snippet
        self.source_title = source_title
        self.url = url


class SearchResults:
    """Represents the results for a single search query."""

    def __init__(self,
                 query: Optional[str] = None,
                 results: Optional[List[PerQueryResult]] = None):
        self.query = query
        self.results = results if results is not None else []


# --- Mock Search Function ---


def search(queries: List[str] | None = None) -> List[SearchResults]:
    """
    Mocks Google Search results for testing purposes.
    In a real application, this would call a search API.

    Args:
        queries (List[str] | None): A list of search queries.

    Returns:
        List[SearchResults]: A list of SearchResults objects, one for each query.
    """
    if not queries:
        logger.warning("No queries provided for Google Search mock.")
        return []

    mock_responses = []
    for query in queries:
        logger.info(f"Simulating Google Search for query: '{query}'")
        time.sleep(0.5)  # Simulate network delay

        # Generate mock results based on the query, especially for stock news
        results_for_query = []
        if "stock news" in query.lower():
            symbol = query.replace("stock news", "").strip().upper()
            if not symbol:
                symbol = "GENERIC"  # Fallback if no specific symbol

            # Generate a few mock headlines
            mock_headlines = [
                f"{symbol} shares surge on strong Q1 earnings report.",
                f"Analyst upgrades {symbol} to 'Buy' amid market recovery.",
                f"{symbol} faces regulatory scrutiny over new product launch.",
                f"Global economic slowdown impacts {symbol}'s export outlook.",
                f"New partnership announced for {symbol} in renewable energy sector.",
                f"Dividend announcement boosts {symbol} investor confidence."
            ]

            # Select a random subset of headlines to simulate varied search results
            selected_headlines = random.sample(
                mock_headlines, min(len(mock_headlines), random.randint(2, 5)))

            for i, headline in enumerate(selected_headlines):
                sentiment_hint = ""
                if "surge" in headline or "upgrades" in headline or "boosts" in headline or "acquisition" in headline:
                    sentiment_hint = " (positive)"
                elif "scrutiny" in headline or "slowdown" in headline or "impacts" in headline:
                    sentiment_hint = " (negative)"

                results_for_query.append(
                    PerQueryResult(
                        index=str(i + 1),
                        publication_time=(
                            datetime.now() -
                            timedelta(days=random.randint(0, 7))).isoformat(),
                        snippet=headline +
                        sentiment_hint,  # Add hint for easier manual verification
                        source_title=f"News Source {random.randint(1, 5)}",
                        url=f"https://example.com/news/{symbol}/{i+1}"))
        else:
            # Generic mock results for other queries
            results_for_query.append(
                PerQueryResult(
                    index="1",
                    snippet=f"Top result for '{query}'.",
                    source_title="Wikipedia",
                    url=
                    f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}")
            )
            results_for_query.append(
                PerQueryResult(
                    index="2",
                    snippet=f"Another relevant article about '{query}'.",
                    source_title="News Site",
                    url=f"https://news.example.com/{query.replace(' ', '-')}"))

        mock_responses.append(
            SearchResults(query=query, results=results_for_query))

    logger.info(
        f"Simulated search completed. Returning {len(mock_responses)} search result sets."
    )
    return mock_responses


# Example Usage / Test
if __name__ == "__main__":
    print("--- Starting Google Search Mock Module Test ---")

    # Test a single query
    queries_1 = ["RELIANCE stock news"]
    results_1 = search(queries_1)
    print(f"\nResults for '{queries_1[0]}':")
    for res_set in results_1:
        print(f"  Query: {res_set.query}")
        for res in res_set.results:
            print(f"    - Snippet: {res.snippet} (Source: {res.source_title})")

    # Test multiple queries
    queries_2 = ["INFY stock news", "TCS latest updates"]
    results_2 = search(queries_2)
    print(f"\nResults for multiple queries:")
    for res_set in results_2:
        print(f"  Query: {res_set.query}")
        for res in res_set.results:
            print(f"    - Snippet: {res.snippet} (Source: {res.source_title})")

    # Test a non-stock query
    queries_3 = ["weather in London"]
    results_3 = search(queries_3)
    print(f"\nResults for '{queries_3[0]}':")
    for res_set in results_3:
        print(f"  Query: {res_set.query}")
        for res in res_set.results:
            print(f"    - Snippet: {res.snippet} (Source: {res.source_title})")

    print("\n--- Google Search Mock Module Test End ---")
