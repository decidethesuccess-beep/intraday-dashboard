# angelone_api_patch.py
# This module provides a patched AngelOne API client for robust login,
# session management, and API request handling with exponential backoff.

import logging
import os
import json
import time
from datetime import datetime, timedelta
from pyotp import TOTP # For TOTP generation
from SmartApi.smartConnect import SmartConnect # The official Angel One SmartAPI client - CORRECTED IMPORT
from SmartApi.smartExceptions import SmartAPIException, DataException # Import specific exceptions
import requests # Import requests for handling network errors in _make_api_request_with_retries
from typing import Any, Dict, Optional, List # ADDED: Import Any, Dict, Optional, List for type hinting
from dotenv import load_dotenv # Import load_dotenv

# Configure logging for this module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AngelOneAPI:
    def __init__(self, api_key_to_use: str = None):
        load_dotenv() # Ensure .env variables are loaded here
        self.api_key = api_key_to_use if api_key_to_use else os.getenv("ANGELONE_API_KEY")
        self.client_code = os.getenv("ANGELONE_CLIENT_CODE")
        self.password = os.getenv("ANGELONE_PASSWORD")
        self.totp_secret = os.getenv("ANGELONE_TOTP_SECRET")
        self.feed_token = None
        self.access_token = None
        self.refresh_token = None
        self.smart_api_client = None
        self.session_expiry = None # To store when the session will expire
        self.login_error_message = None # Added to store specific login errors

        # Exponential backoff settings
        self.MAX_RETRIES = 5
        self.INITIAL_BACKOFF_DELAY = 1 # seconds

        # Debugging: Confirm client_code loaded from .env
        logger.debug(f"DEBUG: AngelOneAPI initialized with client_code: '{self.client_code}'")
        # Added debug statement to check TOTP secret
        logger.debug(f"DEBUG: AngelOneAPI initialized. TOTP_SECRET value: '{self.totp_secret}'")


    def login(self) -> bool:
        """
        Logs into the Angel One API and establishes a session.
        Handles TOTP generation and session token storage.
        """
        self.login_error_message = None # Reset error message on new login attempt

        if self.smart_api_client and self.access_token and self.is_session_active():
            logger.info("Angel One session already active and valid.")
            return True

        logger.info("Attempting to log into Angel One API...")
        try:
            self.smart_api_client = SmartConnect(api_key=self.api_key)

            # Generate TOTP
            if not self.totp_secret:
                self.login_error_message = "ANGELONE_TOTP_SECRET not found in .env. Cannot generate TOTP."
                logger.critical(self.login_error_message)
                return False
            totp = TOTP(self.totp_secret)
            current_totp_pin = totp.now()
            logger.info("Generated TOTP PIN: %s", current_totp_pin)

            data = self.smart_api_client.generateSession(self.client_code, self.password, current_totp_pin)

            if data and data.get("status"):
                self.feed_token = data["data"]["feedToken"]
                self.access_token = data["data"]["jwtToken"]
                self.refresh_token = data["data"]["refreshToken"]

                # Set session expiry (Angel One tokens typically last 24 hours)
                self.session_expiry = datetime.now() + timedelta(hours=23) # Set expiry a bit before 24h
                
                logger.info("Successfully logged into Angel One API.")
                return True
            else:
                error_message = data.get("message", "Unknown error during login.")
                self.login_error_message = f"Angel One API responded with error: {error_message}. Full response: {data}"
                logger.critical("Failed to log into Angel One API: %s", self.login_error_message)
                return False
        except (SmartAPIException, DataException) as e:
            self.login_error_message = f"Angel One API exception during login: {e}"
            logger.critical(self.login_error_message, exc_info=True)
            return False
        except requests.exceptions.RequestException as e:
            self.login_error_message = f"Network or request error during Angel One login: {e}"
            logger.critical(self.login_error_message, exc_info=True)
            return False
        except Exception as e:
            self.login_error_message = f"An unexpected error occurred during Angel One login: {e}"
            logger.critical(self.login_error_message, exc_info=True)
            return False

    def logout(self) -> bool:
        """Logs out from the Angel One API."""
        if self.smart_api_client:
            try:
                logout_data = self.smart_api_client.terminateSession(self.client_code)
                if logout_data and logout_data.get("status"):
                    logger.info("Successfully logged out from Angel One API.")
                    self.feed_token = None
                    self.access_token = None
                    self.refresh_token = None
                    self.session_expiry = None
                    self.smart_api_client = None
                    return True
                else:
                    error_message = logout_data.get("message", "Unknown error during logout.")
                    logger.warning("Failed to log out from Angel One API: %s", error_message)
                    return False
            except Exception as e:
                logger.warning("An error occurred during Angel One logout: %s", e, exc_info=True)
                return False
        return True # Already logged out or never logged in

    def is_session_active(self) -> bool:
        """Checks if the current session token is still valid."""
        if not self.access_token or not self.session_expiry:
            return False
        return datetime.now() < self.session_expiry

    def _make_api_request_with_retries(self, method_name: str, *args, **kwargs) -> Any:
        """
        Internal helper to make API requests with exponential backoff and retries.
        """
        retries = 0
        delay = self.INITIAL_BACKOFF_DELAY

        while retries < self.MAX_RETRIES:
            try:
                if not self.smart_api_client:
                    logger.error("SmartApi client not initialized. Cannot make API request.")
                    return None

                # Call the actual SmartApi method dynamically
                method = getattr(self.smart_api_client, method_name)
                response = method(*args, **kwargs)

                # Check for Angel One specific error status
                if response and response.get('status') is False:
                    error_code = response.get('errorcode')
                    message = response.get('message', 'Unknown API error')
                    logger.warning(f"Angel One API returned error: {error_code} - {message}. Retrying...")
                    # For specific errors that indicate temporary issues, retry.
                    # AB1004 is 'Something Went Wrong, Please Try After Sometime'
                    # You might add other error codes here if they are transient.
                    if error_code in ['AB1004', 'AB1000']: # AB1000 is often a generic failure too
                        retries += 1
                        time.sleep(delay)
                        delay *= 2 # Exponential increase
                        continue # Retry the loop
                    else:
                        logger.error(f"Non-retryable Angel One API error: {error_code} - {message}. Aborting retries.")
                        return response # Return the error response
                
                return response # Successful response

            except (SmartAPIException, DataException) as e:
                logger.warning(f"SmartAPIException/DataException during API call ({method_name}): {e}. Retrying...")
                retries += 1
                time.sleep(delay)
                delay *= 2 # Exponential increase
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network/Request error during API call ({method_name}): {e}. Retrying...")
                retries += 1
                time.sleep(delay)
                delay *= 2 # Exponential increase
            except Exception as e:
                logger.error(f"Unexpected error during API call ({method_name}): {e}. Aborting retries.", exc_info=True)
                return None # Abort for unexpected errors

        logger.critical(f"API call ({method_name}) failed after {self.MAX_RETRIES} retries.")
        return None # Return None if all retries fail

    # --- Wrapper methods for common SmartApi calls ---
    # These methods will now use the retry mechanism

    def get_candle_data(self, params: Dict[str, Any]) -> Any:
        """Fetches candle data using SmartApi's getCandleData with retries."""
        return self._make_api_request_with_retries("getCandleData", params)

    def get_market_data(self, params: Dict[str, Any]) -> Any:
        """Fetches live market data using SmartApi's getMarketData with retries."""
        return self._make_api_request_with_retries("getMarketData", params)

    def get_all_instruments(self) -> Any:
        """Fetches all tradable instruments using SmartApi's get_all_instruments with retries."""
        # Note: SmartApi's get_all_instruments typically doesn't take params
        return self._make_api_request_with_retries("get_all_instruments")

    def get_feed_token(self) -> Optional[str]:
        """Returns the stored feed token."""
        return self.feed_token

    def get_jwt_token(self) -> Optional[str]:
        """Returns the stored JWT token (access token)."""
        return self.access_token

    def get_profile(self) -> Any:
        """Fetches the user profile using SmartApi's getProfile with retries."""
        return self._make_api_request_with_retries("getProfile")


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for detailed output during testing
    from dotenv import load_dotenv
    load_dotenv()

    print("--- Testing AngelOneAPI patch with retries ---")
    
    api_client = AngelOneAPI()
    if api_client.login():
        print("\nLogged in. Testing get_all_instruments with retries...")
        # This will now use the _make_api_request_with_retries method
        instruments = api_client.get_all_instruments() 
        if instruments:
            print(f"Successfully fetched {len(instruments)} instruments.")
            # print(instruments[0]) # Print first instrument to check format
        else:
            print("Failed to fetch instruments after retries.")

        print("\nTesting get_candle_data with retries (e.g., for RELIANCE)...")
        # Example params for getCandleData
        candle_params = {
            "exchange": "NSE",
            "symboltoken": "2885", # RELIANCE token
            "interval": "ONE_MINUTE",
            "fromdate": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d %H:%M"),
            "todate": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        candles = api_client.get_candle_data(candle_params)
        if candles:
            print(f"Successfully fetched {len(candles)} candles for RELIANCE.")
            # print(candles[0])
        else:
            print("Failed to fetch candles for RELIANCE after retries.")

        api_client.logout()
    else:
        print(f"Login failed. Cannot proceed with API tests. Error: {api_client.login_error_message}")
    
    print("\n--- AngelOneAPI patch Test End ---")
