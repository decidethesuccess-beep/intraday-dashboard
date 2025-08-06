# ai_webhook.py
# This module acts as an AI webhook, sending structured feedback and queries
# to the LLM (Gemini) based on trading system events.
# It uses the LLMClient to interact with the Gemini API.

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Import LLMClient for interacting with the Gemini API
from llm_client import LLMClient

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

class AIWebhook:
    """
    Manages sending structured feedback and queries to the LLM (Gemini).
    """
    def __init__(self, llm_client: LLMClient):
        """
        Initializes the AIWebhook with an LLMClient instance.

        Args:
            llm_client (LLMClient): An instance of the LLMClient for AI processing.
        """
        self.llm_client = llm_client
        logger.info("AIWebhook initialized.")

    def _send_feedback_to_gemini(self,
                                 event_type: str,
                                 prompt_details: Dict[str, Any],
                                 expected_response_type: str = "text") -> Optional[str]:
        """
        Constructs a prompt and sends it to Gemini, returning the response.

        Args:
            event_type (str): Type of event (e.g., "TRADE_SUGGESTION", "TRADE_EXIT").
            prompt_details (Dict[str, Any]): Dictionary containing details for the prompt.
            expected_response_type (str): "text" for free-form, "json" for structured.

        Returns:
            Optional[str]: The Gemini's response, or None if an error occurs.
        """
        base_prompt = f"As an AI Trading Assistant, provide concise feedback for the following {event_type} event. "

        if event_type == "TRADE_SUGGESTION":
            symbol = prompt_details.get("symbol")
            direction = prompt_details.get("direction")
            ltp = prompt_details.get("ltp")
            ai_score = prompt_details.get("ai_score")
            sentiment = prompt_details.get("sentiment")
            current_active_positions = prompt_details.get("current_active_positions")
            max_active_positions = prompt_details.get("max_active_positions")
            
            prompt = (
                f"{base_prompt}A {direction} signal for {symbol} at LTP {ltp:.2f} was generated. "
                f"AI Score: {ai_score:.2f}, Sentiment: {sentiment}. "
                f"Current active positions: {current_active_positions}/{max_active_positions}. "
                f"Provide a brief 'AI Suggestion:' or 'AI Caution:' based on these details. "
                f"Focus on market context, potential risks, or confirmation of the signal strength."
            )
            # Define a simple schema for structured feedback if needed later, for now, free text.
            response_schema = None # For now, let LLM return free-form text.
            # If you want structured:
            # response_schema = {
            #     "type": "OBJECT",
            #     "properties": {
            #         "feedback_type": {"type": "STRING", "enum": ["Suggestion", "Caution", "Approval"]},
            #         "message": {"type": "STRING"}
            #     },
            #     "required": ["feedback_type", "message"]
            # }

        elif event_type == "TRADE_REJECTION":
            symbol = prompt_details.get("symbol")
            direction = prompt_details.get("direction")
            reason = prompt_details.get("reason")
            ai_score = prompt_details.get("ai_score")
            sentiment = prompt_details.get("sentiment")
            
            prompt = (
                f"{base_prompt}A {direction} trade for {symbol} was rejected. Reason: '{reason}'. "
                f"AI Score: {ai_score:.2f}, Sentiment: {sentiment}. "
                f"Provide a brief 'AI Feedback:' explaining the rejection or suggesting alternative actions."
            )
            response_schema = None

        elif event_type == "TRADE_EXIT":
            symbol = prompt_details.get("symbol")
            direction = prompt_details.get("direction")
            entry_price = prompt_details.get("entry_price")
            exit_price = prompt_details.get("exit_price")
            pnl = prompt_details.get("pnl")
            exit_reason = prompt_details.get("exit_reason")
            
            prompt = (
                f"{base_prompt}A {direction} trade for {symbol} entered at {entry_price:.2f} was exited at {exit_price:.2f}. "
                f"Realized PnL: {pnl:.2f}. Exit Reason: '{exit_reason}'. "
                f"Provide a brief 'AI Review:' on the outcome, lessons learned, or market conditions."
            )
            response_schema = None
        
        elif event_type == "MISSED_OPPORTUNITY":
            symbol = prompt_details.get("symbol")
            direction = prompt_details.get("direction")
            ltp = prompt_details.get("ltp")
            ai_score = prompt_details.get("ai_score")
            sentiment = prompt_details.get("sentiment")
            missed_reason = prompt_details.get("missed_reason")

            prompt = (
                f"{base_prompt}A potential {direction} trade for {symbol} at LTP {ltp:.2f} was a missed opportunity. "
                f"AI Score: {ai_score:.2f}, Sentiment: {sentiment}. Reason for missing: '{missed_reason}'. "
                f"Provide 'AI Insight:' on what could have been done or market dynamics."
            )
            response_schema = None

        else:
            logger.warning(f"Unknown event type for AI webhook: {event_type}. Skipping feedback.")
            return None

        logger.info(f"[AI Webhook] Triggered for event: {event_type} - Symbol: {prompt_details.get('symbol', 'N/A')}")
        logger.debug(f"[AI Webhook] Sending prompt to LLM:\n{prompt}")

        # Call LLMClient to generate text
        gemini_response = self.llm_client.generate_text(prompt, response_schema=response_schema)

        if gemini_response:
            logger.info(f"[AI Webhook] Received Gemini response for {event_type}: {gemini_response}")
            return gemini_response
        else:
            logger.error(f"[AI Webhook] Failed to get response from Gemini for {event_type}.")
            return None

    # --- Specific Feedback Methods for different events ---

    def send_entry_suggestion_feedback(self,
                                        symbol: str,
                                        direction: str,
                                        ltp: float,
                                        ai_score: float,
                                        sentiment: str,
                                        current_active_positions: int,
                                        max_active_positions: int):
        """Sends feedback when a trade entry signal is generated."""
        details = {
            "symbol": symbol,
            "direction": direction,
            "ltp": ltp,
            "ai_score": ai_score,
            "sentiment": sentiment,
            "current_active_positions": current_active_positions,
            "max_active_positions": max_active_positions
        }
        self._send_feedback_to_gemini("TRADE_SUGGESTION", details)

    def send_trade_rejection_feedback(self,
                                      symbol: str,
                                      direction: str,
                                      reason: str,
                                      ai_score: Optional[float],
                                      sentiment: Optional[str]):
        """Sends feedback when a trade is rejected."""
        details = {
            "symbol": symbol,
            "direction": direction,
            "reason": reason,
            "ai_score": ai_score,
            "sentiment": sentiment
        }
        self._send_feedback_to_gemini("TRADE_REJECTION", details)

    def send_trade_exit_feedback(self,
                                 symbol: str,
                                 direction: str,
                                 entry_price: float,
                                 exit_price: float,
                                 pnl: float,
                                 exit_reason: str):
        """Sends feedback when a trade is exited."""
        details = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "exit_reason": exit_reason
        }
        self._send_feedback_to_gemini("TRADE_EXIT", details)

    def send_missed_opportunity_feedback(self,
                                         symbol: str,
                                         direction: str,
                                         ltp: float,
                                         ai_score: Optional[float],
                                         sentiment: Optional[str],
                                         missed_reason: str):
        """Sends feedback for a missed trading opportunity."""
        details = {
            "symbol": symbol,
            "direction": direction,
            "ltp": ltp,
            "ai_score": ai_score,
            "sentiment": sentiment,
            "missed_reason": missed_reason
        }
        self._send_feedback_to_gemini("MISSED_OPPORTUNITY", details)


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    print("--- Starting AIWebhook Module Test ---")

    # Mock LLMClient for testing
    class MockLLMClientForWebhook:
        def generate_text(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None) -> str:
            print(f"\nMock LLM received prompt (first 100 chars): {prompt[:100]}...")
            if "TRADE_SUGGESTION" in prompt:
                return "AI Suggestion: RELIANCE looks strong. Consider a small position to start."
            elif "TRADE_REJECTION" in prompt:
                return "AI Feedback: Rejection due to cooldown is a good risk management. Wait for next signal."
            elif "TRADE_EXIT" in prompt:
                return "AI Review: Successful TSL exit for INFY. Good profit capture."
            elif "MISSED_OPPORTUNITY" in prompt:
                return "AI Insight: Missed opportunity on TCS due to max positions. Consider adjusting capital allocation."
            return "AI Response: Acknowledged."

    mock_llm_client = MockLLMClientForWebhook()
    ai_webhook = AIWebhook(mock_llm_client)

    # Test Trade Suggestion Feedback
    ai_webhook.send_entry_suggestion_feedback(
        symbol="RELIANCE",
        direction="BUY",
        ltp=2500.0,
        ai_score=0.85,
        sentiment="positive",
        current_active_positions=2,
        max_active_positions=10
    )

    # Test Trade Rejection Feedback
    ai_webhook.send_trade_rejection_feedback(
        symbol="INFY",
        direction="SELL",
        reason="Symbol on cooldown",
        ai_score=-0.75,
        sentiment="negative"
    )

    # Test Trade Exit Feedback
    ai_webhook.send_trade_exit_feedback(
        symbol="TCS",
        direction="BUY",
        entry_price=3400.0,
        exit_price=3450.0,
        pnl=500.0,
        exit_reason="TSL"
    )

    # Test Missed Opportunity Feedback
    ai_webhook.send_missed_opportunity_feedback(
        symbol="SBIN",
        direction="BUY",
        ltp=600.0,
        ai_score=0.9,
        sentiment="positive",
        missed_reason="Max active positions reached"
    )

    print("\n--- AIWebhook Module Test End ---")
