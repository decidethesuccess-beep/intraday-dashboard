# llm_client.py
# This module provides a client for interacting with the Gemini API for text generation.

import os
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional # Import Optional

# Configure logging for the module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("bot_log.log"), # Log to file
                        logging.StreamHandler() # Log to console
                    ])
logger = logging.getLogger(__name__)

class LLMClient:
    """
    A client to interact with the Gemini API for text generation.
    """
    def __init__(self):
        """
        Initializes the LLMClient.
        The API key is typically handled by the Canvas environment for Gemini models.
        """
        load_dotenv() # Load environment variables from .env file
        # For Gemini API calls within the Canvas environment, the API key is often
        # automatically provided or not strictly required in the code if using default models.
        # However, for external testing or if you were to use a different model,
        # you might need to uncomment and set an API_KEY environment variable.
        # self.api_key = os.getenv("GEMINI_API_KEY", "") # Example: if you had a specific API key for Gemini

        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        logger.info("LLMClient initialized for Gemini 2.0 Flash.")

    def generate_text(self, prompt: str, response_schema: Optional[Dict[str, Any]] = None) -> str | None:
        """
        Sends a prompt to the Gemini API and returns the generated text.

        Args:
            prompt (str): The text prompt to send to the LLM.
            response_schema (Optional[Dict[str, Any]]): An optional JSON schema
                                                         to guide the LLM's response structure.

        Returns:
            str | None: The generated text (or JSON string) from the LLM, or None if an error occurs.
        """
        chat_history = []
        chat_history.append({"role": "user", "parts": [{"text": prompt}]})

        payload: Dict[str, Any] = {
            "contents": chat_history
        }

        if response_schema:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }

        headers = {'Content-Type': 'application/json'} # Default for Canvas environment

        try:
            logger.info(f"Sending prompt to LLM: '{prompt[:50]}...'") # Log first 50 chars of prompt
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            result = response.json()

            if result.get('candidates') and len(result['candidates']) > 0 and \
               result['candidates'][0].get('content') and \
               result['candidates'][0]['content'].get('parts') and \
               len(result['candidates'][0]['content']['parts']) > 0:
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                logger.info("Successfully received response from LLM.")
                return generated_text
            else:
                logger.warning(f"LLM response structure unexpected or content missing: {result}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network or API error during LLM call: {e}", exc_info=True)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response from LLM: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM text generation: {e}", exc_info=True)
            return None

# Example usage (for testing purposes)
if __name__ == "__main__":
    print("--- Testing LLMClient Module ---")
    llm_client = LLMClient()

    # Test with a simple text prompt
    test_prompt = "What is the capital of France?"
    print(f"\nPrompt: {test_prompt}")
    response_text = llm_client.generate_text(test_prompt)
    if response_text:
        print(f"LLM Response: {response_text}")
    else:
        print("Failed to get a response from the LLM.")

    # Test with a prompt requiring structured JSON output
    test_structured_prompt = "Provide a recipe for a simple chocolate chip cookie. Include 'recipeName' and 'ingredients' (list of strings)."
    test_schema = {
        "type": "OBJECT",
        "properties": {
            "recipeName": {"type": "STRING"},
            "ingredients": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": ["recipeName", "ingredients"]
    }
    print(f"\nPrompt (Structured): {test_structured_prompt}")
    structured_response_text = llm_client.generate_text(test_structured_prompt, response_schema=test_schema)
    if structured_response_text:
        print(f"LLM Structured Response: {structured_response_text}")
        try:
            parsed_json = json.loads(structured_response_text)
            print(f"Parsed JSON: {json.dumps(parsed_json, indent=2)}")
        except json.JSONDecodeError:
            print("Failed to parse structured response as JSON.")
    else:
        print("Failed to get a structured response from the LLM.")


    print("--- LLMClient Module Test End ---")
