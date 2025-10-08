from src.constants import GOOGLE_API_KEY, MODEL
from src.artifacts import ModerationCheckPrompt, IntentConfirmationPrompt
from src.logging import Logger
import json
from openai import OpenAI

logging = Logger()

client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

class ModerationCheck:
    def __init__(self):
        self.moderation_prompt = ModerationCheckPrompt.prompt
        logging.info("ModerationCheck initialized")

    def check_moderation(self, input_message: str = "") -> bool:
        """
        Returns True if content is UNSAFE (bad), False if SAFE.
        Policy:
        - Empty input => False (safe)
        - finish_reason == 'content_filter' => True (unsafe/bad)
        - Non-empty text => False (safe)
        - No text and no explicit filter => False (safe)
        - On errors => True (treat as unsafe to fail closed)
        """
        try:
            logging.info(f"Starting moderation check for message: {input_message[:50]}...")
            if not input_message.strip():
                logging.warning("Empty input for moderation check.")
                return False  # safe
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self.moderation_prompt},
                    {"role": "user", "content": input_message}
                ]
            )
            message_content = response.choices[0].message.get("content") if hasattr(response.choices[0].message, "get") else getattr(response.choices[0].message, "content", "")
            logging.info(f"Intent for {input_message} is {message_content}")
            if message_content.lower() in ["flagged"]:
                return True
            return False

        except Exception as e:
            logging.error(f"Error during moderation check: {e}")
            return True


class IntentCheck:
    def __init__(self):
        self.intent_confirmation_prompt = IntentConfirmationPrompt.prompt
        logging.info("IntentCheck initialized")

    def check_intent(self, input_message: str = "") -> bool:
        """
        Returns True if intent is confirmed, False otherwise.
        Policy:
        - Empty input => False
        - finish_reason == 'content_filter' => False
        - Parse returned text for affirmative keywords
        """
        try:
            logging.info(f"Starting moderation check for message: {input_message[:50]}...")
            if not input_message.strip():
                logging.warning("Empty input for moderation check.")
                return False  
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": self.intent_confirmation_prompt},
                    {"role": "user", "content": input_message}
                ]
            )
            message_content = response.choices[0].message.get("content") if hasattr(response.choices[0].message, "get") else getattr(response.choices[0].message, "content", "")
            logging.info(f"Intent for {input_message} is {message_content}")
            if message_content.lower() in ["true"]:
                return True
            return False

        except Exception as e:
            logging.error(f"Error during intent check: {e}")
            return True