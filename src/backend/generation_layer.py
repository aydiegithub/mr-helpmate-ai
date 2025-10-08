from src.constants import GOOGLE_API_KEY, MODEL
from src.artifacts import SystemInstruction
from src.logging import Logger
from typing import Dict
import json
from openai import OpenAI

logging = Logger()

client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

class ChatCompletion():
    def __init__(self):
        self.system_instruction = SystemInstruction.prompt
        
    def chat_completion(self, messages: list[Dict[str, str]] = None) -> str:
        try:
            logging.info(f"Starting chat_completion")
            if not messages:
                logging.warning("Empty messages for chat_completion.")
                return ""
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            message_content = response.choices[0].message.get("content") if hasattr(response.choices[0].message, "get") else getattr(response.choices[0].message, "content", "")
            logging.info(f"Gemini response {message_content}")
            return message_content

        except Exception as e:
            logging.error(f"Error during moderation check: {e}")
            raise