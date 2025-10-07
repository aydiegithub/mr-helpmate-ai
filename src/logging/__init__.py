import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, name="mr_helpmate_logger"):
        # Ensure logs directory exists
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # Log file with date and time stamp
        log_filename = datetime.now().strftime('log_%Y-%m-%d_%H-%M-%S.log')
        log_path = os.path.join(log_dir, log_filename)

        # Configure logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler if not already added
        if not self.logger.hasHandlers():
            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

__all__ = ['Logger']