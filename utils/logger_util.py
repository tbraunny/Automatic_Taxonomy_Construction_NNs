import os
import logging
from datetime import datetime

def get_logger(base_name: str = "app") -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file_name = os.path.join(log_dir, f"{base_name}_log_{timestamp}.log")

    logger = logging.getLogger(base_name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if this logger has already been configured
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optional: Add stream handler for console output too
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)

    return logger
