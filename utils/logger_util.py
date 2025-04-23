import os
import logging
from datetime import datetime
import glob

def get_logger(base_name: str = "app", max_logs: int = 3) -> logging.Logger:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    _cleanup_old_logs(log_dir, base_name, max_logs)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_file_name = os.path.join(log_dir, f"{base_name}_log_{timestamp}.log")

    logger = logging.getLogger(base_name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_name)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
        # Add a wrapper to log error and print message
        original_error = logger.error
        def error_with_print(msg, *args, **kwargs):
            print("#################### An error was logged! ####################")
            return original_error(msg, *args, **kwargs)
        logger.error = error_with_print

    logger.propagate = False
    return logger

# def get_logger(base_name: str = "app", max_logs: int = 3) -> logging.Logger:
#     log_dir = "logs"
#     os.makedirs(log_dir, exist_ok=True)

#     # Cleanup old logs
#     _cleanup_old_logs(log_dir, base_name, max_logs)

#     # Create new log file
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
#     log_file_name = os.path.join(log_dir, f"{base_name}_log_{timestamp}.log")

#     logger = logging.getLogger(base_name)
#     logger.setLevel(logging.DEBUG)

#     if not logger.handlers:
#         file_handler = logging.FileHandler(log_file_name)
#         formatter = logging.Formatter(
#             "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#         )
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)
#     logger.propagate = False
#     return logger

def _cleanup_old_logs(log_dir: str, base_name: str, max_logs: int):
    pattern = os.path.join(log_dir, f"{base_name}_log_*.log")
    log_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    for old_file in log_files[max_logs:]:
        try:
            os.remove(old_file)
        except Exception as e:
            print(f"Warning: could not delete old log {old_file}: {e}")
