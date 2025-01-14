import logging
from datetime import datetime


def prepend_datetime(s: str, delimiter: str = "_") -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f"{current_date}{delimiter}{s}"


# # Configure the root logger
# logging.basicConfig(
#     level=logging.WARNING,  # Set to DEBUG to capture all logs
#     format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
# )

# Shared console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set to WARNING for console output
console_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(filename)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
    )
)


def create_logger(name, log_file, logs_path):
    logger = logging.getLogger(name)
    logger.setLevel(
        logging.DEBUG
    )  # Ensure the logger captures all messages at DEBUG level and above
    if not logger.hasHandlers():  # Check if the logger already has handlers
        file_handler = logging.FileHandler(logs_path / log_file)
        file_handler.setLevel(logging.DEBUG)  # Set to DEBUG for file output
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(filename)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # avoid logs from being propagated to the root logger (so that they don't show in the notebook)
        logger.propagate = False
    return logger
