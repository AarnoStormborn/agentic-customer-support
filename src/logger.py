import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# Ensure logs directory exists
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define log file path
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Create a rotating file handler to manage log rotation
file_handler = TimedRotatingFileHandler(LOG_FILE_PATH, when="midnight", interval=1, backupCount=7)
file_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s %(module)s:%(lineno)d - %(message)s"))
file_handler.setLevel(logging.INFO)

# Create a console handler to print logs to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s %(module)s:%(lineno)d - %(message)s"))
console_handler.setLevel(logging.INFO)

# Set up logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=[file_handler, console_handler])

# Export logger
logger = logging.getLogger(__name__)

if __name__=="__main__":
    logger.info("Logger Setup")