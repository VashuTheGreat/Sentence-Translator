import logging
import os
from logging.handlers import RotatingFileHandler

from from_root import from_root

from datetime import datetime

LOG_DIR = 'logs'
# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE=f"log.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# construct log file path

log_dir_path=os.path.join(from_root(),LOG_DIR)

os.makedirs(log_dir_path,exist_ok=True)

log_file_path=os.path.join(log_dir_path,LOG_FILE)


def configure_logger():
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Defining formate
    formater = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # file handleing with rotation
    file_handler=RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT,encoding='utf-8')
    file_handler.setFormatter(formater)
    file_handler.setLevel(logging.DEBUG)

    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formater)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Suppress verbose third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("fsspec").setLevel(logging.WARNING)

    # return logger

configure_logger()    