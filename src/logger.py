import logging
import os
from typing import Optional

def get_logger(
    name: str,
    filename: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    :param name: name of the logger
    :param filename: file path to log to
    :param level: logging level
    :return: logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.FileHandler(filename, mode='w', encoding='utf-8')
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
