"""日志模块"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "shopmind", level: str = "INFO", log_file: str = None, log_format: str = None) -> logging.Logger:
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = setup_logger()
