# portfolio_optimization_app/utils.py

import logging
import sys

def setup_logger(name):
    """通用日志设置函数"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(f'[{name}] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger
