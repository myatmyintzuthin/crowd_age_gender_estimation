"""
@file      : logger.py

@author    : Myat Myint Zu Thin
@date      : 2024/04/18
"""

import logging
from rich.logging import RichHandler

class Logger:

    def __init__(self) -> None:
        logging.basicConfig(level="NOTSET",format="%(message)s",datefmt="[%X]",handlers=[RichHandler(rich_tracebacks=True)])
        self.__logger = logging.getLogger('rich')

    def get_instance(self):
        return self.__logger
