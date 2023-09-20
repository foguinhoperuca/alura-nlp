from typing import Any
import logging
import pprint

from termcolor import colored


LOG_FORMAT_FULL = colored('[%(asctime)s][%(process)d:%(processName)s]', 'green', attrs=['bold', 'dark']) + colored('[%(filename)s#%(funcName)s:%(lineno)d]', 'white', attrs=['bold', 'dark']) + colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
LOG_FORMAT_INFO = colored('[%(filename)s#%(funcName)s:%(lineno)d]', 'white', attrs=['bold', 'dark']) + colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
LOG_FORMAT_SIMPLE = colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
DEFAULT_LOGGER_NAME = "sentiments"
DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"
spp = pprint.PrettyPrinter(indent=4)


def initialize_logger(level: Any = logging.DEBUG, logformat: Any = LOG_FORMAT_FULL):
    """
    Initialize logger to console with default options.
    - level: [WARN | INFO | DEBUG]
    - logformat: [LOG_FORMAT_SIMPLE | LOG_FORMAT_INFO | LOG_FORMAT_FULL]
    """

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.setLevel(level)
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(logging.Formatter(logformat))
    c_handler.setLevel(level)
    logger.addHandler(c_handler)

    return logger

def get_logger_factory(name: str = DEFAULT_LOGGER_NAME):
    """The logging module always return the same reference for a
    logger with same identification. Will use one logger for all
    app."""

    return logging.getLogger(name)
