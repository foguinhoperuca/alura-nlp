#!/usr/bin/env python3

import sys
import logging
from time import sleep

from termcolor import colored
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

LOG_FORMAT_FULL = colored('[%(asctime)s][%(process)d:%(processName)s]', 'green', attrs=['bold', 'dark']) + colored('[%(filename)s#%(funcName)s:%(lineno)d]', 'white', attrs=['bold', 'dark']) + colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
LOG_FORMAT_INFO = colored('[%(filename)s#%(funcName)s:%(lineno)d]', 'white', attrs=['bold', 'dark']) + colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
LOG_FORMAT_SIMPLE = colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
DEFAULT_LOGGER_NAME = "sentiments"
DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"

def get_logger_factory():
    """
        The logging module always return the same reference for a logger with
        same identification. Will use one logger for all app.
    """
    level = logging.DEBUG       # [WARN | INFO | DEBUG]
    logformat = LOG_FORMAT_FULL # [LOG_FORMAT_SIMPLE | LOG_FORMAT_INFO | LOG_FORMAT_FULL]

    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.setLevel(level)
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(logging.Formatter(logformat))
    c_handler.setLevel(level)
    logger.addHandler(c_handler)

    return logger

if __name__ == "__main__":
    logger = get_logger_factory()

    reviews = pd.read_csv("data/imdb-reviews-pt-br.csv")
    logger.info(reviews.head())
    
    train, train_test, train_class, test_class = train_test_split(reviews.text_pt, reviews.sentiment, random_state=42)

    # print(train)
    # print("-----------------------")
    # print("-----------------------")
    # print("")
    # sleep(5)

    # print(train_test)
    # print("-----------------------")
    # print("-----------------------")
    # print("")
    # sleep(5)

    # print(train_class)
    # print("-----------------------")
    # print("-----------------------")
    # print("")
    # sleep(5)

    # print(test_class)
    # print("-----------------------")
    # print("-----------------------")
    # print("")
    # sleep(5)

    # logistic_regression = LogisticRegression()
    # logistic_regression.fit(train, train_class)
    # accuracy = logist_regression.score(train_test, test_class)
    # print(accuracy)

    # print("-----------------------")
    # print("-----------------------")
    # print("")
    # print(reviews.text_pt[189])
    print("-----------------------")
    print("-----------------------")
    print("")
    print(reviews.sentiment.value_counts())

    classification = reviews["sentiment"].replace(["neg", "pos"], [0, 1])
    print("-----------------------")
    print("-----------------------")
    print("")
    print(classification)

    reviews["classification"] = classification

    print(reviews)

    sys.exit(0)
