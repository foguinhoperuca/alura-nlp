#!/usr/bin/env python3

import sys
import logging
from typing import List
from time import sleep
from decimal import Decimal
import matplotlib.pyplot as plt

from termcolor import colored
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

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


logger = get_logger_factory()


def print_data(reviews, train, train_test, train_class, test_class):
    print(train)
    print("-----------------------")
    print("-----------------------")
    print("")
    sleep(5)

    print(train_test)
    print("-----------------------")
    print("-----------------------")
    print("")
    sleep(5)

    print(train_class)
    print("-----------------------")
    print("-----------------------")
    print("")
    sleep(5)

    print(test_class)
    print("-----------------------")
    print("-----------------------")
    print("")
    sleep(5)

    print(reviews.head())
    print(reviews.text_pt[189])
    print("-----------------------")
    print("-----------------------")
    print("")

def simple_example():
    text = ["Assisti um filme ótimo", "Assisti um filme ruim"]

    bw = vectorizer.fit_transform(text)
    logger.info("Bag of words:")
    print(bw)
    print("-----------------------")
    print("-----------------------")
    print("")
    logger.info(vectorizer.get_feature_names_out())

    sparse_matrix = pd.DataFrame.sparse.from_spmatrix(bw, columns=vectorizer.get_feature_names_out())
    print(sparse_matrix)
    print("-----------------------")
    print("-----------------------")
    print("")

    breakpoint()

def classify_text(text: pd.DataFrame, column_text: str, column_classification, max_features: int = 50, random_state: int = 42) -> Decimal:
    """
    Without set max_features the accuracy is 0.8814395471087748. max_features=50 was recomended by tutor. Accuracy is 0.6583097452486858 with max_features=50
    """
    vectorizer = CountVectorizer(lowercase=False, max_features=max_features) if max_features is not None else CountVectorizer(lowercase=False)
    bag_of_words = vectorizer.fit_transform(text[column_text])
    train, train_test, train_class, test_class = train_test_split(bag_of_words, text[column_classification], random_state=random_state)
    logistic_regression = LogisticRegression() # default is: solver='lbfgs'
    logistic_regression.fit(train, train_class)
    accuracy = logistic_regression.score(train_test, test_class)

    logger.info(f"Accuracy found was: {accuracy}")

    return accuracy

def text_cloud_image(reviews: pd.DataFrame, width: int = 1900, height: int = 1500, max_font_size: int = 110):
    all_words: str = ' '.join([text for text in reviews.text_pt])
    word_cloud: WordCloud = WordCloud(width=width, height=height, max_font_size=max_font_size).generate(all_words)
    plt.figure(figsize=(20,15))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.show()


if __name__ == "__main__":
    """
    Vocabulary:
    * Train - teach ML model.
    * Test - use model to our purpose.
    """

    reviews = pd.read_csv("data/imdb-reviews-pt-br.csv")
    count_reviews = reviews.sentiment.value_counts()
    logger.info(f"Sentiment value count --> neg: {count_reviews['neg']} -- pos: {count_reviews['pos']}")
    reviews["classification"] = reviews["sentiment"].replace(["neg", "pos"], [0, 1])

    # print("-----------------------")
    # print("-----------------------")
    # print("")

    print(f"Classified text is: {classify_text(reviews, 'text_pt', 'classification')}")

    text_cloud_image(reviews=reviews)
    text_cloud_image(reviews=reviews.query("sentiment == 'pos'"))
    text_cloud_image(reviews=reviews.query("sentiment == 'neg'"))

    sys.exit(0)
