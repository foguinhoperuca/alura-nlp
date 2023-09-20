#!/usr/bin/env python3

import sys
import logging
from typing import List
from time import sleep
from decimal import Decimal
import matplotlib.pyplot as plt
import argparse
import pprint
import warnings

from termcolor import colored
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk import FreqDist, tokenize, corpus
import seaborn as sns

LOG_FORMAT_FULL = colored('[%(asctime)s][%(process)d:%(processName)s]', 'green', attrs=['bold', 'dark']) + colored('[%(filename)s#%(funcName)s:%(lineno)d]', 'white', attrs=['bold', 'dark']) + colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
LOG_FORMAT_INFO = colored('[%(filename)s#%(funcName)s:%(lineno)d]', 'white', attrs=['bold', 'dark']) + colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
LOG_FORMAT_SIMPLE = colored('[%(levelname)s]', 'magenta', attrs=['bold', 'dark']) + ' %(message)s'
DEFAULT_LOGGER_NAME = "sentiments"
DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"
spp = pprint.PrettyPrinter(indent=4)
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")


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


def print_data(reviews, train, train_test, train_class, test_class) -> None:
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

    count_reviews = reviews.sentiment.value_counts()
    logger.info(f"Sentiment value count --> neg: {count_reviews['neg']} -- pos: {count_reviews['pos']}")

def classify_simple_example() -> None: 
    text = ["Assisti um filme ótimo", "Assisti um filme ruim"]

    vectorizer = CountVectorizer(lowercase=False)
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


def focus_stopwords() -> None:
    focus = reviews.iloc[:5, :]
    print(focus)
    print(".......................")
    focus["pre-processed_removed-stopwords"] = remove_stopwords(opnions=focus.text_pt)
    print(focus)
    spp.pprint(focus.iloc[2:3, 2].tolist())
    print(".......................")
    spp.pprint(focus.iloc[2:3, 5].tolist())


def remove_stopwords_by_tutor_style(opnions: pd.core.series.Series) -> List[str]:
    # Original implementation of tutor
    meaningless_words = corpus.stopwords.words("portuguese")
    wt = tokenize.WhitespaceTokenizer()
    processed_phrase = []
    for opnion in opnions:
        words = wt.tokenize(opnion)
        opnion_without_stopwords = []
        for word in words:
            if word not in meaningless_words:
                opnion_without_stopwords.append(word)

        processed_phrase.append(' '.join(opnion_without_stopwords))

    return processed_phrase


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


def text_cloud_image(reviews: pd.DataFrame, width: int = 1900, height: int = 1500, max_font_size: int = 110, collocations: bool = False) -> None:
    all_words: str = ' '.join([text for text in reviews.text_pt])
    word_cloud: WordCloud = WordCloud(width=width, height=height, max_font_size=max_font_size, collocations=collocations).generate(all_words)
    plt.figure(figsize=(20,15))
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def pareto(phrase: str, figsize: tuple = (12, 8), color_frequence: str = 'grey', color_percentage: str = 'red', quantity: int = 10) -> None:
    wt = tokenize.WhitespaceTokenizer()
    freq_dist = FreqDist(wt.tokenize(phrase))
    df = pd.DataFrame({"Word": list(freq_dist.keys()), "Frequence": list(freq_dist.values())})

    df = df.nlargest(columns="Frequence", n=quantity) # focus on the most meaningful data
    total = df['Frequence'].sum()
    df['Percentage'] = df['Frequence'].cumsum() / total * 100 # cumulative sum to see evolution.

    logger.info("df>:")
    print(df)
    print("-----------------------")
    print("-----------------------")
    print("")
    logger.info(f"{total = }")
    print("-----------------------")
    print("-----------------------")
    print("")

    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df, x="Word", y="Frequence", color=color_frequence)
    ax.set(ylabel="Counting")
    ax2 = ax.twinx()
    sns.lineplot(data=df, x="Word", y="Percentage", color=color_percentage, sort=False, ax=ax2)
    logger.info("Showing pareto data!")
    plt.show()


def remove_stopwords(opnions: pd.core.series.Series, meaningless_words: List[str] = corpus.stopwords.words("portuguese"), wt: tokenize.WhitespaceTokenizer = tokenize.WhitespaceTokenizer()) -> List[str]:
    """
    Pre-process an opnion to be used in classification. By default use basic stopwords from nltk corpus and white space tokenize.
    """

    return [' '.join([word for word in wt.tokenize(opnion) if word not in meaningless_words]) for opnion in opnions]


if __name__ == "__main__":
    """
    Vocabulary:
    * Train - teach ML model.
    * Test - use model to our purpose.
    """

    parser = argparse.ArgumentParser(description="Alura NLP", epilog="Study only")
    parser.add_argument("action", choices=["classify", "tci", "pareto", "stopwords"], help="Available options")
    parser.add_argument("--compl", help="Complement main action")
    args = parser.parse_args()
    logger.info(f"Executing {colored(args.action, 'red', attrs=['bold', 'dark'])} with complement of {colored(args.compl, 'yellow', attrs=['bold', 'dark'])}")

    reviews = pd.read_csv("data/imdb-reviews-pt-br.csv")
    reviews["classification"] = reviews["sentiment"].replace(["neg", "pos"], [0, 1])

    if args.action == "classify":
        print(f"Classified text is: {classify_text(reviews, 'text_pt', 'classification')}")
    elif args.action == "tci":
        if args.compl == "full":
            text_cloud_image(reviews=reviews)
        elif args.compl == "pos":
            text_cloud_image(reviews=reviews.query("sentiment == 'pos'"))
        elif args.compl == "neg":
            text_cloud_image(reviews=reviews.query("sentiment == 'neg'"))
        else:
            logger.error(f"No option found. actions: {args.action} compl: {args.compl}")
    elif args.action == "pareto":
        logger.info("NLTK Process:")
        pareto(phrase=' '.join([text for text in reviews.text_pt]), quantity=20, color_frequence='yellow', color_percentage='green')
    elif args.action == "stopwords":
        reviews["removed_stopwords"] = remove_stopwords(opnions=reviews.text_pt)
        print(f"Classified text is: {classify_text(reviews, 'removed_stopwords', 'classification')}")
        pareto(phrase=' '.join([text for text in reviews.removed_stopwords]), quantity=20, color_frequence='red', color_percentage='blue')

    sys.exit(0)
