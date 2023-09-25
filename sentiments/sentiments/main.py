#!/usr/bin/env python3

import sys
from typing import List, Any
import argparse
from time import sleep

from termcolor import colored
import pandas as pd
from nltk import tokenize

from util import initialize_logger, spp
from preprocess import classify_text, text_cloud_image, pareto, remove_stopwords
from course_examples import count_values_pos_neg, classify_simple_example, focus_stopwords, remove_stopwords_by_tutor_style, tokenize_punctuation, preprocess_punctuation, text_normatization, stemmer_example


if __name__ == "__main__":
    logger = initialize_logger()
    parser = argparse.ArgumentParser(description="Alura NLP", epilog="Study only")
    parser.add_argument("action", choices=["course_example", "classify", "tci", "pareto", "stopwords"], help="Available options")
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
        pareto(phrase=' '.join([text for text in reviews.text_pt]), quantity=20, color_frequence='yellow', color_percentage='green')
    elif args.action == "stopwords":
        logger.info("Removing stopwords...")
        reviews["removed_stopwords"] = remove_stopwords(opnions=reviews.text_pt, meaningless_words=None, wt=tokenize.WordPunctTokenizer())
        logger.info("Removed stopwords")
        reviews["removed_stopwords_without_lower"] = remove_stopwords(opnions=reviews.text_pt, meaningless_words=None, wt=tokenize.WordPunctTokenizer(), lower=False)
        logger.info("Removed stopwords without lower")
        reviews["removed_stopwords_without_stemmer"] = remove_stopwords(opnions=reviews.text_pt, meaningless_words=None, wt=tokenize.WordPunctTokenizer(), lower=True, stemmer=None)
        logger.info("Removed stopwords without stemmer")
        reviews["removed_stopwords_without_lower_stemmer"] = remove_stopwords(opnions=reviews.text_pt, meaningless_words=None, wt=tokenize.WordPunctTokenizer(), lower=False, stemmer=None)
        logger.info("Removed stopwords without lower and stemmer")

        print(reviews["text_pt"][0])
        print("........................................... text_pt")
        print(reviews["removed_stopwords"][0])
        print("........................................... removed_stopwords")
        print(reviews["removed_stopwords_without_lower"][0])
        print("........................................... removed_stopwords_without_lower")
        print(reviews["removed_stopwords_without_stemmer"][0])
        print("........................................... removed_stopwords_without_stemmer")
        print(reviews["removed_stopwords_without_lower_stemmer"][0])
        print("........................................... removed_stopwords_without_lower_stemmer")
        print("")
        print("")

        # FIXME WITHOUT LOWER: 0.6892033966841893; with lower: 0.6878285483218762. Why!? (no stemmer here)
        # Classified text accuracy removed_stopwords is........................: 0.6955115244642135
        # Classified text accuracy removed_stopwords_without_lower is..........: 0.6723008491710473
        # Classified text accuracy removed_stopwords_without_stemmer is........: 0.6878285483218762
        # Classified text accuracy removed_stopwords_without_lower_stemmer is..: 0.6892033966841893
        print(f"Classified text accuracy removed_stopwords is: {classify_text(reviews, 'removed_stopwords', 'classification')}")
        print(f"Classified text accuracy removed_stopwords_without_lower is: {classify_text(reviews, 'removed_stopwords_without_lower', 'classification')}")
        print(f"Classified text accuracy removed_stopwords_without_stemmer is: {classify_text(reviews, 'removed_stopwords_without_stemmer', 'classification')}")
        print(f"Classified text accuracy removed_stopwords_without_lower_stemmer is: {classify_text(reviews, 'removed_stopwords_without_lower_stemmer', 'classification')}")

        logger.info("Text cloud image for negative reviews.")
        text_cloud_image(reviews=reviews.query("sentiment == 'neg'"), column="removed_stopwords", width=800, height=500)
        logger.info("Text cloud image for posite reviews.")
        text_cloud_image(reviews=reviews.query("sentiment == 'pos'"), column="removed_stopwords", width=800, height=500)

        logger.info("Pareto representation.")
        pareto(phrase=" ".join([text for text in reviews.removed_stopwords]), quantity=10, color_frequence="purple", color_percentage="gray", wt=tokenize.WordPunctTokenizer())
    elif args.action == "course_example":
        if args.compl == "count_values_pos_neg":
            count_values_pos_neg(reviews=reviews)
        elif args.compl == "classify_simple_example":
            classify_simple_example()
        elif args.compl == "focus_stopwords":
            focus_stopwords(reviews=reviews)
        elif args.compl == "remove_stopwords_by_tutor_style":
            by_tutor = remove_stopwords_by_tutor_style(opnions=reviews.text_pt)
            by_myself = remove_stopwords(opnions=reviews.text_pt)
            assert by_tutor == by_myself
            logger.info(f"len of remove_stopwords by tutor: {len(by_tutor)} by myself: {len(by_myself)}")
        elif args.compl == "tokenize_punctuation":
            print(f"tokenize_punctuation 01: {tokenize_punctuation()}")
            print(f"tokenize_punctuation 02: {tokenize_punctuation(phrase='Separando a pontuação das palavras. Passo simples!!')}")
        elif args.compl == "preprocess_punctuation":
            preprocess_punctuation()
        elif args.compl == "text_normatization":
            text_normatization()
        elif args.compl == "stemmer_example":
            stemmer_example()

    sys.exit(0)
