#!/usr/bin/env python3

import sys
from typing import List, Any
import argparse
from time import sleep

from termcolor import colored
import pandas as pd

from util import initialize_logger, spp
from preprocess import classify_text, text_cloud_image, pareto, remove_stopwords
from course_examples import count_values_pos_neg, classify_simple_example, focus_stopwords, remove_stopwords_by_tutor_style, tokenize_ponctuation


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
        reviews["removed_stopwords"] = remove_stopwords(opnions=reviews.text_pt)
        print(f"Classified text is: {classify_text(reviews, 'removed_stopwords', 'classification')}")
        pareto(phrase=' '.join([text for text in reviews.removed_stopwords]), quantity=20, color_frequence='red', color_percentage='blue')
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
        elif args.compl == "tokenize_ponctuation":
            print(f"tokenize_ponctuation 01: {tokenize_ponctuation()}")
            print(f"tokenize_ponctuation 02: {tokenize_ponctuation(phrase='Separando a pontuação das palavras. Passo simples!!')}")

    sys.exit(0)
