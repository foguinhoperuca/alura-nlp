#!/usr/bin/env python3

import sys
from typing import List, Any
import argparse
from time import sleep
import logging

from termcolor import colored
import pandas as pd
from nltk import tokenize
from prettytable import PrettyTable, ORGMODE #, ALL
from prettytable.colortable import ColorTable, Themes

from util import initialize_logger, spp
from preprocess import classify_text, text_cloud_image, pareto, remove_stopwords, VectorizerOption
from course_examples import count_values_pos_neg, classify_simple_example, focus_stopwords, remove_stopwords_by_tutor_style, tokenize_punctuation, preprocess_punctuation, text_normatization, stemmer_example, tfidf_example, ngrams_example


if __name__ == "__main__":
    logger = initialize_logger(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Alura NLP", epilog="Study only")
    parser.add_argument("action", choices=["course_example", "classify", "tci", "pareto", "stopwords", "analysis"], help="Available options")
    parser.add_argument("--compl", help="Complement main action")
    args = parser.parse_args()
    logger.info(f"Executing {colored(args.action, 'red', attrs=['bold', 'dark'])} with complement of {colored(args.compl, 'yellow', attrs=['bold', 'dark'])}")

    reviews = pd.read_csv("data/imdb-reviews-pt-br.csv")
    reviews["classification"] = reviews["sentiment"].replace(["neg", "pos"], [0, 1])

    # FIXME ConvergenceWarning: lbfgs failed to converge (status=1):
    # STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    # Increase the number of iterations (max_iter) or scale the data as shown in: https://scikit-learn.org/stable/modules/preprocessing.html
    # Please also refer to the documentation for alternative solver options: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

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
        logger.info("Text cloud image for all reviews.")
        text_cloud_image(reviews=reviews, column="removed_stopwords", width=800, height=500)

        logger.info("Pareto representation.")
        pareto(phrase=" ".join([text for text in reviews.removed_stopwords]), quantity=10, color_frequence="purple", color_percentage="gray", wt=tokenize.WordPunctTokenizer())
    elif args.action == "analysis":
        # TODO make it a function to complet analysis. Put results in table.
        # FIXME why?! More I have preprocess worse will be the final result.
        # | Text               | Vector           | max_features=50 (tutor: 0.6988273352203801)                                                   | max_features=None (tutor: 0.885)                                                              |
        # |--------------------+------------------+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------|
        # | preprocess         | COUNT_VECTORIZER | 0.6955115244642135 (- 0.6954306510311363 unidecode) (= 0.6954306510311363 unidecode + ngrams) | 0.8693894055802669 (- 0.8685806712494946 unidecode) (+ 0.8924383340072786 unidecode + ngrams) |
        # | FULL               | TFIDF_VECTORIZER | 0.6964011322280631 (- 0.6963202587949858 unidecode) (= 0.6963202587949858 unidecode + ngrams) | 0.8822482814395471 (+ 0.8847553578649414 unidecode) (+ 0.8856449656287909 unidecode + ngrams) |
        # |--------------------+------------------+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------|
        # | preprocess WITHOUT | COUNT_VECTORIZER | 0.6892033966841893 (- 0.6887990295188031 unidecode) (- 0.678770723817226  unidecode + ngrams) | 0.8820865345733926 (+ 0.8827335220380105 unidecode) (+ 0.8959158916295997 unidecode + ngrams) |
        # | lower, stemmer     | TFIDF_VECTORIZER | 0.6878285483218762 (+ 0.6879094217549535 unidecode) (- 0.6799838253133845 unidecode + ngrams) | 0.8871815608572584 (+ 0.8882329154872625 unidecode) (- 0.8837848766680145 unidecode + ngrams) |
        # |--------------------+------------------+-----------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------|
        # | text_pt            | COUNT_VECTORIZER | 0.6583097452486858 (- 0.6503032753740396 unidecode + ngrams)                                  | 0.8814395471087748 (+ 0.8994743226849979 unidecode + ngrams)                                  |
        # | NO preprocess      | TFIDF_VECTORIZER | 0.6600889607763849 (- 0.6524868580671249 unidecode + ngrams)                                  | 0.8897695107157298 (- 0.8863728265264861 unidecode + ngrams)                                  |

        logger.info("Processing text - compare all...")
        # analysis_table: ColorTable = ColorTable(theme=Themes.OCEAN, style=ORGMODE)
        analysis_table: ColorTable = ColorTable(theme=Themes.OCEAN)
        analysis_table.set_style(ORGMODE)
        # removed_stopwords_table.field_names = [
        #     'Match',
        #     Util.white_piece('White (1)'),
        #     'Result (1)',
        #     Util.black_piece('Black (1)'),
        #     'Date (1)',
        #     Util.white_piece('White (2)'),
        #     'Result (2)',
        #     Util.black_piece('Black (2)'),
        #     'Date (2)'
        # ]
        analysis_table.field_names = [
            'Text',
            'Vector',
            'max_features=50',
            'max_features=None'
        ]
        analysis_table.add_row([
            'A',
            'B',
            'C',
            'D'
        ], divider=True)
        # ])
        analysis_table.add_row([
            'E',
            'F',
            'G',
            'H'
        ])
        analysis_table.add_row([
            'I',
            'J',
            'K',
            'L'
        ])
        # print(analysis_table)

        # FIXME Why?! My model: 0.6964011322280631 is less than tutor: 0.6988273352203801
        reviews["removed_stopwords"] = remove_stopwords(opnions=reviews.text_pt, meaningless_words=None, wt=tokenize.WordPunctTokenizer())
        print(f"Classified text accuracy removed_stopwords COUNT_VECTORIZER is: {classify_text(reviews, 'removed_stopwords', 'classification')}")                                                                                            # 0.6955115244642135 (- 0.6954306510311363 unidecode) (= 0.6954306510311363 ngrams)
        print(f"Classified text accuracy removed_stopwords TFIDF_VECTORIZER is: {classify_text(reviews, 'removed_stopwords', 'classification', vectorizer_option=VectorizerOption.TFIDF_VECTORIZER)}")                                       # 0.6964011322280631 (- 0.6963202587949858 unidecode) (= 0.6963202587949858 ngrams)
        print(f"Classified text accuracy removed_stopwords COUNT_VECTORIZER max_features=None is: {classify_text(reviews, 'removed_stopwords', 'classification', max_features=None)}")                                                       # 0.8693894055802669 (- 0.8685806712494946 unidecode) (+ 0.8924383340072786 ngrams)
        print(f"Classified text accuracy removed_stopwords TFIDF_VECTORIZER max_features=None is: {classify_text(reviews, 'removed_stopwords', 'classification', max_features=None, vectorizer_option=VectorizerOption.TFIDF_VECTORIZER)}")  # 0.8822482814395471 (+ 0.8847553578649414 unidecode) (+ 0.8856449656287909 ngrams)
        print("----------------------------------------------- removed_stopwords")
        print("")
        print("")

        # FIXME why?! For remove_stopwords using lower=False, stemmer=None --> TFIDF_VECTORIZER is worse than COUNT_VECTORIZER
        reviews["removed_stopwords_less_preprocess"] = remove_stopwords(opnions=reviews.text_pt, meaningless_words=None, wt=tokenize.WordPunctTokenizer(), lower=False, stemmer=None) # FIXME just for test purpose
        print(f"Classified text accuracy removed_stopwords_less_preprocess COUNT_VECTORIZER is: {classify_text(reviews, 'removed_stopwords_less_preprocess', 'classification')}")                                                                                            # 0.6892033966841893 (- 0.6887990295188031 unidecode) (- 0.678770723817226 ngrams)
        print(f"Classified text accuracy removed_stopwords_less_preprocess TFIDF_VECTORIZER is: {classify_text(reviews, 'removed_stopwords_less_preprocess', 'classification', vectorizer_option=VectorizerOption.TFIDF_VECTORIZER)}")                                       # 0.6878285483218762 (+ 0.6879094217549535 unidecode) (- 0.6799838253133845 ngrams)
        print(f"Classified text accuracy removed_stopwords_less_preprocess COUNT_VECTORIZER max_features=None is: {classify_text(reviews, 'removed_stopwords_less_preprocess', 'classification', max_features=None)}")                                                       # 0.8820865345733926 (+ 0.8827335220380105 unidecode) (+ 0.8959158916295997 ngrams)
        print(f"Classified text accuracy removed_stopwords_less_preprocess TFIDF_VECTORIZER max_features=None is: {classify_text(reviews, 'removed_stopwords_less_preprocess', 'classification', max_features=None, vectorizer_option=VectorizerOption.TFIDF_VECTORIZER)}")  # 0.8871815608572584 (+ 0.8882329154872625 unidecode) (- 0.8837848766680145 ngrams)
        print("----------------------------------------------- removed_stopwords_less_preprocess")
        print("")
        print("")

        print(f"Classified text accuracy text_pt COUNT_VECTORIZER is: {classify_text(reviews, 'text_pt', 'classification', vectorizer_option=VectorizerOption.COUNT_VECTORIZER)}")                                       # 0.6583097452486858 (- 0.6503032753740396 unidecode + ngrams)
        print(f"Classified text accuracy text_pt TFIDF_VECTORIZER is: {classify_text(reviews, 'text_pt', 'classification', vectorizer_option=VectorizerOption.TFIDF_VECTORIZER)}")                                       # 0.6600889607763849 (- 0.6524868580671249 unidecode + ngrams)
        print(f"Classified text accuracy text_pt COUNT_VECTORIZER max_features=None is: {classify_text(reviews, 'text_pt', 'classification', max_features=None, vectorizer_option=VectorizerOption.COUNT_VECTORIZER)}")  # 0.8814395471087748 (+ 0.8994743226849979 unidecode + ngrams)
        print(f"Classified text accuracy text_pt TFIDF_VECTORIZER max_features=None is: {classify_text(reviews, 'text_pt', 'classification', max_features=None, vectorizer_option=VectorizerOption.TFIDF_VECTORIZER)}")  # 0.8897695107157298 (- 0.8863728265264861 unidecode + ngrams)
        print("----------------------------------------------- text_pt")
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
        elif args.compl == "tfidf_example":
            tfidf_example()
            print("................................ tfidf_example()")

            phrases = [
                "Assisti um filme ótimo",
                "Assisti um filme péssimo",
                "Assisti um filme horrível",
                "Assisti um filme maravilhoso",
                "Assisti um filme excelente",
                "Assisti um filme de qualidade duvidosa"
            ]
            tfidf_example(phrases=phrases)
            print("................................ tfidf_example(phrases)")
        elif args.compl == "ngrams_example":
            ngrams_example()

            print("-------------------------------------------------- phrases (dual)")
            phrases = [
                "Assisti um filme ótimo",
                "Assisti um filme péssimo"
            ]
            ngrams_example(phrases=phrases)
            
            print("-------------------------------------------------- phrases (various)")
            phrases = [
                "Assisti um filme ótimo",
                "Assisti um filme péssimo",
                "Assisti um filme horrível",
                "Assisti um filme maravilhoso"
            ]
            ngrams_example(phrases=phrases)

            print("-------------------------------------------------- phrases (various complex)")
            phrases = [
                "Assisti um filme ótimo",
                "Assisti um filme péssimo",
                "Assisti um filme horrível",
                "Assisti um filme maravilhoso",
                "Assisti um filme excelente",
                "Assisti um filme de qualidade duvidosa"
            ]
            ngrams_example(phrases=phrases)
            

    sys.exit(0)
