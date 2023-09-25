from time import sleep
from typing import List
from string import punctuation

import unidecode
import pandas as pd
from nltk import FreqDist, tokenize, corpus, RSLPStemmer, ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from util import get_logger_factory, spp
from preprocess import remove_stopwords


logger = get_logger_factory()


def count_values_pos_neg(reviews: pd.DataFrame) -> None:
    """Test method to show main data and classification and count
    neg/pos."""
    count_reviews = reviews.sentiment.value_counts()
    logger.info(f"Sentiment value count --> neg: {count_reviews['neg']} -- pos: {count_reviews['pos']}")

def classify_simple_example() -> None:
    """Test method to classify """

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


def focus_stopwords(reviews: pd.DataFrame) -> None:
    """Test method of stopwords."""

    focus: pd.DataFrame = reviews.iloc[:5, :]
    print(focus)
    print(".......................")
    focus["pre-processed_removed-stopwords"] = remove_stopwords(opnions=focus.text_pt)
    print(focus)
    spp.pprint(focus.iloc[2:3, 2].tolist())
    print(".......................")
    spp.pprint(focus.iloc[2:3, 5].tolist())


def remove_stopwords_by_tutor_style(opnions: pd.core.series.Series) -> List[str]:
    """
    Original implementation of tutor.
    """
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


def tokenize_punctuation(phrase: str = "Olá Mundo!") -> tokenize.WordPunctTokenizer:
    token_ponctuation = tokenize.WordPunctTokenizer()
    tf = token_ponctuation.tokenize(phrase)

    return tf


def preprocess_punctuation() -> None:
    print(punctuation)
    puncts = [punct for punct in punctuation]
    print(puncts)


def text_normatization() -> None:
    accents = "Ótimo pésssimo não é tão são vão lá no Güido ê chapéu"
    print(unidecode.unidecode(accents))
    meaningless_words = [punct for punct in punctuation] + list(set(corpus.stopwords.words("portuguese") + [unidecode.unidecode(stopwords) for stopwords in corpus.stopwords.words("portuguese")]))
    spp.pprint(sorted(meaningless_words))


def stemmer_example(words: List[str] = ["corredor", "corre", "correria"]) -> None:
    stemmer = RSLPStemmer()
    for word in words:
        print(f"Stem from **{word}** is: {stemmer.stem(word)=}")


def tfidf_example(phrases: List[str] = ["Assisti um filme ótimo", "Assisti um filme péssimo"]) -> None:
    tfidf = TfidfVectorizer(lowercase=False, max_features=50)
    characteristics = tfidf.fit_transform(phrases)
    df = pd.DataFrame(characteristics.todense(), columns=tfidf.get_feature_names_out()) # tfidf.get_feature_names() doesn't exist.
    print(df)


def ngrams_example(phrases: List[str] = ["Assisti um filme ótimo"]) -> None:
    wt = tokenize.WhitespaceTokenizer()

    for phrase in phrases:
        phrase_separated: str = wt.tokenize(phrase)
        pairs = ngrams(phrase_separated, 2)
        print(f"pairs is: {list(pairs)}")

    tfidf = TfidfVectorizer(lowercase=False, max_features=50, ngram_range=(1,2))
    characteristics = tfidf.fit_transform(phrases)
    df = pd.DataFrame(characteristics.todense(), columns=tfidf.get_feature_names_out()) # tfidf.get_feature_names() doesn't exist.
    print(df)
